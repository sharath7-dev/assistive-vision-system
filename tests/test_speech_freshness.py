"""Regression tests for disappeared labels and stale frame-derived speech."""

import argparse
from threading import Event
from time import monotonic
from unittest.mock import Mock

import pytest

from assistive_vision import AnnouncementPolicy, Detection
from assistive_vision.__main__ import confidence_value
from assistive_vision.detector import YoloDetector
from assistive_vision.pipeline import VisionPipeline
from assistive_vision.speech import SpeechService


def ready_service(engine: Mock) -> SpeechService:
    service = SpeechService(engine_factory=lambda: engine)
    service.start()
    with service._condition:
        assert service._condition.wait_for(lambda: service.ready, timeout=2)
    return service


def test_disappearing_label_cancels_pending_announcement() -> None:
    engine = Mock()
    service = ready_service(engine)
    try:
        # Hold the worker lock to simulate a delayed speech dequeue.
        with service._condition:
            service.update_current(("person",), observed_at=monotonic())
            assert service.submit(("person",))
            service.update_current((), observed_at=monotonic())
            assert service._pending is None
            assert service.ready
        engine.say.assert_not_called()
    finally:
        service.close()
        service.join()


def test_partial_disappearance_removes_only_missing_labels() -> None:
    engine = Mock()
    spoken = Event()
    engine.runAndWait.side_effect = spoken.set
    service = ready_service(engine)
    try:
        with service._condition:
            service.update_current(("person", "chair"), observed_at=monotonic())
            assert service.submit(("person", "chair"))
            service.update_current(("chair",), observed_at=monotonic())
        assert spoken.wait(2)
        engine.say.assert_called_once_with("Detected: chair.")
    finally:
        service.close()
        service.join()


def test_submit_rejects_labels_absent_from_current_frame() -> None:
    engine = Mock()
    service = ready_service(engine)
    try:
        service.update_current((), observed_at=monotonic())
        assert not service.submit(("person",))
        engine.say.assert_not_called()
    finally:
        service.close()
        service.join()


def test_inference_time_counts_towards_speech_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = Mock()
    service = ready_service(engine)
    try:
        monkeypatch.setattr("assistive_vision.speech.monotonic", lambda: 3.0)
        service.update_current(("person",), observed_at=0.0)
        assert not service.submit(("person",))
        engine.say.assert_not_called()
    finally:
        service.close()
        service.join()


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), 1.0])
def test_invalid_observation_time_is_rejected(
    monkeypatch: pytest.MonkeyPatch, timestamp: float,
) -> None:
    monkeypatch.setattr("assistive_vision.speech.monotonic", lambda: 0.0)
    with pytest.raises(ValueError, match="observed_at"):
        SpeechService().update_current(("person",), observed_at=timestamp)


def test_new_frames_do_not_extend_original_pending_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = Mock()
    service = ready_service(engine)
    clock = [0.0]
    monkeypatch.setattr("assistive_vision.speech.monotonic", lambda: clock[0])
    try:
        with service._condition:
            service.update_current(("person",), observed_at=0.0)
            assert service.submit(("person",))
            clock[0] = 1.5
            service.update_current(("person",), observed_at=1.5)
            assert service._pending[1] == 0.0
            clock[0] = 2.1
            service._condition.notify()
            assert service._condition.wait_for(lambda: service.ready, timeout=2)
        engine.say.assert_not_called()
    finally:
        service.close()
        service.join()


def test_mute_discards_pending_speech_before_playback() -> None:
    engine = Mock()
    service = ready_service(engine)
    try:
        with service._condition:
            service.update_current(("person",), observed_at=monotonic())
            assert service.submit(("person",))
            service.set_muted(True)
            service.set_muted(False)
            assert service._pending is None
        engine.say.assert_not_called()
    finally:
        service.close()
        service.join()


def test_pipeline_cancels_disappearance_even_when_speech_is_busy() -> None:
    engine = Mock()
    service = ready_service(engine)
    camera, detector = Mock(), Mock()
    person = Detection("person", 0.9, (0, 0, 20, 30))
    detector.detect.side_effect = [(person,), ()]
    pipeline = VisionPipeline(camera, detector, service, renderer=lambda frame, _: frame)
    reads = 0

    def read() -> str:
        nonlocal reads
        reads += 1
        if reads == 2:
            # Matching result was published before pending speech was submitted.
            assert pipeline.latest().detections == (person,)
            assert service._pending is not None
        elif reads == 3:
            assert pipeline.latest().detections == ()
            assert service._pending is None
            pipeline.stop()
        return "frame"

    camera.read.side_effect = read
    try:
        with service._condition:
            pipeline._run()  # Same worker logic, deterministically hold speech.
        engine.say.assert_not_called()
    finally:
        service.close()
        service.join()


def test_pause_immediately_cancels_pending_speech() -> None:
    engine = Mock()
    service = ready_service(engine)
    pipeline = VisionPipeline(speech=service)
    try:
        with service._condition:
            service.update_current(("person",), observed_at=monotonic())
            assert service.submit(("person",))
            pipeline.set_paused(True)
            assert service._pending is None
        engine.say.assert_not_called()
    finally:
        service.close()
        service.join()


def test_cooldown_never_announces_a_disappeared_label_after_expiry() -> None:
    policy = AnnouncementPolicy(cooldown_seconds=5)
    person = Detection("person", 0.9, (0, 0, 20, 30))
    assert policy.select_labels((person,), now=0) == ("person",)
    assert policy.select_labels((), now=6) == ()


def test_detector_defensively_filters_below_threshold() -> None:
    boxes = Mock()
    boxes.cpu.return_value = boxes
    boxes.xyxy.tolist.return_value = [[0, 0, 20, 30], [0, 0, 20, 30]]
    boxes.conf.tolist.return_value = [0.49, 0.5]
    boxes.cls.tolist.return_value = [0, 1]
    result = Mock(boxes=boxes, names={0: "person", 1: "chair"})
    model = Mock()
    model.predict.return_value = [result]
    detector = YoloDetector(confidence=0.5, loader=lambda _: model)
    assert detector.detect("frame") == (Detection("chair", 0.5, (0, 0, 20, 30)),)
    assert model.predict.call_args.kwargs["conf"] == 0.5


@pytest.mark.parametrize("value", ["-0.1", "1.1", "nan", "inf"])
def test_cli_rejects_invalid_confidence(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        confidence_value(value)


@pytest.mark.parametrize("value", ["0", "0.5", "0.7", "1"])
def test_cli_preserves_configured_confidence(value: str) -> None:
    assert confidence_value(value) == float(value)
