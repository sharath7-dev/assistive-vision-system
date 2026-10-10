"""Hardware-free adapter and concurrency tests; no real engines or models."""

from threading import Event
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from assistive_vision import AnnouncementPolicy, Detection
from assistive_vision.camera import CameraError, Webcam
from assistive_vision.detector import DetectorError, YoloDetector, load_yolo
from assistive_vision.drawing import draw_detections
from assistive_vision.pipeline import VisionPipeline
from assistive_vision.speech import SpeechService


def detection(label: str = "person") -> Detection:
    return Detection(label, 0.9, (0, 0, 20, 30))


def test_camera_open_read_and_release() -> None:
    cv = Mock()
    frame = SimpleNamespace(size=100)
    capture = cv.VideoCapture.return_value
    capture.isOpened.return_value = True
    capture.read.return_value = (True, frame)
    camera = Webcam(cv=cv)
    camera.open()
    assert camera.read() is frame
    camera.close()
    camera.close()
    capture.release.assert_called_once()


def test_camera_open_failure_releases_device() -> None:
    cv = Mock()
    cv.VideoCapture.return_value.isOpened.return_value = False
    camera = Webcam(2, cv=cv)
    with pytest.raises(CameraError, match="Cannot open camera 2"):
        camera.open()
    cv.VideoCapture.return_value.release.assert_called_once()


@pytest.mark.parametrize("success,frame", [(False, None), (True, None),
                                                    (True, SimpleNamespace(size=0))])
def test_camera_frame_failure(success: bool, frame: object) -> None:
    cv = Mock()
    cv.VideoCapture.return_value.read.return_value = (success, frame)
    camera = Webcam(cv=cv)
    camera.open()
    with pytest.raises(CameraError, match="Cannot read webcam"):
        camera.read()
    camera.close()


def test_read_before_open_is_clear_error() -> None:
    with pytest.raises(CameraError, match="not open"):
        Webcam().read()


def test_camera_dependency_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("assistive_vision.camera.importlib.import_module",
                        Mock(side_effect=ImportError("no cv2")))
    with pytest.raises(CameraError, match="no cv2"):
        Webcam().open()


def model_with_boxes() -> Mock:
    boxes = Mock()
    boxes.cpu.return_value = boxes
    boxes.xyxy.tolist.return_value = [[1, 2, 21, 32]]
    boxes.conf.tolist.return_value = [0.9]
    boxes.cls.tolist.return_value = [0]
    model = Mock()
    model.predict.return_value = [SimpleNamespace(boxes=boxes, names={0: "person"})]
    return model


def test_detector_is_lazy_reuses_model_and_maps_results() -> None:
    model = model_with_boxes()
    loader = Mock(return_value=model)
    detector = YoloDetector(loader=loader)
    loader.assert_not_called()
    result = detector.detect("frame")
    assert result == (Detection("person", 0.9, (1, 2, 21, 32)),)
    detector.detect("next frame")
    loader.assert_called_once_with("yolo11n.pt")
    assert model.predict.call_args.kwargs["device"] == "cpu"
    assert model.predict.call_args.kwargs["save"] is False


def test_loader_prepares_config_directory_before_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    directory = tmp_path / "local-config"
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(directory))
    constructor = Mock(return_value="fake model")
    monkeypatch.setitem(__import__("sys").modules, "ultralytics",
                        SimpleNamespace(YOLO=constructor))
    assert load_yolo("local.pt") == "fake model"
    assert directory.is_dir()
    constructor.assert_called_once_with("local.pt")


def test_detector_empty_result() -> None:
    model = Mock()
    model.predict.return_value = [SimpleNamespace(boxes=None)]
    assert YoloDetector(loader=lambda _: model).detect("frame") == ()


def test_missing_model_is_clear_error() -> None:
    loader = Mock(side_effect=OSError("weights unavailable"))
    with pytest.raises(DetectorError, match="Cannot load yolo11n.pt"):
        YoloDetector(loader=loader).detect("frame")


def test_inference_error_is_wrapped() -> None:
    model = Mock()
    model.predict.side_effect = RuntimeError("bad frame")
    with pytest.raises(DetectorError, match="Object detection failed"):
        YoloDetector(loader=lambda _: model).detect("frame")


@pytest.mark.parametrize("kwargs", [
    {"weights": " "}, {"confidence": -0.1}, {"confidence": float("nan")},
    {"image_size": 0}, {"image_size": 415},
])
def test_invalid_detector_settings(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        YoloDetector(**kwargs)


def test_drawing_copies_frame_and_includes_confidence() -> None:
    cv = Mock()
    cv.getTextSize.return_value = ((90, 12), 3)
    frame = Mock()
    frame.copy.return_value.shape = (480, 640, 3)
    drawn = draw_detections(frame, [detection()], cv=cv)
    assert drawn is frame.copy.return_value
    assert cv.putText.call_args.args[1] == "person 90%"
    assert cv.rectangle.call_count == 2


def test_announcement_limit_does_not_consume_unsaid_labels() -> None:
    policy = AnnouncementPolicy()
    objects = [detection(label) for label in ("person", "chair", "cup", "book")]
    assert policy.select_labels(objects, now=0, max_labels=3) == ("person", "chair", "cup")
    assert policy.select_labels(objects, now=1, max_labels=3) == ("book",)


def test_invalid_announcement_limit() -> None:
    with pytest.raises(ValueError, match="max_labels"):
        AnnouncementPolicy().select_labels([], max_labels=0)


def ready_speech(engine: Mock) -> SpeechService:
    service = SpeechService(engine_factory=lambda: engine)
    service.start()
    # Condition waiting is deterministic and bounded, unlike sleep-based tests.
    with service._condition:
        assert service._condition.wait_for(lambda: service.ready, timeout=2)
    return service


def test_speech_async_drops_busy_submissions_and_stops() -> None:
    entered, release = Event(), Event()
    engine = Mock()

    def speak() -> None:
        entered.set()
        assert release.wait(2)

    engine.runAndWait.side_effect = speak
    service = ready_speech(engine)
    try:
        assert service.submit(("person",))
        assert entered.wait(2)
        assert not service.submit(("chair",))
        assert not service.ready
    finally:
        service.close()
        release.set()
        service.join()
    engine.say.assert_called_once_with("Detected: person.")
    engine.stop.assert_called_once()


def test_muted_speech_rejects_submissions() -> None:
    engine = Mock()
    service = ready_speech(engine)
    try:
        service.set_muted(True)
        assert not service.ready
        assert not service.submit(("person",))
        service.set_muted(False)
        assert service.ready
    finally:
        service.close()
        service.join()
    engine.say.assert_not_called()


def test_speech_initialization_failure_does_not_escape_thread() -> None:
    service = SpeechService(engine_factory=Mock(side_effect=RuntimeError("no audio")))
    service.start()
    service.join()
    assert "Speech unavailable: no audio" in service.status
    assert not service.ready


def test_speech_playback_failure_disables_speech() -> None:
    engine = Mock()
    engine.runAndWait.side_effect = RuntimeError("speaker failed")
    service = ready_speech(engine)
    assert service.submit(("person",))
    service.join()
    assert "speaker failed" in service.status
    assert not service.ready


def test_expired_speech_is_discarded(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = Mock()
    service = ready_speech(engine)
    try:
        with service._condition:
            service._pending = ("old announcement", 0)
            service._busy = True
            monkeypatch.setattr("assistive_vision.speech.monotonic", lambda: 3.0)
            service._condition.notify()
            assert service._condition.wait_for(lambda: not service._busy, timeout=2)
        engine.say.assert_not_called()
    finally:
        service.close()
        service.join()


def test_pipeline_camera_failure_preserves_error_and_cleans_up() -> None:
    camera, detector, speech = Mock(), Mock(), Mock()
    camera.open.side_effect = CameraError("camera missing")
    pipeline = VisionPipeline(camera, detector, speech)
    pipeline.start()
    pipeline.join()
    assert not pipeline.running
    assert "camera missing" in pipeline.status
    camera.close.assert_called_once()
    detector.detect.assert_not_called()
    speech.start.assert_not_called()


def test_pipeline_detection_failure_cleans_up() -> None:
    camera, detector, speech = Mock(), Mock(), Mock()
    detector.detect.side_effect = DetectorError("model missing")
    pipeline = VisionPipeline(camera, detector, speech)
    pipeline.start()
    pipeline.join()
    assert "model missing" in pipeline.status
    camera.close.assert_called_once()
    speech.close.assert_called_once()


def test_pipeline_publishes_latest_fps_and_reuses_policy() -> None:
    camera, detector, speech = Mock(), Mock(), Mock()
    speech.ready = True
    speech.status = "Speech available"
    detector.detect.return_value = (detection(),)
    policy = AnnouncementPolicy()
    rendered = Event()
    second_frame = Event()
    reads = 0

    def read() -> str:
        nonlocal reads
        reads += 1
        if reads == 2:
            assert second_frame.wait(2)
        if reads > 2:
            pipeline.stop()
        return "frame"

    def render(frame: str, objects: tuple[Detection, ...]) -> str:
        rendered.set()
        return "annotated"

    camera.read.side_effect = read
    pipeline = VisionPipeline(camera, detector, speech, policy, renderer=render)
    try:
        pipeline.start()
        assert rendered.wait(2)
        second_frame.set()
        pipeline.join()
        result = pipeline.latest()
        assert result is not None
        assert result.frame == "annotated"
        assert result.detections == (detection(),)
        assert result.fps > 0
        assert speech.submit.call_args_list[0].args == (("person",),)
        assert speech.submit.call_args_list[1].args == ((),)
        assert pipeline.status == "Stopped"
    finally:
        second_frame.set()
        pipeline.stop()
        pipeline.join()


def test_paused_pipeline_skips_detection_and_speech() -> None:
    camera, detector, speech = Mock(), Mock(), Mock()
    pipeline = VisionPipeline(camera, detector, speech, renderer=lambda f, d: f)

    def read() -> str:
        pipeline.set_paused(True)
        pipeline._stop.set() if pipeline.latest() is not None else None
        return "frame"

    camera.read.side_effect = read
    pipeline.start()
    pipeline.join()
    assert pipeline.latest().status == "Detection paused"
    detector.detect.assert_not_called()
    speech.submit.assert_not_called()


def test_stop_returns_while_inference_is_busy() -> None:
    entered, release = Event(), Event()
    camera, detector, speech = Mock(), Mock(), Mock()

    def detect(frame: object) -> tuple[Detection, ...]:
        entered.set()
        assert release.wait(2)
        return (detection(),)

    detector.detect.side_effect = detect
    pipeline = VisionPipeline(camera, detector, speech, renderer=Mock())
    try:
        pipeline.start()
        assert entered.wait(2)
        pipeline.stop()
        assert pipeline.running  # stop did not wait for the native call.
    finally:
        release.set()
        pipeline.join()
    assert not pipeline.running
    camera.close.assert_called_once()
    pipeline._renderer.assert_not_called()


def test_speech_unavailable_does_not_stop_video() -> None:
    camera, detector, speech = Mock(), Mock(), Mock()
    speech.ready = False
    speech.status = "Speech unavailable"
    detector.detect.return_value = (detection(),)
    pipeline = VisionPipeline(camera, detector, speech, renderer=lambda f, d: f)

    def read() -> str:
        if pipeline.latest() is not None:
            pipeline.stop()
        return "frame"

    camera.read.side_effect = read
    pipeline.start()
    pipeline.join()
    assert pipeline.latest().detections == (detection(),)
    assert pipeline.latest().speech_status == "Speech unavailable"
    assert pipeline.status == "Stopped"
    speech.submit.assert_not_called()
