"""Regression tests for fair class announcements rather than person-only output."""

from unittest.mock import Mock

from assistive_vision import AnnouncementPolicy, Detection
from assistive_vision.announcements import AnnouncementScheduler
from assistive_vision.pipeline import VisionPipeline
from assistive_vision.speech import SpeechService


def objects(*labels: str, confidence: float = 0.9) -> tuple[Detection, ...]:
    return tuple(Detection(label, confidence, (0, 0, 20, 30)) for label in labels)


def request(scheduler: AnnouncementScheduler, now: float) -> tuple[str, ...]:
    labels = scheduler.next_labels(now=now)
    scheduler.accepted(labels, now=now)
    return labels


def test_simultaneous_labels_are_each_announced_once_in_order() -> None:
    scheduler = AnnouncementScheduler(AnnouncementPolicy())
    scheduler.update(objects("person", "chair", "bottle"), now=0)
    assert request(scheduler, 0) == ("person",)
    assert request(scheduler, 1) == ("chair",)
    assert request(scheduler, 2) == ("bottle",)
    assert request(scheduler, 3) == ()


def test_persistent_objects_do_not_repeat_even_after_cooldown() -> None:
    scheduler = AnnouncementScheduler(AnnouncementPolicy())
    scheduler.update(objects("person", "chair"), now=0)
    request(scheduler, 0)
    request(scheduler, 1)
    for timestamp in (8, 20, 60):
        scheduler.update(objects("person", "chair"), now=timestamp)
        assert request(scheduler, timestamp) == ()


def test_new_object_is_not_blocked_by_persistent_person() -> None:
    scheduler = AnnouncementScheduler(AnnouncementPolicy())
    scheduler.update(objects("person"), now=0)
    assert request(scheduler, 0) == ("person",)
    scheduler.update(objects("person", "chair"), now=20)
    assert request(scheduler, 20) == ("chair",)


def test_confidence_threshold_applies_to_waiting_objects() -> None:
    policy = AnnouncementPolicy(min_confidence=0.5)
    scheduler = AnnouncementScheduler(policy)
    scheduler.update(objects("person", confidence=0.49) + objects("chair", confidence=0.5), now=0)
    assert request(scheduler, 0) == ("chair",)
    assert request(scheduler, 1) == ()
    scheduler.update(objects("bottle", confidence=0.7), now=2)
    policy.min_confidence = 0.8
    scheduler.update(objects("bottle", confidence=0.7), now=3)
    assert scheduler.waiting_labels == ()
    assert request(scheduler, 3) == ()


def test_disappearance_discards_queued_objects_immediately() -> None:
    scheduler = AnnouncementScheduler(AnnouncementPolicy())
    scheduler.update(objects("person", "chair", "bottle"), now=0)
    request(scheduler, 0)
    scheduler.update(objects("person", "bottle"), now=0.1)
    assert scheduler.waiting_labels == ("bottle",)
    scheduler.update((), now=0.2)
    assert scheduler.waiting_labels == ()
    assert request(scheduler, 1) == ()


def test_reappearance_obeys_disappearance_debounce_and_cooldown() -> None:
    scheduler = AnnouncementScheduler(AnnouncementPolicy(cooldown_seconds=8))
    scheduler.update(objects("chair"), now=0)
    assert request(scheduler, 0) == ("chair",)
    scheduler.update((), now=0.1)
    scheduler.update(objects("chair"), now=0.2)
    assert request(scheduler, 20) == ()  # A brief missed frame is not a new episode.
    scheduler.update((), now=21)
    scheduler.update((), now=22)
    scheduler.update(objects("chair"), now=23)
    assert request(scheduler, 23) == ("chair",)


def test_real_reappearance_before_cooldown_waits_until_eligible() -> None:
    scheduler = AnnouncementScheduler(AnnouncementPolicy(cooldown_seconds=8))
    scheduler.update(objects("person"), now=0)
    request(scheduler, 0)
    scheduler.update((), now=0.5)
    scheduler.update(objects("person"), now=2)
    assert request(scheduler, 2) == ()
    scheduler.update(objects("person"), now=8)
    assert request(scheduler, 8) == ("person",)


def test_bounded_queue_eventually_serves_overflow_labels_fairly() -> None:
    scheduler = AnnouncementScheduler(AnnouncementPolicy(), capacity=2)
    scene = objects("person", "chair", "bottle", "cup", "book")
    spoken = []
    for timestamp in range(6):
        scheduler.update(scene, now=timestamp)
        assert len(scheduler.waiting_labels) <= 2
        spoken.extend(request(scheduler, timestamp))
    assert spoken == ["person", "chair", "bottle", "cup", "book"]


def test_rejected_request_does_not_consume_cooldown_or_queue_entry() -> None:
    policy = AnnouncementPolicy()
    scheduler = AnnouncementScheduler(policy)
    scheduler.update(objects("chair"), now=0)
    assert scheduler.next_labels(now=0) == ("chair",)
    # No accepted() call when mute/busy races reject a submission.
    assert scheduler.next_labels(now=1) == ("chair",)
    assert scheduler.waiting_labels == ("chair",)
    assert request(scheduler, 1) == ("chair",)


def test_reserved_but_cancelled_request_can_retry_without_cooldown() -> None:
    scheduler = AnnouncementScheduler(AnnouncementPolicy())
    scheduler.update(objects("chair"), now=0)
    labels = scheduler.next_labels(now=0)
    scheduler.reserve(labels)
    assert scheduler.next_labels(now=0.1) == ()
    scheduler.cancelled(labels)
    scheduler.update(objects("chair"), now=0.2)
    assert request(scheduler, 0.2) == ("chair",)
    scheduler.update(objects("chair"), now=20)
    assert request(scheduler, 20) == ()


def test_read_only_policy_selection_still_deduplicates() -> None:
    policy = AnnouncementPolicy()
    assert policy.select_labels(objects("chair", "chair"), now=0, record=False) == ("chair",)
    assert policy.select_labels(objects("chair"), now=1) == ("chair",)


def test_pipeline_collects_new_labels_while_speech_is_busy() -> None:
    camera, detector, speech = Mock(), Mock(), Mock()
    speech.submit.return_value = True
    speech.status = "Speech available"
    pipeline = VisionPipeline(camera, detector, speech, renderer=lambda frame, _: frame)
    scenes = [objects("person"), objects("person", "chair", "bottle"),
              objects("person", "chair", "bottle"), objects("person", "chair", "bottle")]
    reads = 0

    def read() -> str:
        nonlocal reads
        reads += 1
        if reads > len(scenes):
            pipeline.stop()
        speech.ready = reads != 2  # Chair/bottle first appear while audio is busy.
        return "frame"

    camera.read.side_effect = read
    detector.detect.side_effect = scenes + [()]
    pipeline._run()
    requested = [call.args[0] for call in speech.submit.call_args_list if call.args[0]]
    assert requested == [("person",), ("chair",), ("bottle",)]
    assert all(call.kwargs["concise"] for call in speech.submit.call_args_list)


def test_muted_frames_do_not_consume_new_object_announcements() -> None:
    camera, detector, speech = Mock(), Mock(), Mock()
    speech.submit.return_value = True
    speech.status = "Speech muted"
    pipeline = VisionPipeline(camera, detector, speech, renderer=lambda frame, _: frame)
    reads = 0

    def read() -> str:
        nonlocal reads
        reads += 1
        speech.ready = reads >= 3  # Unmute with the same objects still visible.
        if reads > 4:
            pipeline.stop()
        return "frame"

    camera.read.side_effect = read
    detector.detect.return_value = objects("person", "chair")
    pipeline._run()
    assert [call.args[0] for call in speech.submit.call_args_list] == [("person",), ("chair",)]


def test_speech_worker_speaks_each_short_class_phrase_asynchronously() -> None:
    engine = Mock()
    service = SpeechService(engine_factory=lambda: engine)
    service.start()
    try:
        for label in ("person", "chair", "bottle"):
            with service._condition:
                assert service._condition.wait_for(lambda: service.ready, timeout=2)
                assert service.submit((label,), concise=True)
                assert service._condition.wait_for(lambda: service.ready, timeout=2)
        assert [call.args[0] for call in engine.say.call_args_list] == [
            "Person detected.", "Chair detected.", "Bottle detected."]
        assert engine.runAndWait.call_count == 3
    finally:
        service.close()
        service.join()


def test_pending_mute_cancellation_reports_cancelled_not_spoken() -> None:
    engine, started, cancelled = Mock(), Mock(), Mock()
    service = SpeechService(engine_factory=lambda: engine)
    service.start()
    try:
        with service._condition:
            assert service._condition.wait_for(lambda: service.ready, timeout=2)
            assert service.submit(("chair",), concise=True,
                                  on_started=started, on_cancelled=cancelled)
            service.set_muted(True)
            service.set_muted(False)
            assert service.ready
        cancelled.assert_called_once_with(("chair",))
        started.assert_not_called()
        engine.say.assert_not_called()
    finally:
        service.close()
        service.join()


def test_pending_disappearance_reports_only_remaining_spoken_label() -> None:
    from time import monotonic

    engine, started, cancelled = Mock(), Mock(), Mock()
    service = SpeechService(engine_factory=lambda: engine)
    service.start()
    try:
        with service._condition:
            assert service._condition.wait_for(lambda: service.ready, timeout=2)
            service.update_current(("person", "chair"), observed_at=monotonic())
            assert service.submit(("person", "chair"), concise=True,
                                  on_started=started, on_cancelled=cancelled)
            service.update_current(("chair",), observed_at=monotonic())
            assert service._condition.wait_for(lambda: service.ready, timeout=2)
        cancelled.assert_called_once_with(("person",))
        started.assert_called_once_with(("chair",))
        engine.say.assert_called_once_with("Chair detected.")
    finally:
        service.close()
        service.join()
