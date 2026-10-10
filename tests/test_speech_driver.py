"""Regression coverage for the Windows event pump and model label fidelity."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from assistive_vision.announcements import AnnouncementPolicy, AnnouncementScheduler
from assistive_vision.detector import YoloDetector
from assistive_vision.drawing import draw_detections
from assistive_vision.speech import WindowsSpeechLoop, announcement_text


def driver() -> Mock:
    engine = Mock()
    callbacks = {}
    engine.connect.side_effect = lambda topic, callback: callbacks.update({topic: callback})
    engine.say.side_effect = lambda text: setattr(engine, "pending_text", text)

    def iterate() -> None:
        if isinstance(engine.pending_text, str):
            engine.pending_text = None
            callbacks["finished-utterance"](None, True)

    engine.iterate.side_effect = iterate
    engine.callbacks = callbacks
    return engine


def test_windows_pump_completes_multiple_phrases_without_restarting_loop() -> None:
    engine = driver()
    speech = WindowsSpeechLoop(engine)
    for label in ("person", "chair", "couch", "bed", "bottle"):
        speech.say(announcement_text((label,), concise=True))
        speech.runAndWait()
    assert [c.args[0] for c in engine.say.call_args_list] == [
        "Person detected.", "Chair detected.", "Couch detected.",
        "Bed detected.", "Bottle detected."]
    engine.startLoop.assert_called_once_with(False)
    engine.runAndWait.assert_not_called()
    engine.endLoop.assert_not_called()
    speech.stop()
    speech.stop()
    engine.stop.assert_called_once()
    engine.endLoop.assert_called_once()


@pytest.mark.parametrize("failure", ["cancelled", "error", "timeout"])
def test_driver_failure_is_reported_instead_of_silent_success(failure, monkeypatch) -> None:
    engine = driver()
    speech = WindowsSpeechLoop(engine)
    speech.say("Chair detected.")
    if failure == "cancelled":
        engine.iterate.side_effect = lambda: engine.callbacks["finished-utterance"](None, False)
    elif failure == "error":
        engine.iterate.side_effect = lambda: engine.callbacks["error"](None, OSError("speaker"))
    else:
        engine.iterate.side_effect = None
        monkeypatch.setattr("assistive_vision.speech.monotonic", Mock(side_effect=[0, 11]))
    with pytest.raises(RuntimeError):
        speech.runAndWait()
    speech.stop()


@pytest.mark.parametrize("label", ["cell phone", "chair", "couch", "bed", "bottle"])
def test_model_label_score_and_box_survive_drawing_and_speech(label: str) -> None:
    boxes = Mock()
    boxes.cpu.return_value = boxes
    boxes.xyxy.tolist.return_value = [[12.5, 20.25, 112.75, 220.5], [1, 1, 2, 2]]
    boxes.conf.tolist.return_value = [0.73, 0.49]
    boxes.cls.tolist.return_value = [67, 0]
    model = Mock()
    model.predict.return_value = [SimpleNamespace(boxes=boxes, names={67: label, 0: "person"})]
    detector = YoloDetector(confidence=0.5, loader=lambda _: model)
    detections = detector.detect("frame")
    assert len(detections) == 1
    actual = detections[0]
    assert (actual.label, actual.confidence, actual.bbox) == (
        label, 0.73, (12.5, 20.25, 112.75, 220.5))
    assert model.predict.call_args.kwargs["conf"] == 0.5
    cv, frame = Mock(), Mock()
    cv.getTextSize.return_value = ((90, 12), 3)
    frame.copy.return_value.shape = (480, 640, 3)
    draw_detections(frame, detections, cv=cv)
    assert cv.putText.call_args.args[1] == f"{label} 73%"
    scheduler = AnnouncementScheduler(AnnouncementPolicy(min_confidence=0.5))
    scheduler.update(detections, now=0)
    labels = scheduler.next_labels(now=0)
    assert labels == (label,)
    assert announcement_text(labels, concise=True) == f"{label.capitalize()} detected."
