"""Rate limiting uses a fake clock, without webcam or audio dependencies."""

from threading import Event
import sys
from unittest.mock import Mock

import pytest

from assistive_vision import Detection
from assistive_vision.diagnostics import DebugReporter
from assistive_vision.pipeline import FrameResult
from assistive_vision.speech import SpeechService
from assistive_vision import __main__ as entry_point


def frame(sequence: int, labels: tuple[str, ...] = ("person",)) -> FrameResult:
    return FrameResult(None, tuple(Detection(label, 0.9, (0, 0, 20, 30))
                                   for label in labels), 30, "Detecting (CPU)",
                       "Speech available", sequence, 0)


def test_normal_mode_does_not_log_or_read_clock() -> None:
    output, clock = Mock(), Mock()
    DebugReporter(clock=clock, output=output).report(
        frame(1), status="Detecting", speech_status="Speech available")
    output.assert_not_called()
    clock.assert_not_called()


def test_high_frame_rate_produces_only_one_summary_per_second() -> None:
    output = Mock()
    clock = [0.0]
    reporter = DebugReporter(True, clock=lambda: clock[0], output=output)
    for sequence in range(61):
        clock[0] = sequence / 30
        reporter.report(frame(sequence), status="Detecting", speech_status="Speech playing")
    assert output.call_count == 3  # t=0, 1, 2, regardless of changing frame numbers.
    assert "person 0.90" in output.call_args.args[0]
    assert "Speech playing" in output.call_args.args[0]


def test_interval_boundary_and_latest_empty_frame() -> None:
    output = Mock()
    clock = [0.0]
    reporter = DebugReporter(True, clock=lambda: clock[0], output=output)
    reporter.report(frame(1), status="Detecting", speech_status="Speech available")
    clock[0] = 0.999
    reporter.report(frame(2, ()), status="Detecting", speech_status="Speech available")
    assert output.call_count == 1
    clock[0] = 1.0
    reporter.report(frame(3, ()), status="Detecting", speech_status="Speech muted")
    assert output.call_count == 2
    assert "detections: none" in output.call_args.args[0]
    assert "person" not in output.call_args.args[0]
    assert "Speech muted" in output.call_args.args[0]


def test_report_without_frame_and_last_request() -> None:
    output = Mock()
    reporter = DebugReporter(True, output=output, clock=lambda: 1)
    reporter.report(None, status="Opening camera", speech_status="Initializing speech",
                    last_request="Detected: chair.")
    assert "no frame" in output.call_args.args[0]
    assert "last request: Detected: chair." in output.call_args.args[0]


def test_crowded_scene_aggregates_labels_and_bounds_summary() -> None:
    output = Mock()
    reporter = DebugReporter(True, output=output)
    reporter.report(frame(1, ("person", "person", "chair", "cup", "book", "car", "dog")),
                    status="Detecting", speech_status="Speech available")
    line = output.call_args.args[0]
    assert line.count("person 0.90") == 1
    assert "+1 labels" in line


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_invalid_interval(interval: float) -> None:
    with pytest.raises(ValueError, match="interval_seconds"):
        DebugReporter(interval_seconds=interval)


def test_debug_speech_keeps_snapshot_without_printing(capsys: pytest.CaptureFixture) -> None:
    engine = Mock()
    entered, release = Event(), Event()

    def speak() -> None:
        entered.set()
        assert release.wait(2)

    engine.runAndWait.side_effect = speak
    service = SpeechService(engine_factory=lambda: engine, debug=True)
    service.start()
    try:
        with service._condition:
            assert service._condition.wait_for(lambda: service.ready, timeout=2)
        assert service.submit(("person",))
        assert entered.wait(2)
        assert service.status == "Speech playing"
        assert "Detected: person." in service.last_request
        assert capsys.readouterr().out == ""
    finally:
        service.close()
        release.set()
        service.join()


@pytest.mark.parametrize("debug,lines", [(False, 0), (True, 1)])
def test_entry_point_quiet_normal_mode_and_sampled_debug(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
    debug: bool, lines: int,
) -> None:
    pipeline = Mock()
    pipeline.running = True
    pipeline.status = "Detecting (CPU)"
    pipeline.latest.return_value = None
    pipeline.speech.status = "Speech available"
    pipeline.speech.last_request = None
    cv = Mock()
    cv.waitKey.return_value = ord("q")
    monkeypatch.setitem(sys.modules, "cv2", cv)
    monkeypatch.setattr(entry_point, "VisionPipeline", Mock(return_value=pipeline))
    monkeypatch.setattr(sys, "argv", ["assistive_vision"] + (["--debug"] if debug else []))
    assert entry_point.main() == 0
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == lines
    assert captured.err == ""
    pipeline.stop.assert_called_once()
