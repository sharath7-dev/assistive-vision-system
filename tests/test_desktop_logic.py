"""Dashboard behavior without a Tk window, webcam, model, or speaker."""

from unittest.mock import Mock

import pytest

from assistive_vision import Detection
from assistive_vision.desktop_logic import DesktopController, RecentDetections, preview_size
from assistive_vision.detector import YoloDetector
from assistive_vision.pipeline import VisionPipeline


def detected(label: str, score: float = 0.9) -> Detection:
    return Detection(label, score, (0, 0, 20, 30))


@pytest.mark.parametrize("dimensions,expected", [
    ((640, 480, 800, 400), (533, 400)),
    ((640, 480, 400, 800), (400, 300)),
    ((1920, 1080, 960, 540), (960, 540)),
])
def test_preview_letterboxing(dimensions: tuple, expected: tuple) -> None:
    assert preview_size(*dimensions) == expected


def test_invalid_preview_dimensions() -> None:
    with pytest.raises(ValueError):
        preview_size(0, 480, 800, 400)


def test_recent_labels_deduplicate_and_use_current_highest_score() -> None:
    recent = RecentDetections()
    recent.update((detected("person", 0.6), detected("person", 0.9)), now=1)
    assert len(recent.rows(1)) == 1
    assert recent.rows(1)[0].confidence == 0.9
    recent.update((detected("person", 0.7),), now=2)
    assert recent.rows(2)[0].confidence == 0.7
    assert recent.rows(2)[0].last_seen == 2


def test_recent_history_is_bounded_and_expires() -> None:
    recent = RecentDetections(limit=2, expiry_seconds=10)
    recent.update((detected("person"),), 0)
    recent.update((detected("chair"),), 1)
    recent.update((detected("cup"),), 2)
    assert [row.label for row in recent.rows(2)] == ["cup", "chair"]
    assert [row.label for row in recent.rows(11)] == ["cup"]
    assert recent.rows(12) == ()


def test_empty_frame_is_history_not_a_current_detection() -> None:
    recent = RecentDetections()
    recent.update((detected("person"),), 0)
    recent.update((), 5)
    assert recent.rows(5)[0].last_seen == 0
    assert recent.rows(10) == ()


def pipeline_mock() -> Mock:
    pipeline = Mock()
    pipeline.running = False
    pipeline.workers_running = False
    return pipeline


def test_controller_reuses_one_pipeline_and_prevents_duplicate_start() -> None:
    pipeline = pipeline_mock()
    controller = DesktopController(pipeline)
    assert controller.start()
    pipeline.workers_running = True
    assert not controller.start()
    pipeline.start.assert_called_once()
    assert controller.pipeline is pipeline


def test_controller_waits_for_both_workers_before_restart_or_exit() -> None:
    pipeline = pipeline_mock()
    pipeline.workers_running = True
    controller = DesktopController(pipeline)
    controller.stop()
    assert not controller.start()
    assert not controller.poll_shutdown()
    assert controller.stopping
    # Vision may have finished while speech still runs.
    pipeline.running = False
    assert not controller.poll_shutdown()
    pipeline.workers_running = False
    assert not controller.poll_shutdown()
    assert controller.start()
    controller.close()
    assert controller.poll_shutdown()
    assert not controller.start()
    pipeline.join.assert_not_called()


def test_pause_and_mute_controls() -> None:
    pipeline = pipeline_mock()
    controller = DesktopController(pipeline, muted=True)
    controller.pause()
    pipeline.set_paused.assert_not_called()
    pipeline.running = True
    controller.pause()
    pipeline.set_paused.assert_called_with(True)
    controller.pause()
    pipeline.set_paused.assert_called_with(False)
    controller.mute()
    pipeline.speech.set_muted.assert_called_with(False)


def test_confidence_setting_is_applied_by_worker_to_detector_and_policy() -> None:
    detector = YoloDetector(loader=Mock())
    speech = Mock()
    pipeline = VisionPipeline(detector=detector, speech=speech)
    pipeline.set_confidence(0.7)
    pipeline.set_confidence(0.8)
    assert detector.confidence == 0.5  # UI caller does not mutate model settings.
    pipeline._apply_confidence()
    assert detector.confidence == 0.8
    assert pipeline.policy.min_confidence == 0.8
    assert speech.update_current.call_args.args == ((),)
    assert speech.update_current.call_args.kwargs["observed_at"] > 0
    detector._loader.assert_not_called()


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_confidence_setting(value: float) -> None:
    with pytest.raises(ValueError):
        VisionPipeline(speech=Mock()).set_confidence(value)


def test_pipeline_reports_initial_and_camera_failure_states() -> None:
    camera, speech = Mock(), Mock()
    camera.open.side_effect = RuntimeError("camera missing")
    pipeline = VisionPipeline(camera=camera, speech=speech)
    assert pipeline.state().camera == "Closed"
    assert pipeline.state().model == "Not loaded"
    pipeline.start()
    pipeline.join()
    assert pipeline.state().camera == "Unavailable"
    assert pipeline.state().detection == "Error"


def test_loading_state_is_available_before_inference_finishes() -> None:
    camera, detector, speech = Mock(), Mock(), Mock()
    pipeline = VisionPipeline(camera, detector, speech, renderer=lambda frame, _: frame)

    def detect(frame: object) -> tuple:
        assert pipeline.state().camera == "Live"
        assert pipeline.state().model == "Loading"
        assert "Loading model" in pipeline.status
        pipeline.stop()
        return ()

    detector.detect.side_effect = detect
    pipeline.start()
    pipeline.join()
    assert pipeline.state().model == "Ready"
    assert pipeline.state().camera == "Closed"


def test_detector_readiness_does_not_load_weights() -> None:
    loader = Mock()
    detector = YoloDetector(loader=loader)
    assert not detector.ready
    loader.assert_not_called()


def test_gui_launcher_reuses_shared_pipeline_without_starting_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from types import SimpleNamespace
    from assistive_vision import __main__ as entry_point

    pipeline = Mock()
    launcher = Mock(return_value=0)
    monkeypatch.setitem(sys.modules, "cv2", Mock())
    monkeypatch.setitem(sys.modules, "assistive_vision.gui", SimpleNamespace(launch_gui=launcher))
    monkeypatch.setattr(entry_point, "VisionPipeline", Mock(return_value=pipeline))
    monkeypatch.setattr(sys, "argv", ["assistive_vision", "--gui", "--confidence", "0.7", "--mute"])
    assert entry_point.main() == 0
    launcher.assert_called_once_with(pipeline, confidence=0.7, muted=True, debug=False)
    pipeline.start.assert_not_called()
