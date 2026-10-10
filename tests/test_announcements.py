"""Deterministic cooldown tests; no sleeps or hardware required."""

import pytest

from assistive_vision import AnnouncementPolicy, Detection


def detected(label: str, confidence: float = 0.9) -> Detection:
    return Detection(label, confidence, (0, 0, 100, 100))


def test_first_detection_is_selected() -> None:
    policy = AnnouncementPolicy(cooldown_seconds=5)
    assert policy.select_labels([detected("person")], now=0) == ("person",)


def test_repeated_frames_are_suppressed_within_cooldown() -> None:
    policy = AnnouncementPolicy(cooldown_seconds=5)
    policy.select_labels([detected("person")], now=0)
    for timestamp in (0, 1, 3, 4.999):
        assert policy.select_labels([detected("person")], now=timestamp) == ()


def test_label_is_selected_at_expiry_and_cooldown_restarts() -> None:
    policy = AnnouncementPolicy(cooldown_seconds=5)
    policy.select_labels([detected("person")], now=0)
    assert policy.select_labels([detected("person")], now=5) == ("person",)
    assert policy.select_labels([detected("person")], now=9) == ()
    assert policy.select_labels([detected("person")], now=11) == ("person",)


def test_multiple_labels_have_independent_cooldowns() -> None:
    policy = AnnouncementPolicy(cooldown_seconds=5)
    assert policy.select_labels([detected("person")], now=0) == ("person",)
    assert policy.select_labels(
        [detected("person"), detected("chair"), detected("chair")], now=2
    ) == ("chair",)
    assert policy.select_labels(
        [detected("person"), detected("chair")], now=5
    ) == ("person",)
    assert policy.select_labels([detected("chair")], now=7) == ("chair",)


def test_duplicates_in_one_frame_preserve_first_occurrence_order() -> None:
    policy = AnnouncementPolicy()
    assert policy.select_labels(
        [detected("chair"), detected("person"), detected("chair")], now=0
    ) == ("chair", "person")


def test_low_confidence_does_not_start_cooldown() -> None:
    policy = AnnouncementPolicy(min_confidence=0.5)
    assert policy.select_labels([detected("chair", 0.49)], now=0) == ()
    assert policy.select_labels([detected("chair", 0.5)], now=1) == ("chair",)


def test_empty_frame_does_not_reset_previous_cooldown() -> None:
    policy = AnnouncementPolicy(cooldown_seconds=5)
    policy.select_labels([detected("person")], now=0)
    assert policy.select_labels([], now=1) == ()
    assert policy.select_labels([detected("person")], now=2) == ()


def test_backwards_time_is_rejected() -> None:
    policy = AnnouncementPolicy()
    policy.select_labels([], now=5)
    with pytest.raises(ValueError, match="backwards"):
        policy.select_labels([], now=4)


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf")])
def test_nonfinite_time_is_rejected(timestamp: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        AnnouncementPolicy().select_labels([], now=timestamp)


@pytest.mark.parametrize("cooldown", [0, -1, float("nan"), float("inf")])
def test_invalid_cooldown_is_rejected(cooldown: float) -> None:
    with pytest.raises(ValueError, match="cooldown"):
        AnnouncementPolicy(cooldown_seconds=cooldown)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_confidence_threshold_is_rejected(threshold: float) -> None:
    with pytest.raises(ValueError, match="min_confidence"):
        AnnouncementPolicy(min_confidence=threshold)
