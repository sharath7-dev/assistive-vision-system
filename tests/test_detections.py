"""Validate the contract that future model adapters must satisfy."""

import pytest

from assistive_vision import Detection


def test_valid_detection_preserves_values() -> None:
    detection = Detection("person", 0.85, (10.0, 20.0, 110.0, 220.0))
    assert detection.label == "person"
    assert detection.confidence == 0.85
    assert detection.bbox == (10.0, 20.0, 110.0, 220.0)


def test_label_whitespace_is_normalized() -> None:
    assert Detection(" person ", 1.0, (0, 0, 1, 1)).label == "person"


@pytest.mark.parametrize("confidence", [0.0, 1.0])
def test_confidence_endpoints_are_valid(confidence: float) -> None:
    assert Detection("chair", confidence, (0, 0, 10, 20)).confidence == confidence


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_confidence_is_rejected(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        Detection("chair", confidence, (0, 0, 10, 20))


@pytest.mark.parametrize("label", ["", "   "])
def test_blank_label_is_rejected(label: str) -> None:
    with pytest.raises(ValueError, match="label"):
        Detection(label, 0.9, (0, 0, 10, 20))


@pytest.mark.parametrize(
    "bbox",
    [(-1, 0, 10, 20), (10, 0, 0, 20), (0, 0, 0, 20),
     (0, 20, 10, 20), (0, 0, float("nan"), 20), (0, 0, 10, float("inf"))],
)
def test_invalid_boxes_are_rejected(bbox: tuple[float, float, float, float]) -> None:
    with pytest.raises(ValueError, match="bbox"):
        Detection("chair", 0.9, bbox)
