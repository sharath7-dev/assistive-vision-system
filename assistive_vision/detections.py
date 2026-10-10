"""Validated object-detection results, independent of any model library."""

from dataclasses import dataclass
from math import isfinite

BoundingBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    """One object in a frame; bbox is (left, top, right, bottom) in pixels.

    Coordinates refer to the original image, with the origin at its top left.
    Confidence is a score between zero and one, not a guarantee of accuracy.
    """

    label: str
    confidence: float
    bbox: BoundingBox

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("label must not be blank")
        object.__setattr__(self, "label", self.label.strip())
        if not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be finite and between 0 and 1")
        if len(self.bbox) != 4:
            raise ValueError("bbox must contain four coordinates")
        if any(not isfinite(value) or value < 0 for value in self.bbox):
            raise ValueError("bbox coordinates must be finite and nonnegative")
        left, top, right, bottom = self.bbox
        if right <= left or bottom <= top:
            raise ValueError("bbox must have positive width and height")
