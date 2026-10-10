"""Readable OpenCV overlays; preserve the original camera frame."""

import importlib
from collections.abc import Iterable
from typing import Any

from .detections import Detection


def draw_detections(
    frame: Any, detections: Iterable[Detection], *, cv: Any = None,
) -> Any:
    cv = cv or importlib.import_module("cv2")
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    for detection in detections:
        left, top, right, bottom = (
            round(value) for value in detection.bbox
        )
        left, right = max(0, min(left, width - 1)), max(0, min(right, width - 1))
        top, bottom = max(0, min(top, height - 1)), max(0, min(bottom, height - 1))
        cv.rectangle(annotated, (left, top), (right, bottom), (60, 220, 100), 2)
        label = f"{detection.label} {detection.confidence:.0%}"
        (text_width, text_height), baseline = cv.getTextSize(
            label, cv.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        text_x = max(0, min(left, width - text_width - 8))
        text_y = min(height - baseline - 3, max(top, text_height + 6))
        cv.rectangle(
            annotated, (text_x, text_y - text_height - 5),
            (min(width - 1, text_x + text_width + 8), text_y + baseline + 3),
            (25, 35, 25), -1,
        )
        cv.putText(annotated, label, (text_x + 4, text_y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv.LINE_AA)
    return annotated
