"""Lazy YOLO11n adapter; returns the Phase 1 Detection data structure."""

from collections.abc import Callable
from math import isfinite
import os
from pathlib import Path
from typing import Any

from .detections import Detection


class DetectorError(RuntimeError):
    """Model dependencies, weights, or inference are unavailable."""


def load_yolo(weights: str) -> Any:
    # Import and possible weight download happen only on the first detection.
    config_dir = os.environ.setdefault("YOLO_CONFIG_DIR", str(
        Path(__file__).resolve().parent.parent / ".local" / "ultralytics"
    ))
    Path(config_dir).expanduser().mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    return YOLO(weights)


class YoloDetector:
    def __init__(
        self, weights: str = "yolo11n.pt", confidence: float = 0.5,
        image_size: int = 416, *, loader: Callable[[str], Any] = load_yolo,
    ) -> None:
        if not weights.strip():
            raise ValueError("weights must not be blank")
        if not isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if image_size <= 0 or image_size % 32:
            raise ValueError("image_size must be a positive multiple of 32")
        self.weights = weights
        self.confidence = confidence
        self.image_size = image_size
        self._loader = loader
        self._model: Any = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    def detect(self, frame: Any) -> tuple[Detection, ...]:
        if self._model is None:
            try:
                self._model = self._loader(self.weights)
            except Exception as exc:
                raise DetectorError(
                    f"Cannot load {self.weights}: {exc}. Install requirements and check "
                    "internet access for the initial download, or supply --weights "
                    "with an existing local detection model."
                ) from exc
        try:
            result = self._model.predict(
                source=frame, conf=self.confidence, imgsz=self.image_size,
                device="cpu", verbose=False, save=False,
            )[0]
            if result.boxes is None:
                return ()
            boxes = result.boxes.cpu()
            detections: list[Detection] = []
            for coordinates, score, class_id in zip(
                boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist(), strict=True
            ):
                detection = Detection(
                    label=result.names[int(class_id)], confidence=float(score),
                    bbox=tuple(float(value) for value in coordinates),
                )
                # Defend the adapter contract even if a model/backend returns
                # a score below the requested predict threshold.
                if detection.confidence >= self.confidence:
                    detections.append(detection)
            return tuple(detections)
        except Exception as exc:
            raise DetectorError(f"Object detection failed: {exc}") from exc
