"""Small, UI-independent helpers for the Tkinter dashboard."""

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite

from .detections import Detection
from .pipeline import VisionPipeline


def preview_size(source_width: int, source_height: int, width: int, height: int) -> tuple[int, int]:
    """Fit a frame without distortion, using letterboxing rather than cropping."""
    if min(source_width, source_height, width, height) <= 0:
        raise ValueError("preview dimensions must be positive")
    scale = min(width / source_width, height / source_height)
    return max(1, int(source_width * scale)), max(1, int(source_height * scale))


@dataclass(frozen=True)
class RecentDetection:
    label: str
    confidence: float
    last_seen: float


class RecentDetections:
    """At most six class labels, with ten-second expiry; not object tracking."""

    def __init__(self, limit: int = 6, expiry_seconds: float = 10.0) -> None:
        if limit <= 0 or not isfinite(expiry_seconds) or expiry_seconds <= 0:
            raise ValueError("recent-detection limits must be positive and finite")
        self.limit = limit
        self.expiry_seconds = expiry_seconds
        self._items: dict[str, RecentDetection] = {}

    def update(self, detections: Iterable[Detection], now: float) -> None:
        scores: dict[str, float] = {}
        for detection in detections:
            scores[detection.label] = max(scores.get(detection.label, 0.0), detection.confidence)
        for label, confidence in scores.items():
            self._items[label] = RecentDetection(label, confidence, now)
        newest = sorted(self._items.values(), key=lambda item: item.last_seen, reverse=True)[:self.limit]
        self._items = {item.label: item for item in newest}
        self.rows(now)

    def rows(self, now: float) -> tuple[RecentDetection, ...]:
        self._items = {label: item for label, item in self._items.items()
                       if now - item.last_seen < self.expiry_seconds}
        return tuple(sorted(self._items.values(), key=lambda item: item.last_seen, reverse=True))

    def clear(self) -> None:
        self._items.clear()


class DesktopController:
    """Own one existing pipeline; never create a second camera or model."""

    def __init__(self, pipeline: VisionPipeline, *, muted: bool = False) -> None:
        self.pipeline = pipeline
        self.paused = False
        self.muted = muted
        self.stopping = False
        self.closing = False
        self.recent = RecentDetections()
        self.pipeline.speech.set_muted(muted)

    def start(self) -> bool:
        if self.closing or self.stopping or self.pipeline.workers_running:
            return False
        self.paused = False
        self.recent.clear()
        self.pipeline.start()
        return True

    def stop(self) -> None:
        self.stopping = True
        self.pipeline.stop()

    def pause(self) -> None:
        if self.pipeline.running and not self.stopping and not self.closing:
            self.paused = not self.paused
            self.pipeline.set_paused(self.paused)

    def mute(self) -> None:
        self.muted = not self.muted
        self.pipeline.speech.set_muted(self.muted)

    def close(self) -> None:
        self.closing = True
        self.stop()

    def poll_shutdown(self) -> bool:
        """No joining on Tk's thread; wait by polling from after()."""
        if self.stopping and not self.pipeline.workers_running:
            self.stopping = False
        return self.closing and not self.pipeline.workers_running
