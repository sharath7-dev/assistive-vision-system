"""Console diagnostics sampled by time, independently of webcam frame rate."""

from collections.abc import Callable
from math import isfinite
from time import monotonic

from .pipeline import FrameResult


class DebugReporter:
    """Print at most one summary per interval; never queue frame logs."""

    def __init__(self, enabled: bool = False, interval_seconds: float = 1.0,
                 *, clock: Callable[[], float] = monotonic,
                 output: Callable[[str], None] = print) -> None:
        if not isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be finite and positive")
        self.enabled = enabled
        self.interval_seconds = interval_seconds
        self._clock = clock
        self._output = output
        self._last_report: float | None = None

    def report(self, result: FrameResult | None, *, status: str,
               speech_status: str, last_request: str | None = None) -> None:
        if not self.enabled:
            return
        now = self._clock()
        if self._last_report is not None and now - self._last_report < self.interval_seconds:
            return
        self._last_report = now
        if result is None:
            frame_info = "no frame"
            objects = "none"
        else:
            scores: dict[str, float] = {}
            for detection in result.detections:
                scores[detection.label] = max(scores.get(detection.label, 0.0),
                                              detection.confidence)
            # Summarize a crowded scene without an unbounded console line.
            objects = ", ".join(f"{label} {score:.2f}"
                                for label, score in list(scores.items())[:5]) or "none"
            if len(scores) > 5:
                objects += f", +{len(scores) - 5} labels"
            frame_info = (f"frame {result.sequence}, {result.fps:.1f} FPS, "
                          f"age {max(0.0, now - result.observed_at):.2f}s")
        request = f" | last request: {last_request}" if last_request else ""
        self._output(f"[debug] {status} | {frame_info} | detections: {objects} | "
                     f"{speech_status}{request}")
