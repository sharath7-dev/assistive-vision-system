"""Per-label announcement selection without camera, model, or audio imports."""

from collections.abc import Iterable
from math import isfinite
from time import monotonic

from .detections import Detection


class AnnouncementPolicy:
    """Select labels once per cooldown, including deduplication within a frame.

    Cooldowns are per class label, not per physical object: two people produce
    one 'person' label. Labels are case-sensitive, as supplied by the model.
    Selection records an announcement immediately. Call only when speech is
    enabled and ready to accept it; this component does not confirm playback.
    Intended for use by one processing thread.
    """

    def __init__(
        self, cooldown_seconds: float = 8.0, min_confidence: float = 0.5
    ) -> None:
        if not isfinite(cooldown_seconds) or cooldown_seconds <= 0:
            raise ValueError("cooldown_seconds must be finite and positive")
        if not isfinite(min_confidence) or not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        self.cooldown_seconds = cooldown_seconds
        self.min_confidence = min_confidence
        self._last_announced: dict[str, float] = {}
        self._last_call: float | None = None

    def select_labels(
        self, detections: Iterable[Detection], *, now: float | None = None
    ) -> tuple[str, ...]:
        """Return eligible labels in input order and start their cooldowns.

        By default use a monotonic clock, unaffected by wall-clock changes.
        Tests may provide nondecreasing timestamps in seconds instead. Keep
        the same clock source for all calls on a policy instance.
        """
        timestamp = monotonic() if now is None else now
        if not isfinite(timestamp):
            raise ValueError("now must be finite")
        if self._last_call is not None and timestamp < self._last_call:
            raise ValueError("now must not move backwards")
        self._last_call = timestamp

        labels: list[str] = []
        for detection in detections:
            if detection.confidence < self.min_confidence:
                continue
            previous = self._last_announced.get(detection.label)
            if previous is None or timestamp - previous >= self.cooldown_seconds:
                labels.append(detection.label)
                self._last_announced[detection.label] = timestamp
        return tuple(labels)
