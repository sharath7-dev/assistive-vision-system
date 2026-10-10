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
        self, detections: Iterable[Detection], *, now: float | None = None,
        max_labels: int | None = None,
        record: bool = True,
    ) -> tuple[str, ...]:
        """Return eligible labels in input order and start their cooldowns.

        By default use a monotonic clock, unaffected by wall-clock changes.
        Tests may provide nondecreasing timestamps in seconds instead. Keep
        the same clock source for all calls on a policy instance.
        max_labels optionally bounds an utterance; labels beyond the limit
        keep their previous cooldown state so they can be selected next time.
        record=False checks eligibility without consuming a cooldown; record
        only after the speech worker accepts the selected request.
        """
        if max_labels is not None and max_labels <= 0:
            raise ValueError("max_labels must be positive")
        timestamp = monotonic() if now is None else now
        if not isfinite(timestamp):
            raise ValueError("now must be finite")
        if self._last_call is not None and timestamp < self._last_call:
            raise ValueError("now must not move backwards")
        self._last_call = timestamp

        labels: list[str] = []
        seen: set[str] = set()
        for detection in detections:
            if detection.confidence < self.min_confidence or detection.label in seen:
                continue
            seen.add(detection.label)
            previous = self._last_announced.get(detection.label)
            if previous is None or timestamp - previous >= self.cooldown_seconds:
                labels.append(detection.label)
                if record:
                    self._last_announced[detection.label] = timestamp
                if max_labels is not None and len(labels) >= max_labels:
                    break
        return tuple(labels)


class AnnouncementScheduler:
    """Fair bounded class queue, using AnnouncementPolicy for confidence/cooldown.

    One announcement per visible episode. A class must be absent for at least
    one second before it counts as a new episode; the policy's eight-second
    cooldown still prevents rapid re-entry chatter. Missing/below-threshold
    labels leave the waiting queue immediately. Only the vision worker uses it.
    """

    def __init__(self, policy: AnnouncementPolicy, capacity: int = 6,
                 disappearance_seconds: float = 1.0) -> None:
        if capacity <= 0 or not isfinite(disappearance_seconds) or disappearance_seconds <= 0:
            raise ValueError("scheduler limits must be positive and finite")
        self.policy = policy
        self.capacity = capacity
        self.disappearance_seconds = disappearance_seconds
        self._current: dict[str, Detection] = {}
        self._waiting: dict[str, Detection] = {}
        self._reserved: dict[str, Detection] = {}
        self._last_seen: dict[str, float] = {}
        self._announced: set[str] = set()
        self._now: float | None = None

    @property
    def waiting_labels(self) -> tuple[str, ...]:
        return tuple(self._waiting)

    def reset(self) -> None:
        self._current.clear()
        self._waiting.clear()
        self._reserved.clear()
        self._last_seen.clear()
        self._announced.clear()
        self._now = None

    def update(self, detections: Iterable[Detection], *, now: float) -> None:
        if not isfinite(now) or (self._now is not None and now < self._now):
            raise ValueError("scheduler time must be finite and nondecreasing")
        self._now = now
        current: dict[str, Detection] = {}
        for detection in detections:
            if detection.confidence >= self.policy.min_confidence:
                previous = current.get(detection.label)
                if previous is None or detection.confidence > previous.confidence:
                    current[detection.label] = detection
        for label, last_seen in tuple(self._last_seen.items()):
            if label not in self._current and now - last_seen >= self.disappearance_seconds:
                self._announced.discard(label)
                del self._last_seen[label]
        self._current = current
        # Refresh queued detection scores but retain arrival order. Never keep
        # a disappeared class just because an earlier frame queued it.
        self._waiting = {label: current[label] for label in self._waiting if label in current}
        for label, detection in current.items():
            self._last_seen[label] = now
            if label not in self._announced and label not in self._waiting and label not in self._reserved:
                if len(self._waiting) < self.capacity:
                    self._waiting[label] = detection

    def next_labels(self, *, now: float) -> tuple[str, ...]:
        return self.policy.select_labels(self._waiting.values(), now=now,
                                         max_labels=1, record=False)

    def accepted(self, labels: tuple[str, ...], *, now: float) -> None:
        """Record actual engine handoff, not acceptance into the pending slot."""
        for label in labels:
            detection = self._reserved.pop(label, None)
            if detection is None:
                detection = self._waiting.pop(label, None)
            if detection is not None:
                self.policy.select_labels((detection,), now=now)
                self._announced.add(label)

    def reserve(self, labels: tuple[str, ...]) -> None:
        for label in labels:
            detection = self._waiting.pop(label, None)
            if detection is not None:
                self._reserved[label] = detection

    def cancelled(self, labels: tuple[str, ...]) -> None:
        """A mute/expiry/disappearance cancellation consumes no cooldown."""
        for label in labels:
            self._reserved.pop(label, None)
