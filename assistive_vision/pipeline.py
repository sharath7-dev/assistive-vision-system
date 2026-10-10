"""Single background camera/inference worker with a latest-frame mailbox."""

from collections.abc import Callable
from collections import deque
from dataclasses import dataclass
from math import isfinite
from threading import Event, Lock, Thread
from time import monotonic
from typing import Any

from .announcements import AnnouncementPolicy, AnnouncementScheduler
from .camera import Webcam
from .detections import Detection
from .detector import YoloDetector
from .drawing import draw_detections
from .speech import SpeechService


@dataclass(frozen=True)
class FrameResult:
    frame: Any
    detections: tuple[Detection, ...]
    fps: float
    status: str
    speech_status: str
    sequence: int = 0
    observed_at: float = 0.0


@dataclass(frozen=True)
class PipelineState:
    camera: str
    detection: str
    model: str


class VisionPipeline:
    """UI clients call start/pause/resume/stop and poll latest().

    Frames are replaced, never queued. CPU inference and camera calls run on
    this worker. Speech runs on its own worker. A slow inference/download or
    camera driver cannot be force-cancelled; stop is observed when it returns.
    """

    def __init__(
        self, camera: Webcam | None = None, detector: YoloDetector | None = None,
        speech: SpeechService | None = None, policy: AnnouncementPolicy | None = None,
        *, renderer: Callable[[Any, tuple[Detection, ...]], Any] = draw_detections,
    ) -> None:
        self.camera = camera if camera is not None else Webcam()
        self.detector = detector if detector is not None else YoloDetector()
        self.speech = speech if speech is not None else SpeechService()
        self.policy = policy if policy is not None else AnnouncementPolicy(
            min_confidence=self.detector.confidence if isinstance(self.detector, YoloDetector) else 0.5
        )
        self.scheduler = AnnouncementScheduler(self.policy)
        self._renderer = renderer
        self._stop = Event()
        self._paused = Event()
        self._lock = Lock()
        self._thread: Thread | None = None
        self._latest: FrameResult | None = None
        self._status = "Not started"
        self._camera_state = "Closed"
        self._detection_state = "Idle"
        self._model_state = "Not loaded"
        self._pending_confidence: float | None = None
        # Speech callbacks only post receipts; scheduler state remains owned
        # by the vision worker. One pending utterance bounds this mailbox.
        self._speech_receipts: deque[tuple[bool, tuple[str, ...]]] = deque(maxlen=12)

    def _speech_started(self, labels: tuple[str, ...]) -> None:
        with self._lock:
            self._speech_receipts.append((True, labels))

    def _speech_cancelled(self, labels: tuple[str, ...]) -> None:
        with self._lock:
            self._speech_receipts.append((False, labels))

    def _drain_speech_receipts(self) -> None:
        with self._lock:
            receipts = tuple(self._speech_receipts)
            self._speech_receipts.clear()
        for started, labels in receipts:
            if started:
                self.scheduler.accepted(labels, now=monotonic())
            else:
                self.scheduler.cancelled(labels)

    @property
    def workers_running(self) -> bool:
        return self.running or self.speech.running

    def state(self) -> PipelineState:
        with self._lock:
            return PipelineState(self._camera_state, self._detection_state, self._model_state)

    def set_confidence(self, confidence: float) -> None:
        """Store the newest setting; only the worker changes detector/policy."""
        if not isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        with self._lock:
            self._pending_confidence = confidence
        self.speech.update_current((), observed_at=monotonic())

    def _apply_confidence(self) -> None:
        with self._lock:
            value, self._pending_confidence = self._pending_confidence, None
        if value is not None:
            self.detector.confidence = value
            self.policy.min_confidence = value

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def latest(self) -> FrameResult | None:
        with self._lock:
            return self._latest

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._paused.clear()
        self.scheduler.reset()
        with self._lock:
            self._speech_receipts.clear()
            self._latest = None
            self._status = "Opening webcam; first detection may load/download YOLO11n"
            self._camera_state = "Opening"
            self._detection_state = "Starting"
        self._thread = Thread(target=self._run, name="vision", daemon=True)
        self._thread.start()

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._paused.set()
            self.speech.update_current((), observed_at=monotonic())
        else:
            self._paused.clear()

    def stop(self) -> None:
        self._stop.set()
        self.speech.close()
        with self._lock:
            if self.running:
                self._status = "Stopping; waiting for camera/inference to finish"
                self._detection_state = "Stopping"

    def join(self, timeout: float = 2.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        failed = False
        sequence = 0
        try:
            self.camera.open()
            with self._lock:
                self._camera_state = "Live"
            if self._stop.is_set():
                return
            self.speech.start()
            while not self._stop.is_set():
                started = monotonic()
                self._drain_speech_receipts()
                self._apply_confidence()
                frame = self.camera.read()
                observed_at = monotonic()
                paused = self._paused.is_set()
                if not paused:
                    with self._lock:
                        if self._model_state != "Ready":
                            self._model_state = "Loading"
                            self._status = "Loading model; first use may download weights"
                detections = () if paused else self.detector.detect(frame)
                if not paused:
                    with self._lock:
                        self._model_state = "Ready"
                if self._stop.is_set():
                    break
                if self._paused.is_set():
                    paused, detections = True, ()
                self.scheduler.update(detections, now=monotonic())
                # Invalidate obsolete speech before drawing/publishing the new
                # frame, even when speech is busy or there are no detections.
                current = tuple(dict.fromkeys(d.label for d in detections
                                              if d.confidence >= self.policy.min_confidence))
                self.speech.update_current(current, observed_at=observed_at)
                annotated = self._renderer(frame, detections)
                status = "Detection paused" if paused else "Detecting (CPU)"
                elapsed = monotonic() - started
                sequence += 1
                result = FrameResult(annotated, detections, 1 / max(elapsed, 1e-6),
                                     status, self.speech.status, sequence, observed_at)
                with self._lock:
                    self._latest = result
                    self._status = status
                    self._detection_state = "Paused" if paused else "Running"
                # Publish the matching frame before submitting new speech.
                if (not paused and not self._paused.is_set() and not self._stop.is_set()
                        and monotonic() - observed_at <= 2.0 and self.speech.ready):
                    timestamp = monotonic()
                    eligible = self.scheduler.next_labels(now=timestamp)
                    if self.speech.submit(eligible, concise=True,
                                          on_started=self._speech_started,
                                          on_cancelled=self._speech_cancelled):
                        self.scheduler.reserve(eligible)
                # Paused capture must not spin at full CPU speed.
                if paused:
                    self._stop.wait(0.03)
        except Exception as exc:
            failed = True
            with self._lock:
                self._status = f"Pipeline unavailable: {exc}"
                self._detection_state = "Error"
                if self._camera_state != "Live":
                    self._camera_state = "Unavailable"
                elif self._model_state == "Loading":
                    self._model_state = "Unavailable"
        finally:
            try:
                self.camera.close()
            except Exception as exc:
                failed = True
                with self._lock:
                    self._status = f"Camera cleanup failed: {exc}"
            self.speech.close()
            with self._lock:
                if self._camera_state != "Unavailable":
                    self._camera_state = "Closed"
                if not failed:
                    self._detection_state = "Idle"
            if not failed:
                with self._lock:
                    self._status = "Stopped"
