"""Single background camera/inference worker with a latest-frame mailbox."""

from collections.abc import Callable
from dataclasses import dataclass
from threading import Event, Lock, Thread
from time import monotonic
from typing import Any

from .announcements import AnnouncementPolicy
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
        self.policy = policy if policy is not None else AnnouncementPolicy()
        self._renderer = renderer
        self._stop = Event()
        self._paused = Event()
        self._lock = Lock()
        self._thread: Thread | None = None
        self._latest: FrameResult | None = None
        self._status = "Not started"

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
        with self._lock:
            self._latest = None
            self._status = "Opening webcam; first detection may load/download YOLO11n"
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

    def join(self, timeout: float = 2.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        failed = False
        sequence = 0
        try:
            self.camera.open()
            if self._stop.is_set():
                return
            self.speech.start()
            while not self._stop.is_set():
                started = monotonic()
                frame = self.camera.read()
                observed_at = monotonic()
                paused = self._paused.is_set()
                detections = () if paused else self.detector.detect(frame)
                if self._stop.is_set():
                    break
                if self._paused.is_set():
                    paused, detections = True, ()
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
                # Publish the matching frame before submitting new speech.
                if (not paused and not self._paused.is_set() and not self._stop.is_set()
                        and monotonic() - observed_at <= 2.0 and self.speech.ready):
                    eligible = self.policy.select_labels(detections, max_labels=3)
                    self.speech.submit(eligible)
                # Paused capture must not spin at full CPU speed.
                if paused:
                    self._stop.wait(0.03)
        except Exception as exc:
            failed = True
            with self._lock:
                self._status = f"Pipeline unavailable: {exc}"
        finally:
            try:
                self.camera.close()
            except Exception as exc:
                failed = True
                with self._lock:
                    self._status = f"Camera cleanup failed: {exc}"
            self.speech.close()
            if not failed:
                with self._lock:
                    self._status = "Stopped"
