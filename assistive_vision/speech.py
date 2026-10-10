"""One speech worker with no FIFO backlog; busy submissions are discarded."""

from collections.abc import Callable
import sys
from math import isfinite
from threading import Condition, Thread
from time import monotonic, sleep
from typing import Any


def create_engine() -> Any:
    import pyttsx3

    engine = pyttsx3.init()
    engine.setProperty("rate", 175)
    return WindowsSpeechLoop(engine) if sys.platform == "win32" else engine


class WindowsSpeechLoop:
    """Keep SAPI's event pump alive across utterances on the speech thread.

    Repeated driver-owned runAndWait loops can cancel the next utterance in
    pyttsx3 2.99. The public external-loop API avoids resetting that driver.
    A completion callback and timeout expose failures instead of silent loss.
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._completed: bool | None = None
        self._error: Exception | None = None
        self._closed = False
        engine.connect("finished-utterance", self._finished)
        engine.connect("error", self._failed)
        engine.startLoop(False)
        engine.iterate()  # Initialize the pump before enqueueing any speech.

    def _finished(self, name: str | None, completed: bool) -> None:
        self._completed = completed

    def _failed(self, name: str | None, exception: Exception) -> None:
        self._error = exception

    def say(self, text: str) -> None:
        self._completed = None
        self._error = None
        self._engine.say(text)

    def runAndWait(self) -> None:
        deadline = monotonic() + 10.0
        while self._completed is None and self._error is None:
            if monotonic() >= deadline:
                raise RuntimeError("Windows speech completion timed out")
            self._engine.iterate()
            sleep(0.01)
        if self._error is not None:
            raise RuntimeError(f"Windows speech failed: {self._error}")
        if not self._completed:
            raise RuntimeError("Windows speech was cancelled by the audio driver")

    def stop(self) -> None:
        if not self._closed:
            self._closed = True
            self._engine.stop()
            self._engine.endLoop()


def announcement_text(labels: tuple[str, ...], concise: bool = False) -> str:
    if concise:
        return " ".join(f"{label.capitalize()} detected." for label in labels)
    return "Detected: " + ", ".join(labels) + "."


class SpeechService:
    """Create and use the audio engine on the same background thread.

    Accept at most one short utterance, with a two-second pending expiry.
    Current detections invalidate pending labels that disappeared. Freshness
    is measured from the frame observation, including inference time.
    Muting drops pending work. An utterance already playing may finish;
    pyttsx3 is never called concurrently from another thread.
    """

    def __init__(self, *, engine_factory: Callable[[], Any] = create_engine,
                 debug: bool = False) -> None:
        self._factory = engine_factory
        self._condition = Condition()
        self._thread: Thread | None = None
        self._pending: tuple[str, float] | None = None
        self._pending_labels: tuple[str, ...] = ()
        self._pending_concise = False
        self._on_started: Callable[[tuple[str, ...]], None] | None = None
        self._on_cancelled: Callable[[tuple[str, ...]], None] | None = None
        self._current_labels: frozenset[str] | None = None
        self._observed_at: float | None = None
        self._debug = debug
        self._last_request: str | None = None
        self._busy = False
        self._closed = True
        self._muted = False
        self._status = "Speech not started"

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def status(self) -> str:
        with self._condition:
            if self._closed and self._status == "Speech available":
                return "Finishing speech" if self.running else "Speech stopped"
            if not self._closed and self._status == "Speech available":
                if self._muted:
                    return "Speech muted"
                if self._busy:
                    return "Speech pending" if self._pending is not None else "Speech playing"
            return self._status

    @property
    def last_request(self) -> str | None:
        """Debug snapshot; the entry point owns all console diagnostics."""
        with self._condition:
            return self._last_request

    @property
    def ready(self) -> bool:
        with self._condition:
            return not (self._closed or self._muted or self._busy)

    def start(self) -> None:
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._closed = False
            self._pending = None
            self._pending_labels = ()
            self._on_started = self._on_cancelled = None
            self._current_labels = None
            self._observed_at = None
            self._last_request = None
            self._busy = True  # No submissions until initialization succeeds.
            self._status = "Initializing speech"
            self._thread = Thread(target=self._run, name="speech", daemon=True)
            self._thread.start()

    def set_muted(self, muted: bool) -> None:
        with self._condition:
            self._muted = muted
            if muted and self._pending is not None:
                self._cancel_labels(self._pending_labels)
                self._pending = None
                self._pending_labels = ()
                self._busy = False

    def submit(self, labels: tuple[str, ...], *, concise: bool = False,
               on_started: Callable[[tuple[str, ...]], None] | None = None,
               on_cancelled: Callable[[tuple[str, ...]], None] | None = None) -> bool:
        with self._condition:
            if not labels or self._closed or self._muted or self._busy:
                return False
            labels = tuple(dict.fromkeys(labels))[:3]
            if self._current_labels is not None:
                labels = tuple(label for label in labels if label in self._current_labels)
            timestamp = monotonic() if self._observed_at is None else self._observed_at
            if not labels or monotonic() - timestamp > 2.0:
                return False
            self._pending_labels = labels
            self._pending_concise = concise
            self._on_started = on_started
            self._on_cancelled = on_cancelled
            self._pending = (announcement_text(labels, concise), timestamp)
            self._busy = True
            self._condition.notify()
            return True

    def _cancel_labels(self, labels: tuple[str, ...]) -> None:
        if labels and self._on_cancelled is not None:
            self._on_cancelled(labels)

    def update_current(self, labels: tuple[str, ...], *, observed_at: float) -> None:
        """Refresh context every processed frame, even while speech is busy.

        Empty labels cancel pending speech on disappearance or pause. A pending
        utterance keeps its original timestamp; new frames never renew its age.
        Standalone submit() remains supported without frame context.
        """
        if not isfinite(observed_at) or observed_at > monotonic():
            raise ValueError("observed_at must be finite and must not be in the future")
        with self._condition:
            self._current_labels = frozenset(labels)
            self._observed_at = observed_at
            if self._pending is not None:
                remaining = tuple(label for label in self._pending_labels
                                  if label in self._current_labels)
                if not remaining or monotonic() - self._pending[1] > 2.0:
                    self._cancel_labels(self._pending_labels)
                    self._pending = None
                    self._pending_labels = ()
                    self._busy = False
                else:
                    self._cancel_labels(tuple(label for label in self._pending_labels
                                              if label not in remaining))
                    self._pending_labels = remaining
                    self._pending = (announcement_text(remaining, self._pending_concise),
                                     self._pending[1])
                self._condition.notify_all()

    def close(self) -> None:
        """Request shutdown without blocking a future UI thread."""
        with self._condition:
            self._closed = True
            if self._pending is not None:
                self._cancel_labels(self._pending_labels)
            self._pending = None
            self._pending_labels = ()
            self._condition.notify_all()

    def join(self, timeout: float = 2.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        engine: Any = None
        com: Any = None
        try:
            if sys.platform == "win32" and self._factory is create_engine:
                # Windows COM initialization belongs to the engine's thread.
                import pythoncom

                pythoncom.CoInitialize()
                com = pythoncom
            engine = self._factory()
            with self._condition:
                self._busy = False
                self._status = "Speech available"
                self._condition.notify_all()
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._closed or self._pending is not None)
                    if self._closed:
                        break
                    pending, self._pending = self._pending, None
                    # Validate and enqueue on the engine while holding the
                    # context lock. runAndWait is blocking and stays outside it.
                    playable = False
                    if pending is not None:
                        text, queued_at = pending
                        playable = not self._muted and monotonic() - queued_at <= 2.0
                    if playable:
                        if self._debug:
                            self._last_request = (f"{text} (source age "
                                                  f"{monotonic() - queued_at:.2f}s)")
                        engine.say(text)
                        if self._on_started is not None:
                            self._on_started(self._pending_labels)
                    else:
                        self._cancel_labels(self._pending_labels)
                    self._pending_labels = ()
                    self._on_started = self._on_cancelled = None
                if playable:
                    engine.runAndWait()
                with self._condition:
                    self._busy = False
                    self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                self._status = f"Speech unavailable: {exc} (video can continue)"
                self._closed = True
                self._pending = None
        finally:
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass
            if com is not None:
                com.CoUninitialize()
