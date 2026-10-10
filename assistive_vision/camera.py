"""OpenCV camera ownership; call from the processing worker only."""

import importlib
import sys
from typing import Any


class CameraError(RuntimeError):
    """The webcam dependency, device, or frame is unavailable."""


class Webcam:
    def __init__(self, index: int = 0, *, cv: Any = None) -> None:
        self.index = index
        self._cv = cv
        self._capture: Any = None

    def open(self) -> None:
        if self._capture is not None:
            return
        try:
            cv = self._cv or importlib.import_module("cv2")
            # DirectShow is a practical Windows webcam backend.
            backend = cv.CAP_DSHOW if sys.platform == "win32" else cv.CAP_ANY
            self._capture = cv.VideoCapture(self.index, backend)
            if not self._capture.isOpened():
                raise CameraError(
                    f"Cannot open camera {self.index}. Check Windows camera permissions, "
                    "close other camera apps, or try another --camera index."
                )
            self._capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
            self._capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        except Exception as exc:
            self.close()
            raise CameraError(f"Webcam unavailable: {exc}") from exc

    def read(self) -> Any:
        if self._capture is None:
            raise CameraError("Camera is not open")
        try:
            success, frame = self._capture.read()
            if not success or frame is None or frame.size == 0:
                raise CameraError("Camera stopped returning frames; reconnect it and restart")
            return frame
        except Exception as exc:
            raise CameraError(f"Cannot read webcam: {exc}") from exc

    def close(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()
