"""Threaded frame grabber: one background thread per camera, always keeping only the latest frame."""

import logging
import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Frame:
    camera_id: str
    image: np.ndarray  # BGR
    timestamp: float   # wall-clock seconds (time.time()) when the frame was read
    index: int         # increasing per grabber; lets consumers skip frames they already processed


def _open_capture(source: int | str) -> cv2.VideoCapture:
    if isinstance(source, int) and sys.platform == "win32":
        # DirectShow opens USB webcams much faster than the default backend on Windows
        capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        capture = cv2.VideoCapture(source)
    # Keep the driver buffer small so read() returns fresh frames instead of queued ones
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


class FrameGrabber:
    """Reads a camera on its own thread; the same class handles a USB index (0) or a stream URL (RTSP)."""

    def __init__(self, source: int | str, camera_id: str):
        self.source = source
        self.camera_id = camera_id
        self._capture: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._next_index = 0

    def start(self) -> "FrameGrabber":
        if self.is_running:
            return self
        self._capture = _open_capture(self.source)
        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            raise RuntimeError(f"{self.camera_id}: could not open source {self._describe_source()}")

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name=f"grabber-{self.camera_id}", daemon=True)
        self._thread.start()
        logger.info("%s: capture started (%s)", self.camera_id, self._describe_source())
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def read_latest(self) -> Frame | None:
        """Return the most recent frame, or None if nothing has been read yet. Never blocks on the camera."""
        with self._lock:
            return self._latest

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def __enter__(self) -> "FrameGrabber":
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.stop()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            ok, image = self._capture.read()
            if not ok:
                # Reconnection is handled in task 2.2; for now the grabber just stops
                logger.warning("%s: stream returned no frame, stopping capture", self.camera_id)
                break
            frame = Frame(self.camera_id, image, time.time(), self._next_index)
            self._next_index += 1
            with self._lock:
                self._latest = frame

    def _describe_source(self) -> str:
        if isinstance(self.source, int):
            return f"usb index {self.source}"
        # Hide credentials embedded in RTSP URLs (rtsp://user:pass@host/...)
        scheme, sep, rest = self.source.partition("://")
        return f"{scheme}{sep}***@{rest.split('@', 1)[1]}" if "@" in rest else self.source
