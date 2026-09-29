"""Threaded frame grabber: one background thread per camera, always keeping only the latest frame."""

import logging
import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Upper bound for opening / reading a network stream, so a dead WiFi camera
# does not block the thread for FFmpeg's default ~30 s
_STREAM_TIMEOUT_MS = 5000


@dataclass(frozen=True)
class Frame:
    camera_id: str
    image: np.ndarray  # BGR
    timestamp: float   # wall-clock seconds (time.time()) when the frame was read
    index: int         # increasing per grabber; lets consumers skip frames they already processed


def _open_capture(source: int | str) -> cv2.VideoCapture:
    if isinstance(source, int):
        # DirectShow opens USB webcams much faster than the default backend on Windows
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        capture = cv2.VideoCapture(source, backend)
    else:
        capture = cv2.VideoCapture(
            source,
            cv2.CAP_FFMPEG,
            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, _STREAM_TIMEOUT_MS, cv2.CAP_PROP_READ_TIMEOUT_MSEC, _STREAM_TIMEOUT_MS],
        )
    # Keep the driver buffer small so read() returns fresh frames instead of queued ones
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


class FrameGrabber:
    """Reads a camera on its own thread; the same class handles a USB index (0) or a stream URL (RTSP).

    If the camera is unavailable or drops, the thread retries every `reconnect_interval_seconds`
    instead of crashing the program.
    """

    def __init__(self, source: int | str, camera_id: str, reconnect_interval_seconds: float = 5.0):
        self.source = source
        self.camera_id = camera_id
        self.reconnect_interval_seconds = reconnect_interval_seconds
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._next_index = 0
        self._connected = False

    def start(self) -> "FrameGrabber":
        if self.is_running:
            return self
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name=f"grabber-{self.camera_id}", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            # A read can block up to _STREAM_TIMEOUT_MS; the thread is a daemon, so don't wait forever
            self._thread.join(timeout=_STREAM_TIMEOUT_MS / 1000 + 1)
            self._thread = None

    def read_latest(self) -> Frame | None:
        """Return the most recent frame, or None if nothing has been read yet. Never blocks on the camera.

        After a disconnect this keeps returning the last frame (same index); check `is_connected`
        or skip frames whose index was already processed.
        """
        with self._lock:
            return self._latest

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_connected(self) -> bool:
        return self._connected

    def __enter__(self) -> "FrameGrabber":
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.stop()

    def _run(self) -> None:
        # The capture lives only in this thread, so stop() never releases it mid-read
        capture: cv2.VideoCapture | None = None
        warned = False
        try:
            while not self._stop_event.is_set():
                if capture is None:
                    capture = self._connect()
                    if capture is None:
                        if not warned:
                            logger.warning(
                                "%s: could not open %s, retrying every %.0f s",
                                self.camera_id, self._describe_source(), self.reconnect_interval_seconds,
                            )
                            warned = True
                        self._stop_event.wait(self.reconnect_interval_seconds)
                        continue
                    logger.info("%s: connected (%s)", self.camera_id, self._describe_source())
                    warned = False
                    self._connected = True

                ok, image = capture.read()
                if not ok:
                    logger.warning(
                        "%s: stream dropped, reconnecting every %.0f s", self.camera_id, self.reconnect_interval_seconds
                    )
                    warned = True
                    self._connected = False
                    capture.release()
                    capture = None
                    self._stop_event.wait(self.reconnect_interval_seconds)
                    continue

                frame = Frame(self.camera_id, image, time.time(), self._next_index)
                self._next_index += 1
                with self._lock:
                    self._latest = frame
        finally:
            self._connected = False
            if capture is not None:
                capture.release()

    def _connect(self) -> cv2.VideoCapture | None:
        try:
            capture = _open_capture(self.source)
        except cv2.error:
            logger.debug("%s: OpenCV error while opening source", self.camera_id, exc_info=True)
            return None
        if capture.isOpened():
            return capture
        capture.release()
        return None

    def _describe_source(self) -> str:
        if isinstance(self.source, int):
            return f"usb index {self.source}"
        # Hide credentials embedded in RTSP URLs (rtsp://user:pass@host/...)
        scheme, sep, rest = self.source.partition("://")
        return f"{scheme}{sep}***@{rest.split('@', 1)[1]}" if "@" in rest else self.source
