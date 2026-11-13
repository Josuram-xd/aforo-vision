"""On-demand camera snapshots (ARCHITECTURE.md 10.6, task 22).

Every 3 s the worker asks Supabase for pending snapshot requests (an admin pressed the button in the web).
For each one it sends ONE still JPEG (max 640 px wide) of the latest frame, or null if the camera is not
connected or its last frame is older than 5 s. Images are never written to disk or to logs.
"""

import base64
import logging
import threading
import time
from collections.abc import Callable, Mapping

import cv2
import httpx
import numpy as np

logger = logging.getLogger(__name__)

PENDING_PATH = "/rest/v1/rpc/pending_snapshot_requests"
SUBMIT_PATH = "/rest/v1/rpc/submit_camera_snapshot"
POLL_INTERVAL_SECONDS = 3.0
MAX_FRAME_AGE_SECONDS = 5.0
MAX_WIDTH = 640
JPEG_QUALITY = 70


def encode_snapshot(image: np.ndarray) -> str | None:
    """Downscale to MAX_WIDTH (keeping the aspect ratio) and return base64 JPEG, or None if encoding fails."""
    height, width = image.shape[:2]
    if width > MAX_WIDTH:
        image = cv2.resize(image, (MAX_WIDTH, round(height * MAX_WIDTH / width)), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return base64.b64encode(buffer.tobytes()).decode("ascii") if ok else None


class SnapshotClient:
    """Talks to the two snapshot RPCs. Never raises on network errors: failures are logged and return []/False."""

    def __init__(self, base_url: str, supabase_key: str | None, device_key: str | None, timeout_seconds: float = 5.0,
                 client: httpx.Client | None = None):
        if not supabase_key:
            raise ValueError("the Supabase publishable key is required (set AFORO_SUPABASE_KEY)")
        if not device_key:
            raise ValueError("the device key is required (set AFORO_DEVICE_KEY)")
        self._base = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self._keys = (supabase_key, device_key)
        self._headers = {"apikey": supabase_key, "X-Device-Key": device_key, "Content-Type": "application/json"}

    def pending(self) -> list[dict]:
        """[{"requestId": uuid, "cameraCode": "camera-outside"}, ...]"""
        try:
            response = self._client.post(self._base + PENDING_PATH, json={}, headers=self._headers)
            response.raise_for_status()
            return list(response.json())
        except (httpx.HTTPError, ValueError, TypeError) as error:
            logger.warning("pending_snapshot_requests failed (%s): %s", type(error).__name__, self._redact(str(error)))
            return []

    def submit(self, request_id: str, image_base64: str | None) -> bool:
        """Sends the image (or null when there is none). True only for HTTP 200."""
        try:
            response = self._client.post(self._base + SUBMIT_PATH, headers=self._headers,
                                         json={"p_request_id": request_id, "p_image": image_base64})
        except httpx.HTTPError as error:
            logger.warning("submit_camera_snapshot failed (%s): %s", type(error).__name__, self._redact(str(error)))
            return False
        if response.status_code != 200:
            logger.warning("submit_camera_snapshot answered %d", response.status_code)
            return False
        return True

    def _redact(self, text: str) -> str:
        for key in self._keys:
            text = text.replace(key, "***")
        return text


class SnapshotWorker:
    """Daemon thread that answers snapshot requests. `grabbers` maps camera_code -> FrameGrabber-like object."""

    def __init__(self, client: SnapshotClient, grabbers: Mapping[str, object],
                 interval_seconds: float = POLL_INTERVAL_SECONDS, clock: Callable[[], float] = time.time):
        self._client = client
        self._grabbers = dict(grabbers)
        self._interval = interval_seconds
        self._clock = clock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def serve_once(self) -> int:
        """Answers every pending request once. Returns how many were submitted successfully."""
        submitted = 0
        for request in self._client.pending():
            image = self._capture(request.get("cameraCode"))
            if self._client.submit(request["requestId"], image):
                submitted += 1
        return submitted

    def _capture(self, camera_code: str | None) -> str | None:
        grabber = self._grabbers.get(camera_code)
        if grabber is None or not grabber.is_connected:
            return None
        frame = grabber.read_latest()
        if frame is None or self._clock() - frame.timestamp > MAX_FRAME_AGE_SECONDS:
            return None
        return encode_snapshot(frame.image)

    def start(self) -> "SnapshotWorker":
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="camera-snapshots", daemon=True)
            self._thread.start()
        return self

    def stop(self, timeout_seconds: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_seconds)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.serve_once()
            except Exception:  # one bad request must not kill the thread; logged without the image
                logger.exception("snapshot round failed, retrying in %.0f s", self._interval)
