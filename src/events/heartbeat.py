"""Camera heartbeats: rpc/camera_heartbeat(p_camera_code, p_online) every 30 s per camera (ARCHITECTURE.md 10.1).

The database turns a missing or `false` heartbeat into a `camera_offline` alert, so the dashboard shows camera
status live. The thread never touches frame processing: a dead network only delays heartbeats.
"""

import logging
import threading
from collections.abc import Callable, Mapping

import httpx

logger = logging.getLogger(__name__)

HEARTBEAT_PATH = "/rest/v1/rpc/camera_heartbeat"
HEARTBEAT_INTERVAL_SECONDS = 30.0


class HeartbeatClient:
    """Posts one heartbeat. Never raises: the outcome is the returned bool (True only for HTTP 200)."""

    def __init__(self, base_url: str, supabase_key: str | None, device_key: str | None, timeout_seconds: float = 5.0,
                 client: httpx.Client | None = None):
        if not supabase_key:
            raise ValueError("the Supabase publishable key is required (set AFORO_SUPABASE_KEY)")
        if not device_key:
            raise ValueError("the device key is required (set AFORO_DEVICE_KEY)")
        self.url = base_url.rstrip("/") + HEARTBEAT_PATH
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self._keys = (supabase_key, device_key)
        self._headers = {"apikey": supabase_key, "X-Device-Key": device_key, "Content-Type": "application/json"}

    def send(self, camera_code: str, online: bool) -> bool:
        try:
            response = self._client.post(
                self.url, json={"p_camera_code": camera_code, "p_online": bool(online)}, headers=self._headers
            )
        except httpx.HTTPError as error:
            logger.warning("camera_heartbeat for %s failed (%s): %s", camera_code, type(error).__name__, self._redact(str(error)))
            return False
        code = response.status_code
        if code == 200:
            return True
        if code in (401, 403):
            logger.error("camera_heartbeat for %s refused with %d: check AFORO_SUPABASE_KEY and AFORO_DEVICE_KEY", camera_code, code)
        elif code == 429:
            logger.warning("camera_heartbeat for %s answered 429 (rate limit)", camera_code)
        else:
            logger.warning("camera_heartbeat for %s answered %d", camera_code, code)
        return False

    def close(self) -> None:
        self._client.close()

    def _redact(self, text: str) -> str:
        for key in self._keys:
            text = text.replace(key, "***")
        return text


class HeartbeatThread:
    """Every `interval_seconds`, report each camera's connection state (`is_connected` of its grabber)."""

    def __init__(self, heartbeat: HeartbeatClient, cameras: Mapping[str, Callable[[], bool]],
                 interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS):
        self._heartbeat = heartbeat
        self._cameras = dict(cameras)
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def beat(self) -> dict[str, bool]:
        """One round: returns {camera_code: delivered}. A failing state callback counts as offline."""
        results = {}
        for code, is_connected in self._cameras.items():
            try:
                online = bool(is_connected())
            except Exception:  # a broken grabber must not kill the heartbeat thread
                logger.exception("could not read the state of %s, reporting it offline", code)
                online = False
            results[code] = self._heartbeat.send(code, online)
        return results

    def start(self) -> "HeartbeatThread":
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="camera-heartbeat", daemon=True)
            self._thread.start()
        return self

    def stop(self, timeout_seconds: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_seconds)
            self._thread = None

    def _run(self) -> None:
        # Wait first: right after start the grabbers are still connecting and would be reported offline.
        while not self._stop.wait(self._interval):
            self.beat()

    def __enter__(self) -> "HeartbeatThread":
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.stop()
