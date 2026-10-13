"""Startup check of the Supabase connection: one camera_heartbeat per camera.

Proves the network, the publishable key and the device key in one go (replaces the old GET /health).
"""

import logging
from collections.abc import Iterable

import httpx

from src.events.heartbeat import HeartbeatClient

logger = logging.getLogger(__name__)


def check_backend_health(
    base_url: str,
    supabase_key: str | None,
    device_key: str | None,
    camera_ids: Iterable[str],
    timeout_seconds: float = 5.0,
    client: httpx.Client | None = None,
) -> bool:
    """True if every camera heartbeat was accepted. Never raises: any failure is logged and returns False.

    Sends `online=True` because only connectivity is being checked (the grabbers have not started yet);
    the heartbeat thread reports the real camera state afterwards. A failed check must not stop the
    pipeline: events are queued on disk and sent when Supabase is reachable again.
    """
    try:
        heartbeat = HeartbeatClient(base_url, supabase_key, device_key, timeout_seconds, client)
    except ValueError as error:
        logger.warning("startup check skipped: %s", error)
        return False
    try:
        results = [heartbeat.send(camera_id, True) for camera_id in camera_ids]
    finally:
        if client is None:
            heartbeat.close()
    return bool(results) and all(results)
