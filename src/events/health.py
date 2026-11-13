"""Startup check of the Supabase connection: one read-only RPC (`camera_sources`) with both keys.

Proves the network, the publishable key and the device key without changing any camera state
(a heartbeat here would show the cameras as online before their grabbers start).
"""

import logging
from collections.abc import Iterable

import httpx

logger = logging.getLogger(__name__)


def check_backend_health(
    base_url: str,
    supabase_key: str | None,
    device_key: str | None,
    camera_ids: Iterable[str],
    timeout_seconds: float = 5.0,
    client: httpx.Client | None = None,
) -> bool:
    """True if Supabase answers `camera_sources` with 2xx. Never raises: any failure is logged and returns False.

    Only reads: no camera_heartbeat is sent, so the dashboard does not mark cameras online before their
    grabbers start; the heartbeat thread reports the real state afterwards. A failed check must not stop
    the pipeline: events are queued on disk and sent when Supabase is reachable again.
    """
    if not list(camera_ids):
        logger.warning("startup check skipped: no cameras configured")
        return False
    if not supabase_key or not device_key:
        logger.warning("startup check skipped: set AFORO_SUPABASE_KEY and AFORO_DEVICE_KEY")
        return False
    headers = {"apikey": supabase_key, "X-Device-Key": device_key, "Content-Type": "application/json"}
    url = base_url.rstrip("/") + "/rest/v1/rpc/camera_sources"
    owns_client = client is None
    http = client or httpx.Client(timeout=timeout_seconds)
    try:
        response = http.post(url, json={}, headers=headers)
    except httpx.HTTPError as error:
        message = str(error).replace(supabase_key, "***").replace(device_key, "***")
        logger.warning("startup check failed (%s): %s", type(error).__name__, message)
        return False
    finally:
        if owns_client:
            http.close()
    if response.is_success:
        return True
    if response.status_code in (401, 403):
        logger.error("camera_sources refused with %d: check AFORO_SUPABASE_KEY and AFORO_DEVICE_KEY", response.status_code)
    else:
        logger.warning("camera_sources answered %d", response.status_code)
    return False
