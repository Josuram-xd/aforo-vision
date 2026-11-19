"""Event time that does not depend on the PC clock: it is corrected with the Supabase server's clock.

The server answers every request with a `Date` header (UTC, NTP-synced). We measure how far the PC is from it
and add that offset to the timestamp of each event. Only event timestamps are corrected; the timing between
two frames still uses the PC clock, because it only needs differences.
"""

import logging
import threading
import time
from email.utils import parsedate_to_datetime

import httpx

logger = logging.getLogger(__name__)

RESYNC_SECONDS = 600.0
IGNORE_BELOW_SECONDS = 2.0  # the Date header has 1 s resolution: smaller gaps are noise
_offset = 0.0


def offset() -> float:
    """Seconds to add to the PC clock to get the real time (0.0 until the first successful sync)."""
    return _offset


def corrected(timestamp: float) -> float:
    return timestamp + _offset


def sync(base_url: str, supabase_key: str, client: httpx.Client | None = None) -> float | None:
    """Measure the PC clock error against the server. Returns the offset, or None if the server is unreachable."""
    global _offset
    http = client or httpx.Client(timeout=5.0)
    before = time.time()
    try:
        response = http.get(base_url.rstrip("/") + "/auth/v1/health", headers={"apikey": supabase_key})
        server = parsedate_to_datetime(response.headers["Date"]).timestamp() + 0.5  # header drops the fraction
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
        logger.warning("could not read the server time (%s); keeping offset %.1f s", type(error).__name__, _offset)
        return None
    measured = server - (before + time.time()) / 2
    _offset = 0.0 if abs(measured) < IGNORE_BELOW_SECONDS else measured
    if _offset:
        logger.warning("el reloj del PC esta %.0f s desfasado: los eventos usan la hora del servidor", -_offset)
    return _offset


def start_sync(base_url: str, supabase_key: str) -> None:
    """Sync now and then every RESYNC_SECONDS in a daemon thread."""
    def loop() -> None:
        while True:
            sync(base_url, supabase_key)
            time.sleep(RESYNC_SECONDS)

    threading.Thread(target=loop, name="clock-sync", daemon=True).start()
