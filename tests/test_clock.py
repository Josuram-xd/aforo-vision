import time
from email.utils import format_datetime
from datetime import datetime, timedelta, timezone

import httpx

from src import clock


def _client(server_time: datetime) -> httpx.Client:
    handler = lambda request: httpx.Response(200, headers={"Date": format_datetime(server_time, usegmt=True)})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_sync_measures_a_pc_clock_that_is_ahead():
    real = datetime.now(timezone.utc) - timedelta(hours=1)  # the PC thinks it is one hour later
    offset = clock.sync("https://x.supabase.co", "key", _client(real))
    assert -3601 < offset < -3599
    assert abs(clock.corrected(time.time()) - real.timestamp()) < 2


def test_small_gaps_are_ignored_and_errors_keep_the_offset():
    assert clock.sync("https://x.supabase.co", "key", _client(datetime.now(timezone.utc))) == 0.0
    broken = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))  # no Date header
    assert clock.sync("https://x.supabase.co", "key", broken) is None
