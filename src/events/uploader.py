"""HTTPS uploader for the Supabase `ingest_event` RPC. Only the event JSON travels, never video.

POST <supabase-url>/rest/v1/rpc/ingest_event with body {"p_event": <event>} (ARCHITECTURE.md 10.1).
What Supabase answers decides what to do with the event:
    200 (duplicate true or false) -> SENT          the database has it (a retry never double-counts: eventId is idempotent)
    400 invalid event / timestamp -> REJECTED      retrying cannot help; keep it aside for inspection
    401 / 403 wrong key           -> UNAUTHORIZED  a configuration problem, not the event's fault; keep it and stop
    429, network error, 5xx, other -> FAILED       try again later (429 also means the 120 req/min limit)
"""

import logging
from enum import Enum

import httpx

logger = logging.getLogger(__name__)

INGEST_PATH = "/rest/v1/rpc/ingest_event"
API_KEY_HEADER = "apikey"
DEVICE_KEY_HEADER = "X-Device-Key"


class SendStatus(Enum):
    SENT = "sent"
    REJECTED = "rejected"
    UNAUTHORIZED = "unauthorized"
    FAILED = "failed"


class EventUploader:
    def __init__(
        self,
        base_url: str,
        supabase_key: str | None,
        device_key: str | None,
        timeout_seconds: float = 5.0,
        client: httpx.Client | None = None,
    ):
        if not supabase_key:
            raise ValueError("the Supabase publishable key is required (set AFORO_SUPABASE_KEY)")
        if not device_key:
            raise ValueError("the device key is required (set AFORO_DEVICE_KEY)")
        self.url = base_url.rstrip("/") + INGEST_PATH
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self._keys = (supabase_key, device_key)
        self._headers = {API_KEY_HEADER: supabase_key, DEVICE_KEY_HEADER: device_key, "Content-Type": "application/json"}

    def send(self, event: dict) -> SendStatus:
        """POST one event. Never raises on network or HTTP errors: the outcome is the returned status."""
        try:
            response = self._client.post(self.url, json={"p_event": event}, headers=self._headers)
        except httpx.HTTPError as error:
            logger.warning("ingest_event failed (%s): %s", type(error).__name__, self._redact(str(error)))
            return SendStatus.FAILED

        code = response.status_code
        if code == 200:
            return SendStatus.SENT
        if code in (401, 403):
            logger.error("ingest_event refused with %d: check AFORO_SUPABASE_KEY and AFORO_DEVICE_KEY (revoked or wrong key?)", code)
            return SendStatus.UNAUTHORIZED
        if code == 400:  # malformed event or timestamp too far in the future
            logger.error("ingest_event rejected event %s with 400: %s", event.get("eventId"), self._redact(_detail(response)))
            return SendStatus.REJECTED
        if code == 429:
            logger.warning("ingest_event answered 429 (rate limit, 120 req/min), backing off")
        else:
            logger.warning("ingest_event answered %d, will retry", code)
        return SendStatus.FAILED

    def close(self) -> None:
        self._client.close()

    def _redact(self, text: str) -> str:
        for key in self._keys:
            text = text.replace(key, "***")
        return text


def _detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(body, dict):
        return str(body.get("message") or body.get("details") or body)[:300]
    return str(body)[:300]
