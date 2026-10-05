"""HTTPS uploader for POST /events on aforo-backend. Only the event JSON travels, never video.

What the backend answers decides what to do with the event (aforo-backend/README.md):
    201 created / 200 duplicate  -> SENT          the backend has it (a retry never double-counts: eventId is idempotent)
    400 invalid event            -> REJECTED      retrying cannot help; keep it aside for inspection
    401 / 403 wrong secret       -> UNAUTHORIZED  a configuration problem, not the event's fault; keep it and stop
    network error, timeout, 5xx  -> FAILED        try again later
"""

import logging
from enum import Enum

import httpx

logger = logging.getLogger(__name__)

SECRET_HEADER = "X-Aforo-Secret"


class SendStatus(Enum):
    SENT = "sent"
    REJECTED = "rejected"
    UNAUTHORIZED = "unauthorized"
    FAILED = "failed"


class EventUploader:
    def __init__(
        self, base_url: str, shared_secret: str | None, timeout_seconds: float = 5.0, client: httpx.Client | None = None
    ):
        if not shared_secret:
            raise ValueError("a shared secret is required (set AFORO_BACKEND_SECRET)")
        self.url = base_url.rstrip("/") + "/events"
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self._headers = {SECRET_HEADER: shared_secret, "Content-Type": "application/json"}

    def send(self, event: dict) -> SendStatus:
        """POST one event. Never raises on network or HTTP errors: the outcome is the returned status."""
        try:
            response = self._client.post(self.url, json=event, headers=self._headers)
        except httpx.HTTPError as error:
            logger.warning("POST /events failed (%s): %s", type(error).__name__, error)
            return SendStatus.FAILED

        code = response.status_code
        if code in (200, 201):
            return SendStatus.SENT
        if code in (401, 403):
            logger.error("POST /events refused with %d: check AFORO_BACKEND_SECRET", code)
            return SendStatus.UNAUTHORIZED
        if 400 <= code < 500 and code not in (408, 425, 429):  # a malformed event; those three are transient
            logger.error("POST /events rejected event %s with %d: %s", event.get("eventId"), code, _detail(response))
            return SendStatus.REJECTED
        logger.warning("POST /events answered %d, will retry", code)
        return SendStatus.FAILED

    def close(self) -> None:
        self._client.close()


def _detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    return str(body.get("invalidFields") or body.get("message") or body)[:300] if isinstance(body, dict) else str(body)[:300]
