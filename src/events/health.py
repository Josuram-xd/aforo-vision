"""Startup check of aforo-backend: GET /health (public, no secret needed)."""

import logging

import httpx

logger = logging.getLogger(__name__)


def check_backend_health(base_url: str, timeout_seconds: float = 5.0, client: httpx.Client | None = None) -> bool:
    """True if the backend answers {"status": "ok"}. Never raises: any failure is logged and returns False.

    A failed check must not stop the pipeline: events are queued on disk and sent when the backend is back.
    """
    url = base_url.rstrip("/") + "/health"
    own_client = client is None
    client = client or httpx.Client(timeout=timeout_seconds)
    try:
        response = client.get(url)
        if response.status_code == 200 and response.json().get("status") == "ok":
            return True
        logger.warning("GET /health answered %d: %s", response.status_code, response.text[:200])
    except (httpx.HTTPError, ValueError, AttributeError) as error:  # network, bad JSON, JSON that is not an object
        logger.warning("GET /health failed (%s): %s", type(error).__name__, error)
    finally:
        if own_client:
            client.close()
    return False
