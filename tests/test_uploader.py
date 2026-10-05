import httpx
import pytest

from src.events.uploader import EventUploader, SendStatus

EVENT = {"eventId": "0b6f1c9e-8a2e-4c55-9d0e-1f2a3b4c5d6e", "direction": "ENTRY"}


def _uploader(handler, secret="s3cret-value-123456", base_url="https://api.example.test/"):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return EventUploader(base_url, secret, client=client)


def test_posts_json_with_the_shared_secret_header_to_events():
    seen = {}

    def handler(request):
        seen["url"], seen["secret"], seen["body"] = str(request.url), request.headers.get("x-aforo-secret"), request.content
        return httpx.Response(201, json={"eventId": EVENT["eventId"]})

    assert _uploader(handler).send(EVENT) is SendStatus.SENT
    assert seen["url"] == "https://api.example.test/events"  # no double slash
    assert seen["secret"] == "s3cret-value-123456"
    assert b'"eventId"' in seen["body"]


def test_duplicate_answer_counts_as_sent():
    assert _uploader(lambda r: httpx.Response(200, json={"duplicate": True})).send(EVENT) is SendStatus.SENT


def test_bad_secret_is_unauthorized_not_a_reason_to_drop_the_event():
    assert _uploader(lambda r: httpx.Response(401, json={"message": "no"})).send(EVENT) is SendStatus.UNAUTHORIZED
    assert _uploader(lambda r: httpx.Response(403)).send(EVENT) is SendStatus.UNAUTHORIZED


def test_invalid_event_is_rejected_and_the_reason_is_logged(caplog):
    uploader = _uploader(lambda r: httpx.Response(400, json={"message": "Evento inválido", "invalidFields": ["confidence"]}))
    with caplog.at_level("ERROR"):
        assert uploader.send(EVENT) is SendStatus.REJECTED
    assert "confidence" in caplog.text and EVENT["eventId"] in caplog.text


@pytest.mark.parametrize("code", [500, 502, 503, 504, 429, 408])
def test_server_errors_and_throttling_are_retryable(code):
    assert _uploader(lambda r: httpx.Response(code)).send(EVENT) is SendStatus.FAILED


def test_network_errors_are_retryable_and_never_raise():
    def down(request):
        raise httpx.ConnectError("network unreachable")

    def slow(request):
        raise httpx.ReadTimeout("timed out")

    assert _uploader(down).send(EVENT) is SendStatus.FAILED
    assert _uploader(slow).send(EVENT) is SendStatus.FAILED


def test_secret_is_required_and_never_logged(caplog):
    with pytest.raises(ValueError, match="AFORO_BACKEND_SECRET"):
        EventUploader("https://api.example.test", None)
    with pytest.raises(ValueError):
        EventUploader("https://api.example.test", "")
    with caplog.at_level("DEBUG"):
        _uploader(lambda r: httpx.Response(401)).send(EVENT)
        _uploader(lambda r: httpx.Response(500)).send(EVENT)
    assert "s3cret-value-123456" not in caplog.text
