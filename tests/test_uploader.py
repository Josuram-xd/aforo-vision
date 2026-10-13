import json

import httpx
import pytest

from src.events.uploader import EventUploader, SendStatus

EVENT = {"eventId": "0b6f1c9e-8a2e-4c55-9d0e-1f2a3b4c5d6e", "direction": "ENTRY"}
SUPABASE_KEY = "sb_publishable_test-value-123"
DEVICE_KEY = "device-key-test-value-456"


def _uploader(handler, base_url="https://proj.supabase.test/", supabase_key=SUPABASE_KEY, device_key=DEVICE_KEY):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return EventUploader(base_url, supabase_key, device_key, client=client)


def test_posts_wrapped_event_to_the_ingest_event_rpc_with_both_keys():
    seen = {}

    def handler(request):
        seen.update(url=str(request.url), headers=request.headers, body=json.loads(request.content), method=request.method)
        return httpx.Response(200, json={"eventId": EVENT["eventId"], "duplicate": False})

    assert _uploader(handler).send(EVENT) is SendStatus.SENT
    assert seen["method"] == "POST"
    assert seen["url"] == "https://proj.supabase.test/rest/v1/rpc/ingest_event"  # no double slash
    assert seen["headers"]["apikey"] == SUPABASE_KEY
    assert seen["headers"]["x-device-key"] == DEVICE_KEY
    assert seen["headers"]["content-type"] == "application/json"
    assert seen["body"] == {"p_event": EVENT}


def test_duplicate_answer_counts_as_sent():
    assert _uploader(lambda r: httpx.Response(200, json={"eventId": "x", "duplicate": True})).send(EVENT) is SendStatus.SENT


def test_200_with_a_body_that_is_not_json_still_counts_as_sent():
    assert _uploader(lambda r: httpx.Response(200, text="ok")).send(EVENT) is SendStatus.SENT


@pytest.mark.parametrize("code", [401, 403])
def test_bad_keys_are_unauthorized_not_a_reason_to_drop_the_event(code, caplog):
    with caplog.at_level("ERROR"):
        assert _uploader(lambda r: httpx.Response(code, json={"message": "no"})).send(EVENT) is SendStatus.UNAUTHORIZED
    assert "AFORO_DEVICE_KEY" in caplog.text


def test_invalid_event_is_rejected_and_the_reason_is_logged(caplog):
    uploader = _uploader(lambda r: httpx.Response(400, json={"code": "P0001", "message": "invalid confidence"}))
    with caplog.at_level("ERROR"):
        assert uploader.send(EVENT) is SendStatus.REJECTED
    assert "invalid confidence" in caplog.text and EVENT["eventId"] in caplog.text


@pytest.mark.parametrize("code", [500, 502, 503, 504, 429, 408, 425, 404])
def test_server_errors_throttling_and_misrouted_calls_are_retryable(code):
    assert _uploader(lambda r: httpx.Response(code)).send(EVENT) is SendStatus.FAILED


def test_network_errors_are_retryable_and_never_raise():
    def down(request):
        raise httpx.ConnectError("network unreachable")

    def slow(request):
        raise httpx.ReadTimeout("timed out")

    assert _uploader(down).send(EVENT) is SendStatus.FAILED
    assert _uploader(slow).send(EVENT) is SendStatus.FAILED


@pytest.mark.parametrize("supabase_key, device_key, missing", [
    (None, DEVICE_KEY, "AFORO_SUPABASE_KEY"), ("", DEVICE_KEY, "AFORO_SUPABASE_KEY"),
    (SUPABASE_KEY, None, "AFORO_DEVICE_KEY"), (SUPABASE_KEY, "", "AFORO_DEVICE_KEY"),
])
def test_both_keys_are_required(supabase_key, device_key, missing):
    with pytest.raises(ValueError, match=missing):
        EventUploader("https://proj.supabase.test", supabase_key, device_key)


def test_keys_are_never_logged(caplog):
    with caplog.at_level("DEBUG"):
        for code in (401, 403, 400, 429, 500):
            _uploader(lambda r, c=code: httpx.Response(c, text=f"echo {SUPABASE_KEY} {DEVICE_KEY}")).send(EVENT)
        _uploader(lambda r: (_ for _ in ()).throw(httpx.ConnectError("boom"))).send(EVENT)
    assert SUPABASE_KEY not in caplog.text and DEVICE_KEY not in caplog.text
