import json

import httpx
import pytest

from src.events.health import check_backend_health

SUPABASE_KEY = "sb_publishable_test-value-123"
DEVICE_KEY = "device-key-test-value-456"
CAMERAS = ("camera-outside", "camera-inside")
SOURCES_URL = "https://proj.supabase.test/rest/v1/rpc/camera_sources"


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def _check(handler, **kwargs):
    return check_backend_health("https://proj.supabase.test/", SUPABASE_KEY, DEVICE_KEY, CAMERAS,
                                client=_client(handler), **kwargs)


def _ok(request):
    return httpx.Response(200, json={})


def test_reads_camera_sources_with_both_keys_and_sends_no_heartbeat():
    seen = []

    def handler(request):
        seen.append((str(request.url), request.headers["apikey"], request.headers["x-device-key"], json.loads(request.content)))
        return _ok(request)

    assert _check(handler) is True
    assert seen == [(SOURCES_URL, SUPABASE_KEY, DEVICE_KEY, {})]


@pytest.mark.parametrize("code", [200, 204])
def test_any_2xx_is_healthy(code):
    assert _check(lambda r: httpx.Response(code)) is True


@pytest.mark.parametrize("code", [400, 401, 403, 429, 500])
def test_unhealthy_answers_are_false_and_logged(code, caplog):
    with caplog.at_level("WARNING"):
        assert _check(lambda r: httpx.Response(code)) is False
    assert "camera_sources" in caplog.text


def test_network_errors_never_raise():
    def down(request):
        raise httpx.ConnectError("unreachable")

    assert _check(down) is False


def test_missing_keys_do_not_raise():
    assert check_backend_health("https://proj.supabase.test", None, DEVICE_KEY, CAMERAS, client=_client(_ok)) is False


def test_no_cameras_is_not_healthy():
    assert check_backend_health("https://proj.supabase.test", SUPABASE_KEY, DEVICE_KEY, [], client=_client(_ok)) is False


def test_keys_never_appear_in_logs(caplog):
    with caplog.at_level("DEBUG"):
        _check(lambda r: httpx.Response(403, text=f"{SUPABASE_KEY} {DEVICE_KEY}"))
        _check(lambda r: (_ for _ in ()).throw(httpx.ConnectError(f"boom {SUPABASE_KEY} {DEVICE_KEY}")))
    assert SUPABASE_KEY not in caplog.text and DEVICE_KEY not in caplog.text
