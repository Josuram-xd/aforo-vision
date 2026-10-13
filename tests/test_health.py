import json

import httpx
import pytest

from src.events.health import check_backend_health

SUPABASE_KEY = "sb_publishable_test-value-123"
DEVICE_KEY = "device-key-test-value-456"
CAMERAS = ("camera-outside", "camera-inside")


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def _check(handler, **kwargs):
    return check_backend_health("https://proj.supabase.test/", SUPABASE_KEY, DEVICE_KEY, CAMERAS,
                                client=_client(handler), **kwargs)


def _ok(request):
    return httpx.Response(200, json={"camera": "x", "status": "online", "lastSeenAt": "2026-10-13T10:00:00Z"})


def test_sends_one_heartbeat_per_camera_with_both_keys():
    seen = []

    def handler(request):
        seen.append((str(request.url), request.headers["apikey"], request.headers["x-device-key"], json.loads(request.content)))
        return _ok(request)

    assert _check(handler) is True
    url = "https://proj.supabase.test/rest/v1/rpc/camera_heartbeat"
    assert seen == [(url, SUPABASE_KEY, DEVICE_KEY, {"p_camera_code": code, "p_online": True}) for code in CAMERAS]


@pytest.mark.parametrize("code", [400, 401, 403, 429, 500])
def test_unhealthy_answers_are_false_and_logged(code, caplog):
    with caplog.at_level("WARNING"):
        assert _check(lambda r: httpx.Response(code)) is False
    assert "camera_heartbeat" in caplog.text


def test_one_rejected_camera_makes_the_check_fail():
    def handler(request):
        return _ok(request) if json.loads(request.content)["p_camera_code"] == "camera-outside" else httpx.Response(400)

    assert _check(handler) is False


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
