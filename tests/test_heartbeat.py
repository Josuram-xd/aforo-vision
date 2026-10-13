import json
import threading

import httpx
import pytest

from src.events.heartbeat import HeartbeatClient, HeartbeatThread

SUPABASE_KEY = "sb_publishable_test-value-123"
DEVICE_KEY = "device-key-test-value-456"


def _heartbeat(handler):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return HeartbeatClient("https://proj.supabase.test", SUPABASE_KEY, DEVICE_KEY, client=client)


def test_reports_the_state_of_each_camera():
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={})

    states = {"camera-outside": lambda: True, "camera-inside": lambda: False}
    results = HeartbeatThread(_heartbeat(handler), states).beat()
    assert sent == [{"p_camera_code": "camera-outside", "p_online": True}, {"p_camera_code": "camera-inside", "p_online": False}]
    assert results == {"camera-outside": True, "camera-inside": True}


def test_a_failing_state_callback_is_reported_offline_and_does_not_stop_the_round():
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={})

    def broken():
        raise RuntimeError("grabber exploded")

    HeartbeatThread(_heartbeat(handler), {"camera-outside": broken, "camera-inside": lambda: True}).beat()
    assert sent == [{"p_camera_code": "camera-outside", "p_online": False}, {"p_camera_code": "camera-inside", "p_online": True}]


def test_failed_heartbeats_are_reported_not_raised():
    thread = HeartbeatThread(_heartbeat(lambda r: httpx.Response(503)), {"camera-outside": lambda: True})
    assert thread.beat() == {"camera-outside": False}


def test_thread_beats_periodically_and_stops():
    calls = threading.Semaphore(0)

    def handler(request):
        calls.release()
        return httpx.Response(200, json={})

    thread = HeartbeatThread(_heartbeat(handler), {"camera-outside": lambda: True}, interval_seconds=0.01)
    with thread:
        assert calls.acquire(timeout=2) and calls.acquire(timeout=2)  # at least two rounds
    assert thread._thread is None


def test_thread_waits_one_interval_before_the_first_beat():
    sent = []
    thread = HeartbeatThread(_heartbeat(lambda r: sent.append(1) or httpx.Response(200, json={})),
                             {"camera-outside": lambda: False}, interval_seconds=60)
    with thread:
        pass
    assert sent == []  # grabbers are still connecting right after start: no false "offline" report


@pytest.mark.parametrize("supabase_key, device_key", [(None, DEVICE_KEY), (SUPABASE_KEY, ""), ("", None)])
def test_both_keys_are_required(supabase_key, device_key):
    with pytest.raises(ValueError):
        HeartbeatClient("https://proj.supabase.test", supabase_key, device_key)


@pytest.mark.parametrize("code, delivered", [(200, True), (400, False), (401, False), (403, False), (429, False), (500, False), (503, False)])
def test_status_codes(code, delivered):
    assert _heartbeat(lambda r: httpx.Response(code, json={})).send("camera-outside", True) is delivered


def test_network_errors_never_raise():
    def down(request):
        raise httpx.ConnectError("unreachable")

    def slow(request):
        raise httpx.ReadTimeout("timed out")

    assert _heartbeat(down).send("camera-outside", True) is False
    assert _heartbeat(slow).send("camera-outside", False) is False


def test_refused_keys_are_logged_loudly(caplog):
    with caplog.at_level("ERROR"):
        _heartbeat(lambda r: httpx.Response(403)).send("camera-inside", True)
    assert "AFORO_DEVICE_KEY" in caplog.text


def test_keys_never_appear_in_logs(caplog):
    echo = f"{SUPABASE_KEY} {DEVICE_KEY}"
    with caplog.at_level("DEBUG"):
        for code in (400, 401, 403, 429, 500):
            _heartbeat(lambda r, c=code: httpx.Response(c, text=echo)).send("camera-outside", True)
        _heartbeat(lambda r: (_ for _ in ()).throw(httpx.ConnectError(f"boom {echo}"))).send("camera-outside", True)
    assert SUPABASE_KEY not in caplog.text and DEVICE_KEY not in caplog.text
