import httpx

from src.capture.cloud_sources import CloudCameraSources, build_rtsp_url


def test_build_rtsp_url_encodes_credentials():
    assert build_rtsp_url("192.168.1.50", "cam", "p@ss:w/rd") == "rtsp://cam:p%40ss%3Aw%2Frd@192.168.1.50:554/stream1"


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_urls_and_headers():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["device"] = request.headers["x-device-key"]
        return httpx.Response(200, json={"camera-outside": {"host": "10.0.0.5", "username": "u", "password": "p"}})

    sources = CloudCameraSources("https://x.supabase.co", "pub", "dev", client=_client(handler))
    assert sources.urls() == {"camera-outside": "rtsp://u:p@10.0.0.5:554/stream1"}
    assert seen["device"] == "dev"
    assert sources.url_for("camera-inside") is None  # not registered yet


def test_url_for_returns_none_when_cloud_fails():
    sources = CloudCameraSources("https://x.supabase.co", "pub", "dev",
                                 client=_client(lambda request: httpx.Response(500)))
    assert sources.url_for("camera-outside") is None
