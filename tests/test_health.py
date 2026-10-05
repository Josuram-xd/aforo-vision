import httpx

from src.events.health import check_backend_health


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_healthy_backend():
    seen = {}

    def handler(request):
        seen["url"], seen["secret"] = str(request.url), request.headers.get("x-aforo-secret")
        return httpx.Response(200, json={"status": "ok"})

    assert check_backend_health("https://api.example.test/", client=_client(handler)) is True
    assert seen == {"url": "https://api.example.test/health", "secret": None}  # public route: no secret is sent


def test_unhealthy_answers_are_false_and_logged(caplog):
    for response in (httpx.Response(500), httpx.Response(200, json={"status": "down"}),
                     httpx.Response(200, text="<html>"), httpx.Response(200, json=["ok"])):
        with caplog.at_level("WARNING"):
            assert check_backend_health("https://api.example.test", client=_client(lambda r, resp=response: resp)) is False
    assert "GET /health" in caplog.text


def test_network_errors_never_raise():
    def down(request):
        raise httpx.ConnectError("unreachable")

    assert check_backend_health("https://api.example.test", client=_client(down)) is False
