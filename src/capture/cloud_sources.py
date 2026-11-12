"""Camera connections registered in the web app (aforo-db task 14): fetched with the classroom's device key."""

from urllib.parse import quote

import httpx

RTSP_PORT = 554
RTSP_PATH = "stream1"  # Tapo C200: stream1 = high quality, stream2 = low quality


def build_rtsp_url(host: str, username: str, password: str) -> str:
    """rtsp://user:pass@host:554/stream1 with the credentials percent-encoded (passwords may contain @ : /)."""
    return f"rtsp://{quote(username, safe='')}:{quote(password, safe='')}@{host}:{RTSP_PORT}/{RTSP_PATH}"


class CloudCameraSources:
    """rpc/camera_sources: {cameraCode: {host, username, password}} for this classroom. Raises httpx errors."""

    def __init__(self, base_url: str, supabase_key: str, device_key: str, timeout: float = 5.0,
                 client: httpx.Client | None = None):
        self._url = base_url.rstrip("/") + "/rest/v1/rpc/camera_sources"
        self._headers = {"apikey": supabase_key, "X-Device-Key": device_key, "Content-Type": "application/json"}
        self._client = client or httpx.Client(timeout=timeout)

    def urls(self) -> dict[str, str]:
        response = self._client.post(self._url, json={}, headers=self._headers)
        response.raise_for_status()
        return {code: build_rtsp_url(c["host"], c["username"], c["password"]) for code, c in response.json().items()}

    def url_for(self, camera_id: str) -> str | None:
        """Source for one camera, or None if it has none or the cloud is unreachable (keep the current one)."""
        try:
            return self.urls().get(camera_id)
        except (httpx.HTTPError, ValueError, KeyError):
            return None
