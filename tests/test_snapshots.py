import base64
import json
import threading
from dataclasses import dataclass

import cv2
import httpx
import numpy as np

from src.capture.frame_grabber import Frame
from src.capture.snapshots import MAX_WIDTH, SnapshotClient, SnapshotWorker

SUPABASE_KEY = "sb_publishable_test-value-123"
DEVICE_KEY = "device-key-test-value-456"
NOW = 1_000_000.0
REQUEST_ID = "7d0c1f2e-0000-4000-8000-000000000001"


@dataclass
class FakeGrabber:
    connected: bool = True
    frame: Frame | None = None

    @property
    def is_connected(self) -> bool:
        return self.connected

    def read_latest(self) -> Frame | None:
        return self.frame


def _frame(width=1280, height=720, age=0.0, seed=0):
    image = np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)  # noise: worst case for JPEG
    return Frame("camera-outside", image, NOW - age, 1)


def _worker(handler, grabbers, interval=3.0):
    client = SnapshotClient("https://proj.supabase.test", SUPABASE_KEY, DEVICE_KEY,
                            client=httpx.Client(transport=httpx.MockTransport(handler)))
    return SnapshotWorker(client, grabbers, interval_seconds=interval, clock=lambda: NOW)


def _handler(sent, pending=(("camera-outside", REQUEST_ID),)):
    def handler(request):
        if request.url.path.endswith("/pending_snapshot_requests"):
            return httpx.Response(200, json=[{"requestId": r, "cameraCode": c} for c, r in pending])
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={})
    return handler


def test_old_frame_is_answered_with_null():
    sent = []
    worker = _worker(_handler(sent), {"camera-outside": FakeGrabber(frame=_frame(age=6.0))})
    assert worker.serve_once() == 1
    assert sent == [{"p_request_id": REQUEST_ID, "p_image": None}]


def test_disconnected_camera_is_answered_with_null():
    sent = []
    worker = _worker(_handler(sent), {"camera-outside": FakeGrabber(connected=False, frame=_frame())})
    worker.serve_once()
    assert sent == [{"p_request_id": REQUEST_ID, "p_image": None}]


def test_unknown_camera_is_answered_with_null():
    sent = []
    worker = _worker(_handler(sent, pending=(("camera-x", REQUEST_ID),)), {})
    worker.serve_once()
    assert sent == [{"p_request_id": REQUEST_ID, "p_image": None}]


def test_fresh_frame_is_sent_as_small_jpeg():
    sent = []
    worker = _worker(_handler(sent), {"camera-outside": FakeGrabber(frame=_frame(width=1920, height=1080))})
    worker.serve_once()
    data = base64.b64decode(sent[0]["p_image"])
    assert len(data) <= 400_000
    assert data[:3] == b"\xff\xd8\xff"
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[1] <= MAX_WIDTH
    assert decoded.shape[1] == MAX_WIDTH  # keeps the aspect ratio: 1920x1080 -> 640x360
    assert decoded.shape[0] == 360


def test_network_error_does_not_kill_the_thread():
    calls = []
    submitted = threading.Event()

    def handler(request):
        if request.url.path.endswith("/pending_snapshot_requests"):
            calls.append(1)
            if len(calls) == 1:
                raise httpx.ConnectError("network down")
            return httpx.Response(200, json=[{"requestId": REQUEST_ID, "cameraCode": "camera-outside"}])
        submitted.set()
        return httpx.Response(200, json={})

    worker = _worker(handler, {"camera-outside": FakeGrabber(frame=_frame(width=320, height=240))}, interval=0.01)
    worker.start()
    try:
        assert submitted.wait(5)
        assert worker._thread.is_alive()
    finally:
        worker.stop()
