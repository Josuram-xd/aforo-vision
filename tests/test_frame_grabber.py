import threading
import time

import numpy as np
import pytest

from src.capture import frame_grabber
from src.capture.frame_grabber import FrameGrabber


class FakeCapture:
    """Stands in for cv2.VideoCapture; yields `n_frames` frames, then reports end of stream."""

    def __init__(self, n_frames: int | None = None, opened: bool = True):
        self.n_frames = n_frames
        self.opened = opened
        self.reads = 0
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        if self.n_frames is not None and self.reads >= self.n_frames:
            return False, None
        self.reads += 1
        time.sleep(0.001)
        return True, np.full((4, 4, 3), self.reads, dtype=np.uint8)

    def set(self, *_):
        return True

    def release(self):
        self.released = True


def use_fake(monkeypatch, capture: FakeCapture) -> None:
    monkeypatch.setattr(frame_grabber, "_open_capture", lambda source: capture)


def wait_until(condition, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.005)


def test_read_latest_returns_none_before_start(monkeypatch):
    use_fake(monkeypatch, FakeCapture())
    assert FrameGrabber(0, "camera-inside").read_latest() is None


def test_read_latest_returns_newest_frame(monkeypatch):
    capture = FakeCapture()
    use_fake(monkeypatch, capture)
    with FrameGrabber(0, "camera-inside") as grabber:
        wait_until(lambda: capture.reads >= 5)
        first = grabber.read_latest()
        wait_until(lambda: capture.reads >= first.index + 5)
        second = grabber.read_latest()

    assert first.camera_id == "camera-inside"
    assert second.index > first.index
    assert second.timestamp >= first.timestamp
    assert capture.released


def test_raises_when_source_cannot_open(monkeypatch):
    capture = FakeCapture(opened=False)
    use_fake(monkeypatch, capture)
    with pytest.raises(RuntimeError, match="camera-outside"):
        FrameGrabber("rtsp://user:secret@10.0.0.1/stream", "camera-outside").start()
    assert capture.released


def test_error_message_hides_rtsp_credentials(monkeypatch):
    use_fake(monkeypatch, FakeCapture(opened=False))
    with pytest.raises(RuntimeError) as exc_info:
        FrameGrabber("rtsp://user:secret@10.0.0.1/stream", "camera-outside").start()
    assert "secret" not in str(exc_info.value)


def test_thread_stops_when_stream_ends_and_keeps_last_frame(monkeypatch):
    use_fake(monkeypatch, FakeCapture(n_frames=3))
    grabber = FrameGrabber(0, "camera-inside").start()
    wait_until(lambda: not grabber.is_running)
    assert grabber.read_latest().index == 2
    grabber.stop()


def test_one_thread_per_grabber(monkeypatch):
    monkeypatch.setattr(frame_grabber, "_open_capture", lambda source: FakeCapture())
    grabbers = [FrameGrabber(0, "camera-inside").start(), FrameGrabber(1, "camera-outside").start()]
    names = {t.name for t in threading.enumerate()}
    assert {"grabber-camera-inside", "grabber-camera-outside"} <= names
    for grabber in grabbers:
        grabber.stop()
