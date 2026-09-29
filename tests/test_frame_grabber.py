import logging
import threading
import time

import numpy as np

from src.capture import frame_grabber
from src.capture.frame_grabber import FrameGrabber

FAST_RETRY = 0.01


class FakeCapture:
    """Stands in for cv2.VideoCapture; yields `n_frames` frames, then reports a dropped stream."""

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


def use_captures(monkeypatch, *captures: FakeCapture) -> list[FakeCapture]:
    """Each open attempt returns the next capture; the last one repeats forever."""
    opened = []

    def fake_open(source):
        capture = captures[min(len(opened), len(captures) - 1)]
        opened.append(capture)
        return capture

    monkeypatch.setattr(frame_grabber, "_open_capture", fake_open)
    return opened


def wait_until(condition, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.005)


def test_read_latest_returns_none_before_start(monkeypatch):
    use_captures(monkeypatch, FakeCapture())
    assert FrameGrabber(0, "camera-inside").read_latest() is None


def test_read_latest_returns_newest_frame(monkeypatch):
    capture = FakeCapture()
    use_captures(monkeypatch, capture)
    with FrameGrabber(0, "camera-inside") as grabber:
        wait_until(lambda: capture.reads >= 5)
        first = grabber.read_latest()
        wait_until(lambda: capture.reads >= first.index + 5)
        second = grabber.read_latest()

    assert first.camera_id == "camera-inside"
    assert second.index > first.index
    assert second.timestamp >= first.timestamp
    assert capture.released


def test_start_does_not_raise_when_source_is_unavailable(monkeypatch):
    opened = use_captures(monkeypatch, FakeCapture(opened=False))
    grabber = FrameGrabber("rtsp://user:secret@10.0.0.1/stream", "camera-outside", FAST_RETRY).start()
    wait_until(lambda: len(opened) >= 3)  # keeps retrying
    assert grabber.is_running
    assert not grabber.is_connected
    assert grabber.read_latest() is None
    grabber.stop()


def test_connects_once_source_becomes_available(monkeypatch):
    live = FakeCapture()
    use_captures(monkeypatch, FakeCapture(opened=False), FakeCapture(opened=False), live)
    with FrameGrabber(1, "camera-outside", FAST_RETRY) as grabber:
        wait_until(lambda: grabber.read_latest() is not None)
        assert grabber.is_connected


def test_reconnects_after_stream_drops(monkeypatch):
    dropping, replacement = FakeCapture(n_frames=3), FakeCapture()
    use_captures(monkeypatch, dropping, replacement)
    with FrameGrabber(0, "camera-inside", FAST_RETRY) as grabber:
        # 3 frames (indices 0-2) from the dropped stream, then indices keep increasing on the new one
        wait_until(lambda: (frame := grabber.read_latest()) is not None and frame.index >= 4)
        assert grabber.is_connected

    assert dropping.released
    assert replacement.reads >= 2


def test_logs_hide_rtsp_credentials(monkeypatch, caplog):
    opened = use_captures(monkeypatch, FakeCapture(opened=False))
    with caplog.at_level(logging.WARNING):
        with FrameGrabber("rtsp://user:secret@10.0.0.1/stream", "camera-outside", FAST_RETRY):
            wait_until(lambda: len(opened) >= 2)
    assert "10.0.0.1" in caplog.text
    assert "secret" not in caplog.text


def test_warns_once_while_source_stays_down(monkeypatch, caplog):
    opened = use_captures(monkeypatch, FakeCapture(opened=False))
    with caplog.at_level(logging.WARNING):
        with FrameGrabber(0, "camera-outside", FAST_RETRY):
            wait_until(lambda: len(opened) >= 5)
    assert caplog.text.count("could not open") == 1


def test_stop_ends_thread_while_waiting_to_retry(monkeypatch):
    use_captures(monkeypatch, FakeCapture(opened=False))
    grabber = FrameGrabber(0, "camera-outside", reconnect_interval_seconds=60).start()
    time.sleep(0.05)
    started = time.monotonic()
    grabber.stop()
    assert time.monotonic() - started < 1
    assert not grabber.is_running


def test_one_thread_per_grabber(monkeypatch):
    monkeypatch.setattr(frame_grabber, "_open_capture", lambda source: FakeCapture())
    grabbers = [FrameGrabber(0, "camera-inside").start(), FrameGrabber(1, "camera-outside").start()]
    names = {t.name for t in threading.enumerate()}
    assert {"grabber-camera-inside", "grabber-camera-outside"} <= names
    for grabber in grabbers:
        grabber.stop()
