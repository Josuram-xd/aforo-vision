"""Local debug preview: one cv2 window per camera with FPS and boxes. Never saves or sends frames."""

import time
from collections.abc import Iterable
from typing import NamedTuple

import cv2
import numpy as np

# cv2.putText only renders ASCII, so on-screen Spanish text avoids accents/ñ
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_GREEN = (0, 255, 0)
_RED = (0, 0, 255)
_BLACK = (0, 0, 0)
_PLACEHOLDER_SHAPE = (480, 640, 3)
_QUIT_KEYS = {ord("q"), 27}  # q or Esc


class Annotation(NamedTuple):
    x1: float
    y1: float
    x2: float
    y2: float
    label: str | None = None


class FpsMeter:
    """Frames per second as an exponential moving average of the interval between ticks."""

    def __init__(self, smoothing: float = 0.9):
        self.smoothing = smoothing
        self._last: float | None = None
        self._interval: float | None = None

    def tick(self, now: float | None = None) -> None:
        now = time.perf_counter() if now is None else now
        if self._last is not None:
            interval = now - self._last
            if self._interval is None:
                self._interval = interval
            else:
                self._interval = self.smoothing * self._interval + (1 - self.smoothing) * interval
        self._last = now

    @property
    def fps(self) -> float:
        return 0.0 if not self._interval else 1.0 / self._interval


def draw_overlay(image: np.ndarray, camera_id: str, fps: float, annotations: Iterable[Annotation] = ()) -> np.ndarray:
    """Return a copy of `image` with boxes and a camera/FPS header drawn on it."""
    canvas = image.copy()
    for box in annotations:
        p1, p2 = (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2))
        cv2.rectangle(canvas, p1, p2, _GREEN, 2)
        if box.label:
            cv2.putText(canvas, box.label, (p1[0], max(p1[1] - 6, 12)), _FONT, 0.5, _GREEN, 1, cv2.LINE_AA)
    _draw_header(canvas, f"{camera_id} | {fps:4.1f} fps", _GREEN)
    return canvas


def draw_no_signal(camera_id: str, shape: tuple[int, ...] = _PLACEHOLDER_SHAPE) -> np.ndarray:
    canvas = np.zeros(shape, dtype=np.uint8)
    _draw_header(canvas, f"{camera_id} | SIN SENAL - reconectando...", _RED)
    return canvas


def _draw_header(canvas: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    (width, height), baseline = cv2.getTextSize(text, _FONT, 0.6, 2)
    cv2.rectangle(canvas, (0, 0), (width + 16, height + baseline + 12), _BLACK, -1)
    cv2.putText(canvas, text, (8, height + 6), _FONT, 0.6, color, 2, cv2.LINE_AA)


class PreviewWindows:
    """Manages one window per camera. Call `update` per camera each loop, then `poll` once."""

    def __init__(self):
        self._meters: dict[str, FpsMeter] = {}
        self._shapes: dict[str, tuple[int, ...]] = {}
        self._showing_no_signal: set[str] = set()
        self._opened: set[str] = set()
        self._closed_by_user = False

    def update(self, camera_id: str, image: np.ndarray | None, connected: bool,
               annotations: Iterable[Annotation] = ()) -> None:
        """Show a new frame, or a 'no signal' placeholder once the camera disconnects.

        Pass image=None when there is no new frame this loop; the window keeps its last content.
        """
        if image is not None and connected:
            meter = self._meters.setdefault(camera_id, FpsMeter())
            meter.tick()
            self._shapes[camera_id] = image.shape
            self._showing_no_signal.discard(camera_id)
            self._show(camera_id, draw_overlay(image, camera_id, meter.fps, annotations))
        elif not connected and camera_id not in self._showing_no_signal:
            self._meters.pop(camera_id, None)  # restart FPS after reconnecting
            self._showing_no_signal.add(camera_id)
            self._show(camera_id, draw_no_signal(camera_id, self._shapes.get(camera_id, _PLACEHOLDER_SHAPE)))

    def poll(self) -> bool:
        """Process window events. Returns False when the user pressed q/Esc or closed a window."""
        key = cv2.waitKey(1) & 0xFF
        return key not in _QUIT_KEYS and not self._closed_by_user

    def close(self) -> None:
        cv2.destroyAllWindows()
        self._opened.clear()

    def _show(self, camera_id: str, canvas: np.ndarray) -> None:
        # Check before imshow: imshow would silently re-create a window the user closed with X
        if camera_id in self._opened and cv2.getWindowProperty(camera_id, cv2.WND_PROP_VISIBLE) < 1:
            self._closed_by_user = True
            return
        cv2.imshow(camera_id, canvas)
        self._opened.add(camera_id)
