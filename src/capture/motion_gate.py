"""Per-camera motion gate: decides whether a frame is worth sending to the (expensive) detector.

Works on a tiny grayscale crop of a configurable region of interest (ROI), compares it with a slow-moving
background and keeps the gate open for `hold_seconds` after the last motion. Only OpenCV and numpy.
"""

import cv2
import numpy as np

_WORK_WIDTH = 160  # width of the full image after downscaling; the ROI crop is scaled proportionally
_BACKGROUND_SECONDS = 20.0  # time constant of the moving-average background (slow lighting changes get absorbed)


class MotionGate:
    def __init__(self, roi=(0.0, 0.0, 1.0, 1.0), pixel_threshold: int = 25, min_area: float = 0.01,
                 hold_seconds: float = 8.0, idle_fps: float = 4.0, warmup_seconds: float = 3.0):
        self.roi = tuple(float(v) for v in roi)
        self.pixel_threshold = pixel_threshold
        self.min_area = min_area  # fraction of the ROI that must change
        self.hold_seconds = hold_seconds
        self.idle_interval = 1.0 / idle_fps
        self.warmup_seconds = warmup_seconds
        self.processed = 0
        self.skipped = 0
        self.change = 0.0  # last measured fraction of changed ROI pixels (for the calibration tool)
        self.idle = False  # True when the last decision was "skip"; main sleeps longer when all cameras are idle
        self._background: np.ndarray | None = None
        self._last_time = 0.0
        self._last_motion = float("-inf")
        self._activated_at = float("-inf")
        self._next_check = float("-inf")

    def update(self, image: np.ndarray, now: float) -> bool:
        """Feed one frame; True while motion is present or was seen within `hold_seconds`."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = self.roi
        crop = image[int(y1 * h):max(int(y1 * h) + 1, int(y2 * h)), int(x1 * w):max(int(x1 * w) + 1, int(x2 * w))]
        size = (max(8, round(_WORK_WIDTH * (x2 - x1))), max(8, round(_WORK_WIDTH * (y2 - y1) * h / w)))
        small = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)  # shrink first: everything below is tiny
        if small.ndim == 3:
            small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        small = cv2.GaussianBlur(small, (5, 5), 0).astype(np.float32)

        if self._background is None or self._background.shape != small.shape:
            self._background = small
            self._last_time = now
            self.change = 0.0
        else:
            self.change = float(np.count_nonzero(cv2.absdiff(small, self._background) > self.pixel_threshold)) / small.size
            cv2.accumulateWeighted(small, self._background, min(0.5, max(0.0, now - self._last_time) / _BACKGROUND_SECONDS))
            self._last_time = now
        if self.change >= self.min_area:
            if now - self._last_motion > self.hold_seconds:
                self._activated_at = now
            self._last_motion = now
        return now - self._last_motion <= self.hold_seconds

    def in_roi(self, box, shape) -> bool:
        """True when the center of a pixel box (x1, y1, x2, y2) lies inside the ROI of an image with this shape."""
        cx, cy = (box[0] + box[2]) / 2 / shape[1], (box[1] + box[3]) / 2 / shape[0]
        return self.roi[0] <= cx <= self.roi[2] and self.roi[1] <= cy <= self.roi[3]

    def should_process(self, image: np.ndarray, now: float, live_boxes=()) -> bool:
        """Decide if this frame goes to the pipeline: motion (or hold), a live track in the ROI, or warmup.

        While idle the frame is only inspected every 1/idle_fps seconds.
        """
        if self.idle and now < self._next_check:
            self.skipped += 1
            return False
        active = self.update(image, now)
        tracked = any(self.in_roi(box, image.shape) for box in live_boxes)
        process = active or tracked or now - self._activated_at < self.warmup_seconds
        self.idle = not process
        self._next_check = now + self.idle_interval
        if process:
            self.processed += 1
        else:
            self.skipped += 1
        return process

    def summary(self) -> str:
        total = self.processed + self.skipped
        return f"procesados {self.processed} / omitidos {self.skipped} ({100 * self.skipped / total:.0f}% omitido)" if total else "sin fotogramas"
