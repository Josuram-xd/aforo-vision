"""Periocular crop: the eye region of a face, cut and straightened so it works with face masks.

This is only a crop. The embedding comes from a standard face model (ArcFace/InsightFace) in
src/identity/ - there is no dedicated "periocular model" (AGENTS.md rule 2).
"""

import math

import cv2
import numpy as np

WIDTH_FACTOR = 2.0  # crop width in inter-ocular distances: both eyes plus a margin on each side
HEIGHT_FACTOR = 1.0  # crop height in inter-ocular distances: brows to upper cheeks, never nose or mouth
MIN_EYE_DISTANCE = 8.0  # pixels; closer eyes are too small to give a usable crop


def crop_periocular(
    frame: np.ndarray, landmarks: np.ndarray, output_size: tuple[int, int] | None = None
) -> np.ndarray | None:
    """Cut the eye region out of a BGR `frame`, rotated so the eyes lie on a horizontal line.

    `landmarks` is an (N, 2) array of x, y points whose first two rows are the eyes (the 5-point
    layout of InsightFace starts with them); their order does not matter. `output_size` is
    (width, height) in pixels; by default the crop keeps its natural size in the frame.

    Returns None when the eyes are missing, not finite, too close together or outside the frame.
    """
    points = np.asarray(landmarks, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2 or not np.isfinite(points[:2, :2]).all():
        return None

    first, second = points[0, :2], points[1, :2]
    left, right = (first, second) if first[0] <= second[0] else (second, first)  # left = smaller x in the image
    distance = float(np.hypot(*(right - left)))
    if distance < MIN_EYE_DISTANCE:
        return None
    height, width = frame.shape[:2]
    if any(not (0 <= x < width and 0 <= y < height) for x, y in (left, right)):
        return None

    natural_size = (round(WIDTH_FACTOR * distance), round(HEIGHT_FACTOR * distance))
    out_width, out_height = output_size or natural_size
    angle = math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))
    center = (left + right) / 2

    # Rotate around the eye midpoint to level the eyes, scale to the output size, then move the midpoint to the middle
    matrix = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle, out_width / natural_size[0])
    matrix[0, 2] += out_width / 2 - center[0]
    matrix[1, 2] += out_height / 2 - center[1]
    return cv2.warpAffine(frame, matrix, (out_width, out_height), flags=cv2.INTER_LINEAR)
