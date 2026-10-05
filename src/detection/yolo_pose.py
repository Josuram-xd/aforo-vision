"""YOLO26n-pose person detector (CPU). Returns boxes, scores and COCO keypoints per frame."""

from typing import NamedTuple

import numpy as np

NUM_KEYPOINTS = 17  # COCO layout: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles


class Detection(NamedTuple):
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    keypoints: np.ndarray  # (17, 3): x, y, confidence per keypoint


class YoloPoseDetector:
    """Wraps an Ultralytics pose model. One instance per camera keeps inference state independent."""

    def __init__(self, model: str = "yolo26n-pose.pt", confidence_threshold: float = 0.5, model_instance=None):
        self.confidence_threshold = confidence_threshold
        if model_instance is None:
            from ultralytics import YOLO  # imported lazily: heavy import, and tests inject a stub

            model_instance = YOLO(model)
        self._model = model_instance

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run the model on a BGR frame and return the people found above the confidence threshold."""
        result = self._model.predict(frame, conf=self.confidence_threshold, device="cpu", verbose=False)[0]
        if result.boxes is None:
            return []

        boxes = _to_numpy(result.boxes.xyxy)
        scores = _to_numpy(result.boxes.conf)
        if len(boxes) == 0:
            return []
        keypoints = _keypoints_array(result.keypoints, len(boxes))
        return [
            Detection(float(x1), float(y1), float(x2), float(y2), float(score), kp)
            for (x1, y1, x2, y2), score, kp in zip(boxes, scores, keypoints)
        ]


def _keypoints_array(keypoints, count: int) -> np.ndarray:
    """Stack keypoint xy and confidence into (count, 17, 3); confidence is 1.0 if the model gives none."""
    if keypoints is None:
        return np.zeros((count, NUM_KEYPOINTS, 3), dtype=np.float32)
    xy = _to_numpy(keypoints.xy)
    conf = _to_numpy(keypoints.conf) if keypoints.conf is not None else np.ones(xy.shape[:2], dtype=np.float32)
    return np.concatenate([xy, conf[..., None]], axis=-1).astype(np.float32)


def _to_numpy(values) -> np.ndarray:
    # Ultralytics returns torch tensors; the stub in tests returns plain arrays
    return values.cpu().numpy() if hasattr(values, "cpu") else np.asarray(values)
