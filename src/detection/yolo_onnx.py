"""YOLO26n-pose person detector on onnxruntime (CPU). Same interface as YoloPoseDetector, no torch needed.

Model file comes from `python -m tools.export_yolo_onnx`. Output layout: (1, 56, N) =
4 box values (cx, cy, w, h in input pixels) + 1 person score + 17 x (x, y, conf) keypoints.
"""

import cv2
import numpy as np

from src.detection.yolo_pose import NUM_KEYPOINTS, Detection

_PAD_VALUE = 114  # same gray as Ultralytics' letterbox
_NMS_IOU = 0.7  # Ultralytics' default predict iou


def letterbox(frame: np.ndarray, size: int) -> tuple[np.ndarray, float, float, float]:
    """Resize keeping aspect ratio and pad to size x size. Returns (image, scale, pad_x, pad_y)."""
    h, w = frame.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - new_w) / 2, (size - new_h) / 2
    top, left = round(pad_y - 0.1), round(pad_x - 0.1)
    out = cv2.copyMakeBorder(resized, top, size - new_h - top, left, size - new_w - left,
                             cv2.BORDER_CONSTANT, value=(_PAD_VALUE,) * 3)
    return out, scale, float(left), float(top)


def decode(output: np.ndarray, confidence_threshold: float, scale: float, pad_x: float, pad_y: float,
           frame_shape: tuple[int, int]) -> list[Detection]:
    """Turn the raw (1, 56, N) model output into NMS-filtered Detections in original frame coordinates."""
    preds = output[0].T  # (N, 56)
    preds = preds[preds[:, 4] >= confidence_threshold]
    if len(preds) == 0:
        return []

    cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    scores = preds[:, 4]
    keep = cv2.dnn.NMSBoxes(
        np.stack([cx - bw / 2, cy - bh / 2, bw, bh], axis=1).tolist(), scores.tolist(), confidence_threshold, _NMS_IOU)
    keep = np.asarray(keep, dtype=int).reshape(-1)

    h, w = frame_shape
    detections = []
    for i in keep:
        x1, y1 = np.clip((cx[i] - bw[i] / 2 - pad_x) / scale, 0, w), np.clip((cy[i] - bh[i] / 2 - pad_y) / scale, 0, h)
        x2, y2 = np.clip((cx[i] + bw[i] / 2 - pad_x) / scale, 0, w), np.clip((cy[i] + bh[i] / 2 - pad_y) / scale, 0, h)
        kp = preds[i, 5:5 + NUM_KEYPOINTS * 3].reshape(NUM_KEYPOINTS, 3).astype(np.float32).copy()
        kp[:, 0] = (kp[:, 0] - pad_x) / scale
        kp[:, 1] = (kp[:, 1] - pad_y) / scale
        detections.append(Detection(float(x1), float(y1), float(x2), float(y2), float(scores[i]), kp))
    return detections


class OnnxPoseDetector:
    """Drop-in replacement for YoloPoseDetector backed by onnxruntime."""

    def __init__(self, model: str = "models/yolo26n-pose.onnx", confidence_threshold: float = 0.5, session=None):
        self.confidence_threshold = confidence_threshold
        if session is None:
            import onnxruntime as ort  # lazy: tests inject a stub

            session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
        self._session = session
        self._input_name = session.get_inputs()[0].name
        self._size = int(session.get_inputs()[0].shape[2])  # fixed square input chosen at export time

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run the model on a BGR frame and return the people found above the confidence threshold."""
        image, scale, pad_x, pad_y = letterbox(frame, self._size)
        blob = np.ascontiguousarray(image[..., ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        output = self._session.run(None, {self._input_name: blob})[0]
        return decode(output, self.confidence_threshold, scale, pad_x, pad_y, frame.shape[:2])
