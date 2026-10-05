from types import SimpleNamespace

import numpy as np

from src.detection.yolo_pose import NUM_KEYPOINTS, YoloPoseDetector


class _StubModel:
    def __init__(self, result):
        self._result = result
        self.calls = []

    def predict(self, frame, **kwargs):
        self.calls.append(kwargs)
        return [self._result]


def _result(boxes, scores, kp_xy=None, kp_conf=None):
    keypoints = None if kp_xy is None else SimpleNamespace(xy=kp_xy, conf=kp_conf)
    return SimpleNamespace(boxes=SimpleNamespace(xyxy=np.array(boxes, dtype=np.float32).reshape(-1, 4),
                                                 conf=np.array(scores, dtype=np.float32)),
                           keypoints=keypoints)


_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def test_returns_empty_list_when_nobody_is_detected():
    detector = YoloPoseDetector(model_instance=_StubModel(_result([], [])))
    assert detector.detect(_FRAME) == []


def test_returns_box_score_and_keypoints_per_person():
    kp_xy = np.random.rand(2, NUM_KEYPOINTS, 2).astype(np.float32)
    kp_conf = np.random.rand(2, NUM_KEYPOINTS).astype(np.float32)
    stub = _StubModel(_result([[10, 20, 110, 220], [300, 40, 380, 300]], [0.9, 0.7], kp_xy, kp_conf))

    detections = YoloPoseDetector(model_instance=stub).detect(_FRAME)

    assert len(detections) == 2
    first = detections[0]
    assert (first.x1, first.y1, first.x2, first.y2) == (10, 20, 110, 220)
    assert first.score == np.float32(0.9)
    assert first.keypoints.shape == (NUM_KEYPOINTS, 3)
    np.testing.assert_allclose(first.keypoints[:, :2], kp_xy[0])
    np.testing.assert_allclose(first.keypoints[:, 2], kp_conf[0])


def test_missing_keypoint_confidence_defaults_to_one():
    kp_xy = np.zeros((1, NUM_KEYPOINTS, 2), dtype=np.float32)
    stub = _StubModel(_result([[0, 0, 50, 100]], [0.8], kp_xy, None))

    (detection,) = YoloPoseDetector(model_instance=stub).detect(_FRAME)

    assert (detection.keypoints[:, 2] == 1.0).all()


def test_passes_threshold_and_cpu_to_the_model():
    stub = _StubModel(_result([], []))
    YoloPoseDetector(confidence_threshold=0.42, model_instance=stub).detect(_FRAME)
    assert stub.calls[0]["conf"] == 0.42
    assert stub.calls[0]["device"] == "cpu"
