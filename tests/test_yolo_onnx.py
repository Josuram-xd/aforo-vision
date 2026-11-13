from types import SimpleNamespace

import numpy as np

from src.detection.yolo_onnx import OnnxPoseDetector, decode, letterbox


def _output(rows):
    """Build a (1, 56, N) tensor from (cx, cy, w, h, score) rows with keypoints at the box center."""
    out = np.zeros((1, 56, max(len(rows), 1)), dtype=np.float32)
    for i, (cx, cy, w, h, score) in enumerate(rows):
        out[0, :5, i] = (cx, cy, w, h, score)
        out[0, 5:, i] = np.tile([cx, cy, 0.9], 17)
    return out


def test_no_detections_below_threshold():
    assert decode(_output([(100, 100, 50, 100, 0.2)]), 0.5, 1.0, 0, 0, (640, 640)) == []
    assert decode(_output([]), 0.5, 1.0, 0, 0, (640, 640)) == []


def test_single_person_box_score_and_keypoints():
    (det,) = decode(_output([(100, 200, 40, 100, 0.9)]), 0.5, 1.0, 0, 0, (640, 640))
    assert (det.x1, det.y1, det.x2, det.y2) == (80, 150, 120, 250)
    assert abs(det.score - 0.9) < 1e-6
    assert det.keypoints.shape == (17, 3) and tuple(det.keypoints[0]) == (100, 200, np.float32(0.9))


def test_overlapping_boxes_merge_and_distant_ones_stay():
    rows = [(100, 100, 60, 120, 0.9), (102, 101, 60, 120, 0.8), (400, 100, 60, 120, 0.7)]
    dets = decode(_output(rows), 0.5, 1.0, 0, 0, (640, 640))
    assert sorted(round(d.score, 1) for d in dets) == [0.7, 0.9]


def test_letterbox_roundtrip_restores_original_coordinates():
    frame = np.zeros((480, 1280, 3), dtype=np.uint8)  # wide frame -> vertical padding
    image, scale, pad_x, pad_y = letterbox(frame, 640)
    assert image.shape == (640, 640, 3) and scale == 0.5 and pad_x == 0 and pad_y == 200
    # a box at original (400, 100)-(600, 300) lives at (200, 250)-(300, 350) in model space
    out = _output([(250, 300, 100, 100, 0.9)])
    (det,) = decode(out, 0.5, scale, pad_x, pad_y, frame.shape[:2])
    assert (det.x1, det.y1, det.x2, det.y2) == (400, 100, 600, 300)
    assert tuple(det.keypoints[0][:2]) == (500, 200)


def test_detector_uses_session_and_scales_to_frame():
    class Session:
        def get_inputs(self):
            return [SimpleNamespace(name="images", shape=[1, 3, 640, 640])]

        def run(self, _, feed):
            assert feed["images"].shape == (1, 3, 640, 640) and feed["images"].dtype == np.float32
            return [_output([(320, 320, 100, 200, 0.95)])]

    dets = OnnxPoseDetector(session=Session()).detect(np.zeros((320, 640, 3), dtype=np.uint8))
    assert len(dets) == 1 and abs(dets[0].x1 - 270) < 1e-3 and abs(dets[0].y1 - 60) < 1e-3
