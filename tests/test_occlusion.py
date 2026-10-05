import numpy as np

from src.detection.yolo_pose import NUM_KEYPOINTS, Detection
from src.tracking.hungarian import associate
from src.tracking.occlusion import keypoint_similarity, occlusion_aware_similarity
from src.tracking.sort_tracker import SortTracker


def _pose(x1, y1, x2, y2, side):
    """Keypoints on a vertical line at 20% (side 'A') or 80% (side 'B') of the box width: a fake body signature."""
    kp = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
    kp[:, 0] = x1 + (x2 - x1) * (0.2 if side == "A" else 0.8)
    kp[:, 1] = np.linspace(y1 + 0.05 * (y2 - y1), y1 + 0.95 * (y2 - y1), NUM_KEYPOINTS)
    kp[:, 2] = 1.0
    return kp


def _det(x1, y1, x2, y2, side=None):
    keypoints = _pose(x1, y1, x2, y2, side) if side else np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
    return Detection(x1, y1, x2, y2, 0.9, keypoints)


def test_identical_keypoints_are_fully_similar():
    kp = _pose(0, 0, 100, 200, "A")
    assert keypoint_similarity(kp, kp, 200) == 1.0


def test_distant_keypoints_have_zero_similarity():
    assert keypoint_similarity(_pose(0, 0, 100, 200, "A"), _pose(0, 0, 100, 200, "B"), 200) == 0.0


def test_too_few_visible_keypoints_gives_no_opinion():
    kp = _pose(0, 0, 100, 200, "A")
    hidden = kp.copy()
    hidden[:, 2] = 0.1
    assert keypoint_similarity(kp, hidden, 200) is None
    few = kp.copy()
    few[2:, 2] = 0.0  # only 2 visible
    assert keypoint_similarity(few, kp, 200) is None


def test_uncontested_pairs_keep_plain_iou():
    boxes = [(0, 0, 10, 10), (100, 100, 110, 110)]
    kps = [_pose(*b, "A") for b in boxes]
    similarity = occlusion_aware_similarity(boxes, kps, boxes, kps)
    assert similarity == [[1.0, 0.0], [0.0, 1.0]]


def test_keypoints_override_iou_when_boxes_overlap():
    track_boxes = [(100, 100, 200, 220), (110, 100, 210, 220)]
    track_kps = [_pose(*track_boxes[0], "A"), _pose(*track_boxes[1], "B")]
    # Same boxes as the tracks, but the people swapped places: IoU alone pairs them wrongly
    det_boxes = track_boxes
    det_kps = [_pose(*det_boxes[0], "B"), _pose(*det_boxes[1], "A")]

    plain, *_ = associate(det_boxes, track_boxes, 0.3)
    aware, *_ = associate(det_boxes, track_boxes, 0.3,
                          occlusion_aware_similarity(det_boxes, det_kps, track_boxes, track_kps))

    assert sorted(plain) == [(0, 0), (1, 1)]
    assert sorted(aware) == [(0, 1), (1, 0)]


def test_tracker_keeps_two_overlapping_people_apart():
    tracker = SortTracker(min_hits=1)
    box_a, box_b = (100, 100, 200, 220), (110, 100, 210, 220)
    first = [_det(*box_a, "A"), _det(*box_b, "B")]
    tracks = tracker.update(first)
    id_of = {t.id: "A" if t.detection is first[0] else "B" for t in tracks}

    second = [_det(*box_a, "B"), _det(*box_b, "A")]  # swapped places
    for track in tracker.update(second):
        who = "B" if track.detection is second[0] else "A"
        assert id_of[track.id] == who


def test_tracker_falls_back_to_iou_without_visible_keypoints():
    tracker = SortTracker(min_hits=1)
    tracker.update([_det(100, 100, 200, 220), _det(400, 100, 500, 220)])
    tracks = tracker.update([_det(102, 100, 202, 220), _det(402, 100, 502, 220)])
    assert sorted(t.id for t in tracks) == [1, 2]
