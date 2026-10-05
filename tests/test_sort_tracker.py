import numpy as np

from src.detection.yolo_pose import NUM_KEYPOINTS, Detection
from src.tracking.sort_tracker import KalmanBoxFilter, SortTracker


def _det(x1, y1, x2, y2, score=0.9):
    return Detection(x1, y1, x2, y2, score, np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32))


def _person(x, y=100, w=50, h=120):
    return _det(x, y, x + w, y + h)


def test_kalman_filter_starts_at_the_first_box():
    box = (10, 20, 60, 140)
    assert np.allclose(KalmanBoxFilter(box).box, box)


def test_kalman_filter_extrapolates_constant_velocity():
    kf = KalmanBoxFilter((0, 0, 50, 100))
    for step in range(1, 8):  # moves 10 px right per frame
        kf.predict()
        kf.update((10 * step, 0, 50 + 10 * step, 100))
    x1_before = kf.box[0]
    x1_predicted = kf.predict()[0]
    assert 7 < x1_predicted - x1_before < 13


def test_track_is_confirmed_only_after_min_hits():
    tracker = SortTracker(min_hits=3)
    assert tracker.update([_person(100)]) == []
    assert tracker.update([_person(102)]) == []
    confirmed = tracker.update([_person(104)])
    assert [t.id for t in confirmed] == [1]


def test_id_stays_stable_while_a_person_walks():
    tracker = SortTracker(min_hits=1)
    ids = {tracker.update([_person(100 + 8 * frame)])[0].id for frame in range(40)}
    assert ids == {1}


def test_two_people_keep_distinct_ids():
    tracker = SortTracker(min_hits=1)
    seen = {}
    for frame in range(20):
        tracks = tracker.update([_person(50 + 5 * frame), _person(400 - 5 * frame)])
        for track in tracks:
            seen.setdefault("left" if track.box[0] < 220 else "right", set()).add(track.id)
    assert seen["left"] == {1}
    assert seen["right"] == {2}


def test_track_survives_a_short_gap_and_keeps_its_id():
    tracker = SortTracker(max_age_frames=10, min_hits=1)
    for frame in range(5):
        tracker.update([_person(100 + 5 * frame)])
    for _ in range(4):  # person hidden
        assert tracker.update([]) == []
    (track,) = tracker.update([_person(100 + 5 * 9)])
    assert track.id == 1


def test_track_is_dropped_after_max_age_and_a_new_id_is_issued():
    tracker = SortTracker(max_age_frames=3, min_hits=1)
    tracker.update([_person(100)])
    for _ in range(5):
        tracker.update([])
    (track,) = tracker.update([_person(100)])
    assert track.id == 2


def test_unmatched_far_detection_creates_a_new_track():
    tracker = SortTracker(min_hits=1)
    tracker.update([_person(100)])
    tracks = tracker.update([_person(102), _person(500)])
    assert sorted(t.id for t in tracks) == [1, 2]


def test_returned_track_keeps_the_last_detection_with_keypoints():
    tracker = SortTracker(min_hits=1)
    detection = _person(100)
    (track,) = tracker.update([detection])
    assert track.detection is detection
    assert track.detection.keypoints.shape == (NUM_KEYPOINTS, 3)


def test_track_keeps_centroids_and_keypoints_of_each_matched_detection():
    tracker = SortTracker(min_hits=1, trajectory_length=10)
    for frame in range(3):
        (track,) = tracker.update([_person(100 + 10 * frame)])
    assert list(track.centroids) == [(125.0, 160.0), (135.0, 160.0), (145.0, 160.0)]
    assert len(track.keypoint_history) == 3
    assert track.keypoint_history[-1].shape == (NUM_KEYPOINTS, 3)


def test_trajectory_is_capped_at_trajectory_length_dropping_the_oldest():
    tracker = SortTracker(min_hits=1, trajectory_length=5)
    for frame in range(12):
        (track,) = tracker.update([_person(100 + 2 * frame)])
    assert len(track.centroids) == 5
    assert len(track.keypoint_history) == 5
    assert track.centroids[-1] == (100 + 2 * 11 + 25.0, 160.0)
    assert track.centroids[0] == (100 + 2 * 7 + 25.0, 160.0)


def test_frames_without_a_detection_do_not_add_to_the_trajectory():
    tracker = SortTracker(max_age_frames=10, min_hits=1)
    (track,) = tracker.update([_person(100)])
    tracker.update([])
    tracker.update([])
    assert len(track.centroids) == 1
