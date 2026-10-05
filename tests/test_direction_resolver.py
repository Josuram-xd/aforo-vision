from collections import deque

import numpy as np
import pytest

from src.detection.yolo_pose import NUM_KEYPOINTS
from src.direction.resolver import (
    APPROACHING,
    ENTRY,
    EXIT,
    RECEDING,
    resolve_direction,
    resolve_from_trajectory,
    torso_height,
)
from src.matching.cross_checkpoint import CrossCheckpointMatcher, CrossingMatch, Sighting


def _match(t_outside, t_inside, **kwargs):
    return CrossingMatch(
        Sighting("camera-outside", 1, t_outside, **kwargs),
        Sighting("camera-inside", 2, t_inside, **kwargs),
        similarity=0.9,
        method="FACE",
    )


def test_outside_before_inside_is_entry():
    assert resolve_direction(_match(10.0, 12.0)) == ENTRY


def test_inside_before_outside_is_exit():
    assert resolve_direction(_match(12.0, 10.0)) == EXIT


def test_simultaneous_sightings_are_undecidable():
    assert resolve_direction(_match(10.0, 10.0)) is None


def test_gap_within_min_gap_is_undecidable_but_larger_gap_is_trusted():
    assert resolve_direction(_match(10.0, 10.4), min_gap_seconds=0.5) is None
    assert resolve_direction(_match(10.4, 10.0), min_gap_seconds=0.5) is None
    assert resolve_direction(_match(10.0, 10.5), min_gap_seconds=0.5) is None  # boundary is not trusted
    assert resolve_direction(_match(10.0, 10.6), min_gap_seconds=0.5) == ENTRY
    assert resolve_direction(_match(10.6, 10.0), min_gap_seconds=0.5) == EXIT


def test_event_values_match_the_shared_contract():
    assert (ENTRY, EXIT) == ("ENTRY", "EXIT")


def test_direction_only_depends_on_timestamps_not_on_the_embeddings():
    # Same person walking backwards looks different to the camera, but the order of checkpoints is the same.
    forwards = np.array([1.0, 0.0], dtype=np.float32)
    backwards = np.array([0.8, 0.6], dtype=np.float32)
    assert resolve_direction(_match(10.0, 12.0, face=forwards)) == ENTRY
    assert resolve_direction(_match(10.0, 12.0, face=backwards)) == ENTRY


def test_works_end_to_end_with_the_cross_checkpoint_matcher():
    face = np.array([1.0, 0.0], dtype=np.float32)
    matcher = CrossCheckpointMatcher(5.0, 0.45, 0.6)
    entering, _, _ = matcher.match([Sighting("camera-outside", 1, 10.0, face=face)], [Sighting("camera-inside", 9, 11.5, face=face)])
    leaving, _, _ = matcher.match([Sighting("camera-outside", 2, 30.0, face=face)], [Sighting("camera-inside", 8, 28.0, face=face)])
    assert resolve_direction(entering[0]) == ENTRY
    assert resolve_direction(leaving[0]) == EXIT


# --- trajectory fallback (6.3) ---

def _pose(torso, confidence=0.9, x=100.0):
    """Keypoints with shoulders at y=100 and hips `torso` pixels lower."""
    keypoints = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
    for index, y in ((5, 100.0), (6, 100.0), (11, 100.0 + torso), (12, 100.0 + torso)):
        keypoints[index] = (x + index, y, confidence)
    return keypoints


def _history(sizes, **kwargs):
    return deque(_pose(size, **kwargs) for size in sizes)


GROWING = np.linspace(40, 90, 12)  # walking towards the camera
SHRINKING = GROWING[::-1]  # walking away


def test_torso_height_needs_all_four_keypoints_visible():
    assert torso_height(_pose(60)) == pytest.approx(60.0)
    assert torso_height(_pose(60, confidence=0.3)) is None
    partial = _pose(60)
    partial[11, 2] = 0.1  # one hip hidden
    assert torso_height(partial) is None
    assert torso_height(np.zeros((3, 3))) is None


def test_outside_camera_receding_is_entry_and_approaching_is_exit():
    assert resolve_from_trajectory(_history(SHRINKING), "camera-outside") == ENTRY
    assert resolve_from_trajectory(_history(GROWING), "camera-outside") == EXIT


def test_inside_camera_mapping_is_the_opposite():
    assert resolve_from_trajectory(_history(GROWING), "camera-inside") == ENTRY
    assert resolve_from_trajectory(_history(SHRINKING), "camera-inside") == EXIT


def test_mapping_can_be_overridden_for_a_different_camera_mounting():
    flipped = {"camera-outside": APPROACHING, "camera-inside": RECEDING}
    assert resolve_from_trajectory(_history(GROWING), "camera-outside", entry_motion=flipped) == ENTRY


def test_walking_backwards_does_not_change_the_result():
    # Facing the other way mirrors the pose left/right but the torso still shrinks as the person walks away.
    mirrored = deque(kp[[0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]] for kp in _history(SHRINKING))
    assert resolve_from_trajectory(mirrored, "camera-outside") == ENTRY


def test_no_conclusion_when_the_person_barely_changes_size():
    assert resolve_from_trajectory(_history(np.linspace(60, 62, 12)), "camera-outside") is None
    assert resolve_from_trajectory(_history([60] * 12), "camera-outside") is None


def test_no_conclusion_with_too_few_visible_frames():
    assert resolve_from_trajectory(_history(GROWING[:4]), "camera-inside") is None
    hidden = deque(_pose(size, confidence=0.2) for size in GROWING)
    assert resolve_from_trajectory(hidden, "camera-inside") is None
    assert resolve_from_trajectory(deque(), "camera-inside") is None


def test_frames_with_hidden_torso_are_skipped_not_fatal():
    history = _history(GROWING)
    for index in (2, 5, 8):
        history[index] = _pose(60, confidence=0.1)
    assert resolve_from_trajectory(history, "camera-inside") == ENTRY


def test_noise_around_a_clear_trend_still_gives_the_trend():
    rng = np.random.default_rng(0)
    noisy = SHRINKING + rng.normal(0, 2, SHRINKING.shape)
    assert resolve_from_trajectory(_history(noisy), "camera-outside") == ENTRY


def test_unknown_camera_gives_no_direction():
    assert resolve_from_trajectory(_history(GROWING), "camera-garage") is None
