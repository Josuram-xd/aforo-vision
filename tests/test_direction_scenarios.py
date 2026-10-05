"""End-to-end direction scenarios with synthetic sequences: matcher -> resolver -> dedup (tasks 6.1-6.4).

Each person has a face vector and a body vector; what a camera "sees" of them depends on whether they
face it. Directions must come out right in every case, including walking backwards (ADR-003).
"""

from collections import deque

import numpy as np

from src.dedup.ttl_set import TTLSet
from src.detection.yolo_pose import NUM_KEYPOINTS
from src.direction.resolver import ENTRY, EXIT, resolve_direction, resolve_from_trajectory
from src.matching.cross_checkpoint import CrossCheckpointMatcher, Sighting

OUTSIDE, INSIDE = "camera-outside", "camera-inside"
WINDOW, FACE_THRESHOLD, BODY_THRESHOLD = 5.0, 0.45, 0.6


def _unit(angle_degrees):
    radians = np.radians(angle_degrees)
    return np.array([np.cos(radians), np.sin(radians)], dtype=np.float32)


class _Person:
    """Identity vectors; `seen_by(...)` builds a sighting with only what the camera could see."""

    def __init__(self, face_angle, body_angle):
        self.face, self.body = _unit(face_angle), _unit(body_angle)

    def seen_by(self, camera, track_id, timestamp, shows_face):
        return Sighting(camera, track_id, timestamp, face=self.face if shows_face else None, body=self.body)


ANA, BETO = _Person(0, 0), _Person(90, 80)


def _crossings(outside, inside, ttl_set=None, min_gap=0.0):
    """What the pipeline would emit: [(direction, method)], one per new crossing."""
    matcher = CrossCheckpointMatcher(WINDOW, FACE_THRESHOLD, BODY_THRESHOLD)
    matches, _, _ = matcher.match(outside, inside)
    events = []
    for match in matches:
        key = (match.outside.track_id, match.inside.track_id)
        if ttl_set is not None and not ttl_set.add(key):
            continue
        direction = resolve_direction(match, min_gap)
        if direction is not None:
            events.append((direction, match.method))
    return events


def test_entering_facing_forward_is_entry_by_face():
    out = ANA.seen_by(OUTSIDE, 1, 10.0, shows_face=True)
    ins = ANA.seen_by(INSIDE, 5, 12.0, shows_face=True)
    assert _crossings([out], [ins]) == [(ENTRY, "FACE")]


def test_leaving_is_exit():
    out = ANA.seen_by(OUTSIDE, 2, 32.0, shows_face=True)
    ins = ANA.seen_by(INSIDE, 6, 30.0, shows_face=True)
    assert _crossings([out], [ins]) == [(EXIT, "FACE")]


def test_entering_backwards_is_still_entry():
    # Walking backwards into the room: the outside camera sees the face, the inside camera only the back.
    # Faces cannot be compared with a body, so the pair is made by body appearance, and the order of
    # checkpoints still says ENTRY.
    out = ANA.seen_by(OUTSIDE, 1, 10.0, shows_face=True)
    ins = ANA.seen_by(INSIDE, 5, 12.0, shows_face=False)
    assert _crossings([out], [ins]) == [(ENTRY, "BODY_ONLY")]


def test_leaving_backwards_is_still_exit():
    out = ANA.seen_by(OUTSIDE, 2, 32.0, shows_face=False)
    ins = ANA.seen_by(INSIDE, 6, 30.0, shows_face=True)
    assert _crossings([out], [ins]) == [(EXIT, "BODY_ONLY")]


def test_entering_with_the_back_to_both_cameras_is_counted_and_entry():
    out = ANA.seen_by(OUTSIDE, 1, 10.0, shows_face=False)
    ins = ANA.seen_by(INSIDE, 5, 12.0, shows_face=False)
    assert _crossings([out], [ins]) == [(ENTRY, "BODY_ONLY")]


def test_one_person_enters_while_another_leaves_at_the_same_time():
    outside = [ANA.seen_by(OUTSIDE, 1, 10.0, True), BETO.seen_by(OUTSIDE, 2, 10.3, True)]
    inside = [BETO.seen_by(INSIDE, 8, 8.0, True), ANA.seen_by(INSIDE, 9, 12.0, True)]
    # Ana: outside 10.0 -> inside 12.0 (enters). Beto: inside 8.0 -> outside 10.3 (leaves).
    assert sorted(_crossings(outside, inside)) == [(ENTRY, "FACE"), (EXIT, "FACE")]


def test_two_people_entering_together_are_not_mixed_up():
    outside = [ANA.seen_by(OUTSIDE, 1, 10.0, True), BETO.seen_by(OUTSIDE, 2, 10.1, True)]
    inside = [BETO.seen_by(INSIDE, 8, 11.0, True), ANA.seen_by(INSIDE, 9, 11.2, True)]
    matcher = CrossCheckpointMatcher(WINDOW, FACE_THRESHOLD, BODY_THRESHOLD)
    matches, _, _ = matcher.match(outside, inside)
    assert sorted((m.outside.track_id, m.inside.track_id) for m in matches) == [(1, 9), (2, 8)]
    assert _crossings(outside, inside) == [(ENTRY, "FACE")] * 2


def test_flickering_detections_of_one_crossing_give_a_single_event():
    ttl_set = TTLSet(10.0, clock=lambda: 0.0)
    out = ANA.seen_by(OUTSIDE, 1, 10.0, True)
    ins = ANA.seen_by(INSIDE, 5, 12.0, True)
    events = []
    for _ in range(5):  # the same pair is matched again in every frame it flickers
        events += _crossings([out], [ins], ttl_set)
    assert events == [(ENTRY, "FACE")]


def test_ambiguous_order_gives_no_event_so_the_fallback_decides():
    out = ANA.seen_by(OUTSIDE, 1, 10.0, True)
    ins = ANA.seen_by(INSIDE, 5, 10.2, True)
    assert _crossings([out], [ins], min_gap=0.5) == []


def test_person_seen_by_one_camera_only_is_resolved_by_trajectory():
    def pose(torso):
        keypoints = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
        for index, y in ((5, 100.0), (6, 100.0), (11, 100.0 + torso), (12, 100.0 + torso)):
            keypoints[index] = (50.0, y, 0.9)
        return keypoints

    out = ANA.seen_by(OUTSIDE, 1, 10.0, True)
    matcher = CrossCheckpointMatcher(WINDOW, FACE_THRESHOLD, BODY_THRESHOLD)
    matches, unmatched_outside, unmatched_inside = matcher.match([out], [])
    assert matches == [] and unmatched_outside == [out] and unmatched_inside == []

    walking_away = deque(pose(size) for size in np.linspace(90, 40, 12))  # shrinking torso on camera-outside
    assert resolve_from_trajectory(walking_away, unmatched_outside[0].camera_id) == ENTRY


def test_a_stranger_with_a_different_look_is_not_paired_with_someone_else():
    stranger = _Person(180, 170)
    out = ANA.seen_by(OUTSIDE, 1, 10.0, True)
    ins = stranger.seen_by(INSIDE, 5, 11.0, True)
    assert _crossings([out], [ins]) == []
