import uuid
from types import SimpleNamespace

import numpy as np

from src.dedup.ttl_set import TTLSet
from src.detection.yolo_pose import NUM_KEYPOINTS
from src.identity.index import IdentityIndex
from src.matching.cross_checkpoint import CrossCheckpointMatcher
from src.pipeline import EMBED_EVERY_FRAMES, READY_BODY_SAMPLES, Pipeline

OUTSIDE, INSIDE = "camera-outside", "camera-inside"
ANA_ID = str(uuid.uuid4())
FRAME_DT = 0.1
# first frame (1) + every Nth after it: the READY_BODY_SAMPLES-th sample happens on this frame
READY_FRAME = 1 + EMBED_EVERY_FRAMES * (READY_BODY_SAMPLES - 1)


def _unit(angle):
    r = np.radians(angle)
    return np.array([np.cos(r), np.sin(r)], dtype=np.float32)


# person code -> (face vector or None, body vector); the code is painted into the frame
PEOPLE = {
    1: (_unit(0), _unit(0)),  # Ana, enrolled
    2: (None, _unit(0)),  # Ana from behind: no face, same body
    3: (_unit(90), _unit(90)),  # a stranger: face and body far from Ana
    4: (_unit(180), _unit(0)),  # a different face on a body that looks like Ana's
    5: (None, _unit(90)),  # the stranger from behind
}


class _Extractor:
    def __init__(self):
        self.calls = 0

    def extract(self, image, box):
        self.calls += 1
        return PEOPLE[int(image[0, 0, 0])]


class _Tracker:
    """Returns whatever tracks the test put in `tracks` for the next frame."""

    tracks = ()

    def update(self, detections):
        return list(self.tracks)


class _Detector:
    def detect(self, image):
        return []


def _pose(torso):
    keypoints = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
    for index, y in ((5, 100.0), (6, 100.0), (11, 100.0 + torso), (12, 100.0 + torso)):
        keypoints[index] = (50.0, y, 0.9)
    return keypoints


def _build(single_camera_fallback=True, enrolled=True):
    events = []
    index = IdentityIndex(0.45, 0.05)
    if enrolled:
        index.add(ANA_ID, _unit(0))
    pipeline = Pipeline(
        detectors={OUTSIDE: _Detector(), INSIDE: _Detector()},
        trackers={OUTSIDE: _Tracker(), INSIDE: _Tracker()},
        extractor=_Extractor(),
        matcher=CrossCheckpointMatcher(5.0, 0.45, 0.6),
        identity_index=index,
        names={ANA_ID: "Ana Pérez"},
        dedup=TTLSet(10.0, clock=lambda: 0.0),
        emit=events.append,
        single_camera_fallback=single_camera_fallback,
    )
    return pipeline, events


def _see(pipeline, camera, person, track_id, start, frames=READY_FRAME + 2, history=()):
    """Feed `frames` frames in which `person` is tracked as `track_id`. Returns the time of the last frame."""
    image = np.full((60, 40, 3), person, dtype=np.uint8)
    track = SimpleNamespace(id=track_id, box=(0, 0, 40, 60), keypoint_history=list(history))
    pipeline.trackers[camera].tracks = [track]
    timestamp = start
    for _ in range(frames):
        pipeline.process_frame(camera, image, timestamp)
        timestamp += FRAME_DT
    pipeline.trackers[camera].tracks = []
    return timestamp


def test_entering_with_face_visible_to_both_cameras_names_the_student():
    pipeline, events = _build()
    _see(pipeline, OUTSIDE, 1, 1, start=10.0)
    _see(pipeline, INSIDE, 1, 5, start=12.0)
    assert len(events) == 1
    event = events[0]
    assert event["direction"] == "ENTRY" and event["method"] == "FACE"
    assert event["personId"] == ANA_ID and event["personName"] == "Ana Pérez"
    assert 0.9 < event["confidence"] <= 1.0


def test_leaving_is_exit():
    pipeline, events = _build()
    _see(pipeline, INSIDE, 1, 5, start=10.0)
    _see(pipeline, OUTSIDE, 1, 1, start=12.0)
    assert [e["direction"] for e in events] == ["EXIT"] and events[0]["personId"] == ANA_ID


def test_unenrolled_person_is_counted_by_face_without_a_name():
    pipeline, events = _build()
    _see(pipeline, OUTSIDE, 3, 1, start=10.0)
    _see(pipeline, INSIDE, 3, 5, start=12.0)
    assert len(events) == 1 and events[0]["method"] == "FACE" and events[0]["personId"] is None


def test_body_only_pair_is_counted_and_never_named():
    pipeline, events = _build()
    _see(pipeline, OUTSIDE, 2, 1, start=10.0)  # Ana's body, no face on either camera
    _see(pipeline, INSIDE, 2, 5, start=12.0)
    assert len(events) == 1
    assert events[0]["method"] == "BODY_ONLY" and events[0]["personId"] is None and events[0]["personName"] is None


def test_walking_backwards_is_still_counted_as_entry_but_not_named():
    # Face seen outside (recognized), only the back inside: paired by body, so no name is attached
    pipeline, events = _build()
    _see(pipeline, OUTSIDE, 1, 1, start=10.0)
    _see(pipeline, INSIDE, 2, 5, start=12.0)
    assert [(e["direction"], e["method"], e["personId"]) for e in events] == [("ENTRY", "BODY_ONLY", None)]


def test_two_faces_that_clearly_differ_are_not_paired_even_if_the_bodies_look_alike():
    # outside: Ana; inside: someone with an Ana-like body but another face (person 4). Faces win over bodies.
    pipeline, events = _build()
    _see(pipeline, OUTSIDE, 1, 1, start=10.0)
    _see(pipeline, INSIDE, 4, 5, start=12.0)
    assert events == []


def test_two_people_crossing_in_opposite_directions_are_not_mixed_up():
    pipeline, events = _build()
    _see(pipeline, OUTSIDE, 1, 1, start=10.0)  # Ana walks in
    _see(pipeline, INSIDE, 3, 6, start=10.5)  # the stranger is already inside and walks out
    _see(pipeline, INSIDE, 1, 5, start=12.0)
    _see(pipeline, OUTSIDE, 3, 2, start=12.5)
    assert sorted((e["direction"], e["personId"]) for e in events) == sorted([("ENTRY", ANA_ID), ("EXIT", None)])


def test_same_student_same_direction_twice_in_a_row_is_one_event():
    pipeline, events = _build()
    _see(pipeline, OUTSIDE, 1, 1, start=10.0)
    _see(pipeline, INSIDE, 1, 5, start=12.0)
    _see(pipeline, OUTSIDE, 1, 2, start=12.2)  # the track broke in two: a new pair for the same crossing
    _see(pipeline, INSIDE, 1, 6, start=12.4)
    assert len(events) == 1


def test_embeddings_are_computed_on_a_cadence_and_stop_once_the_track_is_a_sighting():
    pipeline, _ = _build()
    _see(pipeline, OUTSIDE, 1, 1, start=10.0, frames=READY_FRAME + 30)
    assert pipeline.extractor.calls == READY_BODY_SAMPLES


def test_unmatched_sighting_is_resolved_by_trajectory_after_the_window():
    pipeline, events = _build()
    receding = [_pose(h) for h in np.linspace(90, 40, 12)]  # shrinking torso on camera-outside = walking in
    end = _see(pipeline, OUTSIDE, 3, 1, start=10.0, history=receding)
    pipeline.expire(end + 1.0)
    assert events == []  # the partner could still arrive
    pipeline.expire(end + 6.0)
    assert len(events) == 1
    assert (events[0]["direction"], events[0]["method"], events[0]["confidence"]) == ("ENTRY", "BODY_ONLY", 0.4)
    pipeline.expire(end + 20.0)
    assert len(events) == 1  # not emitted twice


def test_single_camera_fallback_can_be_turned_off():
    pipeline, events = _build(single_camera_fallback=False)
    end = _see(pipeline, OUTSIDE, 3, 1, start=10.0, history=[_pose(h) for h in np.linspace(90, 40, 12)])
    pipeline.expire(end + 20.0)
    assert events == []


def test_unmatched_sighting_without_a_clear_trajectory_makes_no_event():
    pipeline, events = _build()
    end = _see(pipeline, INSIDE, 3, 1, start=10.0, history=[_pose(60)] * 12)  # a student sitting in class
    pipeline.expire(end + 20.0)
    assert events == []


def test_single_camera_event_names_a_recognized_student():
    pipeline, events = _build()
    end = _see(pipeline, OUTSIDE, 1, 1, start=10.0, history=[_pose(h) for h in np.linspace(90, 40, 12)])
    pipeline.expire(end + 20.0)
    assert events[0]["personId"] == ANA_ID and events[0]["method"] == "FACE" and events[0]["confidence"] <= 0.4


def test_sightings_further_apart_than_the_window_are_not_paired():
    pipeline, events = _build()
    _see(pipeline, OUTSIDE, 3, 1, start=10.0)
    _see(pipeline, INSIDE, 3, 5, start=30.0)  # 20 s later: a different crossing
    assert events == []


def test_states_of_long_gone_tracks_are_forgotten():
    pipeline, _ = _build()
    end = _see(pipeline, OUTSIDE, 3, 1, start=10.0)
    assert pipeline._states
    pipeline.expire(end + 500.0)
    assert not pipeline._states and not pipeline._pending


def test_process_frame_returns_the_tracks_for_the_preview():
    pipeline, _ = _build()
    track = SimpleNamespace(id=7, box=(0, 0, 40, 60), keypoint_history=[])
    pipeline.trackers[OUTSIDE].tracks = [track]
    assert pipeline.process_frame(OUTSIDE, np.full((60, 40, 3), 3, dtype=np.uint8), 1.0) == [track]
