import numpy as np
import pytest

from src.matching.cross_checkpoint import CrossCheckpointMatcher, Sighting


def _unit(angle_degrees):
    radians = np.radians(angle_degrees)
    return np.array([np.cos(radians), np.sin(radians)], dtype=np.float32)


def _out(track_id, t, face=None, body=None):
    return Sighting("camera-outside", track_id, t, face=face, body=body)


def _in(track_id, t, face=None, body=None):
    return Sighting("camera-inside", track_id, t, face=face, body=body)


def _matcher(window=5.0, face=0.45, body=0.6):
    return CrossCheckpointMatcher(window, face, body)


def _pairs(matches):
    return sorted((m.outside.track_id, m.inside.track_id) for m in matches)


def test_pairs_the_same_person_seen_by_both_cameras():
    matches, left_out, left_in = _matcher().match([_out(1, 10.0, face=_unit(0))], [_in(7, 12.0, face=_unit(5))])
    assert _pairs(matches) == [(1, 7)] and matches[0].method == "FACE"
    assert matches[0].similarity == pytest.approx(np.cos(np.radians(5)), abs=1e-6)
    assert left_out == [] and left_in == []


def test_global_optimum_beats_greedy_matching():
    # outside 1 resembles inside A best (cos 5 = 0.996), but outside 2 only resembles A (cos 20 = 0.94).
    # Greedy gives A to outside 1 and leaves outside 2 with B (cos 80 = 0.17, invalid): one pair.
    # The global optimum pairs 1-B (cos 55 = 0.57) and 2-A: two valid pairs.
    outside = [_out(1, 10.0, face=_unit(5)), _out(2, 10.5, face=_unit(-20))]
    inside = [_in("A", 11.0, face=_unit(0)), _in("B", 11.5, face=_unit(60))]
    matches, left_out, left_in = _matcher().match(outside, inside)
    assert _pairs(matches) == [(1, "B"), (2, "A")]
    assert left_out == [] and left_in == []


def test_pairing_is_one_to_one_with_leftovers_reported():
    outside = [_out(1, 10.0, face=_unit(0)), _out(2, 10.0, face=_unit(90))]
    inside = [_in(7, 11.0, face=_unit(1))]
    matches, left_out, left_in = _matcher().match(outside, inside)
    assert _pairs(matches) == [(1, 7)]
    assert [s.track_id for s in left_out] == [2] and left_in == []


def test_sightings_outside_the_time_window_are_not_paired_either_way():
    face = _unit(0)
    assert _matcher(window=5.0).match([_out(1, 10.0, face=face)], [_in(7, 15.1, face=face)])[0] == []
    assert _matcher(window=5.0).match([_out(1, 15.1, face=face)], [_in(7, 10.0, face=face)])[0] == []  # symmetric
    assert len(_matcher(window=5.0).match([_out(1, 10.0, face=face)], [_in(7, 15.0, face=face)])[0]) == 1  # edge is in


def test_similarity_below_threshold_is_not_paired():
    matches, left_out, left_in = _matcher(face=0.9).match([_out(1, 10.0, face=_unit(0))], [_in(7, 11.0, face=_unit(40))])
    assert matches == [] and len(left_out) == 1 and len(left_in) == 1


def test_face_embeddings_are_preferred_over_body_and_use_the_face_threshold():
    out = _out(1, 10.0, face=_unit(0), body=_unit(0))
    ins = _in(7, 11.0, face=_unit(50), body=_unit(0))  # faces 0.64 (passes 0.45), bodies identical
    match = _matcher().match([out], [ins])[0][0]
    assert match.method == "FACE" and match.similarity == pytest.approx(np.cos(np.radians(50)), abs=1e-6)


def test_body_only_pairs_use_the_body_threshold():
    out, ins = _out(1, 10.0, body=_unit(0)), _in(7, 11.0, body=_unit(50))  # cos 50 = 0.64
    assert _matcher(body=0.6).match([out], [ins])[0][0].method == "BODY"
    assert _matcher(body=0.7).match([out], [ins])[0] == []


def test_face_and_body_embeddings_are_never_compared_with_each_other():
    matches, left_out, left_in = _matcher().match([_out(1, 10.0, face=_unit(0))], [_in(7, 11.0, body=_unit(0))])
    assert matches == [] and len(left_out) == 1 and len(left_in) == 1


def test_sighting_without_embeddings_stays_unmatched_for_the_trajectory_fallback():
    matches, left_out, _ = _matcher().match([_out(1, 10.0)], [_in(7, 11.0, face=_unit(0))])
    assert matches == [] and len(left_out) == 1


def test_a_valid_pair_is_kept_even_when_an_invalid_one_is_cheaper_to_leave_out():
    # outside 2 has nobody valid; the forced assignment must not steal inside 7 from outside 1
    outside = [_out(1, 10.0, face=_unit(30)), _out(2, 100.0, face=_unit(0))]
    matches, _, left_in = _matcher().match(outside, [_in(7, 11.0, face=_unit(0))])
    assert _pairs(matches) == [(1, 7)] and left_in == []


def test_empty_sides():
    assert _matcher().match([], []) == ([], [], [])
    matches, left_out, left_in = _matcher().match([_out(1, 10.0, face=_unit(0))], [])
    assert matches == [] and len(left_out) == 1 and left_in == []


def test_embedding_size_mismatch_and_zero_vectors_are_errors():
    with pytest.raises(ValueError):
        _matcher().match([_out(1, 10.0, face=np.ones(2))], [_in(7, 11.0, face=np.ones(3))])
    with pytest.raises(ValueError):
        _matcher().match([_out(1, 10.0, face=np.zeros(2))], [_in(7, 11.0, face=np.ones(2))])
