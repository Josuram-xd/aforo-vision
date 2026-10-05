import numpy as np
import pytest

from src.direction.resolver import ENTRY, EXIT, resolve_direction
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
