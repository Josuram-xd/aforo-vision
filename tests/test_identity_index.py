import numpy as np
import pytest

from src.identity.index import IdentityIndex


def _unit(angle_degrees: float) -> np.ndarray:
    """2-d unit vector; cosine similarity between two of them is cos(angle difference)."""
    radians = np.radians(angle_degrees)
    return np.array([np.cos(radians), np.sin(radians)], dtype=np.float32)


def _index(threshold=0.45, **kwargs) -> IdentityIndex:
    index = IdentityIndex(threshold, **kwargs)
    index.add("ana", _unit(0))
    index.add("beto", _unit(90))
    return index


def test_match_returns_closest_person_and_similarity():
    match = _index().lookup(_unit(10))
    assert match.person_id == "ana"
    assert match.similarity == pytest.approx(np.cos(np.radians(10)), abs=1e-6)


def test_no_match_when_best_is_below_threshold():
    index = IdentityIndex(0.9)
    index.add("ana", _unit(0))
    assert index.lookup(_unit(40)) is None  # cos 40 = 0.766 < 0.9


def test_similarity_equal_to_threshold_matches():
    index = IdentityIndex(1.0)
    index.add("ana", _unit(0))
    assert index.lookup(_unit(0)).person_id == "ana"


def test_threshold_is_configurable():
    query = _unit(-60)  # cos 60 = 0.5 against ana, negative against beto
    assert _index(threshold=0.4).lookup(query).person_id == "ana"
    assert _index(threshold=0.6).lookup(query) is None


def test_exact_tie_is_not_named():
    # 45 degrees is equally close to ana (0) and beto (90): naming either would be a guess
    assert _index().lookup(_unit(45)) is None


def test_ambiguity_margin_rejects_near_ties_only():
    query = _unit(40)  # ana 0.766, beto 0.643: gap 0.123
    assert _index(ambiguity_margin=0.2).lookup(query) is None
    assert _index(ambiguity_margin=0.05).lookup(query).person_id == "ana"


def test_tie_between_non_best_people_does_not_block_a_clear_winner():
    index = IdentityIndex(0.1)
    index.add("ana", _unit(0))
    index.add("beto", _unit(90))
    index.add("carla", _unit(-90))  # beto and carla tie for second place
    assert index.lookup(_unit(5)).person_id == "ana"


def test_empty_index_never_matches():
    assert IdentityIndex(0.0).lookup(_unit(0)) is None


def test_unnormalized_vectors_are_normalized_on_add_and_lookup():
    index = IdentityIndex(0.9)
    index.add("ana", np.array([5.0, 0.0]))
    match = index.lookup(np.array([0.2, 0.0]))
    assert match.person_id == "ana"
    assert match.similarity == pytest.approx(1.0)


def test_add_replaces_existing_person_and_remove_forgets_them():
    index = _index()
    index.add("ana", _unit(90))  # re-enroll ana with beto's direction
    assert len(index) == 2
    assert index.lookup(_unit(0)) is None  # nobody is near 0 degrees anymore
    index.remove("beto")
    assert "beto" not in index and len(index) == 1
    assert index.lookup(_unit(90)).person_id == "ana"


def test_lookup_sees_people_added_after_a_previous_lookup():
    index = _index()
    index.lookup(_unit(0))  # builds the cached matrix
    index.add("carla", _unit(180))
    assert index.lookup(_unit(180)).person_id == "carla"


def test_invalid_embeddings_are_rejected():
    index = _index()
    with pytest.raises(ValueError):
        index.add("zero", np.zeros(2))
    with pytest.raises(ValueError):
        index.add("wrong-dim", np.ones(3))
    with pytest.raises(ValueError):
        index.lookup(np.zeros(2))
    with pytest.raises(ValueError):
        index.lookup(np.ones(3))
