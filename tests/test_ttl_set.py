import pytest

from src.dedup.ttl_set import TTLSet


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def _set(ttl=10.0):
    clock = _Clock()
    return TTLSet(ttl, clock=clock), clock


def test_first_add_is_new_and_repeat_within_ttl_is_a_duplicate():
    ttl_set, clock = _set()
    assert ttl_set.add((1, 7)) is True
    clock.now += 3
    assert ttl_set.add((1, 7)) is False
    assert (1, 7) in ttl_set and len(ttl_set) == 1


def test_key_expires_after_ttl_and_counts_as_new_again():
    ttl_set, clock = _set(10.0)
    ttl_set.add("crossing")
    clock.now += 9.99
    assert ttl_set.add("crossing") is False
    clock.now += 0.01  # exactly ttl after the first sighting
    assert "crossing" not in ttl_set
    assert ttl_set.add("crossing") is True


def test_duplicate_does_not_extend_the_ttl():
    ttl_set, clock = _set(10.0)
    ttl_set.add("a")
    clock.now += 6
    ttl_set.add("a")  # duplicate at t=6
    clock.now += 4  # t=10: the window from the first sighting is over
    assert ttl_set.add("a") is True


def test_different_keys_are_independent_and_expire_in_order():
    ttl_set, clock = _set(10.0)
    ttl_set.add("a")
    clock.now += 5
    ttl_set.add("b")
    clock.now += 5  # a expired (t=10), b alive (t=5..15)
    assert "a" not in ttl_set and "b" in ttl_set and len(ttl_set) == 1
    clock.now += 5
    assert len(ttl_set) == 0


def test_expired_entries_are_freed_not_kept_forever():
    ttl_set, clock = _set(1.0)
    for i in range(100):
        ttl_set.add(i)
        clock.now += 0.5
    assert len(ttl_set) <= 2  # only the last ~1 s worth is kept


def test_keys_can_be_any_hashable_such_as_a_pair_of_track_ids():
    ttl_set, _ = _set()
    assert ttl_set.add((1, 7)) and ttl_set.add((7, 1)) and ttl_set.add("1-7")
    assert not ttl_set.add((1, 7))


def test_clear_forgets_everything():
    ttl_set, _ = _set()
    ttl_set.add("a")
    ttl_set.clear()
    assert len(ttl_set) == 0 and ttl_set.add("a") is True


def test_ttl_must_be_positive():
    for bad in (0, -1):
        with pytest.raises(ValueError):
            TTLSet(bad)


def test_uses_the_real_monotonic_clock_by_default():
    ttl_set = TTLSet(60.0)
    assert ttl_set.add("a") is True and ttl_set.add("a") is False
