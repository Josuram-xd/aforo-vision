"""Hash set with a time-to-live, so one physical crossing produces at most one event.

Keyed by the matched cross-checkpoint pair (ARCHITECTURE.md 4.7). If detections flicker, the same pair
shows up again in later frames; the second `add` within the TTL is reported as a duplicate.
"""

import time
from collections import OrderedDict
from collections.abc import Callable, Hashable


class TTLSet:
    def __init__(self, ttl_seconds: float, clock: Callable[[], float] = time.monotonic):
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        # key -> expiry time. Every entry gets the same TTL, so insertion order is also expiry order
        # and expired keys can be dropped from the front without scanning the whole table.
        self._expiry: OrderedDict[Hashable, float] = OrderedDict()

    def add(self, key: Hashable) -> bool:
        """Remember `key`. Returns True if it is new (count the crossing), False if it is a duplicate.

        A duplicate does not extend the TTL: the window runs from the first sighting, and after it
        the same key counts as new again.
        """
        now = self._clock()
        self._purge(now)
        if key in self._expiry:
            return False
        self._expiry[key] = now + self.ttl_seconds
        return True

    def __contains__(self, key: Hashable) -> bool:
        self._purge(self._clock())
        return key in self._expiry

    def __len__(self) -> int:
        self._purge(self._clock())
        return len(self._expiry)

    def clear(self) -> None:
        self._expiry.clear()

    def _purge(self, now: float) -> None:
        while self._expiry:
            key, expires_at = next(iter(self._expiry.items()))
            if expires_at > now:
                break
            del self._expiry[key]
