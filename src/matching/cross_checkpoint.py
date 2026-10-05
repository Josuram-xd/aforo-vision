"""Cross-checkpoint matcher: pair the tracks of camera-outside with those of camera-inside.

A person seen by both cameras within a short time window is one physical crossing. Outside sightings
are the left nodes and inside sightings the right nodes of a bipartite graph; an edge costs
1 - cosine similarity of their appearance embeddings and the minimum-cost matching is found with the
Hungarian algorithm of src/tracking/hungarian.py (ARCHITECTURE.md 4.5).

The match is symmetric: it says "same person", not which way they went. Direction comes from the
order of the two timestamps (src/direction/resolver.py, task 6.2).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.tracking.hungarian import hungarian

_INVALID_COST = 1e6  # dwarfs any real cost (<= 2), so Hungarian keeps as many valid pairs as possible


@dataclass(frozen=True)
class Sighting:
    """What one camera saw of one track, ready to be paired.

    `timestamp` is the wall-clock second the person was at that checkpoint (e.g. when the track was
    confirmed there). `face` is the periocular ArcFace embedding and `body` the OSNet one; either may be
    missing. Embeddings of different kinds are never compared (a face vector says nothing about a body).
    """

    camera_id: str
    track_id: int
    timestamp: float
    face: np.ndarray | None = None
    body: np.ndarray | None = None


@dataclass(frozen=True)
class CrossingMatch:
    outside: Sighting
    inside: Sighting
    similarity: float
    method: str  # "FACE" or "BODY": which embedding scored the pair (same values as the event's `method`)


class CrossCheckpointMatcher:
    def __init__(self, time_window_seconds: float, face_threshold: float, body_threshold: float):
        self.time_window_seconds = time_window_seconds
        self.face_threshold = face_threshold
        self.body_threshold = body_threshold

    def match(
        self, outside: Sequence[Sighting], inside: Sequence[Sighting]
    ) -> tuple[list[CrossingMatch], list[Sighting], list[Sighting]]:
        """Return (matches, unmatched outside sightings, unmatched inside sightings).

        A pair is a candidate only if the sightings are at most `time_window_seconds` apart, share an
        embedding kind and are at least as similar as that kind's threshold. Among candidates the matching
        is globally optimal and one-to-one, so two similar people crossing together are not swapped.
        """
        if not outside or not inside:
            return [], list(outside), list(inside)

        scored = [[self._score(out, ins) for ins in inside] for out in outside]
        cost = [[_INVALID_COST if cell is None else 1.0 - cell[0] for cell in row] for row in scored]

        matches: list[CrossingMatch] = []
        matched_outside: set[int] = set()
        matched_inside: set[int] = set()
        for i, j in hungarian(cost):
            if scored[i][j] is None:
                continue
            similarity, method = scored[i][j]
            matches.append(CrossingMatch(outside[i], inside[j], similarity, method))
            matched_outside.add(i)
            matched_inside.add(j)
        return (
            matches,
            [s for i, s in enumerate(outside) if i not in matched_outside],
            [s for j, s in enumerate(inside) if j not in matched_inside],
        )

    def _score(self, outside: Sighting, inside: Sighting) -> tuple[float, str] | None:
        """(similarity, method) if the pair is a valid candidate, else None."""
        if abs(outside.timestamp - inside.timestamp) > self.time_window_seconds:
            return None
        if outside.face is not None and inside.face is not None:
            similarity, threshold, method = _cosine(outside.face, inside.face), self.face_threshold, "FACE"
        elif outside.body is not None and inside.body is not None:
            similarity, threshold, method = _cosine(outside.body, inside.body), self.body_threshold, "BODY"
        else:
            return None
        return (similarity, method) if similarity >= threshold else None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float32).reshape(-1), np.asarray(b, dtype=np.float32).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"embeddings have different sizes: {a.shape[0]} and {b.shape[0]}")
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm == 0.0:
        raise ValueError("cannot compare a zero embedding")
    return float(a @ b) / norm
