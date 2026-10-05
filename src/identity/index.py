"""Enrolled-roster lookup: hash table `personId -> embedding` with cosine-similarity search.

Embeddings are the L2-normalized vectors from arcface.py, so cosine similarity is a dot product.
The threshold comes from `identity.face_similarity_threshold` in config/pilot.yaml.
"""

from typing import NamedTuple

import numpy as np


class IdentityMatch(NamedTuple):
    person_id: str
    similarity: float


class IdentityIndex:
    def __init__(self, threshold: float, ambiguity_margin: float = 0.0):
        """`ambiguity_margin`: if the best and second-best people are within this similarity gap
        (ties included) no one is named, since naming the wrong person is worse than not naming."""
        self.threshold = threshold
        self.ambiguity_margin = ambiguity_margin
        self._embeddings: dict[str, np.ndarray] = {}  # personId -> embedding (the hash table)
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None  # stacked embeddings, rebuilt lazily after changes

    def __len__(self) -> int:
        return len(self._embeddings)

    def __contains__(self, person_id: str) -> bool:
        return person_id in self._embeddings

    def add(self, person_id: str, embedding: np.ndarray) -> None:
        """Enroll (or replace) a person. The embedding is re-normalized defensively."""
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise ValueError("cannot index a zero embedding")
        if self._embeddings and vector.shape[0] != self._dimension():
            raise ValueError(f"embedding has {vector.shape[0]} dims, index holds {self._dimension()}")
        self._embeddings[person_id] = vector / norm
        self._matrix = None

    def remove(self, person_id: str) -> None:
        del self._embeddings[person_id]
        self._matrix = None

    def lookup(self, embedding: np.ndarray) -> IdentityMatch | None:
        """Return the best match at or above the threshold, or None (no match or ambiguous)."""
        if not self._embeddings:
            return None
        query = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            raise ValueError("cannot look up a zero embedding")
        if query.shape[0] != self._dimension():
            raise ValueError(f"embedding has {query.shape[0]} dims, index holds {self._dimension()}")

        matrix = self._stacked()
        scores = matrix @ (query / norm)
        order = np.argsort(scores)[::-1]
        best = float(scores[order[0]])
        if best < self.threshold:
            return None
        if len(order) > 1 and best - float(scores[order[1]]) <= self.ambiguity_margin:
            return None
        return IdentityMatch(self._ids[order[0]], best)

    def _dimension(self) -> int:
        return next(iter(self._embeddings.values())).shape[0]

    def _stacked(self) -> np.ndarray:
        if self._matrix is None:
            self._ids = list(self._embeddings)
            self._matrix = np.stack([self._embeddings[i] for i in self._ids])
        return self._matrix
