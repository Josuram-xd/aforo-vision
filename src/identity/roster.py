"""Load the enrolled students (data/embeddings/) into an IdentityIndex plus a personId -> name table.

Files come from enrollment/enroll_student.py: <personId>.npy (embedding) and <personId>.json (name).
Only the index lives in memory; nothing is sent anywhere.
"""

import json
import logging
import uuid
from pathlib import Path

import numpy as np

from src.identity.index import IdentityIndex

logger = logging.getLogger(__name__)

# Naming the wrong student is worse than not naming: when the two closest students are this close, name nobody
AMBIGUITY_MARGIN = 0.05


def load_enrolled(
    embeddings_dir: str | Path, threshold: float, ambiguity_margin: float = AMBIGUITY_MARGIN
) -> tuple[IdentityIndex, dict[str, str]]:
    """Return (index, names). Entries that cannot be trusted are skipped with a warning, never fatal."""
    index = IdentityIndex(threshold, ambiguity_margin)
    names: dict[str, str] = {}
    for meta_path in sorted(Path(embeddings_dir).glob("*.json")):
        embedding_path = meta_path.with_suffix(".npy")
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            person_id, name = metadata["personId"], str(metadata["name"]).strip()
            if uuid.UUID(person_id).version != 4 or person_id != meta_path.stem or not name:
                raise ValueError("personId must be the UUID v4 in the file name and name must not be empty")
            index.add(person_id, np.load(embedding_path))
        except (OSError, ValueError, KeyError, TypeError) as error:
            logger.warning("skipping enrolled student %s: %s", meta_path.name, error)
            continue
        names[person_id] = name
    return index, names
