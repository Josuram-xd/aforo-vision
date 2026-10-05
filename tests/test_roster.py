import json
import uuid

import numpy as np

from enrollment.enroll_student import save_enrollment
from src.identity.roster import load_enrolled


def _enroll(root, name, vector):
    person_id = str(uuid.uuid4())
    save_enrollment(person_id, name, np.array(vector, dtype=np.float32), 10, root)
    return person_id


def test_enrolled_students_are_loaded_into_the_index_with_their_names(tmp_path):
    ana, beto = _enroll(tmp_path, "Ana Pérez", [1, 0]), _enroll(tmp_path, "Beto", [0, 1])
    index, names = load_enrolled(tmp_path, threshold=0.45)
    assert len(index) == 2 and names == {ana: "Ana Pérez", beto: "Beto"}
    assert index.lookup(np.array([0.9, 0.1])).person_id == ana


def test_close_students_are_not_named_thanks_to_the_ambiguity_margin(tmp_path):
    _enroll(tmp_path, "Ana", [1, 0.00]), _enroll(tmp_path, "Beto", [1, 0.02])
    index, _ = load_enrolled(tmp_path, threshold=0.45)
    assert index.lookup(np.array([1.0, 0.01])) is None


def test_untrustworthy_entries_are_skipped_not_fatal(tmp_path):
    good = _enroll(tmp_path, "Ana", [1, 0])
    no_npy = _enroll(tmp_path, "Sin embedding", [0, 1])
    (tmp_path / f"{no_npy}.npy").unlink()
    (tmp_path / f"{uuid.uuid4()}.json").write_text("{broken", encoding="utf-8")
    (tmp_path / "p001.json").write_text(json.dumps({"personId": "p001", "name": "X"}), encoding="utf-8")
    index, names = load_enrolled(tmp_path, threshold=0.45)
    assert names == {good: "Ana"} and len(index) == 1


def test_missing_or_empty_folder_gives_an_empty_index(tmp_path):
    assert len(load_enrolled(tmp_path / "nope", 0.45)[0]) == 0
    assert load_enrolled(tmp_path, 0.45)[1] == {}
