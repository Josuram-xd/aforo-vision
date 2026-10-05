import json
import uuid

import numpy as np
import pytest

from enrollment.enroll_student import save_enrollment
from enrollment.export_roster import RosterError, build_roster, write_roster


def _enroll(root, name):
    person_id = str(uuid.uuid4())
    save_enrollment(person_id, name, np.array([1.0, 0.0], dtype=np.float32), 10, root)
    return person_id


def test_roster_has_only_person_id_and_name_sorted_by_name(tmp_path):
    zoe, ana = _enroll(tmp_path, "Zoe"), _enroll(tmp_path, "ana")
    roster, warnings = build_roster(tmp_path)
    assert roster == [{"personId": ana, "name": "ana"}, {"personId": zoe, "name": "Zoe"}]
    assert warnings == []


def test_written_file_contains_no_biometrics_or_enrollment_details(tmp_path):
    _enroll(tmp_path, "Ana Pérez")
    roster, _ = build_roster(tmp_path)
    output = write_roster(roster, tmp_path / "out" / "roster.json")
    text = output.read_text(encoding="utf-8")
    assert "Ana Pérez" in text  # accents kept readable
    data = json.loads(text)
    assert all(set(entry) == {"personId", "name"} for entry in data)
    assert "samples" not in text and "enrolledAt" not in text


def test_person_without_embedding_is_skipped_with_a_warning(tmp_path):
    person_id = _enroll(tmp_path, "Ana")
    (tmp_path / f"{person_id}.npy").unlink()
    roster, warnings = build_roster(tmp_path)
    assert roster == [] and "no embedding file" in warnings[0]


def test_embedding_without_metadata_warns_but_is_not_exported(tmp_path):
    person_id = _enroll(tmp_path, "Ana")
    (tmp_path / f"{person_id}.json").unlink()
    roster, warnings = build_roster(tmp_path)
    assert roster == [] and "without metadata" in warnings[0]


def test_repeated_names_warn(tmp_path):
    _enroll(tmp_path, "Ana"), _enroll(tmp_path, "ana")
    roster, warnings = build_roster(tmp_path)
    assert len(roster) == 2 and "repeated name" in warnings[0]


@pytest.mark.parametrize(
    "metadata",
    [
        {"personId": "p001", "name": "Ana"},  # not a UUID
        {"personId": str(uuid.uuid1()), "name": "Ana"},  # not v4
        {"personId": str(uuid.uuid4()), "name": "  "},  # blank name
        {"name": "Ana"},  # no personId
    ],
)
def test_untrustworthy_metadata_is_rejected(tmp_path, metadata):
    stem = metadata.get("personId", str(uuid.uuid4()))
    (tmp_path / f"{stem}.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RosterError):
        build_roster(tmp_path)


def test_person_id_must_match_the_file_name(tmp_path):
    (tmp_path / f"{uuid.uuid4()}.json").write_text(json.dumps({"personId": str(uuid.uuid4()), "name": "Ana"}), encoding="utf-8")
    with pytest.raises(RosterError, match="does not match"):
        build_roster(tmp_path)


def test_invalid_json_is_rejected(tmp_path):
    (tmp_path / f"{uuid.uuid4()}.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(RosterError, match="invalid JSON"):
        build_roster(tmp_path)


def test_output_loads_with_the_aforo_db_validator(tmp_path):
    import importlib.util
    from pathlib import Path

    seed = Path(__file__).resolve().parents[2] / "aforo-db" / "scripts" / "seed_people.py"
    if not seed.is_file():
        pytest.skip("aforo-db repo not next to this one")
    spec = importlib.util.spec_from_file_location("seed_people", seed)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    _enroll(tmp_path, "Ana"), _enroll(tmp_path, "Beto")
    roster, _ = build_roster(tmp_path)
    output = write_roster(roster, tmp_path / "roster.json")
    assert module.load_roster(output) == roster
