import json
import uuid

import numpy as np
import pytest

from enrollment.enroll_student import save_enrollment
from enrollment.remove_student import parse_person_id, remove_student


@pytest.fixture
def enrolled(tmp_path):
    embeddings = tmp_path / "embeddings"
    keep_id, gone_id = str(uuid.uuid4()), str(uuid.uuid4())
    save_enrollment(keep_id, "Se Queda", np.ones(4, dtype=np.float32), samples=3, root=embeddings)
    save_enrollment(gone_id, "Se Va", np.ones(4, dtype=np.float32), samples=3, root=embeddings)
    roster = tmp_path / "roster.json"
    roster.write_text(
        json.dumps([{"personId": keep_id, "name": "Se Queda"}, {"personId": gone_id, "name": "Se Va"}]), encoding="utf-8"
    )
    return embeddings, roster, keep_id, gone_id


def test_removes_embedding_metadata_and_roster_entry(enrolled):
    embeddings, roster, keep_id, gone_id = enrolled

    result = remove_student(gone_id, embeddings, roster)

    assert sorted(p.name for p in result.deleted) == sorted([f"{gone_id}.npy", f"{gone_id}.json"])
    assert not (embeddings / f"{gone_id}.npy").exists()
    assert not (embeddings / f"{gone_id}.json").exists()
    assert (embeddings / f"{keep_id}.npy").exists()
    assert result.roster_updated
    assert json.loads(roster.read_text(encoding="utf-8")) == [{"personId": keep_id, "name": "Se Queda"}]


def test_unknown_student_changes_nothing(enrolled):
    embeddings, roster, _, _ = enrolled
    before = sorted(p.name for p in embeddings.iterdir())

    result = remove_student(str(uuid.uuid4()), embeddings, roster)

    assert result.deleted == [] and not result.roster_updated
    assert sorted(p.name for p in embeddings.iterdir()) == before


def test_accepts_uppercase_and_spaces():
    person_id = str(uuid.uuid4())
    assert parse_person_id(f"  {person_id.upper()} ") == person_id


@pytest.mark.parametrize(
    "bad",
    ["", "abc", "../../etc/passwd", "../" + str(uuid.uuid4()), str(uuid.uuid1()), "*", str(uuid.uuid4()).replace("-", "")],
)
def test_rejects_anything_but_a_uuid_v4(bad, enrolled):
    embeddings, roster, _, _ = enrolled
    with pytest.raises(ValueError):
        remove_student(bad, embeddings, roster)
