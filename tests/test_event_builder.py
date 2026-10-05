import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.events.builder import BODY_ONLY, FACE, build_event

PERSON = str(uuid.uuid4())
MOMENT = datetime(2026, 9, 30, 14, 32, 0, tzinfo=timezone.utc)


def test_event_has_exactly_the_contract_fields():
    event = build_event("ENTRY", FACE, 0.91, MOMENT, person_id=PERSON, person_name="Ana Pérez")
    assert set(event) == {
        "eventId", "personId", "personName", "direction", "cameraOutsideId", "cameraInsideId",
        "confidence", "method", "timestamp",
    }
    assert event["personId"] == PERSON and event["personName"] == "Ana Pérez"
    assert event["direction"] == "ENTRY" and event["method"] == "FACE" and event["confidence"] == 0.91
    assert event["cameraOutsideId"] == "camera-outside" and event["cameraInsideId"] == "camera-inside"
    assert event["timestamp"] == "2026-09-30T14:32:00Z"
    assert str(uuid.UUID(event["eventId"], version=4)) == event["eventId"]


def test_body_only_event_has_no_identity():
    event = build_event("EXIT", BODY_ONLY, 0.7, MOMENT)
    assert event["personId"] is None and event["personName"] is None and event["method"] == "BODY_ONLY"


def test_body_only_can_never_name_anyone():
    with pytest.raises(ValueError, match="BODY_ONLY"):
        build_event("ENTRY", BODY_ONLY, 0.7, MOMENT, person_id=PERSON)
    with pytest.raises(ValueError, match="BODY_ONLY"):
        build_event("ENTRY", BODY_ONLY, 0.7, MOMENT, person_name="Ana")


def test_face_event_without_a_known_person_is_allowed_but_a_name_needs_an_id():
    assert build_event("ENTRY", FACE, 0.6, MOMENT)["personId"] is None
    with pytest.raises(ValueError, match="personId"):
        build_event("ENTRY", FACE, 0.6, MOMENT, person_name="Ana")


def test_every_event_gets_a_fresh_id_unless_one_is_given():
    assert build_event("ENTRY", FACE, 0.9, MOMENT)["eventId"] != build_event("ENTRY", FACE, 0.9, MOMENT)["eventId"]
    fixed = str(uuid.uuid4())
    assert build_event("ENTRY", FACE, 0.9, MOMENT, event_id=fixed)["eventId"] == fixed


def test_timestamp_is_utc_with_z_from_epoch_or_any_timezone():
    assert build_event("ENTRY", FACE, 0.9, 1790778720.0)["timestamp"] == "2026-09-30T14:32:00Z"
    bogota = MOMENT.astimezone(timezone(timedelta(hours=-5)))
    assert build_event("ENTRY", FACE, 0.9, bogota)["timestamp"] == "2026-09-30T14:32:00Z"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"direction": "IN"},
        {"method": "BODY"},
        {"confidence": 1.1},
        {"confidence": -0.1},
        {"confidence": float("nan")},
        {"confidence": "0.9"},
        {"timestamp": datetime(2026, 9, 30, 14, 32)},  # naive
        {"timestamp": "2026-09-30T14:32:00Z"},
        {"person_id": "p001"},
        {"event_id": "not-a-uuid"},
    ],
)
def test_invalid_values_are_rejected(kwargs):
    arguments = {"direction": "ENTRY", "method": FACE, "confidence": 0.9, "timestamp": MOMENT} | kwargs
    with pytest.raises(ValueError):
        build_event(**arguments)


def test_event_is_accepted_by_the_backend_model():
    models = Path(__file__).resolve().parents[2] / "aforo-backend" / "src"
    if not (models / "models" / "event.py").is_file():
        pytest.skip("aforo-backend repo not next to this one")
    import importlib.util

    try:
        import pydantic  # noqa: F401
    except ImportError:
        pytest.skip("pydantic not installed")
    spec = importlib.util.spec_from_file_location("backend_event", models / "models" / "event.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for event in (
        build_event("ENTRY", FACE, 0.91, MOMENT, person_id=PERSON, person_name="Ana"),
        build_event("EXIT", BODY_ONLY, 0.7, MOMENT),
    ):
        module.AforoEvent.model_validate(event)
