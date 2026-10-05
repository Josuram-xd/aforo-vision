"""Event builder: a resolved crossing as the JSON of the shared contract.

The contract lives in aforo-backend/ARCHITECTURE.md (section 4) and aforo-backend/src/models/event.py
(AforoEvent). Any change to a field must be made in aforo-backend and aforo-frontend too.
"""

import math
import uuid
from datetime import datetime, timezone

from src.direction.resolver import ENTRY, EXIT

FACE = "FACE"
BODY_ONLY = "BODY_ONLY"
CAMERA_OUTSIDE_ID = "camera-outside"
CAMERA_INSIDE_ID = "camera-inside"

_DIRECTIONS = (ENTRY, EXIT)
_METHODS = (FACE, BODY_ONLY)


def build_event(
    direction: str,
    method: str,
    confidence: float,
    timestamp: datetime | float,
    person_id: str | None = None,
    person_name: str | None = None,
    event_id: str | None = None,
) -> dict:
    """Return the event as a JSON-ready dict with the camelCase keys of the contract.

    `timestamp` is an aware datetime or epoch seconds, sent as UTC (e.g. 2026-09-30T14:32:00Z).
    A BODY_ONLY event never carries an identity: body appearance only counts people, it never names them.
    """
    if direction not in _DIRECTIONS:
        raise ValueError(f"direction must be one of {_DIRECTIONS}, got {direction!r}")
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    if not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be a number between 0 and 1, got {confidence!r}")
    if method == BODY_ONLY and (person_id is not None or person_name is not None):
        raise ValueError("a BODY_ONLY event cannot carry personId or personName")
    if person_name is not None and person_id is None:
        raise ValueError("personName requires personId")

    return {
        "eventId": _uuid(event_id, "eventId") if event_id is not None else str(uuid.uuid4()),
        "personId": _uuid(person_id, "personId") if person_id is not None else None,
        "personName": person_name,
        "direction": direction,
        "cameraOutsideId": CAMERA_OUTSIDE_ID,
        "cameraInsideId": CAMERA_INSIDE_ID,
        "confidence": float(confidence),
        "method": method,
        "timestamp": _format_timestamp(timestamp),
    }


def _uuid(value: str, field: str) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except ValueError:
        raise ValueError(f"{field} must be a UUID, got {value!r}") from None


def _format_timestamp(timestamp: datetime | float) -> str:
    if isinstance(timestamp, datetime):
        if timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        moment = timestamp
    elif isinstance(timestamp, (int, float)) and math.isfinite(timestamp):
        moment = datetime.fromtimestamp(timestamp, timezone.utc)
    else:
        raise ValueError(f"timestamp must be a datetime or epoch seconds, got {timestamp!r}")
    return moment.astimezone(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
