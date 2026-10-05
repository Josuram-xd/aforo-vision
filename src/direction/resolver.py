"""Direction resolver: ENTRY or EXIT from the order in which the two checkpoints saw the person.

camera-outside first, then camera-inside -> ENTRY. camera-inside first, then camera-outside -> EXIT.
Body orientation (facing forward or backward) is deliberately not used: walking backwards only changes
how a body looks, not which checkpoint it reached first (ADR-003 in ARCHITECTURE.md).
"""

from src.matching.cross_checkpoint import CrossingMatch

ENTRY = "ENTRY"
EXIT = "EXIT"


def resolve_direction(match: CrossingMatch, min_gap_seconds: float = 0.0) -> str | None:
    """Return "ENTRY" or "EXIT" for a matched crossing, or None when the order cannot be trusted.

    The order is trusted only if the two sightings are more than `min_gap_seconds` apart: with a smaller
    gap the camera delays (a WiFi stream can lag the USB one) may be larger than the real difference.
    Callers treat None like a single-checkpoint track and use the trajectory fallback (task 6.3).
    """
    gap = match.inside.timestamp - match.outside.timestamp  # > 0: seen outside first
    if abs(gap) <= min_gap_seconds or gap == 0.0:
        return None
    return ENTRY if gap > 0 else EXIT
