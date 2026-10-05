"""Direction resolver: ENTRY or EXIT from the order in which the two checkpoints saw the person.

camera-outside first, then camera-inside -> ENTRY. camera-inside first, then camera-outside -> EXIT.
Body orientation (facing forward or backward) is deliberately not used: walking backwards only changes
how a body looks, not which checkpoint it reached first (ADR-003 in ARCHITECTURE.md).
"""

import numpy as np

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


# --- Fallback: tracks seen by a single camera (task 6.3) -------------------------------------------------
# A secondary signal, used only when checkpoint order is missing or untrustworthy. It reads the apparent
# size of the torso over the track's keypoint history: it grows when the person walks toward the camera
# and shrinks when they walk away. Facing direction plays no part, so walking backwards does not fool it.

APPROACHING = "approaching"
RECEDING = "receding"

# Which motion means ENTRY for each camera. Each camera looks at the door from its own side (ARCHITECTURE.md 2):
# the outside one sees an entering person walk away from it, towards the door; the inside one sees them come
# out of the door towards it. Same logic on both cameras (AGENTS.md rule 3), only this mapping differs, and it
# depends on how the cameras are mounted: confirm it in the rehearsal (task 8.2).
DEFAULT_ENTRY_MOTION = {"camera-outside": RECEDING, "camera-inside": APPROACHING}

_SHOULDERS = (5, 6)  # COCO keypoint indices
_HIPS = (11, 12)
MIN_KEYPOINT_CONFIDENCE = 0.5
MIN_SAMPLES = 5  # frames with a visible torso needed to trust a trend
MIN_RELATIVE_CHANGE = 0.15  # torso size must change by at least this fraction over the window


def torso_height(keypoints: np.ndarray, min_confidence: float = MIN_KEYPOINT_CONFIDENCE) -> float | None:
    """Vertical distance between shoulders and hips in pixels, or None if any of the four is not visible."""
    points = np.asarray(keypoints)
    ids = (*_SHOULDERS, *_HIPS)
    if points.ndim != 2 or points.shape[0] <= max(ids) or points.shape[1] < 3 or (points[list(ids), 2] < min_confidence).any():
        return None
    height = float(points[list(_HIPS), 1].mean() - points[list(_SHOULDERS), 1].mean())
    return height if height > 0 else None


def resolve_from_trajectory(
    keypoint_history,
    camera_id: str,
    entry_motion: dict[str, str] | None = None,
    min_samples: int = MIN_SAMPLES,
    min_relative_change: float = MIN_RELATIVE_CHANGE,
) -> str | None:
    """Infer "ENTRY"/"EXIT" for a track seen by one camera from its keypoint history (oldest first).

    Returns None when the history is too short, the torso was rarely visible, the person barely moved
    along the camera's depth axis, or the camera has no configured mapping.
    """
    mapping = DEFAULT_ENTRY_MOTION if entry_motion is None else entry_motion
    if camera_id not in mapping:
        return None
    sizes = [(i, h) for i, kp in enumerate(keypoint_history) if (h := torso_height(kp)) is not None]
    if len(sizes) < min_samples:
        return None
    index, height = np.array(sizes, dtype=float).T
    slope = float(np.polyfit(index, height, 1)[0])
    change = slope * (index[-1] - index[0]) / float(height.mean())  # relative size change over the window
    if abs(change) < min_relative_change:
        return None
    motion = APPROACHING if change > 0 else RECEDING
    return ENTRY if motion == mapping[camera_id] else EXIT
