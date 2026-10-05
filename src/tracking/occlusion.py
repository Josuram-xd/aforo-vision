"""Occlusion handling: when boxes overlap, use visible pose keypoints to decide who is who.

IoU alone cannot tell two overlapping people apart. For pairs that compete for the same box we
blend IoU with how well the detection's visible keypoints line up with the track's last keypoints
(moved along with the track's predicted box). Keypoints only re-rank candidates; the association
gate (min IoU) still applies in hungarian.associate.
"""

import numpy as np

from src.tracking.hungarian import Box, iou

KEYPOINT_CONFIDENCE = 0.5  # a keypoint counts as visible above this confidence
MIN_SHARED_KEYPOINTS = 3  # fewer visible keypoints in common and the comparison is not trusted
KEYPOINT_TOLERANCE = 0.25  # distance (as a fraction of box height) at which similarity reaches 0
CONTESTED_IOU = 0.1  # two candidates overlapping a box by more than this make the pair contested
KEYPOINT_WEIGHT = 0.5  # share of keypoint similarity in the blended score of a contested pair


def keypoint_similarity(det_keypoints: np.ndarray, track_keypoints: np.ndarray, scale: float) -> float | None:
    """Mean closeness (1 = same spot, 0 = far) of keypoints visible in both sets; None if too few are shared."""
    shared = (det_keypoints[:, 2] >= KEYPOINT_CONFIDENCE) & (track_keypoints[:, 2] >= KEYPOINT_CONFIDENCE)
    if shared.sum() < MIN_SHARED_KEYPOINTS or scale <= 0:
        return None
    distance = np.linalg.norm(det_keypoints[shared, :2] - track_keypoints[shared, :2], axis=1)
    return float(np.clip(1.0 - distance / (KEYPOINT_TOLERANCE * scale), 0.0, 1.0).mean())


def occlusion_aware_similarity(
    det_boxes: list[Box], det_keypoints: list[np.ndarray],
    track_boxes: list[Box], track_keypoints: list[np.ndarray],
) -> list[list[float]]:
    """Similarity matrix [detection][track] in 0..1: plain IoU, blended with keypoints for contested pairs."""
    ious = [[iou(d, t) for t in track_boxes] for d in det_boxes]
    similarity = [row[:] for row in ious]

    for i, row in enumerate(ious):
        for j, overlap in enumerate(row):
            if overlap <= 0 or not _is_contested(ious, i, j):
                continue
            height = track_boxes[j][3] - track_boxes[j][1]
            kp = keypoint_similarity(det_keypoints[i], track_keypoints[j], height)
            if kp is not None:
                similarity[i][j] = (1.0 - KEYPOINT_WEIGHT) * overlap + KEYPOINT_WEIGHT * kp
    return similarity


def _is_contested(ious: list[list[float]], det: int, track: int) -> bool:
    """True when this detection overlaps several tracks, or this track overlaps several detections."""
    row_rivals = sum(1 for value in ious[det] if value > CONTESTED_IOU)
    column_rivals = sum(1 for row in ious if row[track] > CONTESTED_IOU)
    return row_rivals > 1 or column_rivals > 1
