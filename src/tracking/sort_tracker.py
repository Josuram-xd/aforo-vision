"""SORT tracker: a Kalman filter per track plus Hungarian association (src/tracking/hungarian.py)."""

import numpy as np

from src.detection.yolo_pose import Detection
from src.tracking.hungarian import associate

_EPS = 1e-6


def _box_to_z(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """Box corners -> measurement [center x, center y, area, aspect ratio w/h]."""
    w, h = x2 - x1, max(y2 - y1, _EPS)
    return np.array([x1 + w / 2, y1 + h / 2, w * h, w / h])


def _state_to_box(x: np.ndarray) -> tuple[float, float, float, float]:
    area, ratio = max(float(x[2]), _EPS), max(float(x[3]), _EPS)
    w = float(np.sqrt(area * ratio))
    h = area / w
    return (float(x[0]) - w / 2, float(x[1]) - h / 2, float(x[0]) + w / 2, float(x[1]) + h / 2)


class KalmanBoxFilter:
    """Constant-velocity Kalman filter over [cx, cy, area, ratio, vcx, vcy, varea]; ratio is kept constant."""

    def __init__(self, box: tuple[float, float, float, float]):
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0
        self.H = np.eye(4, 7)
        # Noise settings from the original SORT paper's reference implementation
        self.R = np.diag([1.0, 1.0, 10.0, 10.0])
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 10000.0, 10000.0, 10000.0])  # unknown initial velocity
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001])
        self.x = np.zeros(7)
        self.x[:4] = _box_to_z(*box)

    def predict(self) -> tuple[float, float, float, float]:
        if self.x[6] + self.x[2] <= 0:  # area would go negative
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.box

    def update(self, box: tuple[float, float, float, float]) -> None:
        innovation = _box_to_z(*box) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.P = (np.eye(7) - K @ self.H) @ self.P

    @property
    def box(self) -> tuple[float, float, float, float]:
        return _state_to_box(self.x)


class Track:
    """One tracked person: stable id, Kalman-filtered box and the last detection (with keypoints)."""

    def __init__(self, track_id: int, detection: Detection):
        self.id = track_id
        self.detection = detection
        self.hits = 1  # total detections matched to this track
        self.time_since_update = 0  # frames since the last matched detection
        self.confirmed = False
        self._filter = KalmanBoxFilter(detection[:4])

    @property
    def box(self) -> tuple[float, float, float, float]:
        return self._filter.box

    def predict(self) -> tuple[float, float, float, float]:
        box = self._filter.predict()
        self.time_since_update += 1
        return box

    def update(self, detection: Detection) -> None:
        self._filter.update(detection[:4])
        self.detection = detection
        self.hits += 1
        self.time_since_update = 0


class SortTracker:
    """Assigns stable ids to detections across frames. Use one instance per camera."""

    def __init__(self, iou_threshold: float = 0.3, max_age_frames: int = 30, min_hits: int = 3):
        self.iou_threshold = iou_threshold
        self.max_age_frames = max_age_frames
        self.min_hits = min_hits
        self._tracks: list[Track] = []
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[Track]:
        """Process one frame. Returns the confirmed tracks that were matched to a detection in it."""
        predicted = [track.predict() for track in self._tracks]
        matches, unmatched_dets, _ = associate([d[:4] for d in detections], predicted, self.iou_threshold)

        for det_index, track_index in matches:
            track = self._tracks[track_index]
            track.update(detections[det_index])
            if track.hits >= self.min_hits:
                track.confirmed = True
        for det_index in unmatched_dets:
            self._tracks.append(self._new_track(detections[det_index]))

        self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age_frames]
        return [t for t in self._tracks if t.confirmed and t.time_since_update == 0]

    def _new_track(self, detection: Detection) -> Track:
        track = Track(self._next_id, detection)
        track.confirmed = self.min_hits <= 1
        self._next_id += 1
        return track
