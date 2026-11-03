import base64
import uuid

import cv2
import numpy as np

from enrollment.sync_web_enrollments import sync_once


class FakeCloud:
    def __init__(self, items):
        self.items, self.completed = items, []

    def pending(self):
        return self.items

    def complete(self, student_id, ok):
        self.completed.append((student_id, ok))


def _jpeg_b64() -> str:
    ok, buffer = cv2.imencode(".jpg", np.zeros((40, 40, 3), dtype=np.uint8))
    assert ok
    return base64.b64encode(buffer.tobytes()).decode()


def _unit(*values: float) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def test_enrolls_and_confirms(tmp_path):
    student = str(uuid.uuid4())
    cloud = FakeCloud([{"studentId": student, "images": [_jpeg_b64()] * 3}])
    assert sync_once(cloud, lambda frame: _unit(1, 0.1, 0), tmp_path) == 1
    assert cloud.completed == [(student, True)]
    assert (tmp_path / f"{student}.npy").exists()


def test_too_few_usable_photos_fails_and_saves_nothing(tmp_path):
    student = str(uuid.uuid4())
    cloud = FakeCloud([{"studentId": student, "images": [_jpeg_b64()] * 3}])
    assert sync_once(cloud, lambda frame: None, tmp_path) == 0
    assert cloud.completed == [(student, False)]
    assert not list(tmp_path.glob("*.npy"))
