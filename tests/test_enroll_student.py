import json
import uuid

import numpy as np
import pytest

from enrollment.enroll_student import (
    EnrollmentError,
    average_embeddings,
    collect_samples,
    periocular_embedding,
    save_enrollment,
)
from src.identity.face_detector import MODEL_FILE, Face, FaceDetector

EYES = np.array([[100, 100], [140, 100], [120, 120], [105, 140], [135, 140]], dtype=np.float32)


def _face(landmarks=EYES, score=0.9):
    return Face((80.0, 70.0, 160.0, 160.0), score, landmarks)


class _StubDetector:
    def __init__(self, faces_per_call):
        self._faces = list(faces_per_call)

    def detect(self, _frame):
        return self._faces.pop(0) if len(self._faces) > 1 else self._faces[0]


def _frame():
    return np.zeros((300, 400, 3), dtype=np.uint8)


def _unit(angle_degrees):
    radians = np.radians(angle_degrees)
    return np.array([np.cos(radians), np.sin(radians)], dtype=np.float32)


def test_average_is_unit_norm_mean_direction():
    mean = average_embeddings([_unit(-10), _unit(10)])
    assert np.linalg.norm(mean) == pytest.approx(1.0)
    np.testing.assert_allclose(mean, _unit(0), atol=1e-6)


def test_average_rejects_a_stray_sample():
    with pytest.raises(EnrollmentError, match="inconsistent"):
        average_embeddings([_unit(0), _unit(2), _unit(1), _unit(120)])  # someone else walked in


def test_average_rejects_samples_that_cancel_out():
    with pytest.raises(EnrollmentError):
        average_embeddings([_unit(0), _unit(180)])


def test_only_a_single_face_gives_an_embedding():
    embed = lambda crop: np.ones(3, dtype=np.float32)  # noqa: E731
    assert periocular_embedding(_frame(), [_face()], embed) is not None
    assert periocular_embedding(_frame(), [], embed) is None
    assert periocular_embedding(_frame(), [_face(), _face()], embed) is None  # could be a bystander


def test_face_with_unusable_eyes_gives_no_embedding():
    tiny_eyes = EYES.copy()
    tiny_eyes[1] = tiny_eyes[0] + 2  # eyes 2 px apart
    assert periocular_embedding(_frame(), [_face(tiny_eyes)], lambda crop: np.ones(3)) is None


def test_collect_skips_frames_without_a_single_face_until_enough_samples():
    detector = _StubDetector([[], [_face(), _face()], [_face()]])  # last answer repeats
    samples = collect_samples(lambda: _frame(), detector, lambda crop: np.ones(3, dtype=np.float32), 3, min_interval=0.0)
    assert len(samples) == 3


def test_collect_times_out_without_faces():
    with pytest.raises(EnrollmentError, match="timed out"):
        collect_samples(lambda: _frame(), _StubDetector([[]]), lambda crop: np.ones(3), 1, timeout=0.05)


def test_collect_can_be_cancelled_from_the_preview():
    with pytest.raises(EnrollmentError, match="cancelled"):
        collect_samples(lambda: _frame(), _StubDetector([[_face()]]), lambda crop: np.ones(3), 1, on_frame=lambda *_: True)


def test_collect_waits_min_interval_between_samples():
    taken = []
    samples = collect_samples(
        lambda: _frame(), _StubDetector([[_face()]]), lambda crop: taken.append(1) or np.ones(3, dtype=np.float32), 2, min_interval=0.05
    )
    assert len(samples) == 2 and len(taken) == 2


def test_save_writes_embedding_and_metadata_without_biometrics_in_json(tmp_path):
    person_id = str(uuid.uuid4())
    embedding = _unit(30)
    path = save_enrollment(person_id, "Ana Pérez", embedding, 10, tmp_path)

    np.testing.assert_array_equal(np.load(path), embedding)
    metadata = json.loads((tmp_path / f"{person_id}.json").read_text(encoding="utf-8"))
    assert metadata["personId"] == person_id and metadata["name"] == "Ana Pérez" and metadata["samples"] == 10
    assert set(metadata) == {"personId", "name", "samples", "enrolledAt"}


def test_save_rejects_non_uuid_v4_ids(tmp_path):
    with pytest.raises(ValueError):
        save_enrollment("not-a-uuid", "Ana", _unit(0), 1, tmp_path)
    with pytest.raises(ValueError):
        save_enrollment(str(uuid.uuid1()), "Ana", _unit(0), 1, tmp_path)


def test_save_overwrites_the_same_person_when_re_enrolling(tmp_path):
    person_id = str(uuid.uuid4())
    save_enrollment(person_id, "Ana", _unit(0), 5, tmp_path)
    path = save_enrollment(person_id, "Ana", _unit(90), 8, tmp_path)
    np.testing.assert_array_equal(np.load(path), _unit(90))
    assert len(list(tmp_path.glob("*.npy"))) == 1


class _StubScrfd:
    def detect(self, _image, input_size=None):
        boxes = np.array([[10, 10, 50, 60, 0.3], [20, 20, 90, 100, 0.95], [5, 5, 30, 30, 0.7]], dtype=np.float32)
        return boxes, np.zeros((3, 5, 2), dtype=np.float32)


def test_detector_filters_by_score_and_sorts_best_first():
    faces = FaceDetector(score_threshold=0.5, model_instance=_StubScrfd()).detect(_frame())
    assert [round(f.score, 2) for f in faces] == [0.95, 0.7]
    assert faces[0].landmarks.shape == (5, 2) and faces[0].area == 70 * 80


def test_missing_detector_file_explains_how_to_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="--download"):
        FaceDetector(model_file=tmp_path / "nope.onnx")


@pytest.mark.skipif(not MODEL_FILE.is_file(), reason="face detector not downloaded")
def test_real_detector_finds_nothing_in_an_empty_frame():
    assert FaceDetector().detect(np.zeros((480, 640, 3), dtype=np.uint8)) == []
