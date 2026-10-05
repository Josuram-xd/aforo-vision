from types import SimpleNamespace

import numpy as np

from src.identity.appearance import AppearanceExtractor

EYES = np.array([[40, 30], [70, 30], [55, 45], [45, 60], [65, 60]], dtype=np.float32)


class _Faces:
    def __init__(self, faces):
        self._faces, self.seen_shapes = faces, []

    def detect(self, crop):
        self.seen_shapes.append(crop.shape)
        return self._faces


class _Embedder:
    def __init__(self, vector):
        self.vector, self.calls = np.array(vector, dtype=np.float32), []

    def embed(self, crop):
        self.calls.append(crop.shape)
        return self.vector


def _extractor(faces):
    face_detector, face_embedder, body_embedder = _Faces(faces), _Embedder([1, 0]), _Embedder([0, 1])
    return AppearanceExtractor(face_detector, face_embedder, body_embedder), face_detector, face_embedder, body_embedder


def _image():
    return np.full((300, 200, 3), 120, dtype=np.uint8)


def test_returns_face_and_body_embeddings_of_the_person_box():
    extractor, faces, face_emb, body_emb = _extractor([SimpleNamespace(landmarks=EYES)])
    face, body = extractor.extract(_image(), (20, 30, 120, 230))
    np.testing.assert_array_equal(face, [1, 0])
    np.testing.assert_array_equal(body, [0, 1])
    assert faces.seen_shapes == [(200, 100, 3)] and body_emb.calls == [(200, 100, 3)]  # the face is looked for inside the person crop
    assert len(face_emb.calls) == 1


def test_no_face_gives_a_body_only_result():
    extractor, _, face_emb, _ = _extractor([])
    face, body = extractor.extract(_image(), (20, 30, 120, 230))
    assert face is None and body is not None and face_emb.calls == []


def test_unusable_eyes_give_no_face():
    close_eyes = EYES.copy()
    close_eyes[1] = close_eyes[0] + 1
    face, body = _extractor([SimpleNamespace(landmarks=close_eyes)])[0].extract(_image(), (20, 30, 120, 230))
    assert face is None and body is not None


def test_boxes_are_clamped_and_tiny_or_outside_boxes_are_ignored():
    extractor, _, _, body_emb = _extractor([])
    assert extractor.extract(_image(), (-50, -50, 100, 100))[1] is not None
    assert body_emb.calls[-1] == (100, 100, 3)  # clamped to the frame
    assert extractor.extract(_image(), (10, 10, 20, 20)) == (None, None)  # too small
    assert extractor.extract(_image(), (500, 500, 600, 600)) == (None, None)  # outside the frame
