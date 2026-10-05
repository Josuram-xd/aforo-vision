from pathlib import Path

import numpy as np
import pytest

from src.identity.body_reid import MODEL_FILE, BodyEmbedder


class _StubInput:
    name = "input"


class _StubSession:
    """Mimics onnxruntime.InferenceSession: records the tensor it receives, returns a fixed feature."""

    def __init__(self, feature):
        self._feature = np.asarray(feature, dtype=np.float32).reshape(1, -1)
        self.received = []

    def get_inputs(self):
        return [_StubInput()]

    def run(self, _outputs, feeds):
        self.received.append(feeds["input"])
        return [self._feature]


def test_embedding_is_unit_norm_float32_vector():
    embedder = BodyEmbedder(session=_StubSession([3.0, 4.0, 0.0]))
    embedding = embedder.embed(np.zeros((200, 80, 3), dtype=np.uint8))
    assert embedding.dtype == np.float32
    np.testing.assert_allclose(embedding, [0.6, 0.8, 0.0], atol=1e-6)


def test_crop_is_resized_and_normalized_to_a_3x256x128_batch():
    stub = _StubSession([1.0, 0.0])
    BodyEmbedder(session=stub).embed(np.full((300, 90, 3), 255, dtype=np.uint8))

    tensor = stub.received[0]
    assert tensor.shape == (1, 3, 256, 128) and tensor.dtype == np.float32
    # white pixel -> (1 - mean) / std per RGB channel
    np.testing.assert_allclose(tensor[0, :, 0, 0], [(1 - 0.485) / 0.229, (1 - 0.456) / 0.224, (1 - 0.406) / 0.225], atol=1e-5)


def test_bgr_crop_is_fed_as_rgb():
    stub = _StubSession([1.0, 0.0])
    crop = np.zeros((20, 10, 3), dtype=np.uint8)
    crop[..., 2] = 255  # pure red in BGR
    BodyEmbedder(session=stub).embed(crop)
    tensor = stub.received[0][0]
    assert tensor[0, 0, 0] == pytest.approx((1 - 0.485) / 0.229, abs=1e-5)  # R channel is first
    assert tensor[2, 0, 0] == pytest.approx((0 - 0.406) / 0.225, abs=1e-5)


def test_empty_crop_and_zero_embedding_are_rejected():
    embedder = BodyEmbedder(session=_StubSession([1.0, 1.0]))
    with pytest.raises(ValueError):
        embedder.embed(np.empty((0, 0, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        embedder.embed(None)
    with pytest.raises(ValueError):
        BodyEmbedder(session=_StubSession([0.0, 0.0])).embed(np.zeros((10, 10, 3), dtype=np.uint8))


def test_missing_model_file_explains_how_to_export(tmp_path):
    with pytest.raises(FileNotFoundError, match="export_osnet_onnx"):
        BodyEmbedder(model_file=tmp_path / "nope.onnx")


@pytest.mark.skipif(not Path(MODEL_FILE).is_file(), reason="OSNet model not exported")
def test_real_model_gives_512d_unit_embeddings_and_separates_inputs():
    embedder = BodyEmbedder()
    rng = np.random.default_rng(1)
    a = embedder.embed(rng.integers(0, 255, (200, 80, 3), dtype=np.uint8))
    b = embedder.embed(rng.integers(0, 255, (200, 80, 3), dtype=np.uint8))
    assert a.shape == (512,)
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-5)
    assert float(a @ b) < 0.99
