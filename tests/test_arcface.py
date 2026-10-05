from pathlib import Path

import numpy as np
import pytest

from src.identity.arcface import MODEL_FILE, ArcFaceEmbedder


class _StubModel:
    """Mimics ArcFaceONNX: records the canvas it receives and returns a fixed raw feature."""

    input_size = (112, 112)

    def __init__(self, feature):
        self._feature = np.asarray(feature, dtype=np.float32).reshape(1, -1)
        self.received = []

    def get_feat(self, image):
        self.received.append(image)
        return self._feature


def test_embedding_is_unit_norm_float32_vector():
    embedder = ArcFaceEmbedder(model_instance=_StubModel([3.0, 4.0, 0.0]))
    embedding = embedder.embed(np.zeros((40, 80, 3), dtype=np.uint8))
    assert embedding.dtype == np.float32
    assert embedding.shape == (3,)
    assert np.linalg.norm(embedding) == pytest.approx(1.0)
    np.testing.assert_allclose(embedding, [0.6, 0.8, 0.0], atol=1e-6)


def test_wide_crop_is_letterboxed_without_stretching():
    stub = _StubModel([1.0, 0.0])
    crop = np.full((40, 80, 3), 200, dtype=np.uint8)  # 2:1, like a periocular crop
    ArcFaceEmbedder(model_instance=stub).embed(crop)

    canvas = stub.received[0]
    assert canvas.shape == (112, 112, 3)
    content_rows = np.nonzero((canvas != 127).any(axis=(1, 2)))[0]
    assert len(content_rows) == 56  # 112 wide -> 56 tall, proportions kept
    assert content_rows[0] == 28  # centered vertically
    assert (canvas[0] == 127).all() and (canvas[-1] == 127).all()  # neutral padding


def test_tall_crop_is_letterboxed_horizontally():
    stub = _StubModel([1.0, 0.0])
    ArcFaceEmbedder(model_instance=stub).embed(np.full((80, 40, 3), 200, dtype=np.uint8))
    content_cols = np.nonzero((stub.received[0] != 127).any(axis=(0, 2)))[0]
    assert len(content_cols) == 56


def test_same_crop_gives_same_embedding():
    embedder = ArcFaceEmbedder(model_instance=_StubModel([1.0, 2.0, 3.0]))
    crop = np.random.default_rng(0).integers(0, 255, (50, 100, 3), dtype=np.uint8)
    np.testing.assert_array_equal(embedder.embed(crop), embedder.embed(crop))


def test_empty_crop_and_zero_embedding_are_rejected():
    embedder = ArcFaceEmbedder(model_instance=_StubModel([1.0, 1.0]))
    with pytest.raises(ValueError):
        embedder.embed(np.empty((0, 0, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        embedder.embed(None)
    with pytest.raises(ValueError):
        ArcFaceEmbedder(model_instance=_StubModel([0.0, 0.0])).embed(np.zeros((10, 10, 3), dtype=np.uint8))


def test_missing_model_file_explains_how_to_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="--download"):
        ArcFaceEmbedder(model_file=tmp_path / "nope.onnx")


@pytest.mark.skipif(not Path(MODEL_FILE).is_file(), reason="ArcFace model not downloaded")
def test_real_model_gives_512d_unit_embeddings_and_separates_inputs():
    embedder = ArcFaceEmbedder()
    rng = np.random.default_rng(1)
    a = embedder.embed(rng.integers(0, 255, (48, 96, 3), dtype=np.uint8))
    b = embedder.embed(rng.integers(0, 255, (48, 96, 3), dtype=np.uint8))
    assert a.shape == (512,)
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-5)
    assert float(a @ b) < 0.99
