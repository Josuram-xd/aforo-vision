"""Body re-id embeddings with OSNet-AIN (CPU, ONNX), for tracks where no face is usable.

These embeddings only count and pair people (method = BODY_ONLY, cross-checkpoint matching).
They are never matched against the enrolled roster, so they never put a name on anyone (task 4.4).

torchreid has no CPython 3.14 wheel, so the network is exported once to ONNX and run with onnxruntime.
The model file is never downloaded implicitly. Run once, with your approval of the network call:
    python -m tools.export_osnet_onnx
"""

from pathlib import Path

import cv2
import numpy as np

MODEL_FILE = Path("data/osnet/osnet_ain_x0_25.onnx")  # 512-d output, input 3x256x128

_INPUT_SIZE = (128, 256)  # (width, height), the usual person re-id crop size
_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)  # ImageNet statistics, RGB
_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


class BodyEmbedder:
    def __init__(self, model_file: str | Path = MODEL_FILE, session=None):
        if session is None:
            model_file = Path(model_file)
            if not model_file.is_file():
                raise FileNotFoundError(
                    f"OSNet model not found at {model_file}. Export it once with: python -m tools.export_osnet_onnx"
                )
            import onnxruntime  # lazy: only needed when the real model is loaded

            session = onnxruntime.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])
        self._session = session
        self._input_name = session.get_inputs()[0].name

    def embed(self, crop: np.ndarray) -> np.ndarray:
        """Return the L2-normalized embedding (float32 vector) of a BGR person crop of any size."""
        if crop is None or crop.size == 0:
            raise ValueError("cannot embed an empty crop")
        feature = np.asarray(self._session.run(None, {self._input_name: _preprocess(crop)[None]})[0]).reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(feature))
        if norm == 0.0:
            raise ValueError("model returned a zero embedding")
        return feature / norm


def _preprocess(crop: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, _INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    normalized = (resized.astype(np.float32) / 255.0 - _MEAN) / _STD
    return normalized.transpose(2, 0, 1)  # CHW


_default_embedder: BodyEmbedder | None = None


def embed_body(crop: np.ndarray) -> np.ndarray:
    """Embed a person crop with the default model (loaded on first use)."""
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = BodyEmbedder()
    return _default_embedder.embed(crop)
