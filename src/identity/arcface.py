"""Face embeddings with the standard InsightFace ArcFace model (CPU, ONNX).

The periocular crop from periocular.py goes through the same ArcFace model used for whole faces;
there is no dedicated "periocular model" (AGENTS.md rule 2). Model files live in data/ (git-ignored).

The model is never downloaded implicitly. Run once, with your approval of the network call:
    python -m src.identity.arcface --download
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

MODEL_ROOT = Path("data/insightface")
MODEL_PACK = "buffalo_l"  # also ships the SCRFD face detector with 5-point landmarks
MODEL_FILE = MODEL_ROOT / "models" / MODEL_PACK / "w600k_r50.onnx"  # ArcFace ResNet50, 512-d output

_PAD_VALUE = 127  # mid gray: becomes 0.0 after the model's (x - 127.5) / 127.5 normalization


class ArcFaceEmbedder:
    def __init__(self, model_file: str | Path = MODEL_FILE, model_instance=None):
        if model_instance is None:
            model_file = Path(model_file)
            if not model_file.is_file():
                raise FileNotFoundError(
                    f"ArcFace model not found at {model_file}. Download it once with: python -m src.identity.arcface --download"
                )
            from insightface.model_zoo import ArcFaceONNX  # lazy: pulls in onnx and onnxruntime

            model_instance = ArcFaceONNX(model_file=str(model_file))
            model_instance.prepare(ctx_id=-1)  # CPU only
        self._model = model_instance

    def embed(self, crop: np.ndarray) -> np.ndarray:
        """Return the L2-normalized embedding (float32 vector) of a BGR crop of any size."""
        if crop is None or crop.size == 0:
            raise ValueError("cannot embed an empty crop")
        side = int(self._model.input_size[0])
        feature = np.asarray(self._model.get_feat(_letterbox(crop, side))).reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(feature))
        if norm == 0.0:
            raise ValueError("model returned a zero embedding")
        return feature / norm


def _letterbox(image: np.ndarray, side: int) -> np.ndarray:
    """Fit `image` inside a side x side canvas keeping its proportions (no stretching of the eyes)."""
    height, width = image.shape[:2]
    scale = side / max(height, width)
    new_width, new_height = max(round(width * scale), 1), max(round(height * scale), 1)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.full((side, side, 3), _PAD_VALUE, dtype=np.uint8)
    top, left = (side - new_height) // 2, (side - new_width) // 2
    canvas[top:top + new_height, left:left + new_width] = resized
    return canvas


_default_embedder: ArcFaceEmbedder | None = None


def embed(crop: np.ndarray) -> np.ndarray:
    """Embed a crop with the default model (loaded on first use)."""
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = ArcFaceEmbedder()
    return _default_embedder.embed(crop)


def download_model(root: str | Path = MODEL_ROOT) -> Path:
    """Fetch the InsightFace model pack into `root`. This is a network call; only run it on purpose."""
    from insightface.utils import ensure_available

    return Path(ensure_available("models", MODEL_PACK, root=str(root)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modelo ArcFace de InsightFace para la identidad.")
    parser.add_argument("--download", action="store_true", help=f"descarga el paquete {MODEL_PACK} a {MODEL_ROOT}")
    if parser.parse_args().download:
        print(f"Descargando {MODEL_PACK} (modelos de InsightFace) a {MODEL_ROOT} ...")
        print(f"Listo: {download_model()}")
    else:
        parser.print_help()
