"""Face detection with 5-point landmarks (SCRFD from the InsightFace buffalo_l pack, CPU, ONNX).

Gives crop_periocular() the eye positions: landmark rows 0 and 1 are the eyes. Uses the same model
pack as arcface.py, so `python -m src.identity.arcface --download` already fetches it.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.identity.arcface import MODEL_ROOT, MODEL_PACK

MODEL_FILE = MODEL_ROOT / "models" / MODEL_PACK / "det_10g.onnx"
INPUT_SIZE = (640, 640)  # (width, height) the detector resizes frames to


@dataclass(frozen=True)
class Face:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in frame pixels
    score: float
    landmarks: np.ndarray  # (5, 2): eyes first, then nose and mouth corners

    @property
    def area(self) -> float:
        return max(self.box[2] - self.box[0], 0.0) * max(self.box[3] - self.box[1], 0.0)


class FaceDetector:
    def __init__(self, score_threshold: float = 0.5, model_file: str | Path = MODEL_FILE, model_instance=None):
        if model_instance is None:
            model_file = Path(model_file)
            if not model_file.is_file():
                raise FileNotFoundError(
                    f"Face detector not found at {model_file}. Download it once with: python -m src.identity.arcface --download"
                )
            from insightface.model_zoo import get_model  # lazy: pulls in onnx and onnxruntime

            model_instance = get_model(str(model_file))
            model_instance.prepare(ctx_id=-1, input_size=INPUT_SIZE, det_thresh=score_threshold)  # CPU only
        self._model = model_instance
        self.score_threshold = score_threshold

    def detect(self, frame: np.ndarray) -> list[Face]:
        """Return the faces found in a BGR frame, best score first."""
        boxes, landmarks = self._model.detect(frame, input_size=INPUT_SIZE)
        if boxes is None or landmarks is None:
            return []
        faces = [
            Face(tuple(float(v) for v in box[:4]), float(box[4]), np.asarray(points, dtype=np.float32))
            for box, points in zip(boxes, landmarks)
            if box[4] >= self.score_threshold
        ]
        return sorted(faces, key=lambda face: face.score, reverse=True)
