"""What the identity models can read from one tracked person in one frame: a face and a body embedding.

Face: SCRFD looks for a face inside the person's box, the eye region is cut (periocular, works with
masks) and ArcFace embeds it. Body: OSNet embeds the whole person crop. Either may be missing.
"""

import numpy as np

from src.identity.periocular import crop_periocular

MIN_CROP_SIDE = 16  # pixels; smaller person boxes are too small to embed


class AppearanceExtractor:
    def __init__(self, face_detector, face_embedder, body_embedder):
        """`face_detector.detect(img)` -> faces with .landmarks; the embedders expose `.embed(crop)`."""
        self._face_detector = face_detector
        self._face_embedder = face_embedder
        self._body_embedder = body_embedder

    def extract(self, image: np.ndarray, box: tuple[float, float, float, float]) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return (face_embedding, body_embedding) of the person in `box` (either can be None)."""
        crop = _crop(image, box)
        if crop is None:
            return None, None
        body = self._body_embedder.embed(crop)
        face = None
        faces = self._face_detector.detect(crop)  # sorted best first; one person, so take the best
        if faces:
            eyes = crop_periocular(crop, faces[0].landmarks)
            if eyes is not None:
                face = self._face_embedder.embed(eyes)
        return face, body


def _crop(image: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None:
    height, width = image.shape[:2]
    x1, y1 = max(int(box[0]), 0), max(int(box[1]), 0)
    x2, y2 = min(int(box[2]), width), min(int(box[3]), height)
    if x2 - x1 < MIN_CROP_SIDE or y2 - y1 < MIN_CROP_SIDE:
        return None
    return image[y1:y2, x1:x2]
