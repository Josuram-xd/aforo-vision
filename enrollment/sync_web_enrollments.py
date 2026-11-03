"""Turn the face photos a teacher took in the web app into local embeddings, then erase the photos.

    python -m enrollment.sync_web_enrollments            # one pass
    python -m enrollment.sync_web_enrollments --watch    # poll every 15 s

Flow (aforo-db task 12): the web app stores 3-8 JPEGs in private.enrollment_captures; this script asks
for them with the device key (RPC pending_enrollments), computes the periocular embedding HERE, saves
data/embeddings/<studentId>.npy and calls complete_enrollment, which deletes the photos in the cloud.
The embedding never leaves this machine (AGENTS.md rule 1; the photos' short trip is its documented exception).

Needs AFORO_BACKEND_URL, AFORO_SUPABASE_KEY and AFORO_DEVICE_KEY, like the event uploader.
"""

import argparse
import base64
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import httpx
import numpy as np

from enrollment.enroll_student import EMBEDDINGS_DIR, EnrollmentError, average_embeddings, save_enrollment, validate_person_id

logger = logging.getLogger(__name__)

MIN_USABLE_PHOTOS = 3
POLL_SECONDS = 15.0

EmbedFrame = Callable[[np.ndarray], np.ndarray | None]


def embed_jpegs(jpegs: list[bytes], embed_frame: EmbedFrame) -> tuple[np.ndarray, int]:
    """Average embedding of the photos that have exactly one usable face. Raises EnrollmentError otherwise."""
    embeddings = []
    for data in jpegs:
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        embedding = embed_frame(frame)
        if embedding is not None:
            embeddings.append(embedding)
    if len(embeddings) < MIN_USABLE_PHOTOS:
        raise EnrollmentError(f"only {len(embeddings)} of {len(jpegs)} photos had one clear face (need {MIN_USABLE_PHOTOS})")
    return average_embeddings(embeddings), len(embeddings)


class CloudEnrollments:
    """The two device RPCs. Raises httpx errors; the caller decides whether to retry."""

    def __init__(self, base_url: str, supabase_key: str, device_key: str, client: httpx.Client | None = None):
        self._base = base_url.rstrip("/") + "/rest/v1/rpc/"
        self._headers = {"apikey": supabase_key, "X-Device-Key": device_key, "Content-Type": "application/json"}
        self._client = client or httpx.Client(timeout=30.0)

    def pending(self) -> list[dict]:
        response = self._client.post(self._base + "pending_enrollments", json={}, headers=self._headers)
        response.raise_for_status()
        return response.json()

    def complete(self, student_id: str, ok: bool) -> None:
        body = {"p_student_id": student_id, "p_ok": ok}
        self._client.post(self._base + "complete_enrollment", json=body, headers=self._headers).raise_for_status()


def sync_once(cloud: CloudEnrollments, embed_frame: EmbedFrame, root: Path = EMBEDDINGS_DIR) -> int:
    """Process every pending enrollment once. Returns how many were enrolled."""
    done = 0
    for item in cloud.pending():
        student_id = item["studentId"]
        try:
            validate_person_id(student_id)
            embedding, samples = embed_jpegs([base64.b64decode(img) for img in item["images"]], embed_frame)
            save_enrollment(student_id, "web-enrollment", embedding, samples, root)
            cloud.complete(student_id, True)
            done += 1
            logger.info("enrolled %s from %d photos", student_id, samples)
        except (EnrollmentError, ValueError) as error:
            logger.warning("enrollment of %s failed: %s", student_id, error)
            cloud.complete(student_id, False)  # also deletes the photos
    return done


def _face_embedder() -> EmbedFrame:
    from enrollment.enroll_student import periocular_embedding
    from src.identity.arcface import ArcFaceEmbedder
    from src.identity.face_detector import FaceDetector

    detector, embedder = FaceDetector(), ArcFaceEmbedder()
    return lambda frame: periocular_embedding(frame, detector.detect(frame), embedder.embed)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convierte las fotos de registro hechas en la web en embeddings locales.")
    parser.add_argument("--watch", action="store_true", help=f"repetir cada {POLL_SECONDS:.0f} s")
    parser.add_argument("--output-dir", type=Path, default=EMBEDDINGS_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    url, key, device = (os.environ.get(n) for n in ("AFORO_BACKEND_URL", "AFORO_SUPABASE_KEY", "AFORO_DEVICE_KEY"))
    if not (url and key and device):
        parser.error("faltan AFORO_BACKEND_URL, AFORO_SUPABASE_KEY o AFORO_DEVICE_KEY")
    cloud, embed_frame = CloudEnrollments(url, key, device), _face_embedder()
    while True:
        try:
            sync_once(cloud, embed_frame, args.output_dir)
        except httpx.HTTPError as error:
            logger.warning("sync failed: %s", type(error).__name__)  # never log the response: it may carry keys
        if not args.watch:
            return 0
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
