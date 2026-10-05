"""Enroll one student: capture N periocular samples from a webcam, average them, save the embedding.

    python -m enrollment.enroll_student --name "Ana Perez"
    python -m enrollment.enroll_student --name "Ana Perez" --person-id <uuid>   # re-enroll the same person

Writes data/embeddings/<personId>.npy (the averaged embedding) and <personId>.json (personId, name,
sample count, date). data/ is git-ignored: embeddings are biometric data and never leave this laptop
(AGENTS.md rule 1). Only enroll people who signed the consent form (task 5.3).
"""

import argparse
import json
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from src.identity.face_detector import Face, FaceDetector
from src.identity.periocular import crop_periocular

EMBEDDINGS_DIR = Path("data/embeddings")
DEFAULT_SAMPLES = 10
MIN_SAMPLE_INTERVAL_SECONDS = 0.5  # spacing between samples so they differ a little (pose, light)
MIN_CONSISTENCY = 0.5  # every sample must be at least this cosine-similar to the average
TIMEOUT_SECONDS = 60.0


class EnrollmentError(RuntimeError):
    pass


def average_embeddings(embeddings: list[np.ndarray]) -> np.ndarray:
    """L2-normalized mean of unit embeddings. Raises if any sample strays from the average
    (e.g. someone else walked into view), so a bad enrollment is retried instead of saved."""
    stacked = np.stack([np.asarray(e, dtype=np.float32).reshape(-1) for e in embeddings])
    mean = stacked.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm == 0.0:
        raise EnrollmentError("samples cancel each other out; capture again")
    mean /= norm
    worst = float((stacked @ mean).min())
    if worst < MIN_CONSISTENCY:
        raise EnrollmentError(
            f"inconsistent samples (lowest similarity to the average {worst:.2f} < {MIN_CONSISTENCY}); "
            "make sure only one person is in front of the camera and capture again"
        )
    return mean


def periocular_embedding(frame: np.ndarray, faces: list[Face], embed: Callable[[np.ndarray], np.ndarray]) -> np.ndarray | None:
    """Embedding of the only face in the frame, or None if there is no usable face.
    With several faces nothing is taken: it could be a bystander."""
    if len(faces) != 1:
        return None
    crop = crop_periocular(frame, faces[0].landmarks)
    return None if crop is None else embed(crop)


def collect_samples(
    read_frame: Callable[[], np.ndarray | None],
    detector: FaceDetector,
    embed: Callable[[np.ndarray], np.ndarray],
    count: int,
    on_frame: Callable[[np.ndarray, list[Face], int], bool] | None = None,
    min_interval: float = MIN_SAMPLE_INTERVAL_SECONDS,
    timeout: float = TIMEOUT_SECONDS,
) -> list[np.ndarray]:
    """Grab `count` embeddings. `on_frame(frame, faces, taken)` draws the preview; returning True cancels."""
    samples: list[np.ndarray] = []
    started = time.monotonic()
    last_taken = -min_interval
    while len(samples) < count:
        if time.monotonic() - started > timeout:
            raise EnrollmentError(f"timed out after {timeout:.0f} s with {len(samples)}/{count} samples")
        frame = read_frame()
        if frame is None:
            time.sleep(0.02)
            continue
        faces = detector.detect(frame)
        if on_frame is not None and on_frame(frame, faces, len(samples)):
            raise EnrollmentError("cancelled")
        if time.monotonic() - last_taken < min_interval:
            continue
        embedding = periocular_embedding(frame, faces, embed)
        if embedding is not None:
            samples.append(embedding)
            last_taken = time.monotonic()
    return samples


def validate_person_id(person_id: str) -> None:
    """The backend contract requires UUID v4 ids (task 5.2). UUID(..., version=4) would not check, only overwrite."""
    if uuid.UUID(person_id).version != 4:
        raise ValueError(f"personId must be a UUID v4: {person_id}")


def save_enrollment(person_id: str, name: str, embedding: np.ndarray, samples: int, root: Path = EMBEDDINGS_DIR) -> Path:
    validate_person_id(person_id)
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / f"{person_id}.npy", np.asarray(embedding, dtype=np.float32))
    metadata = {
        "personId": person_id,
        "name": name,
        "samples": samples,
        "enrolledAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (root / f"{person_id}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return root / f"{person_id}.npy"


def _draw_preview(frame: np.ndarray, faces: list[Face], taken: int, total: int) -> np.ndarray:
    view = frame.copy()
    for face in faces:
        x1, y1, x2, y2 = (int(v) for v in face.box)
        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 200, 0) if len(faces) == 1 else (0, 0, 255), 2)
        for x, y in face.landmarks[:2]:
            cv2.circle(view, (int(x), int(y)), 3, (255, 255, 0), -1)
    note = "Solo una persona" if len(faces) > 1 else f"Muestras {taken}/{total}  (q = cancelar)"
    cv2.putText(view, note, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return view


def main() -> int:
    parser = argparse.ArgumentParser(description="Enrola a un estudiante: promedia su embedding periocular desde la webcam.")
    parser.add_argument("--name", required=True, help="nombre completo (solo queda en data/, nunca en git)")
    parser.add_argument("--person-id", help="UUID v4 existente para volver a enrolar a la misma persona")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES, help=f"muestras a capturar (por defecto {DEFAULT_SAMPLES})")
    parser.add_argument("--camera", type=int, default=0, help="índice de la webcam USB (por defecto 0)")
    parser.add_argument("--output-dir", type=Path, default=EMBEDDINGS_DIR)
    parser.add_argument("--no-preview", action="store_true", help="no abrir la ventana de vista previa")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples debe ser al menos 1")
    person_id = args.person_id or str(uuid.uuid4())
    try:
        validate_person_id(person_id)
    except ValueError:
        parser.error("--person-id debe ser un UUID v4")

    from src.capture.frame_grabber import FrameGrabber
    from src.identity.arcface import ArcFaceEmbedder

    detector, embedder = FaceDetector(), ArcFaceEmbedder()
    print(f"Enrolando a {args.name} ({person_id}). Mira a la cámara; {args.samples} muestras...")
    with FrameGrabber(args.camera, "enrollment-webcam") as grabber:
        latest = {"index": -1}

        def read_frame() -> np.ndarray | None:
            frame = grabber.read_latest()
            if frame is None or frame.index == latest["index"]:
                return None
            latest["index"] = frame.index
            return frame.image

        def on_frame(image: np.ndarray, faces: list[Face], taken: int) -> bool:
            if args.no_preview:
                return False
            cv2.imshow("Enrolamiento", _draw_preview(image, faces, taken, args.samples))
            return cv2.waitKey(1) & 0xFF == ord("q")

        try:
            samples = collect_samples(read_frame, detector, embedder.embed, args.samples, on_frame)
            embedding = average_embeddings(samples)
        except EnrollmentError as error:
            print(f"No se guardó nada: {error}")
            return 1
        finally:
            cv2.destroyAllWindows()

    path = save_enrollment(person_id, args.name, embedding, len(samples), args.output_dir)
    print(f"Listo: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
