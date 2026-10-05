"""Entry point for aforo-vision. Run from the repo root: python -m src.main [--debug] [--dry-run]

Wires the real models, the two cameras and the event delivery around src/pipeline.py.
Needs AFORO_BACKEND_URL and AFORO_BACKEND_SECRET unless --dry-run (events are only printed).
"""

import argparse
import json
import logging
import time
from pathlib import Path

from enrollment.enroll_student import EMBEDDINGS_DIR
from src.capture.frame_grabber import FrameGrabber
from src.config import load_config
from src.dedup.ttl_set import TTLSet
from src.debug.preview import Annotation, PreviewWindows
from src.detection.yolo_pose import YoloPoseDetector
from src.matching.cross_checkpoint import CrossCheckpointMatcher
from src.pipeline import Pipeline
from src.tracking.sort_tracker import SortTracker

logger = logging.getLogger("aforo-vision")

_IDLE_SLEEP_SECONDS = 0.005


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline local de vision del piloto de aforo.")
    parser.add_argument("--config", default="config/pilot.yaml", help="ruta al archivo de configuracion")
    parser.add_argument("--debug", action="store_true", help="abre una ventana local por camara con FPS y cajas")
    parser.add_argument("--dry-run", action="store_true", help="solo muestra los eventos en consola: no los guarda ni los envia")
    parser.add_argument("--embeddings-dir", type=Path, default=EMBEDDINGS_DIR, help="carpeta con los estudiantes enrolados")
    parser.add_argument("--no-single-camera-fallback", action="store_true",
                        help="no generar eventos de personas vistas por una sola camara (solo por trayectoria)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    config = load_config(args.config)
    if not args.dry_run and not config.backend.shared_secret:
        logger.error("falta AFORO_BACKEND_SECRET (el secreto con el que se desplego aforo-backend); usa --dry-run para probar sin enviar")
        return 2

    try:
        from src.identity.appearance import AppearanceExtractor
        from src.identity.arcface import ArcFaceEmbedder
        from src.identity.body_reid import BodyEmbedder
        from src.identity.face_detector import FaceDetector
        from src.identity.roster import load_enrolled

        extractor = AppearanceExtractor(FaceDetector(), ArcFaceEmbedder(), BodyEmbedder())
    except FileNotFoundError as error:
        logger.error("%s", error)
        return 2
    identity_index, names = load_enrolled(args.embeddings_dir, config.identity.face_similarity_threshold)
    if not names:
        logger.warning("no hay estudiantes enrolados en %s: todos los eventos saldran sin nombre", args.embeddings_dir)
    else:
        logger.info("%d estudiante(s) enrolado(s)", len(names))

    delivery = None
    if args.dry_run:
        def emit(event: dict) -> None:
            logger.info("dry-run, no se envia: %s", json.dumps(event, ensure_ascii=False))
    else:
        from src.events.delivery import EventDelivery
        from src.events.health import check_backend_health
        from src.events.retry_queue import RetryQueue
        from src.events.uploader import EventUploader

        if check_backend_health(config.backend.url, config.backend.timeout_seconds):
            logger.info("backend OK (%s)", config.backend.url)
        else:
            logger.warning("NO hay conexion con el backend (%s): los eventos se guardan en la cola y se envian cuando vuelva", config.backend.url)
        delivery = EventDelivery(
            RetryQueue(config.backend.retry_queue_path),
            EventUploader(config.backend.url, config.backend.shared_secret, config.backend.timeout_seconds),
        )
        emit = delivery.submit

    # Same models and thresholds for both cameras (AGENTS.md rule 3)
    pipeline = Pipeline(
        detectors={
            camera_id: YoloPoseDetector(config.detection.model, config.detection.confidence_threshold)
            for camera_id in config.cameras
        },
        trackers={
            camera_id: SortTracker(config.tracking.iou_threshold, config.tracking.max_age_frames,
                                   config.tracking.min_hits, config.tracking.trajectory_length)
            for camera_id in config.cameras
        },
        extractor=extractor,
        matcher=CrossCheckpointMatcher(config.matching.time_window_seconds, config.identity.face_similarity_threshold,
                                       config.identity.body_similarity_threshold),
        identity_index=identity_index,
        names=names,
        dedup=TTLSet(config.dedup.ttl_seconds),
        emit=emit,
        single_camera_fallback=not args.no_single_camera_fallback,
    )

    grabbers = [
        FrameGrabber(camera.source, camera.camera_id, config.capture.reconnect_interval_seconds)
        for camera in config.cameras.values()
    ]
    tracks: dict[str, list] = {}
    preview = PreviewWindows() if args.debug else None
    last_index: dict[str, int] = {}

    if delivery is not None:
        delivery.start()
    for grabber in grabbers:
        grabber.start()
    logger.info("pipeline running%s%s, Ctrl+C to stop", " (dry-run)" if args.dry_run else "",
                " (debug preview: q/Esc to quit)" if preview else "")

    try:
        while True:
            got_new_frame = False
            for grabber in grabbers:
                frame = grabber.read_latest()
                is_new = frame is not None and frame.index != last_index.get(grabber.camera_id)
                if is_new:
                    last_index[grabber.camera_id] = frame.index
                    got_new_frame = True
                    tracks[grabber.camera_id] = pipeline.process_frame(grabber.camera_id, frame.image, frame.timestamp)

                if preview is not None:
                    boxes = [Annotation(*t.box, f"id {t.id}") for t in tracks.get(grabber.camera_id, ())]
                    preview.update(grabber.camera_id, frame.image if is_new else None, grabber.is_connected, boxes)

            pipeline.expire(time.time())
            if preview is not None and not preview.poll():
                break
            if not got_new_frame:
                time.sleep(_IDLE_SLEEP_SECONDS)
    except KeyboardInterrupt:
        pass
    finally:
        for grabber in grabbers:
            grabber.stop()
        if preview is not None:
            preview.close()
        if delivery is not None:
            delivery.stop()
            logger.info("%d evento(s) quedan en la cola para el proximo arranque", len(delivery.queue))
        logger.info("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
