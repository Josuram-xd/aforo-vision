"""Entry point for aforo-vision. Run from the repo root: python -m src.main [--debug] [--dry-run]

Wires the real models, the two cameras and the event delivery around src/pipeline.py.
Needs AFORO_BACKEND_URL, AFORO_SUPABASE_KEY and AFORO_DEVICE_KEY unless --dry-run (events are only printed).
"""

import argparse
import json
import logging
import time
from pathlib import Path

from enrollment.enroll_student import EMBEDDINGS_DIR
from src.capture.frame_grabber import FrameGrabber
from src.capture.motion_gate import MotionGate
from src import performance
from src.config import ConfigError, load_config, load_dotenv
from src.dedup.ttl_set import TTLSet
from src.debug.preview import Annotation, PreviewWindows
from src.matching.cross_checkpoint import CrossCheckpointMatcher
from src.pipeline import Pipeline
from src.tracking.sort_tracker import SortTracker

logger = logging.getLogger("aforo-vision")

_IDLE_SLEEP_SECONDS = 0.005
_MOTION_LOG_SECONDS = 60.0


def process_if_needed(pipeline, gate, camera_id: str, image, timestamp: float, now: float) -> list:
    """Same rule for every camera: no gate means always process; with a gate, only on motion, a live track or warmup."""
    if gate is None or gate.should_process(image, now, pipeline.trackers[camera_id].live_boxes()):
        return pipeline.process_frame(camera_id, image, timestamp)
    return []


def _build_detector(detection):
    """Create one detector for the configured backend; heavy imports stay inside so slim installs never load torch."""
    if detection.backend == "onnx":
        from src.detection.yolo_onnx import OnnxPoseDetector

        return OnnxPoseDetector(detection.onnx_model, detection.confidence_threshold)
    from src.detection.yolo_pose import YoloPoseDetector

    return YoloPoseDetector(detection.model, detection.confidence_threshold)


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
    load_dotenv()
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        logger.error("configuración inválida: %s", exc)
        return 2
    # Once, before any model exists: thread pools read the limit when they are created
    performance.apply(config.performance.cpu_threads, config.performance.low_priority, config.detection.backend)
    if not args.dry_run:
        missing = [name for name, value in (("AFORO_SUPABASE_KEY", config.backend.supabase_key),
                                            ("AFORO_DEVICE_KEY", config.backend.device_key)) if not value]
        if missing:
            logger.error("falta(n) %s (llaves de Supabase); usa --dry-run para probar sin enviar", ", ".join(missing))
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
    heartbeat_client = None
    if args.dry_run:
        def emit(event: dict) -> None:
            logger.info("dry-run, no se envia: %s", json.dumps(event, ensure_ascii=False))
    else:
        from src.events.delivery import EventDelivery
        from src.events.health import check_backend_health
        from src.events.heartbeat import HeartbeatClient
        from src.events.retry_queue import RetryQueue
        from src.events.uploader import EventUploader

        if check_backend_health(config.backend.url, config.backend.supabase_key, config.backend.device_key,
                                config.cameras, config.backend.timeout_seconds):
            logger.info("backend OK (Supabase acepto las llaves)")
        else:
            logger.warning("NO hay conexion con Supabase o rechazo las llaves: los eventos se guardan en la cola y se envian cuando vuelva")
        delivery = EventDelivery(
            RetryQueue(config.backend.retry_queue_path),
            EventUploader(config.backend.url, config.backend.supabase_key, config.backend.device_key, config.backend.timeout_seconds),
        )
        heartbeat_client = HeartbeatClient(config.backend.url, config.backend.supabase_key, config.backend.device_key,
                                           config.backend.timeout_seconds)
        emit = delivery.submit

    # Same models and thresholds for both cameras (AGENTS.md rule 3)
    pipeline = Pipeline(
        detectors={
            camera_id: _build_detector(config.detection)
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

    # Cameras registered in the web app win over pilot.yaml; AFORO_CAMERA_*_SOURCE still wins over both
    cloud_sources = None
    if not args.dry_run:
        from src.capture.cloud_sources import CloudCameraSources

        cloud_sources = CloudCameraSources(config.backend.url, config.backend.supabase_key, config.backend.device_key,
                                           config.backend.timeout_seconds)
    grabbers = []
    for camera in config.cameras.values():
        refresh = None if cloud_sources is None or camera.pinned else (lambda cid=camera.camera_id: cloud_sources.url_for(cid))
        source = (refresh() if refresh else None) or camera.source
        grabbers.append(FrameGrabber(source, camera.camera_id, config.capture.reconnect_interval_seconds, refresh))
    heartbeat = None
    if heartbeat_client is not None:
        from src.events.heartbeat import HeartbeatThread

        heartbeat = HeartbeatThread(heartbeat_client, {g.camera_id: (lambda g=g: g.is_connected) for g in grabbers})
    snapshots = None
    if heartbeat_client is not None:
        from src.capture.snapshots import SnapshotClient, SnapshotWorker

        snapshots = SnapshotWorker(SnapshotClient(config.backend.url, config.backend.supabase_key, config.backend.device_key,
                                                  config.backend.timeout_seconds), {g.camera_id: g for g in grabbers})
    tracks: dict[str, list] = {}
    preview = PreviewWindows() if args.debug else None
    last_index: dict[str, int] = {}
    motion = config.motion
    gates = {cid: MotionGate(motion.roi[cid], motion.pixel_threshold, motion.min_area, motion.hold_seconds,
                             motion.idle_fps, motion.warmup_seconds) for cid in config.cameras} if motion.enabled else {}
    next_motion_log = time.time() + _MOTION_LOG_SECONDS
    if gates:
        logger.info("modo ahorro (movimiento) activado")

    if delivery is not None:
        delivery.start()
    for grabber in grabbers:
        grabber.start()
    if heartbeat is not None:
        heartbeat.start()
    if snapshots is not None:
        snapshots.start()
    logger.info("pipeline running%s%s, Ctrl+C to stop", " (dry-run)" if args.dry_run else "",
                " (debug preview: q/Esc to quit)" if preview else "")

    try:
        while True:
            got_new_frame = False
            processed_any = False
            for grabber in grabbers:
                frame = grabber.read_latest()
                is_new = frame is not None and frame.index != last_index.get(grabber.camera_id)
                if is_new:
                    last_index[grabber.camera_id] = frame.index
                    got_new_frame = True
                    tracks[grabber.camera_id] = process_if_needed(pipeline, gates.get(grabber.camera_id), grabber.camera_id,
                                                                  frame.image, frame.timestamp, time.time())
                    gate = gates.get(grabber.camera_id)
                    processed_any = processed_any or gate is None or not gate.idle

                if preview is not None:
                    boxes = [Annotation(*t.box, f"id {t.id}") for t in tracks.get(grabber.camera_id, ())]
                    preview.update(grabber.camera_id, frame.image if is_new else None, grabber.is_connected, boxes)

            pipeline.expire(time.time())
            if preview is not None and not preview.poll():
                break
            if gates and time.time() >= next_motion_log:
                next_motion_log += _MOTION_LOG_SECONDS
                for camera_id, gate in gates.items():
                    logger.info("%s: %s", camera_id, gate.summary())
            if gates and not processed_any and all(g.idle for g in gates.values()):
                time.sleep(1.0 / motion.idle_fps)  # nothing to do: check again at idle_fps instead of spinning
            elif not got_new_frame:
                time.sleep(_IDLE_SLEEP_SECONDS)
    except KeyboardInterrupt:
        pass
    finally:
        if heartbeat is not None:
            heartbeat.stop()
        if snapshots is not None:
            snapshots.stop()
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
