"""Entry point for aforo-vision. Run from the repo root: python -m src.main [--debug]"""

import argparse
import logging
import time

from src.capture.frame_grabber import FrameGrabber
from src.config import load_config
from src.debug.preview import Annotation, PreviewWindows
from src.detection.yolo_pose import Detection, YoloPoseDetector

logger = logging.getLogger("aforo-vision")

_IDLE_SLEEP_SECONDS = 0.005


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline local de vision del piloto de aforo.")
    parser.add_argument("--config", default="config/pilot.yaml", help="ruta al archivo de configuracion")
    parser.add_argument("--debug", action="store_true", help="abre una ventana local por camara con FPS y cajas")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    config = load_config(args.config)

    grabbers = [
        FrameGrabber(camera.source, camera.camera_id, config.capture.reconnect_interval_seconds)
        for camera in config.cameras.values()
    ]
    # Same model and threshold for both cameras (AGENTS.md rule 3)
    detectors = {
        camera_id: YoloPoseDetector(config.detection.model, config.detection.confidence_threshold)
        for camera_id in config.cameras
    }
    detections: dict[str, list[Detection]] = {}
    preview = PreviewWindows() if args.debug else None
    last_index: dict[str, int] = {}

    for grabber in grabbers:
        grabber.start()
    logger.info("pipeline running%s, Ctrl+C to stop", " (debug preview: q/Esc to quit)" if preview else "")

    try:
        while True:
            got_new_frame = False
            for grabber in grabbers:
                frame = grabber.read_latest()
                is_new = frame is not None and frame.index != last_index.get(grabber.camera_id)
                if is_new:
                    last_index[grabber.camera_id] = frame.index
                    got_new_frame = True
                    detections[grabber.camera_id] = detectors[grabber.camera_id].detect(frame.image)
                    # Tracking plugs in here (tasks 3.2-3.5)

                if preview is not None:
                    boxes = [Annotation(d.x1, d.y1, d.x2, d.y2, f"{d.score:.2f}")
                             for d in detections.get(grabber.camera_id, ())]
                    preview.update(grabber.camera_id, frame.image if is_new else None, grabber.is_connected, boxes)

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
        logger.info("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
