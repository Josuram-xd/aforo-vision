"""Calibrate the motion gate: python -m tools.motion_preview --camera camera-inside [--no-window]

Shows the ROI, the live % of changed pixels and whether the gate would be active. In the window, drag the mouse
to draw a new ROI; press q/Esc to quit. The ROI is printed ready to paste into config/pilot.yaml (motion.roi).
Without a window (--no-window, or no GUI available) it prints one text line per second.
"""

import argparse
import time

import cv2

from src.capture.frame_grabber import _open_capture
from src.capture.motion_gate import MotionGate
from src.config import ConfigError, load_config, load_dotenv


def format_roi(roi) -> str:
    return "[" + ", ".join(f"{v:.2f}" for v in roi) + "]"


def resolve_source(args, config):
    """--source wins; otherwise the camera's source (web-registered connection when .env has the keys, else pilot.yaml)."""
    if args.source is not None:
        return int(args.source) if args.source.isdigit() else args.source
    camera = config.cameras[args.camera]
    if not camera.pinned and config.backend.supabase_key and config.backend.device_key:
        from src.capture.cloud_sources import CloudCameraSources

        url = CloudCameraSources(config.backend.url, config.backend.supabase_key, config.backend.device_key,
                                 config.backend.timeout_seconds).url_for(args.camera)
        if url:
            return url
    return camera.source


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Calibra la ROI y los umbrales del modo ahorro (movimiento).")
    parser.add_argument("--config", default="config/pilot.yaml")
    parser.add_argument("--camera", default="camera-inside", choices=("camera-outside", "camera-inside"))
    parser.add_argument("--source", help="indice USB o URL RTSP (si no, se usa la camara de la configuracion)")
    parser.add_argument("--roi", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"), help="ROI inicial (0-1)")
    parser.add_argument("--no-window", action="store_true", help="modo texto: una linea por segundo")
    parser.add_argument("--seconds", type=float, default=30.0, help="duracion en modo texto")
    args = parser.parse_args(argv)
    load_dotenv()
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"configuracion invalida: {exc}")
        return 2
    motion = config.motion
    roi = tuple(args.roi) if args.roi else motion.roi[args.camera]
    gate = MotionGate(roi, motion.pixel_threshold, motion.min_area, motion.hold_seconds, motion.idle_fps,
                      motion.warmup_seconds)
    capture = _open_capture(resolve_source(args, config))
    if not capture.isOpened():
        print("no se pudo abrir la camara (revisa --source o ejecuta python -m tools.doctor)")
        return 1

    window = not args.no_window
    drag = {"start": None, "end": None}
    shape = [1, 1]
    if window:
        try:
            cv2.namedWindow("motion_preview")

            def on_mouse(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    drag["start"], drag["end"] = (x, y), (x, y)
                elif event == cv2.EVENT_MOUSEMOVE and drag["start"]:
                    drag["end"] = (x, y)
                elif event == cv2.EVENT_LBUTTONUP and drag["start"]:
                    (x1, y1), (x2, y2) = drag["start"], (x, y)
                    drag["start"] = None
                    if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                        h, w = shape
                        gate.roi = tuple(max(0.0, min(1.0, v)) for v in (min(x1, x2) / w, min(y1, y2) / h,
                                                                           max(x1, x2) / w, max(y1, y2) / h))
                        gate._background = None  # new region: restart the background

            cv2.setMouseCallback("motion_preview", on_mouse)
        except cv2.error:
            print("sin ventana disponible, usando modo texto")
            window = False

    started = last_print = time.time()
    try:
        while window or time.time() - started < args.seconds:
            ok, image = capture.read()
            if not ok:
                time.sleep(0.05)
                continue
            now = time.time()
            active = gate.update(image, now)
            if window:
                shape[:] = image.shape[:2]
                h, w = shape
                x1, y1, x2, y2 = gate.roi
                color = (0, 0, 255) if active else (0, 200, 0)
                cv2.rectangle(image, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), color, 2)
                if drag["start"]:
                    cv2.rectangle(image, drag["start"], drag["end"], (255, 200, 0), 1)
                label = f"cambio {gate.change * 100:.1f}% (umbral {gate.min_area * 100:.1f}%)  gate {'ACTIVO' if active else 'inactivo'}"
                cv2.putText(image, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.imshow("motion_preview", image)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            elif now - last_print >= 1.0:
                last_print = now
                print(f"cambio {gate.change * 100:5.1f}%  umbral {gate.min_area * 100:.1f}%  gate {'ACTIVO' if active else 'inactivo'}")
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        if window:
            cv2.destroyAllWindows()
    print(f"\nPega esto en config/pilot.yaml, motion.roi.{args.camera}:\n    {args.camera}: {format_roi(gate.roi)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
