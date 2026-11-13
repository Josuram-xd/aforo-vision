"""One-off: export yolo26n-pose.pt to ONNX so the pipeline can run without torch (detection.backend: onnx).

ESTE PASO SI necesita ultralytics/torch (pip install -r requirements-export.txt). Se hace una sola vez;
despues el equipo de produccion solo necesita requirements-slim.txt.

    python -m tools.export_yolo_onnx [--weights yolo26n-pose.pt] [--imgsz 640] [--output models/yolo26n-pose.onnx]
"""

import argparse
import shutil
from pathlib import Path

DEFAULT_OUTPUT = Path("models/yolo26n-pose.onnx")
OPSET = 17  # stable and supported by onnxruntime 1.30


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="yolo26n-pose.pt")
    parser.add_argument("--imgsz", type=int, default=640, help="fixed square input size (default: 640)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    from ultralytics import YOLO  # needs torch: only this export step

    exported = YOLO(args.weights).export(format="onnx", imgsz=args.imgsz, opset=OPSET, simplify=True, dynamic=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(exported, args.output)
    size_mb = args.output.stat().st_size / 1e6
    print(f"ONNX guardado en {args.output} ({size_mb:.1f} MB)")
    if size_mb > 15:
        print("AVISO: pesa mas de 15 MB; no lo versiones, regeneralo con este script en cada equipo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
