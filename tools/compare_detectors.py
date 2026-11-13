"""Compare the torch and onnx pose detectors on the same frames (needs requirements-export.txt).

    python -m tools.compare_detectors --source foto.jpg | carpeta/ | video.mp4 | 0 | rtsp://...  [--max-frames 100]

Reports people per frame in each backend, mean IoU of matched boxes, mean keypoint distance (px) and FPS.
Verdict "equivalentes" if mean IoU >= 0.9 and the same count in >= 95 % of frames, otherwise "revisar".
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_frames(source: str, max_frames: int):
    path = Path(source)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        for p in files[:max_frames]:
            frame = cv2.imread(str(p))
            if frame is not None:
                yield frame
        return
    if path.suffix.lower() in IMAGE_SUFFIXES:
        frame = cv2.imread(str(path))
        if frame is not None:
            yield frame
        return
    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    try:
        for _ in range(max_frames):
            ok, frame = capture.read()
            if not ok:
                break
            yield frame
    finally:
        capture.release()


def box_iou(a, b) -> float:
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / union if union > 0 else 0.0


def match(torch_dets, onnx_dets):
    """Greedy best-IoU matching; returns [(torch_det, onnx_det, iou)]."""
    pairs, used = [], set()
    for t in torch_dets:
        best = max(((box_iou(t, o), j) for j, o in enumerate(onnx_dets) if j not in used), default=(0.0, -1))
        if best[0] > 0.1:
            used.add(best[1])
            pairs.append((t, onnx_dets[best[1]], best[0]))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--model", default="yolo26n-pose.pt")
    parser.add_argument("--onnx-model", default="models/yolo26n-pose.onnx")
    parser.add_argument("--confidence", type=float, default=0.5)
    args = parser.parse_args()

    from src.detection.yolo_onnx import OnnxPoseDetector
    from src.detection.yolo_pose import YoloPoseDetector

    torch_det = YoloPoseDetector(args.model, args.confidence)
    onnx_det = OnnxPoseDetector(args.onnx_model, args.confidence)

    ious, kp_dists, same_count, frames = [], [], 0, 0
    time_torch = time_onnx = 0.0
    for frame in iter_frames(args.source, args.max_frames):
        t0 = time.perf_counter()
        a = torch_det.detect(frame)
        t1 = time.perf_counter()
        b = onnx_det.detect(frame)
        t2 = time.perf_counter()
        if frames > 0:  # first frame is warm-up, left out of FPS
            time_torch, time_onnx = time_torch + (t1 - t0), time_onnx + (t2 - t1)
        frames += 1
        same_count += len(a) == len(b)
        print(f"frame {frames}: torch={len(a)} onnx={len(b)}")
        for t, o, iou in match(a, b):
            ious.append(iou)
            kp_dists.append(float(np.linalg.norm(t.keypoints[:, :2] - o.keypoints[:, :2], axis=1).mean()))

    if frames == 0:
        print("No se pudo leer ningun frame de la fuente.")
        return 2
    timed = max(frames - 1, 1)
    mean_iou = float(np.mean(ious)) if ious else 0.0
    ratio = same_count / frames
    print(f"frames={frames}  mismo conteo={ratio:.0%}  IoU medio={mean_iou:.3f}  "
          f"dist. media keypoints={np.mean(kp_dists) if kp_dists else float('nan'):.2f} px")
    if frames > 1:
        print(f"FPS torch={timed / time_torch:.1f}  onnx={timed / time_onnx:.1f}")
    if not ious:
        print("Veredicto: revisar (no hubo ninguna persona emparejada; usa una fuente con personas)")
        return 1
    ok = mean_iou >= 0.9 and ratio >= 0.95
    print("Veredicto: " + ("equivalentes" if ok else "revisar"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
