"""Time each stage of the hot path (ms/frame) on synthetic frames. No camera needed.

    python -m tools.bench_pipeline [--size 1920x1080] [--frames 30] [--threads N] [--backend onnx|torch|both]

Stages: video decode (a temp MJPEG/H.264-less file read through cv2, an approximation of RTSP decoding),
letterbox preprocess, detector inference, tracker update, and the embedding models (SCRFD + OSNet; ArcFace on
a fixed 112x112 crop because synthetic frames contain no real face). Models that are not installed are skipped.
"""

import argparse
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from src.detection.yolo_onnx import letterbox
from src.detection.yolo_pose import Detection
from src.tracking.sort_tracker import SortTracker


def synthetic_frame(width: int, height: int, seed: int = 0) -> np.ndarray:
    """Noisy background with a few person-sized blobs so the models have something to look at."""
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    frame = cv2.GaussianBlur(frame, (0, 0), 6)
    for x in (width // 5, width // 2, 4 * width // 5):
        cv2.ellipse(frame, (x, height // 2), (height // 12, height // 4), 0, 0, 360, (60, 90, 200), -1)
        cv2.circle(frame, (x, height // 2 - height // 4), height // 20, (170, 190, 230), -1)
    return frame


def timed(fn, n: int) -> float:
    """Best-case milliseconds per call over n calls (after one warm-up); min resists a busy machine."""
    fn()
    samples = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return min(samples)


def bench_decode(width: int, height: int, n: int) -> float:
    path = Path(tempfile.mkdtemp()) / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 15, (width, height))
    for i in range(n + 2):
        writer.write(synthetic_frame(width, height, i % 3))
    writer.release()
    capture = cv2.VideoCapture(str(path))
    result = timed(lambda: capture.read(), n)
    capture.release()
    return result


def bench_tracker(n: int) -> float:
    tracker = SortTracker()
    kp = np.ones((17, 3), dtype=np.float32) * 100
    state = {"t": 0}

    def step():
        state["t"] += 1
        dx = state["t"] * 3
        tracker.update([Detection(100 + dx + 300 * i, 200, 220 + dx + 300 * i, 600, 0.9, kp + dx) for i in range(4)])

    return timed(step, n)


def bench_embeddings(frame: np.ndarray, n: int) -> dict[str, float]:
    out = {}
    h, w = frame.shape[:2]
    box = (w // 2 - h // 12, h // 4, w // 2 + h // 12, 3 * h // 4)
    crop = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
    try:
        from src.identity.body_reid import BodyEmbedder
        body = BodyEmbedder()
        out["osnet body embed"] = timed(lambda: body.embed(crop), n)
    except Exception as exc:  # noqa: BLE001 - benchmark tool: report and go on
        out["osnet body embed"] = f"skipped ({type(exc).__name__})"
    try:
        from src.identity.face_detector import FaceDetector
        faces = FaceDetector()
        out["scrfd face detect"] = timed(lambda: faces.detect(crop), n)
    except Exception as exc:  # noqa: BLE001
        out["scrfd face detect"] = f"skipped ({type(exc).__name__})"
    try:
        from src.identity.arcface import ArcFaceEmbedder
        arc = ArcFaceEmbedder()
        eyes = np.random.default_rng(1).integers(0, 255, (40, 112, 3), dtype=np.uint8)
        out["arcface embed (112x40)"] = timed(lambda: arc.embed(eyes), n)
    except Exception as exc:  # noqa: BLE001
        out["arcface embed (112x40)"] = f"skipped ({type(exc).__name__})"
    return out


def bench_detector(backend: str, frame: np.ndarray, n: int) -> dict[str, float]:
    out = {}
    if backend == "onnx":
        from src.detection.yolo_onnx import OnnxPoseDetector
        det = OnnxPoseDetector()
        out["onnx: letterbox+blob"] = timed(lambda: letterbox(frame, det._size), n)
        out["onnx: detect total"] = timed(lambda: det.detect(frame), n)
    else:
        from src.detection.yolo_pose import YoloPoseDetector
        det = YoloPoseDetector("yolo26n-pose.pt", 0.5)
        out["torch: detect total"] = timed(lambda: det.detect(frame), n)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", default="1920x1080")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--threads", type=int, default=0, help="0 = library default")
    parser.add_argument("--backend", default="both", choices=["onnx", "torch", "both"])
    args = parser.parse_args()
    width, height = (int(v) for v in args.size.split("x"))
    if args.threads:
        cv2.setNumThreads(args.threads)
        import torch
        torch.set_num_threads(args.threads)
    frame = synthetic_frame(width, height)
    rows = {f"decode {args.size} MJPEG file": bench_decode(width, height, args.frames),
            "tracker update (4 people)": bench_tracker(200)}
    for backend in ("onnx", "torch") if args.backend == "both" else (args.backend,):
        rows.update(bench_detector(backend, frame, args.frames))
    rows.update(bench_embeddings(frame, args.frames))
    for name, value in rows.items():
        print(f"{name:32s} {value:8.2f} ms" if isinstance(value, float) else f"{name:32s} {value}")


if __name__ == "__main__":
    main()
