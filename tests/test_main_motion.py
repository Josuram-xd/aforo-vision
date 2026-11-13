import numpy as np

from src.capture.motion_gate import MotionGate
from src.main import process_if_needed


class FakeTracker:
    def __init__(self):
        self.boxes = []

    def live_boxes(self):
        return self.boxes


class FakePipeline:
    def __init__(self):
        self.calls = 0
        self.trackers = {"camera-inside": FakeTracker()}

    def process_frame(self, camera_id, image, timestamp):
        self.calls += 1
        return ["track"]


IMG = np.full((240, 320, 3), 100, np.uint8)


def test_without_gate_every_frame_is_processed():
    pipeline = FakePipeline()
    for i in range(20):
        assert process_if_needed(pipeline, None, "camera-inside", IMG, i, i) == ["track"]
    assert pipeline.calls == 20


def test_with_gate_quiet_scene_is_skipped_but_live_track_keeps_processing():
    pipeline = FakePipeline()
    gate = MotionGate(warmup_seconds=0.0, idle_fps=4.0)
    for i in range(20):
        process_if_needed(pipeline, gate, "camera-inside", IMG, i, 100 + i)
    assert pipeline.calls == 0 and gate.skipped == 20
    pipeline.trackers["camera-inside"].boxes = [(100, 50, 150, 200)]
    assert process_if_needed(pipeline, gate, "camera-inside", IMG, 0, 200) == ["track"]
    assert pipeline.calls == 1
