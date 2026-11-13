import numpy as np

from src.capture.motion_gate import MotionGate

H, W = 240, 320
DOOR_ROI = (0.0, 0.0, 0.4, 1.0)  # left 40% of the image


def base(level=100, seed=0):
    return np.full((H, W, 3), level, np.uint8)


def noisy(level=100, rng=None):
    noise = rng.normal(0, 3, (H, W, 1))
    return np.clip(level + noise, 0, 255).astype(np.uint8).repeat(3, axis=2)


def warm(gate, rng, seconds=5, fps=10):
    t = 0.0
    for _ in range(seconds * fps):
        gate.update(noisy(100, rng), t)
        t += 1 / fps
    return t


def with_block(x, y=100, size=50):
    img = base()
    img[y:y + size, x:x + size] = 220
    return img


def test_static_noise_does_not_activate():
    rng = np.random.default_rng(1)
    gate = MotionGate(DOOR_ROI)
    t = warm(gate, rng)
    assert not any(gate.update(noisy(100, rng), t + i / 10) for i in range(50))


def test_moving_block_inside_roi_activates():
    rng = np.random.default_rng(1)
    gate = MotionGate(DOOR_ROI)
    t = warm(gate, rng)
    assert gate.update(with_block(20), t)


def test_same_motion_outside_roi_does_not_activate():
    rng = np.random.default_rng(1)
    gate = MotionGate(DOOR_ROI)
    t = warm(gate, rng)
    for i in range(30):  # people moving around the room, far from the door
        assert not gate.update(with_block(200 + i), t + i / 10)


def test_hysteresis_holds_then_releases():
    gate = MotionGate(DOOR_ROI, hold_seconds=5.0)
    gate.update(base(), 0.0)
    assert gate.update(with_block(20), 1.0)
    # block disappears; background still remembers it for a while, so check hold on a quiet scene
    gate._background[:] = 100
    assert gate.update(base(), 5.5)
    assert not gate.update(base(), 6.5)


def test_slow_brightness_change_does_not_activate():
    gate = MotionGate(DOOR_ROI)
    gate.update(base(100), 0.0)
    # +0.5 gray level per second for 2 minutes, one frame per second
    assert not any(gate.update(base(100 + int(0.5 * s)), float(s)) for s in range(1, 120))


def test_live_track_in_roi_keeps_processing_without_motion():
    gate = MotionGate(DOOR_ROI, warmup_seconds=0.0)
    img = base()
    gate.should_process(img, 0.0)
    assert not gate.should_process(img, 100.0)
    inside, outside = (10, 10, 50, 100), (250, 10, 300, 100)
    assert gate.should_process(img, 200.0, [inside])
    assert not gate.should_process(img, 300.0, [outside])  # someone sitting in the room does not count


def test_idle_throttles_and_counts():
    gate = MotionGate(DOOR_ROI, idle_fps=4.0, warmup_seconds=0.0)
    img = base()
    for i in range(100):
        gate.should_process(img, 1000 + i * 0.01)  # 100 fps for 1 s
    assert gate.processed == 0 and gate.skipped == 100
    assert "100% omitido" in gate.summary()
