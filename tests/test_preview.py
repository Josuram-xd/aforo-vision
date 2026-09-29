import numpy as np
import pytest

from src.debug.preview import Annotation, FpsMeter, draw_no_signal, draw_overlay
from src.main import parse_args


def test_fps_meter_is_zero_until_two_ticks():
    meter = FpsMeter()
    assert meter.fps == 0.0
    meter.tick(now=0.0)
    assert meter.fps == 0.0


def test_fps_meter_converges_to_steady_rate():
    meter = FpsMeter()
    for i in range(100):
        meter.tick(now=i * 0.05)  # 20 fps
    assert meter.fps == pytest.approx(20.0)


def test_draw_overlay_does_not_modify_input_and_draws_box():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    canvas = draw_overlay(image, "camera-inside", 15.0, [Annotation(60, 60, 100, 100, "id 1")])
    assert not image.any()
    assert canvas.shape == image.shape
    assert canvas[100, 80].any()  # bottom edge of the box


def test_draw_no_signal_keeps_camera_shape():
    assert draw_no_signal("camera-outside", (720, 1280, 3)).shape == (720, 1280, 3)


def test_debug_flag_defaults_off():
    assert parse_args([]).debug is False
    assert parse_args(["--debug"]).debug is True
