import sys

import pytest

from src import performance


@pytest.fixture(autouse=True)
def _reset_threads():
    yield
    performance._cpu_threads = None


def test_ort_options_disable_spinning_by_default():
    ort = pytest.importorskip("onnxruntime")
    options = performance.ort_session_options()
    assert isinstance(options, ort.SessionOptions)
    assert options.intra_op_num_threads == 0  # library default: not touched


def test_ort_options_apply_thread_cap():
    pytest.importorskip("onnxruntime")
    performance._cpu_threads = 3
    assert performance.ort_session_options().intra_op_num_threads == 3


def test_apply_sets_threads_and_priority(monkeypatch):
    cv2 = pytest.importorskip("cv2")
    seen = {}
    monkeypatch.setattr(cv2, "setNumThreads", lambda n: seen.setdefault("cv2", n))
    monkeypatch.setattr(performance, "lower_process_priority", lambda: seen.setdefault("low", True))
    performance.apply(4, True, backend="onnx")  # onnx backend: torch must not be imported or touched
    assert seen == {"cv2": 4, "low": True}
    assert performance._cpu_threads == 4


def test_apply_defaults_touch_nothing(monkeypatch):
    cv2 = pytest.importorskip("cv2")
    monkeypatch.setattr(cv2, "setNumThreads", lambda n: pytest.fail("cv2 threads must not change"))
    monkeypatch.setattr(performance, "lower_process_priority", lambda: pytest.fail("priority must not change"))
    performance.apply(None, False)


def test_lower_priority_is_noop_outside_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert performance.lower_process_priority() is False


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_lower_priority_on_windows_does_not_raise():
    assert isinstance(performance.lower_process_priority(), bool)


def test_main_applies_performance_once_before_building_models(monkeypatch, tmp_path):
    from src import main as main_module

    class Stop(Exception):
        pass

    calls = []

    def fake_apply(*args):
        calls.append(args)
        raise Stop  # stop here: nothing after it (models, cameras) may run in this test

    monkeypatch.setattr(main_module, "load_dotenv", lambda: None)
    for name in ("AFORO_BACKEND_URL", "AFORO_SUPABASE_KEY", "AFORO_DEVICE_KEY"):
        monkeypatch.delenv(name, raising=False)  # a real .env loaded by another test must not leak in
    monkeypatch.setattr(performance, "apply", fake_apply)
    with pytest.raises(Stop):
        main_module.main(["--dry-run"])
    assert calls == [(None, True, "torch")]  # defaults of config/pilot.yaml
