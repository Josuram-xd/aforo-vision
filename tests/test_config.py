import os
from pathlib import Path

import pytest
import yaml

from src.config import ConfigError, load_config, load_dotenv

PILOT_YAML = Path(__file__).resolve().parent.parent / "config" / "pilot.yaml"


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    for name in ("AFORO_BACKEND_URL", "AFORO_SUPABASE_KEY", "AFORO_DEVICE_KEY", "AFORO_BACKEND_SECRET", "AFORO_CAMERA_OUTSIDE_SOURCE", "AFORO_CAMERA_INSIDE_SOURCE"):
        monkeypatch.delenv(name, raising=False)


def write_config(tmp_path: Path, **overrides) -> Path:
    raw = yaml.safe_load(PILOT_YAML.read_text(encoding="utf-8"))
    for dotted, value in overrides.items():
        *parents, key = dotted.split("__")
        target = raw
        for parent in parents:
            target = target[parent]
        target[key] = value
    path = tmp_path / "pilot.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_loads_repo_pilot_yaml():
    config = load_config(PILOT_YAML)
    assert set(config.cameras) == {"camera-outside", "camera-inside"}
    assert config.cameras["camera-inside"].source == 0
    assert config.matching.time_window_seconds > 0
    assert config.capture.reconnect_interval_seconds > 0


def test_env_overrides_backend_url_and_camera_source(monkeypatch):
    monkeypatch.setenv("AFORO_BACKEND_URL", "https://example.test/prod/")
    monkeypatch.setenv("AFORO_CAMERA_INSIDE_SOURCE", "1")
    config = load_config(PILOT_YAML)
    assert config.backend.url == "https://example.test/prod"
    assert config.cameras["camera-inside"].source == 1


def test_supabase_and_device_keys_come_only_from_the_environment_and_are_never_printed(monkeypatch):
    backend = load_config(PILOT_YAML).backend
    assert backend.supabase_key is None and backend.device_key is None
    monkeypatch.setenv("AFORO_SUPABASE_KEY", "sb_publishable_test-value-123")
    monkeypatch.setenv("AFORO_DEVICE_KEY", "device-key-test-value-456")
    backend = load_config(PILOT_YAML).backend
    assert backend.supabase_key == "sb_publishable_test-value-123"
    assert backend.device_key == "device-key-test-value-456"
    assert "test-value" not in repr(backend) and "test-value" not in str(load_config(PILOT_YAML))


def test_empty_keys_count_as_missing(monkeypatch):
    monkeypatch.setenv("AFORO_SUPABASE_KEY", "")
    monkeypatch.setenv("AFORO_DEVICE_KEY", "")
    backend = load_config(PILOT_YAML).backend
    assert backend.supabase_key is None and backend.device_key is None


def test_the_old_shared_secret_is_no_longer_read(monkeypatch):
    monkeypatch.setenv("AFORO_BACKEND_SECRET", "old-secret")
    assert not hasattr(load_config(PILOT_YAML).backend, "shared_secret")


def test_keys_in_pilot_yaml_are_ignored(tmp_path, monkeypatch):
    path = write_config(tmp_path, backend__device_key="from-yaml", backend__supabase_key="from-yaml")
    backend = load_config(path).backend
    assert backend.device_key is None and backend.supabase_key is None


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.yaml")


def test_rejects_missing_camera(tmp_path):
    path = write_config(tmp_path, cameras={"camera-inside": {"source": 0}})
    with pytest.raises(ConfigError):
        load_config(path)


def test_rejects_threshold_out_of_range(tmp_path):
    path = write_config(tmp_path, identity__face_similarity_threshold=1.5)
    with pytest.raises(ConfigError):
        load_config(path)


def test_rejects_non_http_backend_url(tmp_path):
    path = write_config(tmp_path, backend__url="ftp://nope")
    with pytest.raises(ConfigError):
        load_config(path)


def test_rejects_plain_http_backend_url_outside_localhost(tmp_path):
    path = write_config(tmp_path, backend__url="http://proj.supabase.co")
    with pytest.raises(ConfigError):
        load_config(path)


def test_allows_plain_http_backend_url_on_localhost(tmp_path):
    path = write_config(tmp_path, backend__url="http://127.0.0.1:54321")
    assert load_config(path).backend.url == "http://127.0.0.1:54321"


def test_load_dotenv_reads_file_without_overriding_the_environment(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text('# comment\nAFORO_DEVICE_KEY="from-file"\nAFORO_SUPABASE_KEY=sb-file\n\nbad line\n', encoding="utf-8")
    monkeypatch.setenv("AFORO_SUPABASE_KEY", "from-shell")
    load_dotenv(env_file)
    assert os.environ["AFORO_DEVICE_KEY"] == "from-file"
    assert os.environ["AFORO_SUPABASE_KEY"] == "from-shell"
    monkeypatch.delenv("AFORO_DEVICE_KEY")


def test_load_dotenv_ignores_a_missing_file(tmp_path):
    load_dotenv(tmp_path / "nope.env")


@pytest.mark.parametrize("name", ["AFORO_SUPABASE_KEY", "AFORO_DEVICE_KEY", "AFORO_BACKEND_URL"])
@pytest.mark.parametrize("value", ["<llave del salón>", "con espacio", "clavé"])
def test_placeholder_or_non_ascii_values_are_rejected_without_echoing_them(tmp_path, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigError) as exc:
        load_config(write_config(tmp_path))
    assert name in str(exc.value) and value not in str(exc.value)


def test_main_returns_2_on_placeholder_key(monkeypatch):
    from src import main as main_module

    monkeypatch.setattr(main_module, "load_dotenv", lambda: None)
    monkeypatch.setenv("AFORO_DEVICE_KEY", "<llave del salón>")
    assert main_module.main(["--dry-run"]) == 2


def test_detection_backend_defaults_to_torch():
    config = load_config(PILOT_YAML)
    assert config.detection.backend == "torch"
    assert config.detection.onnx_model == "models/yolo26n-pose.onnx"


def test_accepts_onnx_backend(tmp_path):
    config = load_config(write_config(tmp_path, detection__backend="onnx", detection__onnx_model="m/x.onnx"))
    assert (config.detection.backend, config.detection.onnx_model) == ("onnx", "m/x.onnx")


def test_rejects_unknown_detection_backend(tmp_path):
    with pytest.raises(ConfigError, match="detection.backend"):
        load_config(write_config(tmp_path, detection__backend="tensorrt"))


def test_motion_defaults_to_disabled_with_full_image_roi():
    motion = load_config(PILOT_YAML).motion
    assert motion.enabled is False
    assert motion.roi["camera-inside"] == (0.0, 0.0, 1.0, 1.0)


def test_motion_section_is_optional(tmp_path):
    raw = yaml.safe_load(PILOT_YAML.read_text(encoding="utf-8"))
    del raw["motion"]
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert load_config(path).motion.enabled is False


def test_motion_custom_values_load(tmp_path):
    path = write_config(tmp_path, motion__enabled=True, motion__roi={"camera-inside": [0.2, 0.1, 0.8, 1.0]})
    motion = load_config(path).motion
    assert motion.enabled and motion.roi["camera-inside"] == (0.2, 0.1, 0.8, 1.0)
    assert motion.roi["camera-outside"] == (0.0, 0.0, 1.0, 1.0)


@pytest.mark.parametrize("override", [
    {"motion__enabled": "si"},
    {"motion__hold_seconds": 0},
    {"motion__idle_fps": -1},
    {"motion__pixel_threshold": 300},
    {"motion__min_area": 1.5},
    {"motion__roi": {"camera-inside": [0.8, 0.1, 0.2, 1.0]}},
    {"motion__roi": {"camera-inside": [0, 0, 1]}},
    {"motion__roi": {"camera-x": [0, 0, 1, 1]}},
])
def test_motion_invalid_values_raise_spanish_error(tmp_path, override):
    with pytest.raises(ConfigError, match="motion"):
        load_config(write_config(tmp_path, **override))


def test_performance_defaults_when_section_is_missing(tmp_path):
    raw = yaml.safe_load(PILOT_YAML.read_text(encoding="utf-8"))
    del raw["performance"]
    path = tmp_path / "pilot.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    performance = load_config(path).performance
    assert performance.cpu_threads is None and performance.low_priority is True


def test_repo_pilot_yaml_leaves_threads_untouched():
    assert load_config(PILOT_YAML).performance.cpu_threads is None


def test_performance_accepts_thread_cap_and_priority_off(tmp_path):
    path = write_config(tmp_path, performance__cpu_threads=4, performance__low_priority=False)
    performance = load_config(path).performance
    assert performance.cpu_threads == 4 and performance.low_priority is False


@pytest.mark.parametrize("value", [0, -2, 1.5, "4", True])
def test_performance_rejects_bad_cpu_threads(tmp_path, value):
    path = write_config(tmp_path, performance__cpu_threads=value)
    with pytest.raises(ConfigError, match="performance.cpu_threads"):
        load_config(path)


def test_performance_rejects_non_boolean_priority(tmp_path):
    path = write_config(tmp_path, performance__low_priority="si")
    with pytest.raises(ConfigError, match="performance.low_priority"):
        load_config(path)
