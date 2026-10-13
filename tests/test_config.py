from pathlib import Path

import pytest
import yaml

from src.config import ConfigError, load_config

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
