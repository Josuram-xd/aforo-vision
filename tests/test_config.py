from pathlib import Path

import pytest
import yaml

from src.config import ConfigError, load_config

PILOT_YAML = Path(__file__).resolve().parent.parent / "config" / "pilot.yaml"


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    for name in ("AFORO_BACKEND_URL", "AFORO_BACKEND_SECRET", "AFORO_CAMERA_OUTSIDE_SOURCE", "AFORO_CAMERA_INSIDE_SOURCE"):
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


def test_shared_secret_comes_only_from_the_environment_and_is_never_printed(monkeypatch):
    assert load_config(PILOT_YAML).backend.shared_secret is None
    monkeypatch.setenv("AFORO_BACKEND_SECRET", "s3cret-value-123456")
    backend = load_config(PILOT_YAML).backend
    assert backend.shared_secret == "s3cret-value-123456"
    assert "s3cret" not in repr(backend)


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
