"""Load and validate the pilot configuration (config/pilot.yaml)."""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CAMERA_IDS = ("camera-outside", "camera-inside")

_ENV_BACKEND_URL = "AFORO_BACKEND_URL"
_ENV_BACKEND_SECRET = "AFORO_BACKEND_SECRET"  # shared secret for POST /events; never goes in pilot.yaml
_ENV_CAMERA_SOURCE = {
    "camera-outside": "AFORO_CAMERA_OUTSIDE_SOURCE",
    "camera-inside": "AFORO_CAMERA_INSIDE_SOURCE",
}


class ConfigError(ValueError):
    """Raised when the config file is missing fields or has invalid values."""


@dataclass(frozen=True)
class CameraConfig:
    camera_id: str
    source: int | str  # USB index or stream URL


@dataclass(frozen=True)
class CaptureConfig:
    reconnect_interval_seconds: float


@dataclass(frozen=True)
class DetectionConfig:
    model: str
    confidence_threshold: float


@dataclass(frozen=True)
class TrackingConfig:
    iou_threshold: float
    max_age_frames: int
    min_hits: int
    trajectory_length: int


@dataclass(frozen=True)
class IdentityConfig:
    face_similarity_threshold: float
    body_similarity_threshold: float


@dataclass(frozen=True)
class MatchingConfig:
    time_window_seconds: float


@dataclass(frozen=True)
class DedupConfig:
    ttl_seconds: float


@dataclass(frozen=True)
class BackendConfig:
    url: str
    timeout_seconds: float
    retry_queue_path: Path
    shared_secret: str | None = field(default=None, repr=False)  # repr=False: never print it in logs


@dataclass(frozen=True)
class PilotConfig:
    cameras: dict[str, CameraConfig]
    capture: CaptureConfig
    detection: DetectionConfig
    tracking: TrackingConfig
    identity: IdentityConfig
    matching: MatchingConfig
    dedup: DedupConfig
    backend: BackendConfig


def load_config(path: str | Path) -> PilotConfig:
    """Read the YAML file at `path`, apply env overrides and return a validated PilotConfig."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"config file is empty or not a mapping: {path}")

    return PilotConfig(
        cameras=_parse_cameras(_section(raw, "cameras")),
        capture=CaptureConfig(
            reconnect_interval_seconds=_positive_float(raw, "capture", "reconnect_interval_seconds"),
        ),
        detection=DetectionConfig(
            model=str(_field(raw, "detection", "model")),
            confidence_threshold=_unit_float(raw, "detection", "confidence_threshold"),
        ),
        tracking=TrackingConfig(
            iou_threshold=_unit_float(raw, "tracking", "iou_threshold"),
            max_age_frames=_positive_int(raw, "tracking", "max_age_frames"),
            min_hits=_positive_int(raw, "tracking", "min_hits"),
            trajectory_length=_positive_int(raw, "tracking", "trajectory_length"),
        ),
        identity=IdentityConfig(
            face_similarity_threshold=_unit_float(raw, "identity", "face_similarity_threshold"),
            body_similarity_threshold=_unit_float(raw, "identity", "body_similarity_threshold"),
        ),
        matching=MatchingConfig(
            time_window_seconds=_positive_float(raw, "matching", "time_window_seconds"),
        ),
        dedup=DedupConfig(ttl_seconds=_positive_float(raw, "dedup", "ttl_seconds")),
        backend=_parse_backend(_section(raw, "backend")),
    )


def _section(raw: dict, name: str) -> dict:
    section = raw.get(name)
    if not isinstance(section, dict):
        raise ConfigError(f"missing or invalid section: {name}")
    return section


def _field(raw: dict, section: str, key: str):
    values = _section(raw, section)
    if key not in values or values[key] is None:
        raise ConfigError(f"missing field: {section}.{key}")
    return values[key]


def _number(raw: dict, section: str, key: str) -> float:
    value = _field(raw, section, key)
    # bool is a subclass of int; reject it explicitly
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{section}.{key} must be a number, got {value!r}")
    return value


def _unit_float(raw: dict, section: str, key: str) -> float:
    value = float(_number(raw, section, key))
    if not 0.0 <= value <= 1.0:
        raise ConfigError(f"{section}.{key} must be between 0 and 1, got {value}")
    return value


def _positive_float(raw: dict, section: str, key: str) -> float:
    value = float(_number(raw, section, key))
    if value <= 0:
        raise ConfigError(f"{section}.{key} must be > 0, got {value}")
    return value


def _positive_int(raw: dict, section: str, key: str) -> int:
    value = _number(raw, section, key)
    if not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{section}.{key} must be a positive integer, got {value!r}")
    return value


def _parse_cameras(section: dict) -> dict[str, CameraConfig]:
    if set(section) != set(CAMERA_IDS):
        raise ConfigError(f"cameras must define exactly {CAMERA_IDS}, got {tuple(section)}")

    cameras = {}
    for camera_id in CAMERA_IDS:
        entry = section[camera_id]
        if not isinstance(entry, dict) or "source" not in entry:
            raise ConfigError(f"missing field: cameras.{camera_id}.source")
        env_source = os.environ.get(_ENV_CAMERA_SOURCE[camera_id])
        source = _parse_source(env_source if env_source else entry["source"], camera_id)
        cameras[camera_id] = CameraConfig(camera_id=camera_id, source=source)
    return cameras


def _parse_source(value, camera_id: str) -> int | str:
    if isinstance(value, bool):
        raise ConfigError(f"cameras.{camera_id}.source must be a USB index or URL")
    if isinstance(value, int):
        if value < 0:
            raise ConfigError(f"cameras.{camera_id}.source must be >= 0, got {value}")
        return value
    if isinstance(value, str) and value.strip():
        value = value.strip()
        # Env vars always arrive as strings; "0" means USB index 0
        return int(value) if value.isdigit() else value
    raise ConfigError(f"cameras.{camera_id}.source must be a USB index or URL, got {value!r}")


def _parse_backend(section: dict) -> BackendConfig:
    raw = {"backend": section}
    url = os.environ.get(_ENV_BACKEND_URL) or _field(raw, "backend", "url")
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise ConfigError(f"backend.url must start with http:// or https://, got {url!r}")
    return BackendConfig(
        url=url.rstrip("/"),
        timeout_seconds=_positive_float(raw, "backend", "timeout_seconds"),
        retry_queue_path=Path(_field(raw, "backend", "retry_queue_path")),
        shared_secret=os.environ.get(_ENV_BACKEND_SECRET) or None,
    )
