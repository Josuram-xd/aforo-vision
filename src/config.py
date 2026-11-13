"""Load and validate the pilot configuration (config/pilot.yaml)."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import yaml

CAMERA_IDS = ("camera-outside", "camera-inside")

_ENV_BACKEND_URL = "AFORO_BACKEND_URL"
_ENV_SUPABASE_KEY = "AFORO_SUPABASE_KEY"  # publishable key (apikey header); never goes in pilot.yaml
_ENV_DEVICE_KEY = "AFORO_DEVICE_KEY"  # per-classroom key (X-Device-Key header); never goes in pilot.yaml
_ENV_CAMERA_SOURCE = {
    "camera-outside": "AFORO_CAMERA_OUTSIDE_SOURCE",
    "camera-inside": "AFORO_CAMERA_INSIDE_SOURCE",
}


class ConfigError(ValueError):
    """Raised when the config file is missing fields or has invalid values."""


_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def load_dotenv(path: str | Path | None = None) -> None:
    """Load KEY=VALUE lines from the repo-root .env into os.environ without overriding existing variables.

    shortcut: no `export` prefix, no multiline values, no inline comments; use python-dotenv if that is needed.
    """
    path = Path(path) if path else Path(__file__).resolve().parent.parent / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(name.strip(), value)


def is_secure_url(url: str) -> bool:
    """https:// anywhere; plain http:// only to this machine (local Supabase stack).

    The publishable key, the device key and user JWTs travel in headers, so they must never cross the
    network unencrypted.
    """
    parts = urlsplit(url)
    if parts.scheme == "https":
        return bool(parts.hostname)
    return parts.scheme == "http" and parts.hostname in _LOCAL_HOSTS


@dataclass(frozen=True)
class CameraConfig:
    camera_id: str
    source: int | str  # USB index or stream URL
    pinned: bool = False  # True when AFORO_CAMERA_*_SOURCE is set: the web-registered connection must not override it


@dataclass(frozen=True)
class CaptureConfig:
    reconnect_interval_seconds: float


@dataclass(frozen=True)
class DetectionConfig:
    model: str
    confidence_threshold: float
    backend: str = "torch"  # "torch" (Ultralytics) or "onnx" (onnxruntime, no torch needed)
    onnx_model: str = "models/yolo26n-pose.onnx"


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
    # repr=False: the keys must never show up in logs or tracebacks
    supabase_key: str | None = field(default=None, repr=False)
    device_key: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class MotionConfig:
    enabled: bool = False  # off by default: counting correctly matters more than saving CPU
    hold_seconds: float = 8.0
    idle_fps: float = 4.0
    pixel_threshold: int = 25
    min_area: float = 0.01
    warmup_seconds: float = 3.0
    roi: dict[str, tuple[float, float, float, float]] = field(
        default_factory=lambda: {camera_id: (0.0, 0.0, 1.0, 1.0) for camera_id in CAMERA_IDS})


@dataclass(frozen=True)
class PerformanceConfig:
    cpu_threads: int | None = None  # None = leave every library on its default thread count
    low_priority: bool = True  # BELOW_NORMAL process priority on Windows


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
    motion: MotionConfig = field(default_factory=MotionConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)


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
            backend=_detection_backend(_section(raw, "detection")),
            onnx_model=str(_section(raw, "detection").get("onnx_model") or "models/yolo26n-pose.onnx"),
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
        motion=_parse_motion(raw.get("motion")),
        performance=_parse_performance(raw.get("performance")),
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


def _detection_backend(section: dict) -> str:
    backend = section.get("backend", "torch")
    if backend not in ("torch", "onnx"):
        raise ConfigError(f"detection.backend debe ser 'torch' u 'onnx', se recibio {backend!r}")
    return backend


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


def _parse_performance(section) -> PerformanceConfig:
    if section is None:
        return PerformanceConfig()
    if not isinstance(section, dict):
        raise ConfigError("performance debe ser un bloque con cpu_threads y low_priority")
    threads = section.get("cpu_threads")
    if threads is not None and (isinstance(threads, bool) or not isinstance(threads, int) or threads < 1):
        raise ConfigError(f"performance.cpu_threads debe ser un entero >= 1 (o null para no limitar), se recibio {threads!r}")
    low_priority = section.get("low_priority", True)
    if not isinstance(low_priority, bool):
        raise ConfigError(f"performance.low_priority debe ser true o false, se recibio {low_priority!r}")
    return PerformanceConfig(cpu_threads=threads, low_priority=low_priority)


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
        cameras[camera_id] = CameraConfig(camera_id=camera_id, source=source, pinned=bool(env_source))
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


def _env_value(name: str) -> str | None:
    """Read an env var; reject example text (non-ASCII, whitespace, leading "<") without echoing the value."""
    value = os.environ.get(name)
    if value and (not value.isascii() or any(c.isspace() for c in value) or value.startswith("<")):
        raise ConfigError(f"{name} parece un texto de ejemplo (tiene tildes, espacios o empieza con '<'); pon el valor real en .env")
    return value or None


def _parse_backend(section: dict) -> BackendConfig:
    raw = {"backend": section}
    url = _env_value(_ENV_BACKEND_URL) or _field(raw, "backend", "url")
    if not isinstance(url, str) or not is_secure_url(url):
        raise ConfigError(f"backend.url must be an https:// URL (http:// only for localhost), got {url!r}")
    return BackendConfig(
        url=url.rstrip("/"),
        timeout_seconds=_positive_float(raw, "backend", "timeout_seconds"),
        retry_queue_path=Path(_field(raw, "backend", "retry_queue_path")),
        supabase_key=_env_value(_ENV_SUPABASE_KEY),
        device_key=_env_value(_ENV_DEVICE_KEY),
    )


def _parse_motion(section) -> MotionConfig:
    if section is None:
        return MotionConfig()
    if not isinstance(section, dict):
        raise ConfigError("motion debe ser un bloque con claves (enabled, hold_seconds, ...)")
    defaults = MotionConfig()

    def number(key, low, high=None, integer=False):
        value = section.get(key, getattr(defaults, key))
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"motion.{key} debe ser un numero, se recibio {value!r}")
        if value <= low or (high is not None and value > high):
            rango = f"mayor que {low}" + (f" y como maximo {high}" if high is not None else "")
            raise ConfigError(f"motion.{key} debe ser {rango}, se recibio {value}")
        return int(value) if integer else float(value)

    enabled = section.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ConfigError(f"motion.enabled debe ser true o false, se recibio {enabled!r}")
    roi_section = section.get("roi") or {}
    if not isinstance(roi_section, dict) or not set(roi_section) <= set(CAMERA_IDS):
        raise ConfigError(f"motion.roi solo puede tener las camaras {CAMERA_IDS}")
    roi = {}
    for camera_id in CAMERA_IDS:
        value = roi_section.get(camera_id, defaults.roi[camera_id])
        ok = (isinstance(value, (list, tuple)) and len(value) == 4
              and all(isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 1 for v in value)
              and value[0] < value[2] and value[1] < value[3])
        if not ok:
            raise ConfigError(f"motion.roi.{camera_id} debe ser [x1, y1, x2, y2] entre 0 y 1 con x1<x2 e y1<y2, se recibio {value!r}")
        roi[camera_id] = tuple(float(v) for v in value)
    return MotionConfig(
        enabled=enabled,
        hold_seconds=number("hold_seconds", 0),
        idle_fps=number("idle_fps", 0),
        pixel_threshold=number("pixel_threshold", 0, 255, integer=True),
        min_area=number("min_area", 0, 1),
        warmup_seconds=number("warmup_seconds", -1),
        roi=roi,
    )
