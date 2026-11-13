"""Installation diagnostic: python -m tools.doctor. Prints ✓/✗ lines with the action to fix each failure.

Never prints key values or camera passwords. Exit code 0 only when every check passes.
"""

import importlib.util
import os
import shutil
import socket
import subprocess
import sys
import threading
from pathlib import Path

import httpx
import yaml

from src.capture.cloud_sources import RTSP_PORT, RTSP_PATH, CloudCameraSources
from src.config import ConfigError, _env_value, is_secure_url, load_dotenv

ROOT = Path(__file__).resolve().parent.parent
MIN_FREE_GB = 5
MB = 1024 * 1024

# (import name, pip name); torch-only ones are skipped when detection.backend is "onnx"
_COMMON_DEPS = [("cv2", "opencv-python"), ("numpy", "numpy"), ("insightface", "insightface"),
                ("onnxruntime", "onnxruntime"), ("httpx", "httpx"), ("yaml", "pyyaml"),
                ("jwt", "PyJWT"), ("cryptography", "cryptography")]
_TORCH_DEPS = [("ultralytics", "ultralytics"), ("torch", "torch"), ("torchvision", "torchvision")]

Result = tuple[bool, str, str | None]  # ok, message, action when it failed


def ok(msg: str) -> Result:
    return True, msg, None


def bad(msg: str, action: str) -> Result:
    return False, msg, action


def detection_settings(root: Path = ROOT) -> tuple[str, str]:
    """(backend, model file) from config/pilot.yaml; backend defaults to torch until the key exists."""
    try:
        det = (yaml.safe_load((root / "config" / "pilot.yaml").read_text(encoding="utf-8")) or {}).get("detection", {})
    except (OSError, yaml.YAMLError):
        det = {}
    return str(det.get("backend", "torch")), str(det.get("model", "yolo26n-pose.pt"))


def check_python(version=None, machine: str | None = None) -> list[Result]:
    v = tuple(version or sys.version_info[:3])
    arch = machine or platform_machine()
    out = []
    if v[:2] != (3, 14) or v < (3, 14, 2):
        out.append(bad(f"Python {'.'.join(map(str, v))}: se necesita 3.14.2 o superior (3.14.1 no sirve)",
                       "Instala Python 3.14 (winget install -e --id Python.Python.3.14) y vuelve a correr scripts\\windows\\install.ps1"))
    else:
        out.append(ok(f"Python {'.'.join(map(str, v))}"))
    if arch.lower() in ("amd64", "x86_64"):
        out.append(ok(f"Arquitectura {arch}"))
    else:
        out.append(bad(f"Arquitectura {arch}: se probó solo en x64", "Usa Windows 10/11 de 64 bits (x64); en ARM pueden faltar paquetes"))
    return out


def platform_machine() -> str:
    import platform
    return platform.machine()


def check_dependencies(backend: str, find_spec=importlib.util.find_spec) -> list[Result]:
    deps = _COMMON_DEPS + ([] if backend == "onnx" else _TORCH_DEPS)
    missing = [pip for mod, pip in deps if find_spec(mod) is None]
    if not missing:
        return [ok(f"Dependencias instaladas (backend {backend})")]
    return [bad(f"Faltan paquetes: {', '.join(missing)}",
                "Corre .\\scripts\\windows\\install.ps1 otra vez (o, con el venv activo, pip install -r requirements.txt)")]


def check_models(backend: str, yolo_model: str, root: Path = ROOT) -> list[Result]:
    models = [("YOLO (" + backend + ")", root / yolo_model, "descarga yolo26n-pose.pt y ponlo en la raíz del repo"
               if backend != "onnx" else "genera el .onnx con la herramienta de exportación (ver README) y ponlo en la raíz"),
              ("ArcFace (insightface)", root / "data/insightface/models/buffalo_l/w600k_r50.onnx",
               "python -m src.identity.arcface --download"),
              ("Detector de rostro", root / "data/insightface/models/buffalo_l/det_10g.onnx",
               "python -m src.identity.arcface --download"),
              ("OSNet", root / "data/osnet/osnet_ain_x0_25.onnx", "python -m tools.export_osnet_onnx")]
    out = []
    for name, path, action in models:
        if path.is_file():
            out.append(ok(f"Modelo {name}: {path.relative_to(root)} ({path.stat().st_size / MB:.1f} MB)"))
        else:
            out.append(bad(f"Falta el modelo {name}: {path.relative_to(root)}", action))
    return out


def check_env(root: Path = ROOT) -> list[Result]:
    env_file = root / ".env"
    if not env_file.is_file():
        return [bad("No existe el archivo .env", "Corre .\\scripts\\windows\\install.ps1 (paso 4) o créalo con AFORO_BACKEND_URL, AFORO_SUPABASE_KEY y AFORO_DEVICE_KEY")]
    load_dotenv(env_file)
    out = []
    for name in ("AFORO_BACKEND_URL", "AFORO_SUPABASE_KEY", "AFORO_DEVICE_KEY"):
        try:
            value = _env_value(name)
        except ConfigError as exc:
            out.append(bad(str(exc), f"Corrige {name} en .env"))
            continue
        if not value:
            out.append(bad(f"Falta {name} en .env", f"Agrega {name}=... en .env (pídele la llave del salón a un admin)"))
        elif name == "AFORO_BACKEND_URL" and not is_secure_url(value):
            out.append(bad("AFORO_BACKEND_URL debe empezar con https://", "Usa la URL del proyecto de Supabase (https://<ref>.supabase.co)"))
        else:
            out.append(ok(f"{name} configurada"))
    return out


def fetch_cameras(env=os.environ, client: httpx.Client | None = None) -> tuple[Result, dict]:
    """Call rpc/camera_sources; returns the check result and the cameras ({} when it failed)."""
    try:
        sources = CloudCameraSources(env["AFORO_BACKEND_URL"], env["AFORO_SUPABASE_KEY"], env["AFORO_DEVICE_KEY"],
                                     timeout=8.0, client=client)
        cameras = sources.cameras()
    except KeyError:
        return bad("No se puede probar Supabase sin las 3 variables del .env", "Arregla primero las variables de arriba"), {}
    except (httpx.ConnectError, httpx.TimeoutException):
        return bad("Sin internet o Supabase no responde", "Revisa que el equipo tenga internet y que AFORO_BACKEND_URL esté bien escrita"), {}
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        if code in (401, 403):
            return bad(f"Supabase rechazó la llave ({code})", "Pide a un admin una llave del salón nueva y actualiza AFORO_DEVICE_KEY / AFORO_SUPABASE_KEY en .env"), {}
        return bad(f"Supabase respondió {code}", "Revisa AFORO_BACKEND_URL; si persiste, avisa a un admin"), {}
    except (httpx.HTTPError, ValueError) as exc:
        return bad(f"Respuesta inesperada de Supabase ({type(exc).__name__})", "Revisa AFORO_BACKEND_URL"), {}
    return ok(f"Supabase alcanzable, llaves aceptadas ({len(cameras)} cámara(s) registrada(s) en la web)"), cameras


def tcp_open(host: str, port: int = RTSP_PORT, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def rtsp_opens(url: str, timeout: float = 5.0) -> bool:
    """Open the stream and read one frame with OpenCV; abandoned after `timeout` seconds."""
    import cv2

    result = []

    def worker():
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        result.append(cap.isOpened() and cap.read()[0])
        cap.release()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout)
    return bool(result and result[0])


def check_cameras(cameras: dict, tcp=tcp_open, opener=rtsp_opens) -> list[Result]:
    from src.capture.cloud_sources import build_rtsp_url

    out = []
    for code, cam in cameras.items():
        label = f"Cámara {code} ({cam['host']}, usuario {cam['username']})"
        if not tcp(cam["host"]):
            out.append(bad(f"{label}: no alcanzable", "¿El equipo está en el mismo WiFi que la cámara? Algunos WiFi (instituciones) aíslan los equipos; prueba con el hotspot del celular. Revisa también la IP en la web"))
        elif opener(build_rtsp_url(cam["host"], cam["username"], cam["password"])):
            out.append(ok(f"{label}: conecta"))
        else:
            out.append(bad(f"{label}: puerto abierto pero rechaza usuario/clave", "Revisa la «Cuenta de cámara» de la app Tapo y vuelve a escribirla en la web"))
    if not cameras:
        out.append(bad("No hay cámaras registradas en la web para este salón", "Un admin debe agregarlas en la web (Cámaras)"))
    return out


def running_instances(listing: str | None = None, own_pid: int | None = None) -> int:
    """Number of other `python -m src.main` processes (listing = '<pid> <command line>' lines, injectable for tests)."""
    own_pid = own_pid or os.getpid()
    if listing is None:
        cmd = (["powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_Process | ForEach-Object { \"$($_.ProcessId) $($_.CommandLine)\" }"]
               if sys.platform == "win32" else ["ps", "-eo", "pid,args"])
        try:
            listing = subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
        except (OSError, subprocess.SubprocessError):
            return 0
    count = 0
    for line in listing.splitlines():
        pid, _, args = line.strip().partition(" ")
        if "src.main" in args and "python" in args.lower() and pid != str(own_pid):
            count += 1
    return count


def check_system(root: Path = ROOT, free_bytes: int | None = None, instances: int | None = None) -> list[Result]:
    free = shutil.disk_usage(root).free if free_bytes is None else free_bytes
    out = [ok(f"Espacio libre: {free / 1024 ** 3:.1f} GB") if free >= MIN_FREE_GB * 1024 ** 3
           else bad(f"Poco espacio libre: {free / 1024 ** 3:.1f} GB", f"Libera al menos {MIN_FREE_GB} GB")]
    n = running_instances() if instances is None else instances
    out.append(ok("No hay otra instancia de aforo-vision corriendo") if n == 0
               else bad(f"Hay {n} proceso(s) `python -m src.main` corriendo (dos instancias duplican eventos)",
                        "Ciérralos (Administrador de tareas) o detén la tarea: Stop-ScheduledTask -TaskName AforoVision"))
    return out


def report(title: str, results: list[Result]) -> bool:
    print(f"\n{title}")
    for good, msg, action in results:
        print(f"  {'✓' if good else '✗'} {msg}")
        if action:
            print(f"      → {action}")
    return all(r[0] for r in results)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    backend, model = detection_settings()
    print(f"aforo-vision doctor (backend de detección: {backend})")
    passed = [report("1. Python", check_python()),
              report("2. Dependencias", check_dependencies(backend)),
              report("3. Modelos", check_models(backend, model)),
              report("4. Archivo .env", check_env())]
    result, cameras = fetch_cameras()
    passed.append(report("5. Supabase", [result]))
    passed.append(report("6. Cámaras", check_cameras(cameras) if result[0] else [bad("No se pueden probar sin Supabase", "Arregla el paso 5")]))
    passed.append(report("7. Sistema", check_system()))
    print("\n" + ("Todo en orden." if all(passed) else "Hay problemas: corrige los ✗ de arriba y vuelve a correr python -m tools.doctor"))
    return 0 if all(passed) else 1


if __name__ == "__main__":
    sys.exit(main())
