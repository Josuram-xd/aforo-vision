import httpx
import pytest

from tools import doctor

CAMS = {"camera-outside": {"host": "10.0.0.5", "username": "u", "password": "s3cret"}}


def test_python_version_and_arch():
    assert all(r[0] for r in doctor.check_python((3, 14, 2), "AMD64"))
    assert not doctor.check_python((3, 14, 1), "AMD64")[0][0]
    assert not doctor.check_python((3, 13, 5), "AMD64")[0][0]
    assert not doctor.check_python((3, 14, 2), "ARM64")[1][0]


def test_dependencies_skip_torch_for_onnx():
    only_torch_missing = lambda name: None if name in ("torch", "ultralytics", "torchvision") else object()
    assert doctor.check_dependencies("onnx", only_torch_missing)[0][0]
    res = doctor.check_dependencies("torch", only_torch_missing)[0]
    assert not res[0] and "torch" in res[1]


def test_models_report_missing_and_size(tmp_path):
    (tmp_path / "yolo26n-pose.pt").write_bytes(b"x" * 10)
    res = doctor.check_models("torch", "yolo26n-pose.pt", tmp_path)
    assert res[0][0] and "MB" in res[0][1]
    assert not any(r[0] for r in res[1:])


def test_detection_backend_defaults_to_torch(tmp_path):
    assert doctor.detection_settings(tmp_path) == ("torch", "yolo26n-pose.pt")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "pilot.yaml").write_text("detection:\n  backend: onnx\n  model: m.onnx\n", encoding="utf-8")
    assert doctor.detection_settings(tmp_path) == ("onnx", "m.onnx")


def test_env_validation_never_prints_values(tmp_path, monkeypatch):
    for n in ("AFORO_BACKEND_URL", "AFORO_SUPABASE_KEY", "AFORO_DEVICE_KEY"):
        monkeypatch.delenv(n, raising=False)
    assert not doctor.check_env(tmp_path)[0][0]
    (tmp_path / ".env").write_text("AFORO_BACKEND_URL=https://x.supabase.co\nAFORO_SUPABASE_KEY=<tu llave>\nAFORO_DEVICE_KEY=abc123\n", encoding="utf-8")
    res = doctor.check_env(tmp_path)
    assert [r[0] for r in res] == [True, False, True]
    assert all("abc123" not in r[1] and "tu llave" not in r[1] for r in res)


def _env():
    return {"AFORO_BACKEND_URL": "https://x.supabase.co", "AFORO_SUPABASE_KEY": "pub", "AFORO_DEVICE_KEY": "dev"}


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_cameras_ok():
    res, cams = doctor.fetch_cameras(_env(), _client(lambda r: httpx.Response(200, json=CAMS)))
    assert res[0] and cams == CAMS


@pytest.mark.parametrize("code", [401, 403])
def test_fetch_cameras_key_rejected(code):
    res, cams = doctor.fetch_cameras(_env(), _client(lambda r: httpx.Response(code)))
    assert not res[0] and "rechazó" in res[1] and cams == {}


def test_fetch_cameras_no_internet():
    def boom(request):
        raise httpx.ConnectError("down")

    res, _ = doctor.fetch_cameras(_env(), _client(boom))
    assert not res[0] and "internet" in res[1]


def test_camera_states_and_no_password():
    unreachable = doctor.check_cameras(CAMS, tcp=lambda h: False)[0]
    assert not unreachable[0] and "WiFi" in unreachable[2]
    refused = doctor.check_cameras(CAMS, tcp=lambda h: True, opener=lambda u: False)[0]
    assert not refused[0] and "rechaza" in refused[1]
    good = doctor.check_cameras(CAMS, tcp=lambda h: True, opener=lambda u: True)[0]
    assert good[0] and "conecta" in good[1]
    assert all("s3cret" not in " ".join(map(str, r)) for r in (unreachable, refused, good))


def test_running_instances_ignores_self():
    listing = "100 python.exe -m src.main\n200 python.exe -m src.main --debug\n300 python.exe -m pytest\n"
    assert doctor.running_instances(listing, own_pid=100) == 1


def test_system_checks():
    assert all(r[0] for r in doctor.check_system(free_bytes=50 * 1024 ** 3, instances=0))
    res = doctor.check_system(free_bytes=1024 ** 3, instances=2)
    assert not res[0][0] and not res[1][0]
