"""Configuration parsing tests (no model involved)."""

import pytest

from app.config import DEFAULT_MAX_UPLOAD_BYTES, Settings, load_settings

_ENV_VARS = (
    "YOLO_MODEL", "YOLO_IMGSZ", "YOLO_CONF", "YOLO_DEVICE",
    "DEMO_FPS", "HOST", "PORT", "MAX_UPLOAD_MB",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_defaults():
    s = load_settings()
    assert s.model == "yolo26n.pt"
    assert s.imgsz == 640
    assert s.conf == 0.25
    assert s.device == "cpu"
    assert s.host == "127.0.0.1"
    assert s.port == 8000
    assert s.demo_fps == 3
    assert s.max_upload_bytes == DEFAULT_MAX_UPLOAD_BYTES


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("YOLO_MODEL", "yolo26n.pt")
    monkeypatch.setenv("YOLO_IMGSZ", "320")
    monkeypatch.setenv("YOLO_CONF", "0.5")
    monkeypatch.setenv("YOLO_DEVICE", "cpu")
    monkeypatch.setenv("DEMO_FPS", "5")
    monkeypatch.setenv("HOST", "127.0.0.1")
    monkeypatch.setenv("PORT", "9000")
    monkeypatch.setenv("MAX_UPLOAD_MB", "2")
    s = load_settings()
    assert s.imgsz == 320
    assert s.conf == 0.5
    assert s.demo_fps == 5
    assert s.port == 9000
    assert s.max_upload_bytes == 2 * 1024 * 1024


def test_non_cpu_device_rejected(monkeypatch):
    # GPU auto-selection is forbidden by AGENTS.md; fail clearly instead.
    monkeypatch.setenv("YOLO_DEVICE", "cuda")
    with pytest.raises(ValueError, match="YOLO_DEVICE"):
        load_settings()


def test_invalid_int_rejected(monkeypatch):
    monkeypatch.setenv("YOLO_IMGSZ", "not-a-number")
    with pytest.raises(ValueError, match="YOLO_IMGSZ"):
        load_settings()


def test_invalid_float_rejected(monkeypatch):
    monkeypatch.setenv("YOLO_CONF", "high")
    with pytest.raises(ValueError, match="YOLO_CONF"):
        load_settings()


def test_settings_immutable():
    s = Settings()
    with pytest.raises(Exception):
        s.model = "other.pt"
