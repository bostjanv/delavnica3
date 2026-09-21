"""Explicit, environment-driven configuration.

Only the settings the first release needs. No framework; defaults follow
the AGENTS.md strategic defaults (CPU, yolo26n.pt, imgsz 640, ~3 FPS).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MiB per submitted frame


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for the demo service."""

    model: str = "yolo26n.pt"
    imgsz: int = 640
    conf: float = 0.25
    device: str = "cpu"
    host: str = "127.0.0.1"
    port: int = 8000
    demo_fps: int = 3
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES


def load_settings() -> Settings:
    """Build Settings from environment variables.

    Supported variables:

        YOLO_MODEL    pretrained weights (default: yolo26n.pt)
        YOLO_IMGSZ    inference input size in pixels (default: 640)
        YOLO_CONF     confidence threshold (default: 0.25)
        YOLO_DEVICE   inference device; "cpu" is the only supported value
        DEMO_FPS      requested browser detection rate (default: 3)
        HOST          bind address (default: 127.0.0.1, loopback only)
        PORT          bind port (default: 8000)
        MAX_UPLOAD_MB maximum /api/detect body size in MiB (default: 10)
    """
    env = os.environ
    settings = Settings()

    if "YOLO_MODEL" in env:
        settings = replace(settings, model=_env_str("YOLO_MODEL", settings.model))
    if "YOLO_IMGSZ" in env:
        settings = replace(settings, imgsz=_env_int("YOLO_IMGSZ", settings.imgsz))
    if "YOLO_CONF" in env:
        settings = replace(settings, conf=_env_float("YOLO_CONF", settings.conf))
    if "YOLO_DEVICE" in env:
        device = _env_str("YOLO_DEVICE", settings.device).lower()
        if device != "cpu":
            # CPU is the only supported baseline device. Fail clearly instead of
            # silently auto-selecting another backend (e.g. CUDA).
            raise ValueError(
                f"YOLO_DEVICE={device!r} is not supported in this release; use YOLO_DEVICE=cpu"
            )
        settings = replace(settings, device=device)
    if "DEMO_FPS" in env:
        settings = replace(settings, demo_fps=_env_int("DEMO_FPS", settings.demo_fps))
    if "HOST" in env:
        settings = replace(settings, host=_env_str("HOST", settings.host))
    if "PORT" in env:
        settings = replace(settings, port=_env_int("PORT", settings.port))
    if "MAX_UPLOAD_MB" in env:
        max_mb = _env_float("MAX_UPLOAD_MB", settings.max_upload_bytes / (1024 * 1024))
        settings = replace(settings, max_upload_bytes=int(max_mb * 1024 * 1024))
    return settings
