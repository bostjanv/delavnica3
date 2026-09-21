"""API tests using a stub detector: fast, deterministic, no model download."""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.detector import Detection, DetectionResult
from app.main import create_app


class FakeDetector:
    """Deterministic stand-in for YoloDetector implementing the Detector protocol."""

    model_name = "fake.pt"
    device = "cpu"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def detect(self, image: np.ndarray) -> DetectionResult:
        self.calls += 1
        if self.fail:
            raise RuntimeError("simulated inference failure")
        h, w = image.shape[:2]
        det = Detection(
            class_id=0,
            class_name="person",
            confidence=0.91,
            x1=10.0,
            y1=20.0,
            x2=max(30.0, w - 10),
            y2=max(40.0, h - 10),
        )
        return DetectionResult(
            image_width=w,
            image_height=h,
            inference_ms=12.5,
            detections=(det,),
        )


def make_jpeg(width=64, height=48) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(120, 90, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def make_png(width=64, height=48) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(10, 200, 30)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def fake():
    return FakeDetector()


@pytest.fixture()
def client(fake):
    app = create_app(Settings(), detector=fake)
    with TestClient(app) as c:
        yield c


def test_root_serves_demo_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "CPU YOLO Browser Demo" in resp.text


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_info_reports_configuration(client):
    resp = client.get("/api/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "yolo26n.pt"  # default settings
    assert body["device"] == "cpu"
    assert body["imgsz"] == 640
    assert body["conf"] == 0.25
    assert body["demo_fps"] == 3
    assert "backend" in body
    assert "max_upload_mb" in body


def test_detect_valid_jpeg(client, fake):
    resp = client.post(
        "/api/detect",
        content=make_jpeg(64, 48),
        headers={"Content-Type": "image/jpeg"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["image_width"] == 64
    assert body["image_height"] == 48
    assert body["model"] == "fake.pt"
    assert body["inference_ms"] == 12.5
    assert len(body["detections"]) == 1
    det = body["detections"][0]
    assert det["class_id"] == 0
    assert det["class_name"] == "person"
    assert det["confidence"] == 0.91
    assert det["x1"] == 10.0
    assert det["y1"] == 20.0
    assert det["x2"] == 54.0
    assert det["y2"] == 40.0
    assert fake.calls == 1


def test_detect_valid_png(client):
    resp = client.post(
        "/api/detect",
        content=make_png(),
        headers={"Content-Type": "image/png"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["image_width"] == 64
    assert body["image_height"] == 48


def test_detect_rejects_empty_body(client):
    resp = client.post("/api/detect", content=b"", headers={"Content-Type": "image/jpeg"})
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"]


def test_detect_rejects_malformed_image(client):
    resp = client.post(
        "/api/detect",
        content=b"\x00\x01\x02 this is not an image",
        headers={"Content-Type": "image/jpeg"},
    )
    assert resp.status_code == 400
    assert "decode" in resp.json()["detail"]


def test_detect_rejects_unsupported_media_type(client):
    resp = client.post(
        "/api/detect",
        content=make_jpeg(),
        headers={"Content-Type": "text/plain"},
    )
    assert resp.status_code == 415
    assert "media type" in resp.json()["detail"]


def test_detect_rejects_missing_media_type(client):
    resp = client.post("/api/detect", content=make_jpeg())
    assert resp.status_code == 415


def test_detect_rejects_oversized_body(fake):
    settings = Settings(max_upload_bytes=16)  # tiny limit for the test
    app = create_app(settings, detector=fake)
    with TestClient(app) as c:
        resp = c.post(
            "/api/detect",
            content=make_jpeg(),
            headers={"Content-Type": "image/jpeg"},
        )
    assert resp.status_code == 413
    assert fake.calls == 0  # oversized body must not reach the detector


def test_detect_maps_inference_failure_to_controlled_500():
    app = create_app(Settings(), detector=FakeDetector(fail=True))
    with TestClient(app, raise_server_exceptions=False) as c:
        resp = c.post(
            "/api/detect",
            content=make_jpeg(),
            headers={"Content-Type": "image/jpeg"},
        )
    assert resp.status_code == 500
    assert "inference failed" in resp.json()["detail"]
    assert "Traceback" not in resp.text  # no raw traceback leaked


def test_startup_fails_clearly_when_model_cannot_load(monkeypatch):
    import app.main as main_module

    class BoomDetector:
        def __init__(self, *a, **k):
            raise RuntimeError("simulated model load failure")

    monkeypatch.setattr(main_module, "YoloDetector", BoomDetector)
    app = create_app(Settings())  # no injected detector -> real load path
    with pytest.raises(RuntimeError, match="simulated model load failure"):
        with TestClient(app):
            pass
