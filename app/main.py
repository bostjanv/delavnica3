"""FastAPI application: serves the browser UI and the detection API.

Single process, single loaded model. The detector is created once in the
application lifespan; if it cannot be initialized, startup fails clearly.
"""

from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .config import Settings, load_settings
from .detector import YoloDetector
from .schemas import DetectionOut, DetectionResponse, HealthResponse, InfoResponse

if TYPE_CHECKING:
    from .detector import Detector

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
ALLOWED_MEDIA_TYPES = ("image/jpeg", "image/png")


def decode_image(body: bytes) -> np.ndarray:
    """Decode a JPEG/PNG body to an RGB numpy array (H, W, 3).

    Dimensions are always taken from the decoded image, never from client
    metadata. Raises ValueError for undecodable content.
    """
    try:
        with Image.open(io.BytesIO(body)) as img:
            img.verify()
        with Image.open(io.BytesIO(body)) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"could not decode image: {exc}") from exc


def _backend_info() -> str:
    try:
        import torch

        return f"pytorch {torch.__version__} (cpu)"
    except Exception:  # torch unavailable (e.g. stub-detector test runs)
        return "unknown"


def create_app(settings: Settings | None = None, detector: "Detector | None" = None) -> FastAPI:
    """Application factory.

    ``detector`` may be injected (tests use a fake); otherwise a
    ``YoloDetector`` is loaded once at startup.
    """
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if detector is not None:
            app.state.detector = detector
        else:
            try:
                app.state.detector = YoloDetector(
                    model=settings.model,
                    device=settings.device,
                    imgsz=settings.imgsz,
                    conf=settings.conf,
                )
                app.state.detector.warmup()
            except Exception:
                logger.exception("Detector initialization failed; refusing to start")
                raise  # fail startup clearly instead of pretending to be healthy
        app.state.settings = settings
        logger.info(
            "Service ready: model=%s device=%s imgsz=%d conf=%.2f listen=%s:%d",
            settings.model,
            settings.device,
            settings.imgsz,
            settings.conf,
            settings.host,
            settings.port,
        )
        yield

    app = FastAPI(title="CPU YOLO Browser Demo", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        # Deliberately cheap: no inference on the health path.
        return HealthResponse(status="ok")

    @app.get("/api/info", response_model=InfoResponse)
    async def info() -> InfoResponse:
        s: Settings = app.state.settings
        return InfoResponse(
            model=s.model,
            device=s.device,
            imgsz=s.imgsz,
            conf=s.conf,
            demo_fps=s.demo_fps,
            backend=_backend_info(),
            max_upload_mb=round(s.max_upload_bytes / (1024 * 1024), 1),
        )

    @app.post("/api/detect", response_model=DetectionResponse)
    async def detect(request: Request) -> JSONResponse | DetectionResponse:
        s: Settings = app.state.settings
        det: "Detector" = app.state.detector

        media_type = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
        if media_type not in ALLOWED_MEDIA_TYPES:
            raise HTTPException(
                status_code=415,
                detail=f"unsupported media type {media_type!r}; send image/jpeg or image/png",
            )

        body = b"".join([chunk async for chunk in _stream_limited(request, s.max_upload_bytes)])
        if not body:
            raise HTTPException(status_code=400, detail="empty request body")

        try:
            image = decode_image(body)
        except ValueError as exc:
            logger.warning("Rejected undecodable frame (%d bytes)", len(body))
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            result = det.detect(image)
        except Exception:
            logger.exception("Inference failed")
            raise HTTPException(
                status_code=500, detail="inference failed on the server; see server logs"
            ) from None

        return DetectionResponse(
            image_width=result.image_width,
            image_height=result.image_height,
            inference_ms=result.inference_ms,
            model=det.model_name,
            detections=[DetectionOut(**d.__dict__) for d in result.detections],
        )

    return app


async def _stream_limited(request: Request, limit: int):
    """Yield the request body, raising 413 before reading more than ``limit`` bytes."""
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=413,
                detail=f"request body exceeds {limit // (1024 * 1024)} MiB limit",
            )
        yield chunk


app = create_app()


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    _s = load_settings()
    uvicorn.run("app.main:app", host=_s.host, port=_s.port)
