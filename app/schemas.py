"""Pydantic models for the public JSON API.

The API representation is deliberately stable and simple; the frontend and
any other client can rely on these field names.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DetectionOut(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionResponse(BaseModel):
    image_width: int
    image_height: int
    inference_ms: float
    model: str
    detections: list[DetectionOut]


class HealthResponse(BaseModel):
    status: str


class InfoResponse(BaseModel):
    model: str
    device: str
    imgsz: int
    conf: float
    demo_fps: int
    backend: str
    max_upload_mb: float
