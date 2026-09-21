"""Small detector abstraction around Ultralytics.

The HTTP layer only sees ``Detection`` / ``DetectionResult`` dataclasses;
Ultralytics result internals never leak into the API response. The model is
loaded once (``YoloDetector.__init__``) and reused for every request.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """One bounding box in pixel coordinates of the submitted image."""

    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class DetectionResult:
    """Outcome of one inference call."""

    image_width: int
    image_height: int
    inference_ms: float
    detections: tuple[Detection, ...]


class Detector(Protocol):
    """The only interface the application needs from a detector."""

    model_name: str
    device: str

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Run inference on a single RGB numpy image (H, W, 3)."""
        ...


def boxes_to_detections(
    boxes: Sequence[tuple[Sequence[float], float, int, str]],
    image_width: int,
    image_height: int,
) -> tuple[Detection, ...]:
    """Map raw boxes ``(xyxy, confidence, class_id, class_name)`` to Detection.

    Pure function so the mapping is unit-testable without a model.
    Coordinates are clamped to the image rectangle and x1/x2, y1/y2 ordered.
    """
    detections: list[Detection] = []
    for xyxy, confidence, class_id, class_name in boxes:
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        x1, x2 = sorted((min(max(x1, 0.0), image_width), min(max(x2, 0.0), image_width)))
        y1, y2 = sorted((min(max(y1, 0.0), image_height), min(max(y2, 0.0), image_height)))
        detections.append(
            Detection(
                class_id=int(class_id),
                class_name=class_name,
                confidence=round(float(confidence), 4),
                x1=round(x1, 1),
                y1=round(y1, 1),
                x2=round(x2, 1),
                y2=round(y2, 1),
            )
        )
    return tuple(detections)


class YoloDetector:
    """Ultralytics YOLO detector, loaded once and reused across requests."""

    def __init__(
        self,
        model: str,
        device: str = "cpu",
        imgsz: int = 640,
        conf: float = 0.25,
    ) -> None:
        from ultralytics import YOLO  # imported lazily so importing app works without torch

        start = time.perf_counter()
        self._model = YOLO(model)
        self.model_name = model
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self._names: dict[int, str] = {int(k): str(v) for k, v in self._model.names.items()}
        logger.info(
            "Loaded model %r on device %r in %.1f s",
            model,
            device,
            time.perf_counter() - start,
        )

    def warmup(self, iterations: int = 2) -> None:
        """Run inferences on a synthetic image so the first real frame is not
        slowed down by one-time initialization (also makes timing stable)."""
        dummy = np.full((self.imgsz, self.imgsz, 3), 128, dtype=np.uint8)
        for _ in range(iterations):
            self.detect(dummy)
        logger.info("Detector warm-up complete (%d iterations)", iterations)

    def detect(self, image: np.ndarray) -> DetectionResult:
        start = time.perf_counter()
        results = self._model.predict(
            image,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )
        inference_ms = (time.perf_counter() - start) * 1000.0

        raw_boxes: list[tuple[Sequence[float], float, int, str]] = []
        result = results[0]
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy().astype(int)
            for i in range(len(cls)):
                class_id = int(cls[i])
                raw_boxes.append(
                    (
                        xyxy[i],
                        float(confs[i]),
                        class_id,
                        self._names.get(class_id, f"class_{class_id}"),
                    )
                )

        detections = boxes_to_detections(raw_boxes, int(image.shape[1]), int(image.shape[0]))
        return DetectionResult(
            image_width=int(image.shape[1]),
            image_height=int(image.shape[0]),
            inference_ms=round(inference_ms, 1),
            detections=detections,
        )
