"""Detector result-mapping tests (pure logic, no model download)."""

import numpy as np
import pytest

from app.detector import Detection, DetectionResult, boxes_to_detections


def test_maps_fields():
    raw = [([10.0, 20.0, 110.0, 220.0], 0.912345, 0, "person")]
    out = boxes_to_detections(raw, 640, 480)
    assert len(out) == 1
    d = out[0]
    assert isinstance(d, Detection)
    assert d.class_id == 0
    assert d.class_name == "person"
    assert d.confidence == 0.9123  # rounded
    assert (d.x1, d.y1, d.x2, d.y2) == (10.0, 20.0, 110.0, 220.0)


def test_empty_boxes_gives_empty_tuple():
    assert boxes_to_detections([], 640, 480) == ()


def test_out_of_range_clamped_to_image():
    raw = [([-5.0, -10.0, 9999.0, 9999.0], 0.5, 2, "car")]
    out = boxes_to_detections(raw, 640, 480)
    d = out[0]
    assert d.x1 == 0.0
    assert d.y1 == 0.0
    assert d.x2 == 640.0
    assert d.y2 == 480.0


def test_unordered_coordinates_reordered():
    raw = [([300.0, 400.0, 100.0, 200.0], 0.7, 5, "bicycle")]
    out = boxes_to_detections(raw, 640, 480)
    d = out[0]
    assert d.x1 == 100.0
    assert d.x2 == 300.0
    assert d.y1 == 200.0
    assert d.y2 == 400.0


def test_multiple_detections_preserve_order():
    raw = [
        ([0, 0, 10, 10], 0.9, 0, "person"),
        ([5, 5, 15, 15], 0.8, 1, "bicycle"),
    ]
    out = boxes_to_detections(raw, 100, 100)
    assert [d.class_name for d in out] == ["person", "bicycle"]
    assert [d.class_id for d in out] == [0, 1]


def test_result_structure():
    r = DetectionResult(image_width=640, image_height=480, inference_ms=42.0, detections=())
    assert r.image_width == 640
    assert r.image_height == 480
    assert r.inference_ms == 42.0
    assert r.detections == ()


# ---------------------------------------------------------------------------
# RGB -> BGR boundary regression (Finding 1)
#
# The detector contract is RGB; Ultralytics assumes OpenCV-style BGR for
# numpy HWC input and flips channels internally. The flip must happen exactly
# once, at the Ultralytics boundary, so a red pixel must reach predict() as
# blue. This test fakes the model (no weights, no download) and inspects the
# exact array handed to predict().
# ---------------------------------------------------------------------------


class _FakeBoxes:
    pass


class _FakeResult:
    def __init__(self):
        self.boxes = None  # no detections


class _FakeUltralyticsModel:
    """Stands in for YOLO(); records every array passed to predict()."""

    names = {0: "person"}

    def __init__(self):
        self.predicted_inputs = []

    def predict(self, image, **kwargs):
        self.predicted_inputs.append(np.asarray(image))
        return [_FakeResult()]


def _yolo_detector_with_fake_model(fake_model):
    """Build a YoloDetector without loading real weights (skips __init__)."""
    from app.detector import YoloDetector

    det = YoloDetector.__new__(YoloDetector)
    det._model = fake_model
    det.model_name = "fake.pt"
    det.device = "cpu"
    det.imgsz = 640
    det.conf = 0.25
    det._names = {0: "person"}
    return det


def test_detect_converts_rgb_to_bgr_at_ultralytics_boundary():
    fake_model = _FakeUltralyticsModel()
    det = _yolo_detector_with_fake_model(fake_model)

    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[..., 0] = 10  # R
    rgb[..., 1] = 20  # G
    rgb[..., 2] = 30  # B

    result = det.detect(rgb)

    assert len(fake_model.predicted_inputs) == 1
    sent = fake_model.predicted_inputs[0]
    assert sent.shape == (4, 6, 3)
    assert sent.dtype == np.uint8
    # Channel order at the model boundary must be B, G, R:
    assert sent[0, 0, 0] == 30, "channel 0 must be blue (B)"
    assert sent[0, 0, 1] == 20, "channel 1 must be green (G)"
    assert sent[0, 0, 2] == 10, "channel 2 must be red (R)"
    assert sent.flags["C_CONTIGUOUS"], "model input should be contiguous"
    # The RGB input array must not be mutated in place.
    assert rgb[0, 0, 0] == 10 and rgb[0, 0, 1] == 20 and rgb[0, 0, 2] == 30
    # Mapping still works with the faked (empty) boxes.
    assert result.detections == ()
    assert (result.image_width, result.image_height) == (6, 4)
