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
