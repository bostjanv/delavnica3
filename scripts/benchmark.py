#!/usr/bin/env python3
"""Benchmark the configured YOLO detector on the local machine.

Measures real inference latency (not documentation numbers). The model is
warmed up before steady-state timing is recorded.

Examples:
    python scripts/benchmark.py
    python scripts/benchmark.py --imgsz 320 --iterations 20
    python scripts/benchmark.py --url http://127.0.0.1:8000/api/detect
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image


def default_settings():
    from app.config import load_settings

    return load_settings()


def resolve_test_image(path: str | None) -> np.ndarray:
    """Load a test image: explicit path > bundled ultralytics asset > synthetic."""
    if path:
        with Image.open(path) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)
    asset = Path(__import__("ultralytics", fromlist=["__file__"]).__file__).parent / "assets" / "bus.jpg"
    if asset.exists():
        with Image.open(asset) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)
    print(f"note: no test image available; using synthetic {640}x{480} image")
    return np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)


def measure_inference(detector, image: np.ndarray, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        detector.detect(image)
    latencies: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = detector.detect(image)
        latencies.append((time.perf_counter() - start) * 1000.0)
    return latencies


def measure_http(url: str, image: np.ndarray, iterations: int) -> list[float]:
    """End-to-end latency against a running service (requires the server to be up)."""
    buf = __import__("io").BytesIO()
    Image.fromarray(image).save(buf, format="JPEG", quality=70)
    payload = buf.getvalue()
    latencies: list[float] = []
    last_body: bytes | None = None
    for _ in range(iterations):
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "image/jpeg"}, method="POST"
        )
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=60) as resp:
            last_body = resp.read()
        latencies.append((time.perf_counter() - start) * 1000.0)
    if last_body is not None:
        print(f"last HTTP response: {last_body[:200]!r}{'...' if len(last_body) > 200 else ''}")
    return latencies


def report(label: str, latencies: list[float]) -> None:
    mean = statistics.mean(latencies)
    median = statistics.median(latencies)
    p95 = statistics.quantiles(latencies, n=20)[-1] if len(latencies) >= 2 else median
    print(f"{label}: n={len(latencies)} mean={mean:.1f} ms median={median:.1f} ms "
          f"p95={p95:.1f} ms  ->  approx {1000.0 / mean:.2f} fps")


def main() -> int:
    settings = default_settings()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=settings.model)
    p.add_argument("--imgsz", type=int, default=settings.imgsz)
    p.add_argument("--conf", type=float, default=settings.conf)
    p.add_argument("--device", default=settings.device)
    p.add_argument("--warmup", type=int, default=3, help="un-timed warm-up iterations")
    p.add_argument("--iterations", type=int, default=10, help="timed iterations")
    p.add_argument("--image", default=None, help="path to a test image (default: ultralytics bus.jpg)")
    p.add_argument("--url", default=None, help="run end-to-end HTTP measurement against this /api/detect URL")
    args = p.parse_args()

    if args.device != "cpu":
        print(f"refusing to benchmark on {args.device!r}; this baseline is CPU-only", file=sys.stderr)
        return 2

    from app.detector import YoloDetector

    try:
        import torch

        backend = f"pytorch {torch.__version__} (cpu)"
    except Exception:
        backend = "pytorch (version unknown)"

    print("=== YOLO CPU benchmark ===")
    print(f"model:        {args.model}")
    print(f"backend:      {backend}")
    print(f"device:       {args.device}")
    print(f"imgsz:        {args.imgsz}")
    print(f"conf:         {args.conf}")
    print(f"cpu:          {platform.processor() or platform.machine()} ({os.cpu_count()} threads)")
    print(f"warm-up:      {args.warmup} iterations (un-timed)")

    image = resolve_test_image(args.image)
    print(f"test image:   {image.shape[1]}x{image.shape[0]} px")

    detector = YoloDetector(model=args.model, device=args.device, imgsz=args.imgsz, conf=args.conf)
    latencies = measure_inference(detector, image, args.warmup, args.iterations)
    report("inference", latencies)

    if args.url:
        print(f"\n=== End-to-end HTTP vs {args.url} ===")
        try:
            http_latencies = measure_http(args.url, image, min(args.iterations, 5))
            report("end-to-end HTTP", http_latencies)
        except Exception as exc:  # server down is a diagnostic, not a crash
            print(f"end-to-end HTTP measurement failed: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
