# delavnica3 — Local CPU Object-Detection Browser Demo

A small, credible demo that proves the complete loop: **a browser captures
camera frames, sends them over HTTP to a CPU-only inference service, and gets
pretrained YOLO object detections back** — fast enough for a live
demonstration, without a GPU.

It is a local demo, not a product: no training, no custom datasets, no
tracking, no storage, no authentication, no cloud. See
[`AGENTS.md`](AGENTS.md) for scope, invariants, and explicit non-goals.

## Purpose

The project targets a single promise:

> A browser can capture camera frames, send them to a Linux/WSL CPU inference
> service, and receive correct pretrained YOLO object detections quickly
> enough for an interactive demonstration.

## Architecture

```text
Browser camera
    |
    | getUserMedia()
    v
HTML5 video + canvas
    |
    | JPEG frames over HTTP (one in flight, stale frames skipped)
    v
FastAPI service (serves UI + API, single process)
    |
    v
Detector abstraction (app/detector.py)
    |
    v
Ultralytics YOLO26n pretrained detection model
    |
    v
CPU inference
    |
    | JSON detections (pixel coordinates of the submitted frame)
    v
Browser overlay canvas (boxes, class labels, confidence)
```

- **Backend:** Python + FastAPI + Uvicorn, single process. The same process
  serves the static UI and the detection API.
- **Frontend:** plain HTML5/CSS/JavaScript — `getUserMedia()`, `<video>`,
  `<canvas>`, `fetch()`. No frontend framework, no build step.
- **Detector:** pretrained `yolo26n.pt` (COCO classes), loaded **once** at
  startup and reused for every request. Ultralytics-specific code lives
  behind a small detector abstraction (`app/detector.py`).
- **Frame policy:** the browser requests ~3 FPS, keeps at most **one**
  inference request in flight, and skips a capture tick if the previous
  request is still running. Stale frames never queue up.
- **Transport:** one HTTP `POST` per selected frame with a raw JPEG/PNG body;
  a stable JSON response with pixel coordinates.

## Supported Environment

- **OS:** Linux, or Windows via WSL2
- **Hardware:** any multicore CPU; **no GPU required** (CPU is the default and
  only supported device in this release)
- **Browser:** any modern browser with `getUserMedia()` support (Chrome,
  Edge, Firefox). For same-host access over `http://localhost`, no HTTPS is
  needed.
- **Python:** 3.11 or newer (verified on 3.14)
- **Network:** the app binds to `127.0.0.1` by default — same-host only.
  Camera frames never leave the machine.

## Repository Layout

```text
.
├── AGENTS.md          # project constitution
├── README.md
├── pyproject.toml
├── app/
│   ├── __init__.py
│   ├── main.py        # FastAPI app, endpoints, request validation
│   ├── config.py      # environment-driven settings
│   ├── detector.py    # small detector abstraction + Ultralytics wrapper
│   ├── schemas.py     # JSON API models
│   └── static/
│       ├── index.html
│       ├── app.js
│       └── style.css
├── scripts/
│   └── benchmark.py   # local CPU latency/FPS measurement
└── tests/             # pytest suite (stub detector; no model download)
```

## Setup

Tested on Ubuntu 26.04 / Python 3.14:

```bash
cd delavnica3
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"   # app + dev extras (pytest, httpx)
```

The first run of anything that touches the model will download the pretrained
weights (see below).

## Model Prefetch / Warm-up

The demo may occur without reliable internet access, so make sure the weights
are present **before** the presentation:

```bash
python -c "from app.detector import YoloDetector; d = YoloDetector('yolo26n.pt'); d.warmup(); print('model ready')"
```

This downloads `yolo26n.pt` on first use (Ultralytics caches it in the
current working directory) and runs two warm-up inferences. If it prints
`model ready` offline after the first successful download, the weights are
local and no network is needed at demo time.

## Run

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

or equivalently `python -m app.main` (reads `HOST`/`PORT` from the
environment). The model loads and warms up at startup; if it cannot load,
startup fails with a clear error instead of serving a broken demo.

## Browser

Open:

```text
http://localhost:8000/
```

1. Press **Start Detection** and grant camera permission.
2. Point the camera at ordinary objects (people, cars, bicycles, animals,
   furniture).
3. Bounding boxes with class name and confidence percentage are drawn over
   the video; a stats line shows round-trip latency, inference latency, and
   the achieved detections/s.
4. Press **Stop Detection** to stop requests and release the camera; **Start**
   works again at any time.
5. If the backend dies, the error is shown on the page (no silent failure).

## API

| Method | Path           | Description                                                        |
| ------ | -------------- | ------------------------------------------------------------------ |
| GET    | `/`            | Serves the browser demo page.                                      |
| GET    | `/api/health`  | `{"status": "ok"}` — cheap, no inference.                          |
| GET    | `/api/info`    | Runtime config: model, device, imgsz, conf, demo_fps, backend, max upload size. |
| POST   | `/api/detect`  | Accepts a raw `image/jpeg` or `image/png` body; returns JSON detections. |

`POST /api/detect` response shape (coordinates are pixels of the submitted
frame):

```json
{
  "image_width": 640,
  "image_height": 480,
  "inference_ms": 57.4,
  "model": "yolo26n.pt",
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.93,
      "x1": 104.2,
      "y1": 51.3,
      "x2": 311.9,
      "y2": 468.0
    }
  ]
}
```

Error handling: empty body → `400`; undecodable image → `400`; unsupported
content type → `415`; body larger than the limit (default 10 MiB) → `413`;
inference failure → controlled `500` (traceback stays in the server log).

## Configuration

All settings are environment variables with safe defaults:

| Variable        | Default     | Meaning                                    |
| --------------- | ----------- | ------------------------------------------ |
| `YOLO_MODEL`    | `yolo26n.pt`| Pretrained weights (detection, COCO).      |
| `YOLO_IMGSZ`    | `640`       | Inference input size in pixels.            |
| `YOLO_CONF`     | `0.25`      | Confidence threshold.                      |
| `YOLO_DEVICE`   | `cpu`       | Only `cpu` is supported in this release.   |
| `DEMO_FPS`      | `3`         | Requested browser detection rate.          |
| `HOST`          | `127.0.0.1` | Bind address (loopback by default).        |
| `PORT`          | `8000`      | Bind port.                                 |
| `MAX_UPLOAD_MB` | `10`        | Maximum `/api/detect` body size in MiB.    |

Examples:

```bash
YOLO_IMGSZ=320 uvicorn app.main:app --port 8000   # faster on weak CPUs
DEMO_FPS=5 uvicorn app.main:app                   # more aggressive browser rate
```

## CPU Performance Notes

This is a CPU baseline. Expectations on a typical 4-core laptop/server CPU:
inference-only latency of roughly 50–250 ms per 640×480 frame (i.e. ~4–20
inference fps), with the browser effectively limited to the lower of the
requested `DEMO_FPS` and the measured round-trip rate. **Run the benchmark
below on your machine and trust that number, not this one.**

Tuning levers (in risk order, all configuration changes, no code changes):

1. Lower `YOLO_IMGSZ` (e.g. 320) — biggest lever.
2. Lower `DEMO_FPS`.
3. Lower the capture resolution in the browser (camera ideal size is 640×480).
4. Lower the JPEG quality in `app.js` (currently 0.7).

## Benchmark

```bash
python scripts/benchmark.py                          # inference-only
python scripts/benchmark.py --imgsz 320 --iterations 20
python scripts/benchmark.py --url http://127.0.0.1:8000/api/detect   # + end-to-end HTTP
```

The script prints model, backend, device, imgsz, CPU info, warm-up count, and
mean/median/p95 inference latency with the approximate FPS. Warm-up
iterations are un-timed. It uses the bundled Ultralytics test image
(`bus.jpg`) by default, or `--image path/to/img.jpg`.

## Demo Checklist

Run shortly before the presentation, on the actual machine:

- [ ] `source .venv/bin/activate` in a fresh terminal.
- [ ] `python -c "from app.detector import YoloDetector; d = YoloDetector('yolo26n.pt'); d.warmup(); print('model ready')"` — model present, no download.
- [ ] `uvicorn app.main:app --host 127.0.0.1 --port 8000` starts and logs `Service ready`.
- [ ] Open `http://localhost:8000/`; status line shows the model name.
- [ ] Grant camera permission; preview appears.
- [ ] **Start Detection**; boxes appear within a second or two on a person/object.
- [ ] **Stop Detection**; request activity stops (watch server logs).
- [ ] **Start Detection** again; still works.
- [ ] Note the stats line (latency + detections/s) — this is your demo number.

## Troubleshooting

- **Model download/loading failures:** check network access on first run;
  the prefetch command above must succeed once. Re-run it offline to confirm
  the weights are cached. A wrong model name or a corrupt download should be
  fixed by re-running the prefetch (delete the cached `*.pt` in the working
  directory first).
- **Camera permission denied:** the browser prompt must be accepted;
  `http://localhost` is treated as a secure context by modern browsers.
  Check the site's camera permission and that no other app is using the
  camera exclusively.
- **Browser from another device / WSL:** the service binds to loopback, so it
  is reachable only from the same machine. In WSL2, opening
  `http://localhost:8000` from a Windows browser works via WSL2's
  localhost-forwarding. Access from *other* physical devices requires a
  non-loopback bind (`HOST=0.0.0.0`), which exposes the service to the
  reachable network, and browser camera access from non-localhost origins
  requires HTTPS — both are out of scope for this local demo.
- **Slow inference:** lower `YOLO_IMGSZ` (e.g. 320) and/or `DEMO_FPS`; run the
  benchmark to measure the effect. Do not compare against published GPU
  numbers.
- **Port already in use:** pick another port (`uvicorn app.main:app --port 8001`)
  and open that URL, or stop the other listener.
- **Boxes look misaligned:** ensure the page was loaded from the served app
  (not a copy) — the overlay tracks the actual video size at runtime.

## Limitations

- **CPU-focused:** no GPU/CUDA path in this release; `YOLO_DEVICE` accepts
  `cpu` only.
- **Detection only:** no tracking, no custom classes, no re-identification.
- **Simple inference architecture:** one process, one loaded model, one
  in-flight request per browser session; no queueing, no scaling.
- **Demo-grade robustness:** input validation is present, but this is not
  production hardened (no auth, no rate limiting, no TLS, no observability
  stack).
- **Performance varies with hardware:** see the benchmark section.

## Ultralytics Licensing

This project uses Ultralytics software and pretrained model weights
(`yolo26n.pt`). Ultralytics distributes its code and weights under specific
AGPL/commercial terms that must be reviewed before any commercial or
production deployment. This repository is **not** cleared for proprietary
production use; it is a local technical demo. Preserve upstream license
notices where applicable and consult the Ultralytics license before
reusing anything outside this demo.

## WSL Notes

- Run the service inside WSL2 (Ubuntu). From Windows, `http://localhost:8000`
  reaches it via WSL2 localhost-forwarding — no extra configuration.
- A WSL2 webcam is available to the browser running on Windows; the browser
  does the camera capture, the WSL2 service only sees JPEG frames, so no
  device passthrough into WSL is needed.
