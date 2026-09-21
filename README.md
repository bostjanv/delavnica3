# delavnica3 — Local CPU Object-Detection Browser Demo

A small, credible demo that proves the complete loop: **a browser captures camera frames, sends them over HTTP to a CPU-only inference service, and gets pretrained YOLO object detections back** — fast enough for a live demonstration, without a GPU.

> **Status: design complete, implementation pending.**
> This repository currently contains only the project constitution ([`AGENTS.md`](AGENTS.md)).
> All setup commands below are derived from that document and are **UNVERIFIED** — they describe the intended first release, not working code.

---

## Purpose

The project targets a single promise:

> A browser can capture camera frames, send them to a Linux/WSL CPU inference service, and receive correct pretrained YOLO object detections quickly enough for an interactive demonstration.

It is a local demo, not a product: no training, no custom datasets, no tracking, no storage, no authentication, no cloud. See [`AGENTS.md`](AGENTS.md) for scope, invariants, and explicit non-goals.

## Architecture

```text
Browser camera
    |
    | getUserMedia()
    v
HTML5 video + canvas
    |
    | JPEG frames over HTTP
    v
FastAPI service
    |
    v
Detector abstraction
    |
    v
Ultralytics YOLO26n pretrained detection model
    |
    v
CPU inference
    |
    | JSON detections
    v
Browser overlay canvas
```

- **Backend:** Python + FastAPI + Uvicorn, single process. The same process serves the static UI and the detection API.
- **Frontend:** plain HTML5/CSS/JavaScript — `getUserMedia()`, `<video>`, `<canvas>`, `fetch()`. No frontend framework, no build step.
- **Detector:** pretrained `yolo26n.pt` (COCO classes), loaded **once** at startup and reused for every request. Ultralytics-specific code is kept behind a small detector abstraction.
- **Frame policy:** the browser requests ~3 FPS, keeps at most **one** inference request in flight, and skips a capture tick if the previous request is still running. Stale frames never queue up.
- **Transport:** one HTTP `POST` per selected frame with a raw JPEG/PNG body; a stable JSON response with pixel coordinates.

## Supported Environment

- **OS:** Linux, or Windows via WSL2
- **Hardware:** any multicore CPU; **no GPU required** (CPU is the default and only assumed device)
- **Browser:** any modern browser with `getUserMedia()` support (Chrome, Edge, Firefox)
- **Python:** 3.11 or another currently supported Python 3
- **Network:** the app binds to `127.0.0.1` by default — same-host only. Camera frames never leave the machine.

## Planned Repository Layout

```text
.
├── AGENTS.md          # project constitution (governing document)
├── README.md
├── pyproject.toml
├── app/
│   ├── main.py        # FastAPI app, routes, static file serving
│   ├── detector.py    # detector abstraction + Ultralytics implementation
│   ├── config.py      # environment-based configuration
│   ├── schemas.py     # API request/response models
│   └── static/        # index.html, app.js, style.css
├── scripts/
│   └── benchmark.py   # CPU inference + end-to-end benchmark
└── tests/
    ├── test_api.py
    ├── test_detector.py
    └── fixtures/
```

## Setup

> **UNVERIFIED** — intended first-release procedure, per `AGENTS.md`.

```bash
# 1. Clone and enter the repository
git clone https://github.com/bostjanv/delavnica3
cd delavnica3

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install project dependencies (expected: fastapi, uvicorn, ultralytics,
#    pillow, numpy, pytest)
python -m pip install --upgrade pip
python -m pip install -e .
```

Do not install project dependencies globally when a venv is available.

## Model Warm-Up / Prefetch

> **UNVERIFIED** — the model is not committed to the repository.

The app lets Ultralytics download `yolo26n.pt` on first use. Because the demo may run without reliable internet, prefetch before the presentation and verify the weights load offline:

```bash
# Planned warm-up step (loads the configured model, runs one inference)
python -c "from ultralytics import YOLO; m = YOLO('yolo26n.pt'); print('model OK')"
```

The demo-readiness check must prove the model is already present or loadable **before** the presentation begins.

## Starting the Service

> **UNVERIFIED**

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Browser URL

Open **`http://localhost:8000`** in a browser on the same host as the service.

- Same-host `localhost` works over plain HTTP — no HTTPS needed.
- If you open the page from **another device**, browser camera APIs require a secure context (HTTPS or localhost), so remote-device access is a separate deployment concern, not part of the demo.
- On WSL2, open `http://localhost:8000` in the **Windows** browser; WSL2's localhost forwarding reaches the service.

## Using the Camera Demo (Planned UX)

1. Page loads and shows the video element.
2. Click **Start** — the browser asks for camera permission.
3. Live preview appears; detection requests begin at the configured rate (~3 FPS).
4. Bounding boxes, class names, and confidence scores are drawn over the video.
5. Click **Stop** — inference requests stop.
6. Basic performance info (latency / FPS) is shown.
7. Camera, network, or inference errors are displayed visibly, never swallowed.

## API (Planned)

| Method | Path           | Description |
|--------|----------------|-------------|
| GET    | `/`            | Browser UI (static assets) |
| GET    | `/api/health`  | Liveness check |
| GET    | `/api/info`    | Model, device, configuration |
| POST   | `/api/detect`  | Detect objects in one image frame |

`POST /api/detect` accepts a raw JPEG/PNG body with the matching `Content-Type` and returns JSON:

```json
{
  "image_width": 640,
  "image_height": 480,
  "inference_ms": 42.1,
  "model": "yolo26n.pt",
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.91,
      "x1": 120.0,
      "y1": 50.0,
      "x2": 310.0,
      "y2": 460.0
    }
  ]
}
```

Coordinates are in pixels relative to the submitted frame. The server decodes the image itself and never trusts client-supplied dimensions.

Expected robustness: empty bodies, unsupported media types, and malformed images are rejected with clean HTTP errors — no Python tracebacks in API responses, and image bodies are never logged.

## Configuration (Planned)

Configuration is explicit and minimal, via environment variables:

| Variable         | Default        | Meaning |
|------------------|----------------|---------|
| `YOLO_MODEL`     | `yolo26n.pt`   | Pretrained detector weights |
| `YOLO_IMGSZ`     | `640`          | Inference input size |
| `YOLO_CONF`      | `0.25`         | Confidence threshold |
| `YOLO_DEVICE`    | `cpu`          | Inference device — **never** auto-selected to CUDA |
| `DEMO_FPS`       | `3`            | Requested browser detection rate |
| `HOST`           | `127.0.0.1`    | Bind address (loopback by default) |
| `PORT`           | `8000`         | Bind port |

Binding to a LAN address (e.g. `0.0.0.0`) must be a deliberate, documented choice — it exposes the service to the reachable network.

## CPU Performance Notes

- The target is an **interactive demo** (a few detections per second), not a benchmark record.
- The model is loaded once; per-request work is decode → infer → encode the JSON response.
- Permitted tuning, in order: input `imgsz`, browser capture resolution, JPEG quality, skipping capture ticks while inference is in flight, model warm-up.
- OpenVINO is an **optional** backend, to be considered only after the PyTorch/Ultralytics CPU baseline is functional **and measured**, and only if it materially improves measured performance without destabilizing the demo.
- Performance numbers are never quoted from documentation: anything reported must be measured on the actual machine, with model, input size, and hardware stated.

### Benchmark Command

> **UNVERIFIED** — planned `scripts/benchmark.py` behavior: warms the model, then records model, backend, input size, CPU info, average inference latency, approximate inference FPS, and end-to-end HTTP latency where practical.

```bash
python scripts/benchmark.py
```

## Demo Checklist

> **UNVERIFIED** — to be completed on the actual target machine before the demo.

- [ ] Virtual environment creates successfully
- [ ] Dependencies install from documented commands
- [ ] Model weights are present (prefetched), load offline
- [ ] Server starts from the documented command
- [ ] Root page opens in the browser
- [ ] Camera permission granted; video preview appears
- [ ] Detection starts; boxes/labels/confidence appear aligned with the video
- [ ] Stop/start works repeatedly
- [ ] No ever-growing request backlog in the browser
- [ ] Malformed input is rejected cleanly (server stays up)
- [ ] Measured latency/FPS recorded with hardware + settings
- [ ] Restart from a clean terminal works
- [ ] README demo procedure is current and accurate

## WSL Notes

- Run the service **inside WSL2**; open the UI in the **Windows** browser at `http://localhost:8000` (WSL2 forwards localhost).
- Camera access is a **browser** concern: the Windows browser sees the Windows camera. The WSL service only ever sees the JPEG frames the browser sends over HTTP.
- Do not add Windows-specific frontend code.

## Troubleshooting

> **UNVERIFIED** — expected issue list for the first release.

| Symptom | Likely cause / check |
|---------|----------------------|
| Page won't load | Is the service running? Check `http://localhost:8000` and `/api/health` from the same host. |
| No video preview | Camera permission denied, camera in use elsewhere, or insecure context (non-localhost URL). |
| Boxes don't appear | Check `/api/info` for model status; watch the browser console and server logs for inference errors. |
| Very slow | Lower `DEMO_FPS`, `YOLO_IMGSZ`, or browser capture resolution. Run the benchmark to see where time goes. |
| `0.0.0.0` binding question | Intentional for LAN access? Document it — the default is loopback only. |

## Current Limitations

- **No application code exists in this repository yet** — the implementation follows the plan in `AGENTS.md`.
- CPU inference only; GPU use is explicitly out of scope.
- Single camera, single browser session at a time by design (one in-flight request).
- Detection-only: no tracking, no custom classes, no recording, no persistence.
- Pretrained COCO classes only — objects outside the 80 COCO classes are not detected.
- Local demo only: no TLS, no authentication, no multi-device story.

## Ultralytics Licensing Note

This project uses Ultralytics software and pretrained model weights. The Ultralytics license permits the intended local demonstration, but **must be reviewed before any commercial or production deployment**. This repository is **not** cleared for proprietary production use. Upstream attribution is preserved in the implementation.

## Project Constitution

[`AGENTS.md`](AGENTS.md) is the governing document: it defines purpose, architecture, non-negotiable invariants, non-goals, testing and reporting rules, and the workflow for all work on this repository. When this README and `AGENTS.md` disagree, `AGENTS.md` wins.
