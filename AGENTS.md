# AGENTS.md

## Purpose

This file is the project constitution for the coding agent.

Read this file before making any change. Treat it as durable repository law, not as a suggestion.
If a work order conflicts with this file, stop and report the conflict instead of silently choosing one.
The human lead owns product intent, risk acceptance, and release decisions. The coding agent implements bounded work and returns evidence.

---

## Discovery Summary

### Domain problem

Build a small, credible object-detection demo that runs without a discrete GPU and can be demonstrated from a web browser.

The demo must:

- run on Linux or Windows Subsystem for Linux (WSL2);
- use a pretrained Ultralytics YOLO detector;
- require no model training or fine-tuning;
- detect ordinary COCO objects such as people, cars, bicycles, animals, furniture, etc.;
- expose inference as a local network service;
- provide an HTML5 browser page that can capture camera frames and send them to the service;
- display returned detections as bounding boxes and labels over the browser video;
- remain responsive on a multicore CPU at a practical demo rate of a few frames per second;
- be simple enough to build, understand, test, and demonstrate quickly.

### Product shape

This is a **local CPU object-detection web service with a browser demo UI**.

It is not a training system, dataset-management system, surveillance platform, production video analytics product, distributed inference cluster, or general computer-vision framework.

### Chosen architecture

First release:

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

The FastAPI process should serve both:

1. the browser UI/static assets; and
2. the object-detection API.

Keep the first release as a single-process application unless measurements prove that another process is necessary.

### Baseline model

Use:

```text
yolo26n.pt
```

as the default pretrained detector.

Rules:

- Detection only.
- Pretrained weights only.
- COCO classes only.
- No training.
- No fine-tuning.
- No custom datasets.
- No GPU requirement.
- Do not silently replace YOLO26 with another YOLO family.
- Do not silently replace the nano model with a larger model.

A larger model such as `yolo26s.pt` may be benchmarked only as an explicit optional comparison after the baseline demo works.

### CPU optimization strategy

Correctness and a complete browser-to-detector-to-browser loop come first.

Optimization order:

1. Make `yolo26n.pt` CPU inference work correctly.
2. Measure end-to-end and inference-only latency.
3. Avoid frame queues and unnecessary copies.
4. Tune input image size and browser frame rate.
5. Only then evaluate OpenVINO as an optional CPU backend if it materially improves measured performance without destabilizing the demo.

Do not introduce OpenVINO, ONNX Runtime, multiprocessing, Redis, Celery, message queues, or a separate inference service merely because they might be faster.

### First release scope

The first release is complete when a user can:

1. start the service under Linux or WSL2;
2. open the served page in a modern browser;
3. grant camera permission;
4. start detection;
5. see bounding boxes, class names, and confidence scores over the camera image;
6. stop detection;
7. see basic performance information;
8. restart the application from documented commands.

---

## Mission

Deliver the smallest robust CPU-only YOLO browser demo that proves the complete system works.

Preserve this core promise:

> A browser can capture camera frames, send them to a Linux/WSL CPU inference service, and receive correct pretrained YOLO object detections quickly enough for an interactive demonstration.

Prefer boring, readable, testable code over framework complexity.

The demo deadline matters. Do not expand scope to make the project look more production-like.

---

## Human / Strategic / Execution Roles

### Human lead

The human lead owns:

- the demo goal;
- acceptance of architecture changes;
- risk acceptance;
- release/demo readiness;
- decisions to expand scope.

### Strategic model

The strategic model owns:

- architecture guidance;
- scope sequencing;
- work orders;
- interpretation of execution reports;
- review of evidence;
- identification of missing tests or risks.

### Coding agent

The coding agent owns:

- repository inspection;
- implementation;
- local dependency installation inside the approved environment;
- running the application;
- running tests;
- measuring performance;
- updating documentation;
- committing scoped changes;
- reporting exact evidence.

The coding agent must not redefine the product.

---

## Architecture

### Backend

Use Python and FastAPI.

Recommended first-release components:

- Python 3.11 or another currently supported Python 3 version available in the environment;
- `fastapi`;
- `uvicorn`;
- `ultralytics`;
- Pillow and/or NumPy where needed for image decoding and conversion;
- pytest for automated tests.

Avoid adding dependencies unless they solve a demonstrated requirement.

### Frontend

Use plain HTML5, CSS, and JavaScript.

Use browser APIs directly:

- `navigator.mediaDevices.getUserMedia()`;
- `<video>`;
- `<canvas>`;
- `fetch()`.

Do not introduce React, Vue, Angular, Svelte, Node.js build tooling, npm, Vite, webpack, TypeScript, or another frontend framework for the first release.

### API

The first release should expose at least:

```text
GET  /
GET  /api/health
GET  /api/info
POST /api/detect
```

`POST /api/detect` should accept one encoded image frame and return JSON.

Prefer a raw JPEG/PNG request body with the correct `Content-Type` over multipart form upload if that keeps the implementation simpler.

A detection response should be stable and machine-readable. A reasonable shape is:

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

Coordinates are in pixels relative to the submitted frame.

Do not add API authentication to the first local demo unless explicitly requested.

### Detector ownership

The model must be loaded once and reused across requests.

Do not reload model weights for every frame.

Keep Ultralytics-specific code behind a small detector abstraction so that optimization or export backends can be changed later without rewriting HTTP and browser logic.

The detector abstraction should expose only what the application needs.

### Browser frame flow

The browser must not create an unbounded inference queue.

Required behavior:

- configurable requested detection rate;
- default around 3 FPS;
- no more than one inference request in flight per browser session for the simple first release;
- if the previous request has not completed, skip the next capture tick rather than queueing frames;
- overlay detections on the frame dimensions from which they were produced;
- show connection/inference errors visibly rather than silently failing.

Responsiveness matters more than processing every captured frame.

### Concurrency

Assume CPU inference is the bottleneck.

Start simple:

- one application process;
- one loaded model;
- bounded inference concurrency;
- no background queue.

Do not assume multiple Uvicorn workers are better: multiple workers can duplicate model memory and compete for CPU.

Add concurrency complexity only after measurement.

---

## Configuration

Configuration should be explicit and minimal.

Environment variables or command-line configuration may include:

```text
YOLO_MODEL=yolo26n.pt
YOLO_IMGSZ=640
YOLO_CONF=0.25
YOLO_DEVICE=cpu
DEMO_FPS=3
HOST=127.0.0.1
PORT=8000
```

Exact names may differ, but configuration must be documented.

Default to CPU.

Do not auto-select CUDA merely because PyTorch reports a CUDA device.

Do not hard-code paths tied to one developer machine.

---

## Non-Negotiable Invariants

1. **CPU-only baseline**
   - The application must function without a discrete GPU.
   - GPU support is not required for acceptance.

2. **No training**
   - Do not add training code, labeling tools, dataset downloads, augmentation, fine-tuning, or experiment tracking.

3. **Pretrained detector**
   - Use pretrained Ultralytics detection weights.
   - Default to `yolo26n.pt`.

4. **Model loaded once**
   - Never reload the model per HTTP request.

5. **No unbounded frame queue**
   - Stale frames must not accumulate.

6. **Browser-visible failures**
   - Camera, HTTP, decoding, model, and inference errors must be surfaced clearly.

7. **Repository reproducibility**
   - A clean checkout must be runnable from documented setup commands.

8. **No secret dependency on developer state**
   - Do not rely on globally installed private packages, shell aliases, undocumented model paths, or manually modified system files.

9. **No fake performance claims**
   - Report measured FPS/latency with hardware and settings.
   - Never claim a target FPS was achieved unless it was measured.

10. **No fake test claims**
    - Passed, failed, skipped, and not-run tests are different states and must be reported separately.

---

## Explicit Non-Goals

For the first release, do not implement:

- model training or fine-tuning;
- custom object classes;
- object tracking;
- face recognition or identification;
- person re-identification;
- persistent video recording;
- database storage;
- user accounts;
- authentication/authorization;
- cloud deployment;
- Kubernetes;
- Docker unless it directly solves an environment problem;
- Redis;
- Celery;
- Kafka;
- distributed inference;
- multi-camera management;
- RTSP ingestion;
- production TLS termination;
- production observability stacks;
- Prometheus/Grafana;
- WebRTC;
- a native desktop application;
- a mobile application;
- a JavaScript frontend framework;
- a general plugin architecture.

Do not implement WebSockets in the first slice unless HTTP frame submission has been measured and shown inadequate for the required demo.

Do not build speculative abstractions for possible future requirements.

---

## Performance Requirements

The target is an interactive demo, not a benchmark record.

### Functional target

Aim for approximately a few detections per second on the available multicore CPU.

The initial browser request rate should be conservative, around 3 FPS.

### Measurement

Provide a benchmark or diagnostic command that records at least:

- model;
- backend;
- input size;
- CPU information if readily available;
- warm-up behavior;
- average inference latency;
- approximate inference FPS;
- end-to-end HTTP latency when practical.

Warm the model before recording steady-state timing.

Do not use benchmark numbers from documentation as proof of performance on the local machine.

### Optimization rules

Before adding a new inference backend, identify where time is spent.

Permitted low-risk tuning includes:

- smaller `imgsz`;
- lower browser capture resolution;
- JPEG quality adjustment;
- skipping capture ticks while inference is in flight;
- sensible CPU thread configuration when measurement supports it;
- avoiding duplicate image conversions;
- model warm-up.

OpenVINO may be added as an optional backend only after the PyTorch/Ultralytics baseline is functional and measured.

If OpenVINO is added:

- keep the ordinary baseline working;
- document export/setup commands;
- do not commit generated model binaries unless explicitly requested;
- benchmark both backends on the same input/settings;
- select the faster backend only through explicit configuration or a clearly documented default.

---

## Browser and WSL Requirements

Support the normal case where:

- the backend runs inside WSL2 or Linux;
- the browser runs on the same Windows/Linux host;
- the user opens the application through `localhost`.

Document the exact URL.

Do not introduce HTTPS for a same-host localhost demo.

If the browser is accessed from another physical device, note that browser camera APIs may require a secure context. Treat remote-device HTTPS as a separate deployment concern unless explicitly requested.

Do not add Windows-specific frontend code.

---

## Input Validation and Robustness

The API must fail cleanly.

At minimum:

- reject empty request bodies;
- reject unsupported media types;
- reject malformed images;
- impose a reasonable maximum upload size;
- return useful HTTP error codes;
- avoid exposing Python tracebacks to the browser as normal API responses;
- log enough information for local debugging;
- never log entire submitted image bodies.

Do not trust dimensions or metadata supplied by the browser; decode the image server-side.

---

## Security and Privacy

This is a local demo, but basic discipline still applies.

- Never commit secrets, tokens, credentials, cookies, or private keys.
- Do not require external API keys.
- Do not send camera frames to third-party services.
- Inference must run locally.
- Do not persist submitted camera frames by default.
- Do not add analytics or telemetry.
- Do not expose the service on `0.0.0.0` by default; default to loopback.
- If LAN binding is explicitly enabled, document that this exposes the service to the reachable network.
- Do not make production-security claims.

### Ultralytics licensing

Do not silently make licensing assumptions.

The project uses Ultralytics software/model weights whose licensing must be reviewed before commercial or production deployment.

For this demo:

- preserve upstream license notices/attribution where required;
- mention the licensing consideration in the README;
- do not claim that the repository is cleared for proprietary production deployment;
- do not let licensing uncertainty block a local technical demo unless instructed by the human lead.

---

## Repository Structure

Prefer a small structure such as:

```text
.
├── AGENTS.md
├── README.md
├── pyproject.toml
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── detector.py
│   ├── config.py
│   ├── schemas.py
│   └── static/
│       ├── index.html
│       ├── app.js
│       └── style.css
├── scripts/
│   └── benchmark.py
└── tests/
    ├── test_api.py
    ├── test_detector.py
    └── fixtures/
```

This is guidance, not a requirement to create empty files.

Keep the structure smaller if a file has no real responsibility.

---

## Dependency Management

Prefer `pyproject.toml`.

Pin or constrain dependencies enough to make the demo reproducible without freezing every transitive dependency unnecessarily.

Do not install project dependencies globally when a virtual environment can be used.

Recommended local setup pattern:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The coding agent may install missing packages inside the approved execution environment without asking the human to act as a terminal operator.

Document any OS packages installed.

Do not modify unrelated host configuration.

---

## Model Files

Do not commit downloaded YOLO weight files or exported runtime artifacts to Git unless explicitly requested.

The application may let Ultralytics download the configured pretrained model on first use.

Because the demo may occur without reliable internet access, document a pre-demo warm-up/prefetch step and run it during verification when network access is available.

A demo-readiness check should prove that the model is already present or can be loaded before the presentation begins.

---

## Testing

Tests are evidence, not decoration.

### Unit tests

Cover logic that does not require real model inference where practical:

- configuration parsing;
- response serialization;
- bounding-box conversion;
- validation of detector result mapping;
- error handling.

### API tests

Test at least:

- `/api/health`;
- `/api/info`;
- valid `/api/detect`;
- malformed image;
- empty body;
- unsupported media type;
- detector failure mapped to a controlled error.

Use a fake/stub detector for most API tests so the test suite remains fast and deterministic.

### Real-model smoke test

Provide at least one clearly marked real-model smoke test or verification command that:

- loads the configured pretrained YOLO model;
- runs one known image through it;
- proves inference completes;
- reports detections or a valid empty result.

Do not make the normal unit suite download model weights unexpectedly.

### Browser verification

For the first release, a documented manual browser smoke test is acceptable.

The smoke test must cover:

1. page loads;
2. camera permission works;
3. video preview appears;
4. detection starts;
5. requests reach backend;
6. boxes/labels appear for detectable objects;
7. stopping detection stops requests;
8. errors are visible if backend disappears.

Do not add Playwright/Selenium solely to automate a one-day demo unless browser regressions justify it.

### Static quality checks

Use a formatter/linter only if configured in the repository.

Do not spend substantial demo time constructing a complex lint stack.

### Test reporting

The final report must distinguish:

- PASS;
- FAIL;
- SKIPPED;
- BLOCKED;
- NOT RUN.

Never describe a skipped or unexecuted test as passing.

---

## Demo Readiness

Before calling the demo ready, verify all of the following on the actual target environment:

- virtual environment can be created;
- dependencies install;
- model weights are available;
- server starts from documented command;
- root page opens;
- camera permission works;
- detection works;
- bounding boxes align with displayed video;
- stop/start works repeatedly;
- browser does not build an ever-growing request backlog;
- server survives malformed input;
- measured latency/FPS is recorded;
- restart from a clean terminal works;
- README contains the exact demo procedure.

Create a short **Demo Checklist** in the README.

---

## Logging

Use ordinary Python logging.

Useful events:

- startup;
- configured model/backend;
- model load duration;
- service listening address;
- inference error;
- image decode error;
- optionally periodic latency statistics.

Avoid per-frame INFO logging if it floods the terminal.

Never log binary image data.

---

## Error Handling

Prefer explicit errors over silent fallback.

Examples:

- If the requested model cannot load, startup should fail clearly.
- If OpenVINO is configured but unavailable, do not silently use a different backend unless the fallback is explicitly documented and visible.
- If the browser cannot access a camera, display the browser error.
- If inference fails, return a controlled API error and show it in the UI.

Do not catch broad exceptions merely to return success-like responses.

---

## Coding Style

Priorities:

1. readability;
2. correctness;
3. small functions with clear ownership;
4. type hints for public Python interfaces;
5. minimal abstraction;
6. explicit configuration;
7. useful error messages.

Avoid:

- premature design patterns;
- inheritance hierarchies for one implementation;
- generic service containers;
- dependency-injection frameworks;
- generated boilerplate;
- large utility modules with unrelated helpers.

Comments should explain non-obvious reasoning, not restate code.

---

## Workflow

Before coding:

1. Read this file.
2. Inspect the current repository and working tree.
3. Report any pre-existing uncommitted changes before modifying them.
4. Identify the exact work-order scope.
5. Preserve unrelated user changes.

During coding:

1. Work only within the requested slice.
2. Install routine dependencies inside the approved environment when needed.
3. Run relevant tests after meaningful changes.
4. Do not ask the human to perform routine package installation or command execution that the agent can safely perform itself.
5. Do not expand scope merely because additional features are easy.

Version control:

- Work on a feature branch when Git is available.
- Do not commit directly to protected/main branches.
- Commit only files related to the task.
- Use concise, descriptive commit messages.
- Do not rewrite unrelated history.
- Do not merge your own pull request.
- If no remote repository is configured, do not invent one; produce local commits and report that PR creation is blocked by the missing remote.

---

## Forbidden Actions

Do not:

- train a YOLO model;
- download large training datasets;
- commit model weights;
- add a GPU requirement;
- silently switch model family;
- add cloud inference;
- send camera frames off-machine;
- persist camera frames without an explicit requirement;
- introduce user tracking or analytics;
- add a database for the first release;
- add authentication for the first local demo;
- add a frontend framework;
- add a message broker;
- add Docker/Kubernetes merely for appearance;
- bind publicly by default;
- disable TLS/security settings on unrelated systems;
- modify firewall rules without an explicit work order;
- access production services;
- use real credentials;
- fabricate benchmark results;
- claim tests passed when they were skipped or not run;
- claim production readiness;
- hide known limitations;
- rewrite unrelated files;
- delete user data or unrelated repository content.

---

## Documentation Contract

`README.md` is part of the deliverable.

It should contain:

- project purpose;
- architecture summary;
- supported environment;
- Python setup;
- dependency installation;
- model warm-up/prefetch;
- start command;
- browser URL;
- camera/detection usage;
- API endpoint summary;
- configuration;
- CPU performance notes;
- benchmark command;
- demo checklist;
- troubleshooting;
- WSL notes;
- current limitations;
- Ultralytics licensing note.

Commands in the README must be tested or explicitly marked unverified.

Do not write aspirational features as though they exist.

---

## Definition of Done

A work item is done only when all of the following applicable conditions hold:

- requested behavior is implemented;
- code runs in the intended Linux/WSL environment;
- relevant automated tests pass;
- real-model smoke verification has been performed when the task affects inference;
- browser smoke verification has been performed when the task affects the UI;
- documentation matches actual behavior;
- no unrelated changes are included;
- performance claims are backed by measurements;
- known limitations are reported;
- the final report contains reproducible evidence.

A visually convincing page without a verified inference path is not done.

A passing mocked test suite without a real-model smoke test is not enough to call inference done.

A working detector without the browser round trip is not enough to call the demo done.

---

## Final Report

Every execution task must end with a compact report using this structure:

```text
Summary
- What was implemented.

Scope
- Files/components changed.
- Anything intentionally not changed.

Environment
- OS / WSL distribution:
- Python:
- CPU:
- Model:
- Backend:
- Input size:

Verification
- Commands run:
- Unit/API tests:
- Real-model smoke test:
- Browser smoke test:
- Benchmark:

Measured performance
- Model:
- Image size:
- Inference latency:
- End-to-end latency:
- Approximate FPS:
- Measurement method:

Dependencies / setup changes
- Python packages installed:
- OS packages installed:
- Generated/downloaded artifacts:

Version control
- Branch:
- Commit(s):
- PR:
- Working tree status:

Risks / limitations
- Known issues.
- Unverified behavior.
- Licensing/deployment considerations.

Recommended next step
- One narrow follow-up task only.
```

Do not omit failures or blocked checks.

---

## Current Strategic Defaults

Unless a later approved work order changes them:

```text
Product: local browser-based object detection demo
Backend: Python + FastAPI + Uvicorn
Frontend: plain HTML/CSS/JavaScript
Capture: getUserMedia + canvas
Transport: HTTP request per selected frame
API payload: encoded JPEG/PNG image
API result: JSON detections
Detector: Ultralytics YOLO26 detection
Default model: yolo26n.pt
Training: forbidden / not required
Classes: pretrained COCO classes
Device: CPU
Initial image size: 640
Initial requested browser detection rate: ~3 FPS
Frame scheduling: max one inference request in flight; skip stale frames
Persistence: none
Database: none
Authentication: none for localhost demo
Default bind address: 127.0.0.1
Optimization follow-up: measure first; OpenVINO optional
```

If any of these defaults proves technically impossible in the target environment, do not improvise a major redesign. Report the evidence and propose the smallest viable change.

---

## Strategic Principle

The coding agent is implementation labor, not product authority.

When uncertain:

1. preserve the smallest working architecture;
2. measure rather than guess;
3. keep the demo path working;
4. report uncertainty explicitly;
5. ask for a strategic decision only when the decision changes product scope, architecture, safety, licensing posture, or release criteria.
