// CPU YOLO browser demo: camera capture, one-in-flight HTTP inference, overlay.
// No framework; plain DOM + canvas + fetch.

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");
const stage = document.getElementById("stage");
const placeholder = document.getElementById("cam-placeholder");
const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const statusEl = document.getElementById("status");
const errorEl = document.getElementById("error");
const statsEl = document.getElementById("stats");

const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d", { willReadFrequently: false });

const state = {
  running: false,
  inFlight: false, // hard limit: at most one inference request at a time
  stream: null,
  tickTimer: null,
  lastTickAt: 0,
  intervalMs: 1000 / 3, // updated from /api/info demo_fps
  roundTrips: [], // recent end-to-end ms, for the stats line
};

function setStatus(text) {
  statusEl.textContent = text;
}

function showError(text) {
  errorEl.hidden = !text;
  errorEl.textContent = text || "";
}

function updateStats(roundTripMs, inferenceMs) {
  state.roundTrips.push(roundTripMs);
  if (state.roundTrips.length > 20) state.roundTrips.shift();
  const mean = state.roundTrips.reduce((a, b) => a + b, 0) / state.roundTrips.length;
  const fps = 1000 / mean;
  statsEl.textContent =
    `round-trip ~${mean.toFixed(0)} ms · inference ${inferenceMs.toFixed(0)} ms · ` +
    `~${fps.toFixed(1)} detections/s (last ${state.roundTrips.length} frames)`;
}

// Draw boxes scaled from the submitted frame's pixel space to the overlay canvas.
function drawDetections(data) {
  if (!state.running) return; // stale response after Stop: ignore
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return;
  overlay.width = vw; // resetting width also clears the canvas
  overlay.height = vh;
  const sx = overlay.width / data.image_width;
  const sy = overlay.height / data.image_height;
  overlayCtx.lineWidth = 2;
  overlayCtx.font = "14px system-ui, sans-serif";
  for (const det of data.detections) {
    const x = det.x1 * sx;
    const y = det.y1 * sy;
    const w = (det.x2 - det.x1) * sx;
    const h = (det.y2 - det.y1) * sy;
    const hue = (det.class_id * 47) % 360;
    const color = `hsl(${hue}, 85%, 60%)`;
    overlayCtx.strokeStyle = color;
    overlayCtx.strokeRect(x, y, w, h);
    const label = `${det.class_name} ${Math.round(det.confidence * 100)}%`;
    const tw = overlayCtx.measureText(label).width;
    const ly = Math.max(0, y - 18);
    overlayCtx.fillStyle = color;
    overlayCtx.fillRect(x, ly, tw + 8, 18);
    overlayCtx.fillStyle = "#0b0f14";
    overlayCtx.fillText(label, x + 4, ly + 13);
  }
}

async function sendFrame() {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return;
  // Capture at the video's intrinsic pixel size so encoded frame == response coords.
  captureCanvas.width = vw;
  captureCanvas.height = vh;
  captureCtx.drawImage(video, 0, 0, vw, vh);

  const blob = await new Promise((resolve) =>
    captureCanvas.toBlob(resolve, "image/jpeg", 0.7)
  );
  if (!blob) {
    showError("Could not encode camera frame to JPEG.");
    return;
  }

  setStatus("Inference in flight…");
  const t0 = performance.now();
  try {
    const resp = await fetch("/api/detect", {
      method: "POST",
      headers: { "Content-Type": "image/jpeg" },
      body: blob,
    });
    if (!resp.ok) {
      let detail = `HTTP ${resp.status}`;
      try {
        const body = await resp.json();
        if (body && body.detail) detail += `: ${body.detail}`;
      } catch (_) {
        /* non-JSON error body */
      }
      throw new Error(detail);
    }
    const data = await resp.json();
    const roundTrip = performance.now() - t0;
    updateStats(roundTrip, data.inference_ms);
    drawDetections(data);
    setStatus("Detecting");
  } catch (err) {
    showError(err.message || String(err));
    setStatus("Error — see message below");
  }
}

// Tick scheduler: fires on the requested-FPS grid; skips the tick (no queueing)
// if the previous inference is still in flight.
function tick() {
  if (!state.running) return;
  if (state.inFlight) {
    scheduleTick();
    return;
  }
  state.lastTickAt = performance.now();
  state.inFlight = true;
  Promise.resolve(sendFrame()).finally(() => {
    state.inFlight = false;
    scheduleTick();
  });
}

function scheduleTick() {
  if (!state.running) return;
  clearTimeout(state.tickTimer);
  const elapsed = performance.now() - state.lastTickAt;
  state.tickTimer = setTimeout(tick, Math.max(0, state.intervalMs - elapsed));
}

async function ensureStream() {
  if (state.stream) return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("This browser does not support camera capture (getUserMedia).");
  }
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
  } catch (err) {
    if (err.name === "NotAllowedError") {
      throw new Error("Camera permission denied. Allow camera access in the browser and retry.");
    }
    if (err.name === "NotFoundError") {
      throw new Error("No camera found on this machine.");
    }
    throw new Error(`Camera error: ${err.name || err}`);
  }
  video.srcObject = state.stream;
  await video.play().catch(() => {});
  // Match the stage's aspect ratio to the real video so overlay == video exactly.
  if (video.videoWidth && video.videoHeight) {
    stage.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
    placeholder.style.display = "none";
  }
}

async function startDetection() {
  showError("");
  setStatus("Requesting camera…");
  try {
    await ensureStream();
  } catch (err) {
    showError(err.message);
    setStatus("Camera error");
    return;
  }
  state.running = true;
  state.lastTickAt = 0;
  startBtn.disabled = true;
  stopBtn.disabled = false;
  scheduleTick();
  setStatus("Detecting");
}

function stopDetection() {
  state.running = false;
  clearTimeout(state.tickTimer);
  if (state.stream) {
    for (const track of state.stream.getTracks()) track.stop();
    state.stream = null;
  }
  video.srcObject = null;
  overlay.width = 0; // clear
  startBtn.disabled = false;
  stopBtn.disabled = true;
  setStatus("Stopped");
}

async function checkBackend() {
  try {
    const resp = await fetch("/api/info");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const info = await resp.json();
    state.intervalMs = 1000 / (info.demo_fps || 3);
    setStatus(`Backend online — ${info.model} on ${info.device} (imgsz ${info.imgsz})`);
  } catch (err) {
    showError("Backend unreachable: is the service running?");
    setStatus("Backend offline");
  }
}

startBtn.addEventListener("click", startDetection);
stopBtn.addEventListener("click", stopDetection);
checkBackend();
