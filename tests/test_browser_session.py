"""Browser Stop/Start race regression tests (Finding 2 + PR#3 repair), Playwright.

Loads the real page + app.js in headless Chromium with a synthetic camera
(``--use-fake-device-for-media-stream``) against a tiny local HTTP server that
stands in for the FastAPI service. The mock can hold a /api/detect response
until the test releases it, which makes the in-flight-request race
deterministic. No YOLO model and no project runtime dependency on Playwright
is required: the test is skipped when Playwright is not installed.

Covered scenarios:

1. ``test_stale_response_cannot_mutate_session`` — a response from a stopped
   session cannot change status, update stats, draw boxes, or send further
   requests; an immediately restarted session works normally.

2. ``test_immediate_restart_while_request_outstanding`` — the exact race from
   the PR#3 repair: Stop, then *immediately* Start while the previous
   session's request is still outstanding. While it is held: no second
   request is sent, no zero-delay timer busy loop occurs, and the new
   session stays logically active. When the stale response finally lands, it
   must not touch the new session's UI, and the new session must resume
   normal inference on its own.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class QuietThreadingHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that stays silent on aborted connections.

    In-flight requests are legitimately aborted when the test closes the
    browser; the resulting BrokenPipe must not pollute test output.
    """

    def handle_error(self, request, client_address):
        pass

import pytest

pytest.importorskip("playwright.sync_api")

from playwright.sync_api import sync_playwright  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parents[1] / "app" / "static"

INFO = {
    "model": "fake.pt",
    "device": "cpu",
    "imgsz": 640,
    "conf": 0.25,
    "demo_fps": 20,  # 50 ms tick so the tests move quickly
    "backend": "fake",
    "max_upload_mb": 10,
}

DETECTION = {
    "class_id": 0,
    "class_name": "person",
    "confidence": 0.93,
    "x1": 160.0,
    "y1": 96.0,
    "x2": 480.0,
    "y2": 384.0,
}

CONTENT_TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}

# Arrow-function body (no wrapping braces) checking that the overlay canvas
# has any painted pixels.
OVERLAY_HAS_CONTENT_JS = """
    const c = document.getElementById('overlay');
    if (!c.width || !c.height) return false;
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    for (let i = 3; i < d.length; i += 4) if (d[i] !== 0) return true;
    return false;
"""


def _overlay_js() -> str:
    return f"() => {{ {OVERLAY_HAS_CONTENT_JS} }}"

# Instrumentation installed before app.js runs:
#  * __zeroTimers — count of setTimeout() calls with delay < 1 ms. The tick
#    scheduler is the only page code using setTimeout; a stale in-flight
#    request must not cause repeated zero-delay ticks (busy loop).
#  * __detectResponses — count of /api/detect responses received by the page,
#    so the test can prove a stale response was actually processed.
INSTRUMENT_JS = """
    window.__zeroTimers = 0;
    window.__detectResponses = 0;
    const _setTimeout = window.setTimeout;
    window.setTimeout = function (fn, delay, ...args) {
        if (delay === undefined || delay < 1) window.__zeroTimers += 1;
        return _setTimeout.call(window, fn, delay, ...args);
    };
    const _fetch = window.fetch;
    window.fetch = function (url, opts) {
        if (String(url).includes('/api/detect')) {
            return _fetch.apply(this, arguments).then((r) => {
                window.__detectResponses += 1;
                return r;
            });
        }
        return _fetch.apply(this, arguments);
    };
"""


class MockServer:
    """Threading HTTP server standing in for the FastAPI demo service.

    ``begin_hold()`` makes the next /api/detect request wait for
    ``release_hold()`` before answering — i.e. the test controls the moment a
    response lands while a request is provably in flight.
    """

    def __init__(self, static_dir: Path) -> None:
        self.static_dir = static_dir
        self.lock = threading.Lock()
        self.count = 0
        self.hold_release: threading.Event | None = None
        self.held = threading.Event()
        # Hold events that a handler is currently blocked on; release_hold()
        # sets all of them (normally exactly one).
        self.waiting: set[threading.Event] = set()
        self.auto_delay_s = 0.03

        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep test output clean
                pass

            def _send(self, status: int, body: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/api/info":
                    self._send(200, json.dumps(INFO).encode(), "application/json")
                elif self.path == "/":
                    self._send(200, (static_dir / "index.html").read_bytes(), "text/html")
                elif self.path.startswith("/static/"):
                    p = static_dir / self.path.rsplit("/", 1)[-1]
                    if p.exists():
                        self._send(
                            200,
                            p.read_bytes(),
                            CONTENT_TYPES.get(p.suffix, "application/octet-stream"),
                        )
                    else:
                        self._send(404, b"not found", "text/plain")
                else:
                    self._send(404, b"not found", "text/plain")

            def do_POST(self):
                if self.path != "/api/detect":
                    self._send(404, b"not found", "text/plain")
                    return
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)  # drain the (JPEG) body
                with server.lock:
                    server.count += 1
                    # Consume the armed hold atomically at arrival so a
                    # pre-armed next hold survives this request's release.
                    hold = server.hold_release
                    server.hold_release = None
                if hold is not None:
                    server.held.set()  # request fully received -> in flight
                    with server.lock:
                        server.waiting.add(hold)
                    try:
                        hold.wait(timeout=15)
                    finally:
                        with server.lock:
                            server.waiting.discard(hold)
                else:
                    time.sleep(server.auto_delay_s)
                payload = {
                    "image_width": 640,
                    "image_height": 480,
                    "inference_ms": 42.0,
                    "model": "fake.pt",
                    "detections": [DETECTION],
                }
                self._send(200, json.dumps(payload).encode(), "application/json")

        self.httpd = QuietThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def begin_hold(self) -> None:
        """Arm holding for the next /api/detect request.

        Resets the held marker from any earlier hold cycle.
        """
        with self.lock:
            self.hold_release = threading.Event()
            self.held.clear()

    def wait_held(self, timeout: float = 15) -> bool:
        """True once the armed request has fully arrived and is being held."""
        return self.held.wait(timeout)

    def release_hold(self) -> None:
        """Release every request currently held (normally exactly one).

        This targets the events handlers are actually blocked on, not the
        currently armed hold, so pre-arming the next hold before releasing
        the previous request is safe.
        """
        with self.lock:
            for ev in list(self.waiting):
                ev.set()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def _launch():
    """Headless Chromium with a synthetic camera and the page instrumentation.

    Returns (playwright, browser, page); the caller must close the browser
    and stop playwright in a finally block.
    """
    pw = sync_playwright().start()
    browser = pw.chromium.launch(
        headless=True,
        args=[
            "--use-fake-device-for-media-stream",
            "--use-fake-ui-for-media-stream",
            "--autoplay-policy=no-user-gesture-required",
        ],
    )
    page = browser.new_page()
    page.add_init_script(INSTRUMENT_JS)
    return pw, browser, page


def test_stale_response_cannot_mutate_session():
    server = MockServer(STATIC_DIR)
    pw, browser, page = _launch()
    try:
        # localhost origin: getUserMedia requires a secure context.
        page.goto(f"http://localhost:{server.port}/")
        page.wait_for_function("statusEl.textContent.includes('Backend online')", timeout=10_000)

        # --- Session 1: start, then stop while a request is in flight ---
        page.click("#start-btn")
        page.wait_for_function("video.videoWidth > 0", timeout=15_000)

        server.begin_hold()
        assert server.held.wait(timeout=15), "no /api/detect request reached the server"
        held_count = server.count

        page.click("#stop-btn")
        page.wait_for_function("statusEl.textContent === 'Stopped'", timeout=5_000)
        stats_before_stale = page.inner_text("#stats")

        # The stale response now completes. It must change nothing.
        server.release_hold()
        page.wait_for_timeout(500)  # several tick intervals: enough for any stray work

        assert page.inner_text("#status") == "Stopped", "stale response changed the status"
        assert page.inner_text("#stats") == stats_before_stale, "stale response updated stats"
        assert not page.evaluate(_overlay_js()), "stale response drew boxes"
        assert server.count == held_count, "requests were sent after Stop"

        # --- Session 2: immediate restart must work and be unaffected ---
        page.click("#start-btn")
        page.wait_for_function("statusEl.textContent === 'Detecting'", timeout=15_000)
        page.wait_for_function("statsEl.textContent.includes('round-trip')", timeout=15_000)
        assert page.evaluate(_overlay_js()), "current-session boxes were not drawn"
        assert server.count > held_count, "new session did not send requests"

        # Stop again: activity must stop cleanly (no backlog).
        page.click("#stop-btn")
        page.wait_for_function("statusEl.textContent === 'Stopped'", timeout=5_000)
        count_after_stop2 = server.count
        page.wait_for_timeout(500)
        assert server.count == count_after_stop2, "requests accumulated after Stop"
    finally:
        browser.close()
        pw.stop()
        server.close()


def test_immediate_restart_while_request_outstanding():
    """Stop -> immediate Start while the old request is still outstanding.

    Fully event-driven sequence (no timing-only assertions for state changes):

        session 1 request A held in flight
        -> Stop session 1
        -> immediately Start session 2 (A still outstanding)
        -> while A is held: no second request, no zero-delay timer churn,
           session 2 stays logically active
        -> release A, and hold the next request B (armed before release)
        -> stale A response is processed while B is in flight: it must not
           touch status/stats/overlay/errors
        -> B is released: session 2 works normally (boxes, stats, status)
        -> final Stop: no further requests, overlay cleared
    """
    server = MockServer(STATIC_DIR)
    pw, browser, page = _launch()
    try:
        page.goto(f"http://localhost:{server.port}/")
        page.wait_for_function("statusEl.textContent.includes('Backend online')", timeout=10_000)

        # Hold from the very start so request A is the first (and only)
        # request: no response has completed, so stats stay "—".
        server.begin_hold()

        # --- Session 1: start; request A becomes in flight and is held ---
        page.click("#start-btn")
        page.wait_for_function("video.videoWidth > 0", timeout=15_000)
        assert server.wait_held(), "request A never reached the server"
        assert server.count == 1, "expected request A to be the only request so far"

        # --- Stop session 1, then IMMEDIATELY start session 2 ---
        page.click("#stop-btn")
        page.wait_for_function("statusEl.textContent === 'Stopped'", timeout=5_000)
        zero_timers_before = page.evaluate("window.__zeroTimers")

        page.click("#start-btn")
        page.wait_for_function("statusEl.textContent === 'Detecting'", timeout=15_000)
        state = page.evaluate("({running: state.running, generation: state.generation})")
        assert state["running"] is True and state["generation"] >= 2, "session 2 is not active"

        # Request A is still outstanding. Give a buggy scheduler time to show
        # its behavior (repeated zero-delay ticks), then check.
        page.wait_for_timeout(1000)
        assert server.count == 1, "a second inference request was sent concurrently"
        assert page.inner_text("#status") == "Detecting", "new session reverted to Stopped"
        zero_timers_delta = page.evaluate("window.__zeroTimers") - zero_timers_before
        # The tick scheduler is the only page setTimeout user; while the stale
        # request is outstanding there must be no (near-)zero-delay timer
        # churn. A buggy scheduler re-schedules a 0 ms tick every task
        # iteration (hundreds per second).
        assert zero_timers_delta <= 10, (
            f"zero-delay timer churn while stale request outstanding: {zero_timers_delta}"
        )

        # Arm a hold for the next request B before releasing A, so B (sent by
        # session 2 the moment A finishes) is deterministically in flight
        # while we verify the stale-response invariants.
        server.begin_hold()
        server.release_hold()  # release A

        # The stale session-1 response arrives and is processed by the page.
        page.wait_for_function("window.__detectResponses === 1", timeout=10_000)
        assert server.wait_held(timeout=10), "session 2 did not resume sending after the stale request finished"
        assert server.count == 2, "expected exactly one new request (B) from session 2"

        # Stale response A was processed, session 2's own request B is in
        # flight: the stale response must not have mutated any session-2 UI.
        assert page.inner_text("#status") == "Inference in flight…", (
            "stale response changed the status (session 2 must stay active)"
        )
        assert page.inner_text("#stats") == "—", "stale response updated stats"
        assert not page.evaluate(_overlay_js()), "stale response drew boxes"
        assert page.evaluate("errorEl.hidden") is True, "stale response displayed an error"

        # --- Release B: session 2 now works normally ---
        server.release_hold()
        page.wait_for_function("statsEl.textContent.includes('round-trip')", timeout=10_000)
        assert page.evaluate(_overlay_js()), "session 2 boxes were not drawn"
        # Status alternates between "Detecting" and "Inference in flight…" on
        # the 50 ms tick grid; both are the healthy active-session states.
        assert page.inner_text("#status") in ("Detecting", "Inference in flight…")

        # --- Final stop: clean shutdown, no backlog ---
        page.click("#stop-btn")
        page.wait_for_function("statusEl.textContent === 'Stopped'", timeout=5_000)
        count_after_stop = server.count
        page.wait_for_timeout(500)
        assert server.count == count_after_stop, "requests accumulated after Stop"
        assert not page.evaluate(_overlay_js()), "overlay was not cleared on Stop"
        assert page.inner_text("#status") == "Stopped"
    finally:
        browser.close()
        pw.stop()
        server.close()
