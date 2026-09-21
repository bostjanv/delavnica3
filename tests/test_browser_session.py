"""Browser Stop/Start race regression test (Finding 2), driven with Playwright.

Loads the real page + app.js in headless Chromium with a synthetic camera
(``--use-fake-device-for-media-stream``) against a tiny local HTTP server that
stands in for the FastAPI service. The mock can hold a /api/detect response
until the test releases it, which makes the Stop-during-in-flight race
deterministic. No YOLO model and no project runtime dependency on Playwright
is required: the test is skipped when Playwright is not installed.

It proves that a response from a stopped detection session cannot:
  * flip the status back to "Detecting",
  * update the stats line,
  * draw boxes on the overlay,
and that a freshly started session works normally afterwards.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")

from playwright.sync_api import sync_playwright  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parents[1] / "app" / "static"

INFO = {
    "model": "fake.pt",
    "device": "cpu",
    "imgsz": 640,
    "conf": 0.25,
    "demo_fps": 20,  # 50 ms tick so the test moves quickly
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
                    hold = server.hold_release
                if hold is not None:
                    server.held.set()  # request fully received -> in flight
                    hold.wait(timeout=15)
                    with server.lock:
                        server.hold_release = None
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

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def begin_hold(self) -> None:
        with self.lock:
            self.hold_release = threading.Event()

    def release_hold(self) -> None:
        with self.lock:
            if self.hold_release is not None:
                self.hold_release.set()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def _overlay_has_content(page) -> bool:
    return page.evaluate(
        """() => {
            const c = document.getElementById("overlay");
            if (!c.width || !c.height) return false;
            const data = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
            for (let i = 3; i < data.length; i += 4) if (data[i] !== 0) return true;
            return false;
        }"""
    )


def test_stale_response_cannot_mutate_session():
    server = MockServer(STATIC_DIR)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=True,
                args=[
                    "--use-fake-device-for-media-stream",
                    "--use-fake-ui-for-media-stream",
                    "--autoplay-policy=no-user-gesture-required",
                ],
            )
            page = browser.new_page()
            # localhost origin: getUserMedia requires a secure context.
            page.goto(f"http://localhost:{server.port}/")

            # Backend check succeeds via /api/info.
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
            assert not _overlay_has_content(page), "stale response drew boxes"
            assert server.count == held_count, "requests were sent after Stop"

            # --- Session 2: immediate restart must work and be unaffected ---
            page.click("#start-btn")
            page.wait_for_function("statusEl.textContent === 'Detecting'", timeout=15_000)
            page.wait_for_function("statsEl.textContent.includes('round-trip')", timeout=15_000)
            assert _overlay_has_content(page), "current-session boxes were not drawn"
            assert server.count > held_count, "new session did not send requests"

            # Stop again: activity must stop cleanly (no backlog).
            page.click("#stop-btn")
            page.wait_for_function("statusEl.textContent === 'Stopped'", timeout=5_000)
            count_after_stop2 = server.count
            page.wait_for_timeout(500)
            assert server.count == count_after_stop2, "requests accumulated after Stop"

            browser.close()
    finally:
        server.close()
