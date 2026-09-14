"""Screenshot a running web UI with headless Chrome, driven over CDP.

    python tools/screenshot.py http://localhost:8601 docs/images/ui.png [--settle 8]

Why not ``chrome --screenshot``
-------------------------------
Chrome's one-shot ``--screenshot`` flag fires as soon as the load event does.
Streamlit renders its page over a websocket *after* that, so the capture is a
skeleton loader every time. ``--virtual-time-budget`` does not help either: it
fast-forwards timers, not a real network round trip to a real server.

Driving the DevTools Protocol directly lets us wait for the load event, then
wait a further settle period for the websocket render, then capture. It needs no
extra dependency — ``websockets`` already ships with Streamlit.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]


def find_chrome() -> str:
    """Locate a Chrome or Edge binary, or raise with a useful message."""
    for name in ("google-chrome", "chromium", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    for path in CHROME_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No Chrome or Edge binary found. Screenshots are optional; the "
        "comparison figures in docs/images/ are produced by run.py without a browser."
    )


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_for_devtools(port: int, timeout: float = 25.0) -> str:
    """Poll the DevTools HTTP endpoint until a page target appears."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=2) as r:
                targets = json.loads(r.read().decode())
            for t in targets:
                if t.get("type") == "page" and t.get("webSocketDebuggerUrl"):
                    return t["webSocketDebuggerUrl"]
        except Exception as e:  # noqa: BLE001 - chrome is simply not up yet
            last = e
        time.sleep(0.4)
    raise TimeoutError(f"Chrome DevTools never became available on port {port} ({last})")


async def capture(
    ws_url: str, url: str, out_path: Path, width: int, height: int, settle: float
) -> None:
    import websockets

    async with websockets.connect(ws_url, max_size=200 * 1024 * 1024) as ws:
        msg_id = 0

        async def send(method: str, params: dict | None = None) -> dict:
            nonlocal msg_id
            msg_id += 1
            await ws.send(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
            while True:
                reply = json.loads(await ws.recv())
                if reply.get("id") == msg_id:
                    return reply

        # No Emulation.setDeviceMetricsOverride here. Combining an override with
        # an explicit capture clip makes Chrome interpret the clip in the
        # pre-override coordinate space, which shifts the image left and crops a
        # sidebar off the edge. The --window-size flag already sets the viewport.
        await send("Page.enable")
        await send("Page.navigate", {"url": url})

        # wait for the load event, then for the app to actually draw
        try:
            deadline = time.time() + 30
            while time.time() < deadline:
                event = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                if event.get("method") == "Page.loadEventFired":
                    break
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001
            pass

        await asyncio.sleep(settle)

        # Scroll back to the document origin. Streamlit lays its sidebar out to
        # the left of the main pane, and a page that has been scrolled at all
        # crops the sidebar out of the capture.
        await send("Runtime.enable")
        await send(
            "Runtime.evaluate",
            {"expression": "window.scrollTo(0, 0); document.documentElement.scrollLeft = 0;"},
        )
        await asyncio.sleep(0.6)

        # Measure the document in CSS pixels from the page itself, which is the
        # same coordinate space the clip uses.
        measured = await send(
            "Runtime.evaluate",
            {
                "expression": (
                    "JSON.stringify({w: document.documentElement.scrollWidth,"
                    " h: document.documentElement.scrollHeight})"
                ),
                "returnByValue": True,
            },
        )
        try:
            size = json.loads(measured["result"]["result"]["value"])
        except (KeyError, TypeError, json.JSONDecodeError):
            size = {"w": width, "h": height}

        clip = {
            "x": 0,
            "y": 0,
            "width": int(size.get("w") or width),
            "height": int(size.get("h") or height),
            "scale": 1,
        }

        shot = await send(
            "Page.captureScreenshot",
            {"format": "png", "captureBeyondViewport": True, "clip": clip},
        )
        data = shot.get("result", {}).get("data")
        if not data:
            raise RuntimeError(f"Chrome returned no image data: {shot}")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(base64.b64decode(data))


def screenshot(
    url: str,
    out_path: str | Path,
    width: int = 1600,
    height: int = 1200,
    settle: float = 8.0,
) -> Path:
    """Capture ``url`` to ``out_path``. Returns the path written."""
    out_path = Path(out_path)
    chrome = find_chrome()
    port = free_port()
    profile = tempfile.mkdtemp(prefix="cv-shot-")

    proc = subprocess.Popen(
        [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            "--hide-scrollbars",
            "--force-color-profile=srgb",
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile}",
            f"--window-size={width},{height}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        ws_url = wait_for_devtools(port)
        asyncio.run(capture(ws_url, url, out_path, width, height, settle))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()
        shutil.rmtree(profile, ignore_errors=True)

    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1200)
    ap.add_argument("--settle", type=float, default=8.0, help="seconds to wait after load")
    args = ap.parse_args()

    path = screenshot(args.url, args.out, args.width, args.height, args.settle)
    size = path.stat().st_size
    print(f"wrote {path} ({size / 1024:.0f} KB)")
    if size < 30_000:
        print("WARNING: the file is small; the page may not have finished rendering.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
