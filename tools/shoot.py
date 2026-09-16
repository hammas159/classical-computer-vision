"""Launch a project's Streamlit app, photograph it, and shut it down.

    python tools/shoot.py 01_document_scanner \
        00_real_photo \
        "01_pixel_matrix:Pixel matrix" \
        "02_comparison:Comparison matrix"

Each argument after the project is ``OUTPUT_NAME`` or ``OUTPUT_NAME:TAB LABEL``.
Shots are written to ``projects/<project>/results/screenshots/<name>.png``.

Why this exists
---------------
Three things went wrong often enough by hand to be worth automating:

1. **A stale server answers.** Streamlit exits with "Port N is not available"
   when something already holds the port — and the *old* server keeps serving,
   so the screenshot silently shows the previous build. This picks a free port
   and fails loudly if the server does not come up on it.
2. **The theme comes from the working directory.** Streamlit reads
   ``.streamlit/config.toml`` relative to the cwd, so launching from the repo
   root gives every project the stock theme. This always launches from inside
   the project folder.
3. **Servers leaked.** Each manual run left a server running; a dozen had
   accumulated. This tears its own server down in a ``finally``.

It also refuses to write two screenshots that are near-identical, which is the
mistake that put five copies of one page in three READMEs.
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.screenshot import screenshot  # noqa: E402

#: Two captures whose first N rows agree this closely are the same picture with
#: a different heading. 0.35 = 35% of the taller image.
DUPLICATE_PREFIX_LIMIT = 0.35


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_for_app(port: int, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    last: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}", timeout=3) as r:
                if r.status == 200:
                    return
        except Exception as e:  # noqa: BLE001 - the server is simply not up yet
            last = e
        time.sleep(0.5)
    raise TimeoutError(f"the app never answered on port {port} ({last})")


def identical_prefix(a: Path, b: Path) -> float:
    """Fraction of the taller image over which two captures agree exactly.

    Row-by-row from the top, because that is how the duplication actually
    appeared: the same title, controls and pipeline strip above the fold, and
    only the panel below it different.
    """
    import cv2
    import numpy as np

    ia, ib = cv2.imread(str(a)), cv2.imread(str(b))
    if ia is None or ib is None:
        return 0.0
    h = min(ia.shape[0], ib.shape[0])
    w = min(ia.shape[1], ib.shape[1])
    same = 0
    for r in range(h):
        if float(np.abs(ia[r, :w].astype(int) - ib[r, :w].astype(int)).mean()) > 1.0:
            break
        same += 1
    return same / max(ia.shape[0], ib.shape[0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("project", help="folder name under projects/, e.g. 01_document_scanner")
    ap.add_argument("shots", nargs="+", metavar="NAME[:TAB]")
    ap.add_argument("--settle", type=float, default=16.0)
    ap.add_argument("--click-wait", type=float, default=5.0)
    ap.add_argument("--width", type=int, default=1600)
    args = ap.parse_args()

    proj = REPO / "projects" / args.project
    if not (proj / "ui" / "app.py").exists():
        print(f"no ui/app.py under {proj}", file=sys.stderr)
        return 1
    out_dir = proj / "results" / "screenshots"
    out_dir.mkdir(parents=True, exist_ok=True)

    port = free_port()
    python = REPO / ".venv" / "Scripts" / "python.exe"
    if not python.exists():
        python = Path(sys.executable)

    # cwd is the project folder, so Streamlit picks up its own
    # .streamlit/config.toml and therefore its own palette.
    server = subprocess.Popen(
        [str(python), "-m", "streamlit", "run", "ui/app.py", f"--server.port={port}"],
        cwd=str(proj),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    written: list[Path] = []
    try:
        wait_for_app(port)
        url = f"http://localhost:{port}"
        for spec in args.shots:
            name, _, tab = spec.partition(":")
            out = out_dir / f"{name}.png"
            screenshot(
                url,
                out,
                width=args.width,
                settle=args.settle,
                clicks=[tab] if tab else None,
                click_wait=args.click_wait,
            )
            size = out.stat().st_size
            print(f"  {out.name:34s} {size / 1024:6.0f} KB" + (f"  [tab: {tab}]" if tab else ""))
            if size < 30_000:
                print(f"  WARNING: {out.name} is small; the page may not have rendered", file=sys.stderr)
            written.append(out)
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            server.kill()

    # The check that projects 01-04 needed and did not have.
    problems = 0
    for i in range(len(written)):
        for j in range(i + 1, len(written)):
            frac = identical_prefix(written[i], written[j])
            if frac > DUPLICATE_PREFIX_LIMIT:
                print(
                    f"  DUPLICATE: {written[i].name} and {written[j].name} agree over "
                    f"{frac:.0%} of the taller image — keep one",
                    file=sys.stderr,
                )
                problems += 1
    if problems:
        print(f"\n{problems} near-duplicate pair(s). Drop the redundant capture.", file=sys.stderr)
        return 2
    print(f"\n{len(written)} shot(s), all visually distinct.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
