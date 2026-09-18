"""Download the non-photograph data the remaining projects need.

    python tools/fetch_assets.py            # everything not already cached
    python tools/fetch_assets.py --set video

`tools/fetch_images.py` handles the photograph pools. This handles everything
else: a video clip, a calibration series, a wide-baseline pair with a published
homography. Those are the assets that decide whether a project can be built at
all, so what is reachable is recorded here rather than discovered again later.

Only two hosts answer from this machine -- ``raw.githubusercontent.com`` and
``sipi.usc.edu`` -- which rules out most standard datasets and is why several
projects below synthesise their ground truth from a real image rather than
downloading an annotated one. Where that happens it is stated in the project's
own README, because "generated from a photograph" and "annotated by a person"
are different claims.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CACHE = Path.home() / ".cache" / "classical-cv-images" / "assets"
OPENCV = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/"
EXTRA = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/"

#: Three more hosts on the same CDN, found by asking the GitHub API what a
#: repository actually contains rather than guessing at filenames. Everything
#: here is a real photograph taken by somebody else for the same purpose this
#: repository needs it for.
LANES = "https://raw.githubusercontent.com/udacity/CarND-LaneLines-P1/master/test_images/"
LANES2 = "https://raw.githubusercontent.com/udacity/CarND-Advanced-Lane-Lines/master/test_images/"
PLATES = "https://raw.githubusercontent.com/openalpr/benchmarks/master/endtoend/"

#: The opencv repository itself, for data files that are not in the wheel.
OPENCV_REPO = "https://raw.githubusercontent.com/opencv/opencv/4.x/"

#: The HGR1 hand-gesture subset, for project 55.
HGR1 = ("https://raw.githubusercontent.com/"
        "lds217/Hand-Gesture-Recognition-using-small-HGR1-dataset/main/")

USER_AGENT = "Mozilla/5.0 (classical-computer-vision asset fetch)"
RETRIES = 5
TIMEOUT = 240

#: Measured on this machine: ``raw.githubusercontent.com`` answers, but at a few
#: kilobytes a second on a bad run -- 28 KB took 12 s. A single 8 MB file
#: therefore cannot be fetched in one request, and a plain retry loop throws away
#: everything it downloaded before the timeout. Anything above this size is
#: fetched in ranged chunks and resumed from a ``.part`` file instead.
RESUME_ABOVE = 1 << 20
CHUNK = 256 * 1024

#: What each still-unbuilt project needs, and where it comes from. Grouped by
#: asset rather than by project because several projects share a source -- the
#: one video clip serves tracking, background subtraction and the alarm project.
#: Ordered smallest-first on purpose. The link to this host runs at a few
#: kilobytes a second on a bad day, and putting the 8 MB video first meant
#: nothing at all was on disk after half an hour. The three small sets unblock
#: seven projects between them and finish in seconds.
SETS: dict[str, dict[str, str]] = {
    # Two synchronised series of a chessboard at different poses. The board's
    # geometry is known exactly, which is what makes reprojection error a real
    # number rather than a self-report. Projects 35 and 46.
    "calibration": {
        **{f"left{i:02d}.jpg": OPENCV + f"left{i:02d}.jpg" for i in range(1, 15)},
        **{f"right{i:02d}.jpg": OPENCV + f"right{i:02d}.jpg" for i in range(1, 15)},
    },
    # The Oxford graffiti sequence: six views of one wall with the true
    # homography between each pair published alongside. Project 38 needs exactly
    # this -- a panorama scored against a known transform rather than by eye.
    "graf": {
        **{f"graf{i}.png": OPENCV + f"graf{i}.png" for i in range(1, 7)},
        **{f"H1to{i}p": OPENCV + f"H1to{i}p" for i in (2, 3, 4, 5, 6)},
    },
    # Odds and ends each used by one project: a plate photographed in motion,
    # a blob field, a textured still life.
    "misc": {
        "licenseplate_motion.jpg": OPENCV + "licenseplate_motion.jpg",
        "detect_blob.png": OPENCV + "detect_blob.png",
        "stuff.jpg": OPENCV + "stuff.jpg",
        "blox.jpg": OPENCV + "blox.jpg",
        "pic1.png": OPENCV + "pic1.png",
        "pic3.png": OPENCV + "pic3.png",
        "board.jpg": OPENCV + "board.jpg",
        "baboon.jpg": OPENCV + "baboon.jpg",
    },
    # A 768x576 clip of people crossing a plaza: a static camera, real moving
    # objects, real shadows. Projects 29, 30 and 55. 8 MB, and the only file
    # here that needs the resumable path.
    "video": {
        "vtest.avi": OPENCV + "vtest.avi",
    },
    # Fourteen dashcam photographs of real roads with painted lane markings,
    # from the two Udacity self-driving course repositories. Project 06. Six are
    # 960x540 and eight are 1280x720, which is useful rather than annoying: a
    # region of interest expressed in pixels has to become one expressed in
    # fractions of the frame, and that is the first thing a lane finder gets
    # wrong.
    "lanes": {
        **{f"lane_{n}": LANES + n for n in (
            "solidWhiteCurve.jpg", "solidWhiteRight.jpg", "solidYellowCurve.jpg",
            "solidYellowCurve2.jpg", "solidYellowLeft.jpg",
            "whiteCarLaneSwitch.jpg")},
        **{f"lane_{n}": LANES2 + n for n in (
            "straight_lines1.jpg", "straight_lines2.jpg", "test1.jpg", "test2.jpg",
            "test3.jpg", "test4.jpg", "test5.jpg", "test6.jpg")},
    },
    # Photographs of cars with a licence plate, **and a human-annotated box and
    # plate text for each one**, from the openalpr benchmark. Project 48.
    #
    # This is the rarest thing in the whole repository: a real photograph with a
    # real annotation, so the project can report an accuracy instead of scoring
    # against something it planted itself. The annotation is one tab-separated
    # line -- filename, x, y, width, height, text -- and the text means
    # localisation and reading can be measured separately, which is the same
    # split project 52 found decisive for barcodes.
    "plates": {
        **{f"eu{i}.jpg": PLATES + f"eu/eu{i}.jpg" for i in range(1, 41)},
        **{f"eu{i}.txt": PLATES + f"eu/eu{i}.txt" for i in range(1, 41)},
    },
    # The three LBP face cascades, which are in the opencv repository but NOT
    # inside the `opencv-python` wheel -- `cv2.data.haarcascades` has only the
    # Haar ones. Project 49 needs them because the interesting comparison is
    # between cascades trained at different window sizes, and the 45x45
    # `_improved` cascade is the whole point of that project's first finding.
    "cascades": {
        n: OPENCV_REPO + "data/lbpcascades/" + n for n in (
            "lbpcascade_frontalface.xml",
            "lbpcascade_frontalface_improved.xml",
            "lbpcascade_profileface.xml")
    },
    # 27 photographs of hands from the HGR1 gesture set -- eight people, the
    # gesture named in the folder the dataset put each file in. They arrive cut
    # out against pure black, which is somebody else's segmentation and is
    # inherited rather than invented: it gives project 55 an **exact** matte, and
    # therefore an exact truth once each hand is composited onto a real
    # background.
    "hands": {
        "1_P__1_P_hgr1_id01_3.jpg": HGR1 + "label/test/1_P/1_P_hgr1_id01_3.jpg",
        "2_P__2_P_hgr1_id01_1.jpg": HGR1 + "label/test/2_P/2_P_hgr1_id01_1.jpg",
        "3_P__3_P_hgr1_id02_2.jpg": HGR1 + "label/test/3_P/3_P_hgr1_id02_2.jpg",
        "4_P__4_P_hgr1_id05_1.jpg": HGR1 + "label/test/4_P/4_P_hgr1_id05_1.jpg",
        "5_P__5_P_hgr1_id04_2.jpg": HGR1 + "label/test/5_P/5_P_hgr1_id04_2.jpg",
        "A_P__A_P_hgr1_id02_8.jpg": HGR1 + "label/test/A_P/A_P_hgr1_id02_8.jpg",
        "B_P__B_P_hgr1_id01_3.jpg": HGR1 + "label/test/B_P/B_P_hgr1_id01_3.jpg",
        "C_P__C_P_hgr1_id07_1.jpg": HGR1 + "label/test/C_P/C_P_hgr1_id07_1.jpg",
        "D_P__D_P_hgr1_id03_3.jpg": HGR1 + "label/test/D_P/D_P_hgr1_id03_3.jpg",
        "E_P__E_P_hgr1_id03_9.jpg": HGR1 + "label/test/E_P/E_P_hgr1_id03_9.jpg",
        "F_P__F_P_hgr1_id03_6.jpg": HGR1 + "label/test/F_P/F_P_hgr1_id03_6.jpg",
        "G_P__G_P_hgr1_id03_4.jpg": HGR1 + "label/test/G_P/G_P_hgr1_id03_4.jpg",
        "H_P__H_P_hgr1_id06_2.jpg": HGR1 + "label/test/H_P/H_P_hgr1_id06_2.jpg",
        "I_P__I_P_hgr1_id01_2.jpg": HGR1 + "label/test/I_P/I_P_hgr1_id01_2.jpg",
        "K_P__K_P_hgr1_id01_3.jpg": HGR1 + "label/test/K_P/K_P_hgr1_id01_3.jpg",
        "L_P__L_P_hgr1_id01_1.jpg": HGR1 + "label/test/L_P/L_P_hgr1_id01_1.jpg",
        "M_P__M_P_hgr1_id01_1.jpg": HGR1 + "label/test/M_P/M_P_hgr1_id01_1.jpg",
        "N_P__N_P_hgr1_id01_1.jpg": HGR1 + "label/test/N_P/N_P_hgr1_id01_1.jpg",
        "O_P__O_P_hgr1_id01_2.jpg": HGR1 + "label/test/O_P/O_P_hgr1_id01_2.jpg",
        "P_P__P_P_hgr1_id01_2.jpg": HGR1 + "label/test/P_P/P_P_hgr1_id01_2.jpg",
        "R_P__R_P_hgr1_id01_1.jpg": HGR1 + "label/test/R_P/R_P_hgr1_id01_1.jpg",
        "S_P__S_P_hgr1_id08_3.jpg": HGR1 + "label/test/S_P/S_P_hgr1_id08_3.jpg",
        "T_P__T_P_hgr1_id01_2.jpg": HGR1 + "label/test/T_P/T_P_hgr1_id01_2.jpg",
        "U_P__U_P_hgr1_id06_2.jpg": HGR1 + "label/test/U_P/U_P_hgr1_id06_2.jpg",
        "W_P__W_P_hgr1_id03_10.jpg": HGR1 + "label/test/W_P/W_P_hgr1_id03_10.jpg",
        "Y_P__Y_P_hgr1_id08_2.jpg": HGR1 + "label/test/Y_P/Y_P_hgr1_id08_2.jpg",
        "Z_P__Z_P_hgr1_id02_4.jpg": HGR1 + "label/test/Z_P/Z_P_hgr1_id02_4.jpg",
    },
    # Eleven group photographs used by OpenCV's own cascade tests: several
    # faces per frame, at different scales, some in profile, one a painting.
    # Project 49.
    "faces": {
        n: EXTRA + "cv/cascadeandhog/images/" + n for n in (
            "addams-family.png", "audrybt1.png", "bttf301.png",
            "churchill-downs.png", "class57.png", "er.png",
            "karen-and-rob.png", "larroquette.png", "mona-lisa.png",
            "rehg-thanksgiving-1994.png", "waynesworld2.png")
    },
}


def fetch(url: str, *, retries: int = RETRIES, timeout: int = TIMEOUT) -> bytes | None:
    last = ""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            last = f"HTTP {e.code}"
        except Exception as e:  # noqa: BLE001
            last = type(e).__name__
        time.sleep(2.0 * (attempt + 1))
    print(f"  FAILED ({last}): {url[:95]}")
    return None


def remote_size(url: str) -> int | None:
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as r:
            return int(r.headers.get("Content-Length") or 0) or None
    except Exception:  # noqa: BLE001
        return None


def fetch_resumable(url: str, dest: Path, *, attempts: int = 60) -> bool:
    """Download in ranged chunks, resuming a ``.part`` file between attempts.

    A whole-file GET over a 2 KB/s link times out and loses everything. A range
    request that dies mid-chunk loses at most one chunk, so progress is
    monotonic and the file eventually completes even on a link this bad.
    """
    total = remote_size(url)
    part = dest.with_suffix(dest.suffix + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    have = part.stat().st_size if part.exists() else 0
    if total:
        print(f"  {dest.name}: {total:,} bytes, {have:,} already on disk", flush=True)

    stalled = 0
    for _ in range(attempts):
        if total and have >= total:
            break
        end = have + CHUNK - 1
        req = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={have}-{end}"})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                block = r.read()
                partial = r.status == 206
        except Exception as e:  # noqa: BLE001
            print(f"  {dest.name}: {type(e).__name__} at {have:,}, retrying", flush=True)
            time.sleep(3.0)
            stalled += 1
            if stalled >= 8:
                return False
            continue

        if not block:
            break
        stalled = 0
        with open(part, "ab") as fh:
            fh.write(block)
        have += len(block)
        if total:
            print(f"  {dest.name}: {have:,}/{total:,} "
                  f"({100 * have / total:.1f}%)", flush=True)
        if not partial:  # server ignored the Range header and sent the whole file
            break

    if total and have < total:
        return False
    part.replace(dest)
    return True


def cache_set(name: str, workers: int = 3) -> tuple[int, int]:
    items = SETS[name]
    todo = {k: v for k, v in items.items() if not (CACHE / name / k).exists()}
    print(f"{name}: {len(items)} files, {len(items) - len(todo)} cached, "
          f"{len(todo)} to fetch")

    def one(item):
        key, url = item
        path = CACHE / name / key
        size = remote_size(url) or 0
        if size > RESUME_ABOVE:
            return fetch_resumable(url, path)
        data = fetch(url)
        if not data:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return True

    got = 0
    with ThreadPoolExecutor(workers) as ex:
        for i, ok in enumerate(ex.map(one, todo.items()), start=1):
            got += bool(ok)
            if i % 5 == 0:
                print(f"  {i}/{len(todo)} ...", flush=True)
    return got, len(items)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="which", default="all",
                    choices=["all", *SETS])
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    wanted = list(SETS) if args.which == "all" else [args.which]
    for name in wanted:
        got, total = cache_set(name)
        have = len(list((CACHE / name).glob("*"))) if (CACHE / name).exists() else 0
        print(f"{name}: fetched {got}, {have}/{total} present\n", flush=True)
    print(f"cache: {CACHE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
