"""Download and cache the photograph pools the projects draw their samples from.

    python tools/fetch_images.py cache            # fill the local cache
    python tools/fetch_images.py cache --set bsds
    python tools/fetch_images.py describe --set bsds --limit 40

Why a cache
-----------
Every project needs **10-12 candidate images that no other project uses**, and
they have to span whatever axis that project actually measures. Picking them by
hand from a browser does not scale to 58 projects, and picking them at random
produces four pictures of the same thing.

So the pools are cached **outside the repository** (see ``CACHE_DIR``), measured
once, and each project then selects from the cache on a stated, computed axis and
copies only its own winners into ``assets/real/``. The repository therefore
carries only images that are actually used, and no image reaches two projects --
``tools/check_image_reuse.py`` enforces that.

Sources
-------
Only two hosts are reachable from this machine, which narrowed the options a
great deal: ``raw.githubusercontent.com`` and ``sipi.usc.edu``.
``upload.wikimedia.org`` returns 403 and ``commons.wikimedia.org`` does not
resolve. Both reachable hosts time out intermittently, so every request retries
with a backoff rather than failing the run.

**Licence.** Neither pool is public domain, and that is stated at the point of
use rather than left to be discovered -- see ``assets/real/README.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

#: Outside the repo on purpose: 500 photographs are not a dependency of this
#: project, only the handful each experiment keeps are.
CACHE_DIR = Path.home() / ".cache" / "classical-cv-images"

USER_AGENT = "Mozilla/5.0 (classical-computer-vision dataset fetch)"

#: Both reachable hosts drop connections under any parallelism. Retrying is not
#: optional here -- a plain urlopen fails on roughly one request in five.
RETRIES = 5
TIMEOUT = 60

BSDS_SPLITS = ("test", "train", "val")
BSDS_RAW = ("https://raw.githubusercontent.com/BIDS/BSDS500/master/"
            "BSDS500/data/images/{split}/{name}.jpg")

#: The human segmentations. Five annotators per image, each giving a full
#: region labelling and a boundary map. This is the only *real* ground truth
#: available to this repository -- everywhere else the truth is either generated
#: or defined by construction -- and it brings something generated truth cannot:
#: the annotators disagree with each other, so their agreement is a ceiling no
#: algorithm has any business passing.
BSDS_GT = ("https://raw.githubusercontent.com/BIDS/BSDS500/master/"
           "BSDS500/data/groundTruth/{split}/{name}.mat")
BSDS_API = ("https://api.github.com/repos/BIDS/BSDS500/contents/"
            "BSDS500/data/images/{split}")

#: USC-SIPI volumes worth having. ``textures`` is the Brodatz set, which is the
#: right pool for the texture and FFT projects; ``aerials`` is the only source
#: here of top-down imagery; ``misc`` is the classic photographic test set.
SIPI_VOLUMES = {
    "textures": [f"1.1.{i:02d}" for i in range(1, 14)]
    + [f"1.2.{i:02d}" for i in range(1, 14)]
    + [f"1.3.{i:02d}" for i in range(1, 14)]
    + [f"1.4.{i:02d}" for i in range(1, 14)]
    + [f"1.5.{i:02d}" for i in range(1, 14)],
    "aerials": [f"2.1.{i:02d}" for i in range(1, 13)]
    + [f"2.2.{i:02d}" for i in range(1, 25)],
    "misc": [f"4.1.{i:02d}" for i in range(1, 9)]
    + [f"4.2.{i:02d}" for i in range(1, 8)]
    + [f"5.1.{i:02d}" for i in range(9, 15)]
    + [f"5.2.{i:02d}" for i in range(8, 11)]
    + [f"5.3.{i:02d}" for i in (1, 2)]
    + [f"7.1.{i:02d}" for i in range(1, 11)]
    + ["boat.512", "elaine.512", "gray21.512", "numbers.512", "ruler.512",
       "testpat.1k"],
}
SIPI_URL = "https://sipi.usc.edu/database/download.php?vol={vol}&img={img}"


def fetch(url: str, *, retries: int = RETRIES, timeout: int = TIMEOUT) -> bytes | None:
    """GET with a backoff. Returns ``None`` rather than raising on a hard 404.

    A missing image is expected -- the SIPI volume listings above are built from
    the published numbering and it is not perfectly dense -- so a 404 is a fact
    about the catalogue, not an error. A timeout is a fact about the network and
    is retried.
    """
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
        except Exception as e:  # noqa: BLE001 - urllib raises a wide variety
            last = type(e).__name__
        time.sleep(1.5 * (attempt + 1))
    print(f"  FAILED after {retries} tries ({last}): {url[:90]}")
    return None


def bsds_ids() -> dict[str, list[str]]:
    """The 500 BSDS image ids, listed from the GitHub API and then cached."""
    index = CACHE_DIR / "bsds_ids.json"
    if index.exists():
        return json.loads(index.read_text())
    out: dict[str, list[str]] = {}
    for split in BSDS_SPLITS:
        raw = fetch(BSDS_API.format(split=split))
        if raw is None:
            raise SystemExit(f"could not list the BSDS {split} split")
        out[split] = sorted(e["name"][:-4] for e in json.loads(raw)
                            if e["name"].endswith(".jpg"))
        print(f"  {split}: {len(out[split])} images")
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps(out))
    return out


def _save(path: Path, data: bytes | None) -> bool:
    if not data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return True


def cache_bsds(workers: int = 6) -> int:
    ids = bsds_ids()
    jobs = [(split, name) for split in BSDS_SPLITS for name in ids[split]]
    todo = [(s, n) for s, n in jobs if not (CACHE_DIR / "bsds" / f"{n}.jpg").exists()]
    print(f"bsds: {len(jobs)} total, {len(jobs) - len(todo)} cached, {len(todo)} to fetch")

    def one(job):
        split, name = job
        return _save(CACHE_DIR / "bsds" / f"{name}.jpg",
                     fetch(BSDS_RAW.format(split=split, name=name)))

    got = 0
    with ThreadPoolExecutor(workers) as ex:
        for i, ok in enumerate(ex.map(one, todo), start=1):
            got += bool(ok)
            if i % 25 == 0:
                print(f"  {i}/{len(todo)} ...")
    return got


def cache_bsds_gt(workers: int = 6) -> int:
    """Cache the human segmentations alongside the images."""
    ids = bsds_ids()
    jobs = [(split, name) for split in BSDS_SPLITS for name in ids[split]]
    todo = [(s, n) for s, n in jobs
            if not (CACHE_DIR / "bsds_gt" / f"{n}.mat").exists()]
    print(f"bsds_gt: {len(jobs)} total, {len(jobs) - len(todo)} cached, "
          f"{len(todo)} to fetch")

    def one(job):
        split, name = job
        return _save(CACHE_DIR / "bsds_gt" / f"{name}.mat",
                     fetch(BSDS_GT.format(split=split, name=name)))

    got = 0
    with ThreadPoolExecutor(workers) as ex:
        for i, ok in enumerate(ex.map(one, todo), start=1):
            got += bool(ok)
            if i % 50 == 0:
                print(f"  {i}/{len(todo)} ...")
    return got


def cache_sipi(workers: int = 4) -> int:
    """SIPI ships TIFFs; they are converted to PNG on the way into the cache."""
    import cv2
    import numpy as np

    jobs = [(vol, img) for vol, imgs in SIPI_VOLUMES.items() for img in imgs]
    todo = [(v, i) for v, i in jobs if not (CACHE_DIR / "sipi" / f"{v}_{i}.png").exists()]
    print(f"sipi: {len(jobs)} total, {len(jobs) - len(todo)} cached, {len(todo)} to fetch")

    def one(job):
        vol, img = job
        raw = fetch(SIPI_URL.format(vol=vol, img=img))
        if not raw:
            return False
        arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        if arr is None:
            return False
        out = CACHE_DIR / "sipi" / f"{vol}_{img}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(out), arr))

    got = 0
    with ThreadPoolExecutor(workers) as ex:
        for i, ok in enumerate(ex.map(one, todo), start=1):
            got += bool(ok)
            if i % 25 == 0:
                print(f"  {i}/{len(todo)} ...")
    return got


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["cache", "list"])
    ap.add_argument("--set", dest="which", default="all",
                    choices=["all", "bsds", "bsds_gt", "sipi"])
    args = ap.parse_args()

    if args.action == "list":
        for pool in ("bsds", "bsds_gt", "sipi"):
            files = sorted((CACHE_DIR / pool).glob("*")) if (CACHE_DIR / pool).exists() else []
            print(f"{pool:8s} {len(files)} cached at {CACHE_DIR / pool}")
        return 0

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if args.which in ("all", "bsds"):
        print(f"fetched {cache_bsds()} new BSDS images")
    if args.which in ("all", "bsds_gt"):
        print(f"fetched {cache_bsds_gt()} new BSDS ground truths")
    if args.which in ("all", "sipi"):
        print(f"fetched {cache_sipi()} new SIPI images")
    print(f"\ncache: {CACHE_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
