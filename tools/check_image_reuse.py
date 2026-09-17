"""Enforce the one rule that cannot be checked by looking at a single project.

    python tools/check_image_reuse.py            # report
    python tools/check_image_reuse.py --strict   # exit 1 if any image is shared

**No photograph may appear in two projects.** 58 projects x ~10 images means the
repository should contain ~580 *distinct* pictures, and the moment one is shared
the comparison figures start to look like each other.

Two different things are checked, because there are two ways to break it:

1. **By name** -- two projects referencing the same entry in `shared.io`.
   This is the common case and the cheap one.
2. **By content** -- the same photograph committed twice under two names, or
   re-downloaded from the cache after it was already used. Names cannot catch
   that, so every image is fingerprinted with a downscaled DCT hash. The
   fingerprint survives resizing and JPEG re-encoding, which is exactly how a
   duplicate gets in without anyone noticing.

The content check is also what maps `assets/real/` back to the download cache,
so `tools/select_images.py` can exclude images a project already took even
though they were renamed on the way in.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.fetch_images import CACHE_DIR  # noqa: E402

REAL = REPO / "assets" / "real"
MANIFEST = REPO / "assets" / "real" / "manifest.json"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

#: Hamming distance below which two DCT hashes are the same picture. 64-bit
#: hash; 5 tolerates JPEG re-encoding and a resize without merging genuinely
#: different photographs.
HASH_THRESHOLD = 5


def phash(img: np.ndarray) -> int:
    """64-bit perceptual hash: low-frequency DCT signs against the median.

    Chosen over a pixel hash because every image here is resized and re-encoded
    on the way into the repository, which changes every byte and none of the
    content.
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    small = cv2.resize(g, (32, 32)).astype(np.float32)
    d = cv2.dct(small)[:8, :8]
    flat = d.flatten()[1:]  # drop DC, which is just brightness
    bits = flat > np.median(flat)
    return int("".join("1" if b else "0" for b in bits[:64]), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def hash_dir(paths) -> dict[str, int]:
    out = {}
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None:
            out[p.stem] = phash(img)
    return out


def project_usage() -> dict[str, set[str]]:
    """Which project references which `shared.io` image name."""
    from shared import io

    names = set(io.REAL_PHOTOS)
    used: dict[str, set[str]] = defaultdict(set)
    for f in (REPO / "projects").rglob("*.py"):
        if "__pycache__" in str(f):
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        project = f.relative_to(REPO / "projects").parts[0]
        for name in names:
            if re.search(rf'["\']{re.escape(name)}["\']', text):
                used[name].add(project)
    return used


def build_manifest() -> dict:
    """Map every committed photograph back to the cache entry it came from.

    Without this, an image renamed on the way into `assets/real/` is invisible
    to the selector and can be handed to a second project.
    """
    repo_hashes = hash_dir(p for p in REAL.iterdir() if p.suffix.lower() in EXTS)
    cache_hashes = {}
    for pool in ("bsds", "sipi"):
        d = CACHE_DIR / pool
        if d.exists():
            cache_hashes |= {f"{pool}/{k}": v for k, v in hash_dir(d.iterdir()).items()}

    mapping = {}
    for stem, h in sorted(repo_hashes.items()):
        best, best_d = None, 99
        for cid, ch in cache_hashes.items():
            d = hamming(h, ch)
            if d < best_d:
                best, best_d = cid, d
        mapping[stem] = {"phash": h,
                         "source": best if best_d <= HASH_THRESHOLD else None,
                         "distance": best_d if best_d <= HASH_THRESHOLD else None}
    return mapping


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--no-manifest", action="store_true")
    args = ap.parse_args()

    failures = 0

    # ---- 1. the same name in two projects ---------------------------------- #
    shared_names = {n: sorted(p) for n, p in project_usage().items() if len(p) > 1}
    print(f"== name reuse ==  {len(shared_names)} image(s) referenced by >1 project")
    for name, projects in sorted(shared_names.items()):
        print(f"  {name:24s} {projects}")
    failures += len(shared_names)

    # ---- 2. the same picture under two names ------------------------------- #
    repo_hashes = hash_dir(p for p in REAL.iterdir() if p.suffix.lower() in EXTS)
    dupes = []
    stems = sorted(repo_hashes)
    for i, a in enumerate(stems):
        for b in stems[i + 1:]:
            d = hamming(repo_hashes[a], repo_hashes[b])
            if d <= HASH_THRESHOLD:
                dupes.append((a, b, d))
    print(f"\n== content duplicates in assets/real ==  {len(dupes)} pair(s)")
    for a, b, d in dupes:
        print(f"  {a} == {b}  (hamming {d})")
    failures += len(dupes)

    # ---- 3. refresh the cache mapping -------------------------------------- #
    if not args.no_manifest:
        mapping = build_manifest()
        MANIFEST.write_text(json.dumps(mapping, indent=1))
        matched = sum(1 for v in mapping.values() if v["source"])
        print(f"\n== manifest ==  {matched}/{len(mapping)} traced to the download cache")
        print(f"  wrote {MANIFEST}")

    print(f"\n{failures} problem(s)")
    return 1 if (args.strict and failures) else 0


if __name__ == "__main__":
    sys.exit(main())
