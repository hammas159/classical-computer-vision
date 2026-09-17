"""Copy chosen cache images into `assets/real/` and register them in `shared.io`.

    python tools/import_images.py --spec spec.json

The spec is a list of `{id, pool, name, description}`, normally written straight
out of a `tools/select_images.py` run once the contact sheet has been looked at
and each picture has been given a name that says what it is **of**.

Three things happen here, and all three matter:

* the image is **resized to a common long edge** (`MAX_EDGE`) and re-encoded as
  JPEG. A comparison figure whose rows are different pixel sizes silently gives
  each row a different amount of detail, and every per-pixel metric in this
  repository would then be measuring resolution as much as method;
* the entry is **appended to `shared.io.REAL_PHOTOS`**, so a project refers to
  an image by a name that means something rather than by `113009.jpg`;
* nothing is imported that is already in the repository, under any name --
  checked by perceptual hash, not filename.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.check_image_reuse import EXTS, HASH_THRESHOLD, hamming, hash_dir, phash  # noqa: E402
from tools.fetch_images import CACHE_DIR  # noqa: E402

REAL = REPO / "assets" / "real"
IO_PY = REPO / "shared" / "io.py"

#: Long edge every imported photograph is resized to. 481x321 is the BSDS
#: native size; 640 keeps the SIPI and Kodak material from dwarfing it while
#: staying large enough that a 31 px local window still means something.
MAX_EDGE = 640

JPEG_QUALITY = 92


def load_spec(path: Path) -> list[dict]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    for entry in spec:
        missing = {"id", "pool", "name", "description"} - set(entry)
        if missing:
            raise SystemExit(f"spec entry {entry} is missing {sorted(missing)}")
    return spec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    spec = load_spec(Path(args.spec))
    existing = hash_dir(p for p in REAL.iterdir() if p.suffix.lower() in EXTS)

    from shared import io

    added, skipped = [], []
    for entry in spec:
        src = CACHE_DIR / entry["pool"] / f"{entry['id']}.jpg"
        if not src.exists():
            src = CACHE_DIR / entry["pool"] / f"{entry['id']}.png"
        if not src.exists():
            skipped.append((entry["name"], "not in cache"))
            continue
        if entry["name"] in io.REAL_PHOTOS:
            skipped.append((entry["name"], "name already registered"))
            continue

        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            skipped.append((entry["name"], "unreadable"))
            continue

        h = phash(img)
        clash = next((k for k, v in existing.items() if hamming(h, v) <= HASH_THRESHOLD), None)
        if clash:
            skipped.append((entry["name"], f"already in the repo as {clash}"))
            continue

        scale = MAX_EDGE / max(img.shape[:2])
        if scale < 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        dest = REAL / f"{entry['name']}.jpg"
        if not args.dry_run:
            cv2.imwrite(str(dest), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        existing[entry["name"]] = h
        added.append((entry, img.shape))
        print(f"  + {entry['name']:26s} {img.shape[1]}x{img.shape[0]}  <- {entry['pool']}/{entry['id']}")

    for name, why in skipped:
        print(f"  - {name:26s} SKIPPED: {why}")

    if added and not args.dry_run:
        lines = [f'    "{e["name"]}": ("{e["name"]}.jpg", "{e["description"]}"),'
                 for e, _ in added]
        text = IO_PY.read_text(encoding="utf-8")
        marker = "}\n\n\ndef real_photo("
        if marker not in text:
            raise SystemExit("could not find the end of REAL_PHOTOS in shared/io.py")
        block = "\n".join(lines) + "\n" + marker
        IO_PY.write_text(text.replace(marker, block), encoding="utf-8")
        print(f"\nregistered {len(added)} name(s) in shared/io.py")

    print(f"\n{len(added)} imported, {len(skipped)} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
