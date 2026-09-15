"""Resize your own image content-aware — inference from the command line.

    python infer.py photo.jpg --width 800
    python infer.py photo.jpg --reduction 0.25 --out narrow.png
    python infer.py photo.jpg --reduction 0.25 --compare        # also write the rescale
    python infer.py photo.jpg --energy "Laplacian"
    python infer.py photo.jpg --seams seams.png                 # draw what will be removed

There is no ground truth for a photo you supply, so nothing here is scored
against a reference. What *is* reported needs none: how much of the image's own
gradient energy survived — which is the objective seam carving optimises — and
how long it took against the plain rescale it is meant to beat.

**Carving is slow, and knowing that is the point.** It runs one sequential
dynamic programme per removed column. Measured on the sample set, a 20% reduction
costs about **2,800x** a plain `cv2.resize` for **11.6 points** more of the
high-energy region retained — and on one of the four sample images it retains
*less*.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import seam_carving as sc  # noqa: E402
from shared.bench import timeit  # noqa: E402
from shared.io import imread, imwrite  # noqa: E402
from shared.report import init_console  # noqa: E402

MAX_SIDE = 900


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("image")
    ap.add_argument("--width", type=int, help="target width in px (overrides --reduction)")
    ap.add_argument("--reduction", type=float, default=0.20, help="fraction of width to remove")
    ap.add_argument("--energy", default="Gradient |dx|+|dy|", choices=list(sc.ENERGIES))
    ap.add_argument("--out", default="carved.png")
    ap.add_argument("--compare", action="store_true", help="also write the plain rescale")
    ap.add_argument("--seams", help="write an image with the removed seams drawn on it")
    ap.add_argument("--max-side", type=int, default=MAX_SIDE,
                    help="downscale first; carving is O(columns removed) sequential DPs")
    args = ap.parse_args()

    if not Path(args.image).exists():
        print(f"error: no such file: {args.image}")
        return 2

    img = imread(args.image)
    original_shape = img.shape
    if max(img.shape[:2]) > args.max_side:
        s = args.max_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    h, w = img.shape[:2]
    target = args.width if args.width else int(w * (1.0 - args.reduction))
    target = max(8, min(target, w - 1))
    reduction = 1.0 - target / w

    print(f"input   : {args.image}  {original_shape[1]}x{original_shape[0]}", end="")
    print(f"  (working at {w}x{h})" if img.shape != original_shape else "")
    print(f"target  : {target}px  ({reduction:.1%} narrower)")
    print(f"energy  : {args.energy}")

    if reduction > 0.45:
        print(
            f"\nwarning : {reduction:.0%} is past the point where this method degrades sharply.\n"
            "          Seam carving works by having somewhere to route AROUND; past roughly\n"
            "          45% of the width there are no low-energy paths left, every remaining\n"
            "          seam crosses something, and the distortion arrives as bent edges\n"
            "          rather than as uniform squashing."
        )

    fn = sc.ENERGIES[args.energy]
    (carved, _), carve_time = timeit(lambda: sc.carve(img, target, fn), runs=1, warmup=0)
    rescaled, rescale_time = timeit(
        lambda: cv2.resize(img, (target, h), interpolation=cv2.INTER_AREA), runs=1, warmup=0
    )

    total = max(float(sc.energy_gradient(img).sum()), 1e-9)
    e_carved = float(sc.energy_gradient(carved).sum()) / total
    e_plain = float(sc.energy_gradient(rescaled).sum()) / total

    imwrite(args.out, carved)
    print(f"\ncarved  : {carve_time.median_ms:.0f} ms   energy kept {e_carved:.1%}")
    print(f"rescale : {rescale_time.median_ms:.2f} ms   energy kept {e_plain:.1%}")
    print(f"cost    : {carve_time.median_ms / max(rescale_time.median_ms, 1e-9):.0f}x a plain resize")
    print(f"wrote   : {args.out}")

    if args.compare:
        stem = Path(args.out).with_suffix("")
        imwrite(f"{stem}_rescaled.png", rescaled)
        print(f"wrote   : {stem}_rescaled.png")

    if args.seams:
        imwrite(args.seams, sc.seam_overlay(img, fn, n=min(w - target, 200)))
        print(f"wrote   : {args.seams}")

    # ------------------------------------------------------------------ #
    # how to read the result
    # ------------------------------------------------------------------ #
    gain = (e_carved - e_plain) * 100
    print()
    if gain < 2.0:
        print(
            f"verdict : carving retained only {gain:+.1f} points more of this image's gradient\n"
            "          energy than a plain resize did, at many times the cost. That is what\n"
            "          happens when the interesting content spans the full width -- there are\n"
            "          no low-energy paths to route around it. A plain resize is the better\n"
            "          tool for this image."
        )
    else:
        print(
            f"verdict : carving retained {gain:+.1f} points more of the image's gradient energy.\n"
            "          Note this is seam carving's OWN objective, not a perceptual score --\n"
            "          measured on the sample set it wins this column on 4 images out of 4 and\n"
            "          the region-retention column on only 3, so a good number here is\n"
            "          necessary and not sufficient. Look at the picture."
        )

    if args.energy != "Gradient |dx|+|dy|":
        print(
            "\nnote    : the energy function barely matters. Measured over four images at 20%\n"
            "          reduction the four in this project span 0.5 points of region retention,\n"
            "          while the gap to a plain rescale is 11.6 -- 23x larger. Switching\n"
            "          energies is not where the quality is."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
