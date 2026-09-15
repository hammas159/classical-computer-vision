"""Scan a page from the command line — inference on your own image.

    python infer.py my_photo.jpg
    python infer.py my_photo.jpg --detector "Saturation (HSV)" --binariser Sauvola
    python infer.py my_photo.jpg --out scanned.png --all-detectors

No ground truth exists for a photo you supply, so no accuracy is reported. What
*is* reported is everything measurable without it: which detector found a page,
the recovered aspect ratio, and the time each stage took.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import document_scanner as ds  # noqa: E402
from shared.bench import timeit  # noqa: E402
from shared.io import imread, imwrite  # noqa: E402
from shared.report import init_console  # noqa: E402

MAX_SIDE = 1600


def load(path: str) -> np.ndarray:
    """Read an image and cap its size.

    A modern phone photo is 4000 px wide. Every detector here is O(pixels), and
    nothing in the pipeline benefits from that resolution, so it is capped — the
    homography is scale-free, so this changes the recovered geometry not at all.
    """
    img = imread(path)
    if max(img.shape[:2]) > MAX_SIDE:
        scale = MAX_SIDE / max(img.shape[:2])
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", help="path to a photo of a page")
    ap.add_argument("--detector", default="Otsu + contour", choices=list(ds.DETECTORS))
    ap.add_argument("--binariser", default="Sauvola", choices=list(ds.BINARISERS))
    ap.add_argument("--out", default="scanned.png", help="where to write the result")
    ap.add_argument("--save-stages", action="store_true",
                    help="also write the detection overlay and the rectified page")
    ap.add_argument("--all-detectors", action="store_true",
                    help="run all six and report which found a page, and how fast")
    ap.add_argument("--edge-aspect", action="store_true",
                    help="use the edge-length heuristic instead of the closed form")
    args = ap.parse_args()

    if not Path(args.image).exists():
        print(f"error: no such file: {args.image}")
        return 2

    img = load(args.image)
    print(f"input : {args.image}  {img.shape[1]}x{img.shape[0]}")

    if args.all_detectors:
        print(f"\n{'Detector':<26} {'Found':<7} {'Recovered w/h':<15} {'Time (ms)':>9}")
        print("-" * 60)
        for name, fn in ds.DETECTORS.items():
            corners, timing = timeit(lambda f=fn: f(img), runs=3, warmup=1)
            if corners is None:
                print(f"{name:<26} {'no':<7} {'-':<15} {timing.median_ms:>9.2f}")
                continue
            ratio = ds.aspect_from_perspective(corners, img.shape)
            shown = "degenerate" if ratio is None else f"{ratio:.3f}"
            print(f"{name:<26} {'yes':<7} {shown:<15} {timing.median_ms:>9.2f}")
        print(
            "\nNo accuracy column: a photo you supplied has no ground truth, so "
            "there is nothing to score against.\nRun `python run.py` for the "
            "measured comparison on generated scenes."
        )
        return 0

    (corners, rectified, binary), timing = timeit(
        lambda: ds.scan(img, args.detector, args.binariser,
                        use_perspective_aspect=not args.edge_aspect),
        runs=3, warmup=1,
    )

    if corners is None:
        print(f"\n{args.detector} found no page in this image.")
        print("Try another detector, or --all-detectors to see which ones do:")
        for name in ds.DETECTORS:
            print(f"    --detector \"{name}\"")
        return 1

    ratio = ds.aspect_from_perspective(corners, img.shape)
    imwrite(args.out, binary)

    print(f"detector : {args.detector}")
    print(f"binariser: {args.binariser}")
    print(f"page     : found, corners at {np.round(corners, 1).tolist()}")
    print(f"aspect   : {'degenerate (near-affine view)' if ratio is None else f'{ratio:.4f} w/h'}")
    print(f"output   : {rectified.shape[1]}x{rectified.shape[0]} px")
    print(f"time     : {timing.median_ms:.1f} ms (median of 3)")
    print(f"wrote    : {args.out}")

    if args.save_stages:
        overlay = img.copy()
        cv2.polylines(overlay, [np.int32(corners)], True, (0, 200, 0), 3)
        stem = Path(args.out).with_suffix("")
        imwrite(f"{stem}_detected.png", overlay)
        imwrite(f"{stem}_rectified.png", rectified)
        print(f"wrote    : {stem}_detected.png, {stem}_rectified.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
