"""Measure every claimed invariance on a shape of your own.

    python infer.py mask.png
    python infer.py mask.png --rotate 45
    python infer.py --object collie_standing        # one of the project's silhouettes

Give it a binary mask (or any image, which is thresholded). It applies exact
transforms and prints how far each descriptor moves — and, next to that, how far
apart two people already are when they trace the same object.

That second number is the one that decides anything. An invariance far below the
human spread is real arithmetic of no practical use: nothing upstream defines the
shape that precisely. An invariance above it is a genuine failure. On this
project's twelve silhouettes the spread is 0.11 to 0.75 depending on the
descriptor, and Hu moments are invariant to a 180 degree rotation about a
thousand times more tightly than that.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from shared import figures, io  # noqa: E402
from shared.io import ensure_rgb, to_gray  # noqa: E402
from shared.report import init_console  # noqa: E402

import shapes_desc as sd  # noqa: E402

#: The spread between two people tracing the same object, per descriptor. Copied
#: from `annotator_variation()` so that a single-shape run does not have to
#: recompute it over twelve images and 44 pairs.
HUMAN_SPREAD = {
    "Hu moments (log)": 0.753,
    "Hu moments (textbook log)": 0.753,
    "Fourier descriptors": 0.578,
    "Chain code histogram": 0.173,
    "Simple geometry": 0.108,
}


def load_mask(path: str) -> np.ndarray:
    img = io.imread(path)
    gray = to_gray(img) if img.ndim == 3 else img
    if len(np.unique(gray)) > 2:
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if (gray > 0).mean() > 0.5:      # keep the smaller side as the object
            gray = 255 - gray
    return (gray > 127).astype(np.uint8) * 255


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mask", nargs="?", default=None)
    ap.add_argument("--object", default=None, choices=list(sd.REAL_IMAGES),
                    help="use one of the project's human-traced silhouettes")
    ap.add_argument("--rotate", type=float, default=30.0)
    ap.add_argument("--scale", type=float, default=0.7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    init_console()
    if args.object:
        mask = sd.load_silhouette(args.object)
        source = f"{args.object} (traced by annotator {sd.SILHOUETTES[args.object][0]})"
    elif args.mask:
        mask = load_mask(args.mask)
        source = args.mask
    else:
        ap.error("give a mask image, or --object with one of the project's silhouettes")

    contours, _ = cv2.findContours((mask > 0).astype(np.uint8),
                                   cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise SystemExit("no shape found in that mask")
    print(f"{source}  {mask.shape[1]}x{mask.shape[0]}  "
          f"{(mask > 0).mean():.1%} filled  {len(contours)} contours")
    if len(contours) > 1:
        print("  three of the five descriptors read only the largest contour, so "
              f"{len(contours) - 1} of those are invisible to them")

    base = sd.transform_mask(mask)
    variants = {
        f"rotated {args.rotate:g} deg": sd.transform_mask(mask, rotation=args.rotate),
        "rotated 180 deg": sd.transform_mask(mask, rotation=180.0),
        f"scaled {args.scale:g}x": sd.transform_mask(mask, scale=args.scale),
        "translated (30, 30)": sd.transform_mask(mask, translate=(30, 30)),
        "mirrored": cv2.flip(base, 1),
    }

    print(f"\n{'descriptor':28s} " + " ".join(f"{k:>16s}" for k in variants)
          + f" {'people differ by':>17s}")
    verdicts = {}
    for name, fn in sd.DESCRIPTORS.items():
        reference = fn(base)
        changes = {k: sd.relative_change(reference, fn(v)) for k, v in variants.items()}
        verdicts[name] = changes
        print(f"{name:28s} " + " ".join(f"{changes[k]:16.5f}" for k in variants)
              + f" {HUMAN_SPREAD.get(name, float('nan')):17.3f}")

    print("\n--- what that means for this shape ---")
    for name, changes in verdicts.items():
        floor = HUMAN_SPREAD.get(name)
        rotation = changes[f"rotated {args.rotate:g} deg"]
        if floor is None:
            continue
        if rotation < 0.1 * floor:
            print(f"{name:28s} rotation-invariant with room to spare "
                  f"({rotation:.4f} against a human spread of {floor:.3f})")
        elif rotation < floor:
            print(f"{name:28s} rotation error {rotation:.4f} is inside the human "
                  f"spread ({floor:.3f}) — not the thing that will limit you")
        else:
            print(f"{name:28s} rotation error {rotation:.4f} EXCEEDS the human "
                  f"spread ({floor:.3f}) — a real failure, not rounding")

    hu_a, hu_b = sd.desc_hu(base), sd.desc_hu(variants["mirrored"])
    flipped = bool(np.sign(hu_a[6]) != np.sign(hu_b[6]))
    symmetric = sd.relative_change(sd.desc_fourier(base),
                                   sd.desc_fourier(variants["mirrored"])) < 0.01
    print(f"\nreflection: the 7th Hu moment {'FLIPS' if flipped else 'does not flip'} sign.")
    if flipped:
        print("  so this shape is distinguishable from its mirror image — by h7 alone. "
              "Every other descriptor here scores the reflection as identical.")
    else:
        print("  this shape is close to mirror-symmetric, so its reflection really is "
              "the same shape and h7 is right not to move.")
    if symmetric and not flipped:
        print("  (the Fourier descriptors agree: under 0.01 change under reflection)")

    raw = cv2.HuMoments(cv2.moments(base, binaryImage=True)).ravel()
    if np.abs(raw).min() < sd.EPS:
        i = int(np.argmin(np.abs(raw)))
        print(f"\nCAUTION: Hu moment h{i + 1} is {raw[i]:.2e}, below the 1e-12 floor. "
              "This shape is symmetric enough to sit in the hole the textbook "
              "log transform has at zero — `desc_hu` floors it, "
              "`desc_hu_textbook` would not.")

    out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "invariance.png"
    figures.grid(
        [("as given", ensure_rgb(base))] + [(k, ensure_rgb(v)) for k, v in variants.items()],
        out_path, ncols=3,
        suptitle=f"{Path(source).name} under each transform the descriptors claim not to notice",
    )
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
