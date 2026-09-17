"""Binarise your own image with every method at once.

    python infer.py page.jpg
    python infer.py page.jpg --method Sauvola --out mask.png
    python infer.py x --simulate --kind thin --illum 0.3   # score them on a known scene

With a real image there is no ground truth, so **no IoU is printed**. What is
printed instead is the thing that predicts which family will work: **the width
of the foreground relative to the local window**. A stroke narrower than the
window is what adaptive thresholding needs; a filled region wider than it is
what adaptive thresholding hollows out.

That measurement needs no annotation, and it is the whole finding of this
project made usable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from shared import io  # noqa: E402
from shared.io import to_gray  # noqa: E402
from shared.metrics import iou  # noqa: E402

import thresholding as th  # noqa: E402

#: The window every local method here uses. The comparison in this file is
#: against this number, so it is named once.
LOCAL_WINDOW = 31


def foreground_thickness(mask: np.ndarray) -> float:
    """Typical width of the detected foreground, in pixels.

    Twice the 90th-percentile distance-to-background: the 90th rather than the
    maximum, so one large blob in an otherwise thin image does not decide the
    answer.
    """
    m = (mask > 0).astype(np.uint8)
    if not m.any():
        return float("nan")
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    return float(np.percentile(dist[m > 0], 90)) * 2.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--method", default=None, choices=list(th.METHODS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--simulate", action="store_true",
                    help="ignore the image and score every method on a generated scene")
    ap.add_argument("--kind", default="thin", choices=["solid", "thin"])
    ap.add_argument("--illum", type=float, default=1.0,
                    help="with --simulate, illumination at the darkest corner")
    args = ap.parse_args()

    truth = None
    if args.simulate:
        img, truth = th.synthetic_scene(kind=args.kind, fg_fraction=0.12,
                                        illum_min=args.illum, seed=0)
        print(f"generated a '{args.kind}' scene, illumination {args.illum:g}")
    else:
        img = io.imread(args.image)
        print(f"{args.image}  {img.shape[1]}x{img.shape[0]}")

    gray = to_gray(img)
    names = [args.method] if args.method else list(th.METHODS)

    print(f"\n{'method':24s} {'fg %':>7s} {'width px':>9s}" +
          (f" {'IoU':>7s}" if truth is not None else ""))
    masks = {}
    for name in names:
        m = th.METHODS[name](gray)
        if truth is not None and iou(m > 0, truth > 0) < iou(m == 0, truth > 0):
            m = (m == 0).astype(np.uint8) * 255
        masks[name] = m
        line = f"{name:24s} {100 * float((m > 0).mean()):6.2f}% {foreground_thickness(m):9.1f}"
        if truth is not None:
            line += f" {max(iou(m > 0, truth > 0), iou(m == 0, truth > 0)):7.3f}"
        print(line)

    if truth is None:
        # Otsu is the least assumption-laden estimate of "where is the
        # foreground", so its thickness is the one used to advise.
        width = foreground_thickness(masks.get("Otsu", next(iter(masks.values()))))
        print(f"\nforeground width (from Otsu): {width:.1f} px, "
              f"local window: {LOCAL_WINDOW} px")
        if np.isnan(width):
            print("  -> nothing detected; the image may be uniform.")
        elif width < LOCAL_WINDOW * 0.6:
            print("  -> thinner than the window. Local methods (Sauvola, adaptive) "
                  "should do well here, especially if the lighting is uneven.")
        else:
            print("  -> wider than the window. Local methods will hollow this out; "
                  "prefer a global method, or raise the window above the "
                  f"{width:.0f} px foreground.")
        print("\nNo IoU is reported. Nobody recorded which pixels of a real image "
              "are foreground, so any number here would be invented. Run with "
              "--simulate to see the methods scored.")

    out = Path(args.out) if args.out else PROJECT_DIR / "results" / "binarised.png"
    panel = np.hstack([
        np.repeat((masks[n] > 0).astype(np.uint8)[..., None] * 255, 3, axis=2) for n in names
    ])
    io.imwrite(out, panel)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
