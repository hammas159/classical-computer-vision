"""Run every edge operator on your own image.

    python infer.py photo.jpg
    python infer.py photo.jpg --operator Sobel --threshold 0.2
    python infer.py photo.jpg --simulate       # add known edges and score them

With a real photograph there is no ground truth, so **no precision or recall is
printed**. What is printed is what needs no reference: how much of the frame each
operator marked as an edge, and how thin those edges are.

The edge fraction is the useful one. An operator marking 25% of a photograph as
edge has not found more detail than one marking 4% — it has found noise, and the
auto-threshold recipe does exactly that as soon as the image is grainy.
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
from shared.metrics import edge_prf  # noqa: E402

import edges as ed  # noqa: E402

ALL_OPERATORS = list(ed.GRADIENTS) + ["Canny (fixed 50/150)", "Canny (auto median)"]


def run_operator(name: str, gray: np.ndarray, threshold: float) -> np.ndarray:
    if name in ed.GRADIENTS:
        return ed.threshold_magnitude(ed.GRADIENTS[name](gray), threshold)
    if name.startswith("Canny (fixed"):
        return ed.edges_canny(gray)
    return ed.edges_canny_auto(gray)


def mean_thickness(e: np.ndarray) -> float:
    """Edge pixels divided by skeleton pixels — 1.0 is a one-pixel-wide edge.

    Worth reporting because a thick edge inflates recall for free: at a 2 px
    matching tolerance, a 3 px wide edge covers three chances to be counted
    correct. Canny thins to one pixel and the gradient operators do not.
    """
    mask = (e > 0).astype(np.uint8)
    if not mask.any():
        return float("nan")
    thin = cv2.ximgproc.thinning(mask * 255) if hasattr(cv2, "ximgproc") else None
    if thin is None:
        # opencv-contrib is not a dependency here; erode-based proxy instead
        eroded = cv2.erode(mask, np.ones((3, 3), np.uint8))
        boundary = int((mask & ~eroded).sum())
        return float(mask.sum()) / max(boundary, 1)
    return float(mask.sum()) / max(int((thin > 0).sum()), 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--operator", default=None, choices=ALL_OPERATORS,
                    help="run just this one; the default runs all seven")
    ap.add_argument("--threshold", type=float, default=0.15,
                    help="magnitude threshold for the gradient operators")
    ap.add_argument("--simulate", action="store_true",
                    help="ignore the image and score the operators on a generated scene")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="with --simulate, the Gaussian noise sigma to add")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    truth = None
    if args.simulate:
        img, truth = ed.scene(noise_sigma=args.noise, seed=0)
        print(f"generated shapes with exact edges, noise sigma {args.noise:g}")
    else:
        img = io.imread(args.image)
        print(f"{args.image}  {img.shape[1]}x{img.shape[0]}")

    gray = to_gray(img)
    names = [args.operator] if args.operator else ALL_OPERATORS

    print(f"\n{'operator':24s} {'edge %':>8s} {'thickness':>10s}" +
          (f" {'F1':>7s} {'P':>7s} {'R':>7s}" if truth is not None else ""))
    results = {}
    for name in names:
        e = run_operator(name, gray, args.threshold)
        results[name] = e
        line = f"{name:24s} {100 * float((e > 0).mean()):7.2f}% {mean_thickness(e):10.2f}"
        if truth is not None:
            prf = edge_prf(e, truth, ed.TOLERANCE)
            line += f" {prf['f1']:7.3f} {prf['precision']:7.3f} {prf['recall']:7.3f}"
        print(line)

    if truth is None:
        print("\nNo precision or recall is reported. Nobody recorded where the "
              "edges of a real scene are, so any number here would be invented. "
              "Run with --simulate to see the operators scored.")
        noisy = [n for n, e in results.items() if float((e > 0).mean()) > 0.20]
        if noisy:
            print(f"\n  -> {', '.join(noisy)} marked over 20% of the frame as edge. "
                  "That is noise, not detail.")

    out = Path(args.out) if args.out else PROJECT_DIR / "results" / "edges.png"
    panel = np.hstack([
        np.repeat((results[n] > 0).astype(np.uint8)[..., None] * 255, 3, axis=2)
        for n in names
    ])
    io.imwrite(out, panel)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
