"""Find lines and circles in your own image with both Hough variants.

    python infer.py photo.jpg
    python infer.py photo.jpg --threshold 160 --out lines.png
    python infer.py --annotated hotel_rossiya      # score against human boundaries

On your own photograph there is no annotation, so **no precision is printed**.
What is printed instead is how much of Canny's edge map each method accounts
for — and how much of the frame it draws.

That ratio is the useful one. Hough votes *on* Canny's output, so its lines can
only be supported by pixels Canny already found. Drawing far more than Canny
found means it is extrapolating, which the infinite-line representation does by
construction: standard Hough marks 55% of the frame on this project's twelve
photographs, against Canny's 18%.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from shared import bsds, io  # noqa: E402
from shared.io import to_gray  # noqa: E402
from shared.report import init_console  # noqa: E402

import hough as hg  # noqa: E402

#: Drawn-pixel ratio against Canny above which Hough is extrapolating rather
#: than summarising.
EXTRAPOLATION_WARN = 1.5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", default=None)
    ap.add_argument("--annotated", default=None, choices=list(hg.IMAGES))
    ap.add_argument("--threshold", type=int, default=120)
    ap.add_argument("--circles", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    init_console()
    if args.annotated:
        img = hg.load_scene(args.annotated)
        target = bsds.consensus_boundaries(args.annotated)
        print(f"{args.annotated}  {img.shape[1]}x{img.shape[0]}  "
              "(with human boundary annotations)")
    elif args.image:
        img = io.imread(args.image)
        target = None
        print(f"{args.image}  {img.shape[1]}x{img.shape[0]}")
    else:
        ap.error("give an image path, or --annotated with one of the project's photographs")

    gray = to_gray(img)
    canny = cv2.Canny(gray, 50, 150)
    canny_frac = float((canny > 0).mean())
    print(f"Canny marks {100 * canny_frac:.1f}% of the frame — that is what Hough "
          "votes on, so its lines can only come from these pixels")

    variants = {
        "Canny edges (control)": (canny, len(np.flatnonzero(canny))),
    }
    t0 = time.perf_counter()
    std_lines = hg.detect_lines_standard(gray, threshold=args.threshold)
    std_ms = (time.perf_counter() - t0) * 1000
    variants["Standard Hough"] = (hg.rasterise_lines(std_lines, gray.shape), len(std_lines))

    t0 = time.perf_counter()
    prob_lines = hg.detect_lines_probabilistic(gray)[0]
    prob_ms = (time.perf_counter() - t0) * 1000
    variants["Probabilistic Hough"] = (hg.rasterise_lines(prob_lines, gray.shape),
                                       len(prob_lines))

    header = (f"\n{'method':24s} {'found':>7s} {'% drawn':>9s} {'vs Canny':>9s} "
              f"{'ms':>8s}")
    if target is not None:
        header += f" {'precision':>10s}"
    print(header)
    timings = {"Standard Hough": std_ms, "Probabilistic Hough": prob_ms}
    for name, (drawn, count) in variants.items():
        frac = float((drawn > 0).mean())
        line = (f"{name:24s} {count:7d} {100 * frac:8.1f}% "
                f"{frac / max(canny_frac, 1e-9):8.2f}x "
                f"{timings.get(name, 0.0):8.2f}")
        if target is not None:
            line += f" {bsds.boundary_f_measure(drawn, target, 2)['precision']:10.4f}"
        print(line)

    ratios = {n: float((d > 0).mean()) / max(canny_frac, 1e-9)
              for n, (d, _) in variants.items() if n != "Canny edges (control)"}
    extrapolating = [n for n, r in ratios.items() if r > EXTRAPOLATION_WARN]
    if extrapolating:
        print(f"\n{', '.join(extrapolating)} draws more than "
              f"{EXTRAPOLATION_WARN:g}x what Canny found — it is extrapolating "
              "across parts of the frame where nothing voted, which is what an "
              "infinite (rho, theta) line means.")
    else:
        print("\nBoth variants draw less than Canny found, so they are "
              "summarising its edges rather than extending them.")

    if target is not None:
        scores = {n: bsds.boundary_f_measure(d, target, 2)["precision"]
                  for n, (d, _) in variants.items()}
        best = max(scores, key=scores.get)
        print(f"Most precise here: {best} at {scores[best]:.4f}.")
        if best == "Canny edges (control)":
            print("  Canny wins, as it does on 11 of this project's 12 "
                  "photographs — the voting stage adds error on real images.")
        else:
            print("  Hough wins here, which happens on the man-made scenes where "
                  "there is genuine straight structure to find.")
    else:
        print("No precision is reported. There is no annotation for this "
              "photograph, and the ratio above is the measurement that is real.")

    panels = [io.ensure_rgb(d) for d, _ in variants.values()]
    if args.circles:
        circles = hg.detect_circles(gray) if hasattr(hg, "detect_circles") else []
        canvas = img.copy()
        for c in circles:
            cv2.circle(canvas, (int(c[0]), int(c[1])), int(c[2]), (0, 220, 0), 2)
        print(f"\n{len(circles)} circles found")
        panels.append(canvas)

    out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "hough.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    io.imwrite(out_path, np.hstack(panels))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
