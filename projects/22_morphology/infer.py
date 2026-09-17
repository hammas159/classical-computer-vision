"""Run every morphological operation on your own image at once.

    python infer.py photo.jpg
    python infer.py photo.jpg --size 7 --shape cross
    python infer.py photo.jpg --simulate          # add known noise, then score
    python infer.py photo.jpg --operation Open --out cleaned.png

Morphology is defined on binary images, so the photograph is binarised with Otsu
first and that binarisation is what the operations act on.

Without `--simulate` there is nothing to score against, so **no IoU is printed**
— only how much foreground each operation kept or added, and the size of the
smallest structure in your image, which is the number that decides whether
opening will help or destroy it.
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
from shared.report import init_console  # noqa: E402

import morphology as mo  # noqa: E402


def structure_width(mask: np.ndarray) -> float:
    """Typical foreground width in pixels: twice the 90th-percentile distance
    to background. The 90th rather than the maximum, so one large blob does not
    speak for a picture made of twigs."""
    m = (mask > 0).astype(np.uint8)
    if not m.any():
        return float("nan")
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    return float(np.percentile(dist[m > 0], 90)) * 2.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--operation", default=None, choices=list(mo.OPERATIONS))
    ap.add_argument("--shape", default="ellipse", choices=["rect", "ellipse", "cross"])
    ap.add_argument("--size", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--simulate", action="store_true",
                    help="treat the binarisation as truth, add salt-and-pepper "
                         "noise, and score every operation at removing it")
    ap.add_argument("--noise", type=float, default=mo.PHOTO_NOISE)
    args = ap.parse_args()

    init_console()
    img = io.imread(args.image)
    gray = to_gray(img)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"{args.image}  {img.shape[1]}x{img.shape[0]}  "
          f"binarised: {100 * float((binary > 0).mean()):.1f}% foreground")

    truth = None
    if args.simulate:
        truth = binary
        rng = np.random.default_rng(0)
        work = truth.copy()
        r = rng.random(truth.shape)
        work[r < args.noise / 2] = 255
        work[r > 1.0 - args.noise / 2] = 0
        print(f"added {args.noise:.0%} salt-and-pepper — the input is now "
              f"{iou(work > 0, truth > 0):.4f} IoU from the truth")
    else:
        work = binary

    width = structure_width(work)
    se = mo.element(args.shape, args.size)
    print(f"typical foreground width: {width:.1f} px, element: "
          f"{args.shape} {args.size}x{args.size}")
    if width < args.size:
        print("  -> the element is WIDER than the typical structure. Opening "
              "will delete most of the image; that is the definition, not a bug.")
    else:
        print("  -> the element fits inside the typical structure, so opening "
              "will keep it and remove smaller things.")

    names = [args.operation] if args.operation else list(mo.OPERATIONS)
    header = f"\n{'operation':16s} {'foreground %':>13s} {'vs input':>9s}"
    if truth is not None:
        header += f" {'IoU':>8s}"
    print(header)

    base_fg = float((work > 0).mean())
    outs = {}
    if truth is not None:
        med = cv2.medianBlur(work, args.size if args.size % 2 else args.size + 1)
        outs["Median filter"] = med
        print(f"{'Median filter':16s} {100 * float((med > 0).mean()):12.2f}% "
              f"{float((med > 0).mean()) / max(base_fg, 1e-9):8.2f}x "
              f"{iou(med > 0, truth > 0):8.4f}")
        print(f"{'Do nothing':16s} {100 * base_fg:12.2f}% {1.0:8.2f}x "
              f"{iou(work > 0, truth > 0):8.4f}")

    for name in names:
        out = mo.OPERATIONS[name](work, se)
        outs[name] = out
        fg = float((out > 0).mean())
        line = (f"{name:16s} {100 * fg:12.2f}% "
                f"{fg / max(base_fg, 1e-9):8.2f}x")
        if truth is not None:
            line += f" {iou(out > 0, truth > 0):8.4f}"
        print(line)

    if truth is not None:
        scored = {n: iou(o > 0, truth > 0) for n, o in outs.items()}
        control = iou(work > 0, truth > 0)
        best = max(scored, key=scored.get)
        print(f"\nbest: {best} at {scored[best]:.4f}; doing nothing scores {control:.4f}.")
        if scored[best] <= control:
            print("  Nothing beat leaving it alone — which is what happens on all "
                  "twelve of this project's photographs, at every element size. "
                  "A real binarisation has genuine single-pixel structure, and "
                  "removing noise removes that too.")
    else:
        print("\nNo IoU is reported. Without the added noise there is nothing to "
              "score against — the binarisation is the only version that exists.")
        print("Run with --simulate to treat the binarisation as truth, add known "
              "noise, and see every operation scored at removing it.")

    out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "morphology.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panels = [io.ensure_rgb(outs[n]) for n in names]
    io.imwrite(out_path, panels[0] if args.operation else np.hstack(panels))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
