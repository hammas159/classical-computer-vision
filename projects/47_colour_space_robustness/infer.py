"""Threshold a colour in five spaces and see which survives a change of light.

    python infer.py photo.jpg --x 320 --y 200
    python infer.py --photo red_sports_car --cast warm
    python infer.py photo.jpg --x 320 --y 200 --balance

You name a pixel; a threshold is built around that colour in each space and then
applied **unchanged** to a degraded copy. With `--photo` the region a person
traced is available, so IoU is real. On your own image there is none, so what is
printed instead is the **agreement between the degraded and undegraded
selections** — the same question the IoU column asks without needing a truth.

Two things worth knowing before reading the output:

* OpenCV's hue is **0-179**, not 0-359. A threshold written in degrees picks the
  wrong colour at every angle except red, silently.
* No colour space is invariant to a change of illuminant. On this project's
  twelve photographs a strong cast takes **every** space to an IoU of 0.000, and
  an explicit white balance step restores four of five to about 0.66. `--balance`
  runs that step so the difference is visible.
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
from shared.io import ensure_rgb  # noqa: E402
from shared.metrics import iou  # noqa: E402
from shared.report import init_console  # noqa: E402

import colour_spaces as cs  # noqa: E402

CASTS = {
    "warm": (1.25, 1.0, 0.75),
    "cool": (0.78, 1.0, 1.28),
    "none": (1.0, 1.0, 1.0),
}

#: Agreement below which a selection has effectively moved somewhere else.
UNSTABLE = 0.5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", default=None)
    ap.add_argument("--photo", default=None, choices=list(cs.IMAGES))
    ap.add_argument("--x", type=int, default=None)
    ap.add_argument("--y", type=int, default=None)
    ap.add_argument("--cast", default="warm", choices=list(CASTS))
    ap.add_argument("--brightness", type=float, default=1.0)
    ap.add_argument("--balance", action="store_true",
                    help="grey-world white balance the degraded image before thresholding")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    init_console()
    if args.photo:
        clean = cs.load_scene(args.photo)
        truth = cs.load_target(args.photo)
        source = args.photo
        print(f"{args.photo}: target is annotator "
              f"{cs.SEGMENTS[args.photo][0]}'s region {cs.SEGMENTS[args.photo][1]}, "
              f"chroma distance {cs.chroma_distance(clean, truth):.1f}")
    elif args.image:
        clean = io.imread(args.image)
        if args.x is None or args.y is None:
            ap.error("with your own image, give --x and --y to name the colour to find")
        seed = np.zeros(clean.shape[:2], np.uint8)
        cv2.circle(seed, (args.x, args.y), max(6, min(clean.shape[:2]) // 40), 255, -1)
        truth = seed
        source = args.image
        colour = clean[args.y, args.x]
        print(f"{args.image}: sampling around ({args.x}, {args.y}), "
              f"RGB {tuple(int(v) for v in colour)}")
    else:
        ap.error("give an image path with --x/--y, or --photo")

    degraded = clean
    if args.brightness != 1.0:
        degraded = cs.degrade_brightness(degraded, args.brightness)
    if args.cast != "none":
        degraded = cs.degrade_colour_cast(degraded, CASTS[args.cast])
    label = f"brightness x{args.brightness:g}, {args.cast} cast"

    if args.balance:
        mean = degraded.reshape(-1, 3).mean(0).astype(np.float64)
        gains = mean.mean() / np.maximum(mean, 1e-9)
        degraded = np.clip(degraded.astype(np.float64) * gains, 0, 255).astype(np.uint8)
        label += ", grey-world balanced"

    print(f"degradation: {label}")

    header = f"\n{'space':16s} {'tol':>5s} {'clean':>8s} {'degraded':>9s} {'agreement':>10s}"
    if args.photo:
        header += f" {'IoU kept':>9s}"
    print(header)

    panels = [("photograph", clean), ("the target", ensure_rgb(truth)),
              ("degraded", degraded)]
    results = {}
    for space in cs.SPACES:
        tol = cs.PHOTO_TOLERANCE[space]
        before = cs.threshold_in_space(clean, space, clean, truth, tol)
        after = cs.threshold_in_space(degraded, space, clean, truth, tol)
        agreement = iou(after, before)
        results[space] = (before, after, agreement)

        line = (f"{space:16s} {tol:5.0f} {iou(before, truth):8.3f} "
                f"{iou(after, truth):9.3f} {agreement:10.3f}")
        if args.photo:
            kept = iou(after, truth) / max(iou(before, truth), 1e-9)
            line += f" {kept:9.3f}"
        print(line)
        panels.append((f"{space}\nagreement {agreement:.3f}", ensure_rgb(after)))

    # ------------------------------------------------------------------ #
    # what to make of it
    # ------------------------------------------------------------------ #
    print()
    stable = [s for s, (_, _, a) in results.items() if a > UNSTABLE]
    print(f"{len(stable)} of {len(results)} spaces kept more than {UNSTABLE:g} agreement "
          f"with their own undegraded selection: {', '.join(stable) or 'none'}.")

    best = max(results, key=lambda s: results[s][2])
    worst = min(results, key=lambda s: results[s][2])
    print(f"  most stable here: {best} ({results[best][2]:.3f}); "
          f"least: {worst} ({results[worst][2]:.3f})")

    if args.cast != "none" and not args.balance:
        print("\nThis is a colour cast, which is the degradation no space survives — on "
              "this project's twelve photographs a strong one takes every space to "
              "0.000. Re-run with --balance to see what an explicit white balance step "
              "is worth; there it restores four of five to about 0.66.")
    elif args.balance:
        print("\nWith the cast corrected first, the selections should be close to their "
              "undegraded versions. That is the point: correct the illuminant, then pick "
              "a colour space for accuracy rather than for robustness.")

    if args.brightness != 1.0 and args.cast == "none":
        hsv = results["HSV"][2]
        print(f"\nBrightness only: HSV should be near 1.000 here and is {hsv:.3f}. Hue is "
              "an angle that scaling all three channels does not rotate — the half of "
              "the folklore that is true.")

    out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "spaces.png"
    figures.grid(panels, out_path, ncols=4,
                 suptitle=f"Colour thresholds on {Path(source).name} after {label}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
