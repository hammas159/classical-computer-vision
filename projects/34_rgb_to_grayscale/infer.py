"""Convert your own image to grayscale six ways, and find what the choice costs.

    python infer.py photo.jpg
    python infer.py photo.jpg --blind-spot
    python infer.py photo.jpg --method "BT.709 (HDTV)" --out gray.png

The six conversions almost always agree, and this prints by how much rather than
asserting it. On the twelve photographs in this project the whole
BT.601-versus-BT.709 argument is worth 2.18 grey levels.

`--blind-spot` is the interesting one. Every fixed weighting maps a whole plane
of colours to a single grey, so for any photograph there is a *worst* colour
edge — the one whose contrast the projection discards most. This finds it,
reports how much is left, and writes a picture of where those pixels are. On
this project's pool the worst pixel of twelve photographs still keeps 45% of its
contrast; if yours keeps much less, the conversion is a real decision for that
image.
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
from shared.io import ensure_rgb, to_float  # noqa: E402
from shared.report import init_console  # noqa: E402

import grayscale as gs  # noqa: E402

#: Retention below which a pixel's colour contrast has genuinely been damaged by
#: the projection rather than merely reweighted. The worst pixel across this
#: project's twelve photographs is 0.45, so anything under this is unusual.
DAMAGED = 0.5

#: Grey levels of separation Canny's default hysteresis needs on a step edge,
#: measured by `sweep_isoluminance`. Used only to say what a shortfall means.
CANNY_NEEDS = 28.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", default=None)
    ap.add_argument("--photo", default=None, choices=list(gs.IMAGES),
                    help="use one of the project's own photographs")
    ap.add_argument("--method", default=None, choices=list(gs.METHODS))
    ap.add_argument("--blind-spot", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    init_console()
    if args.photo:
        img = gs.load_scene(args.photo)
        source = args.photo
    elif args.image:
        img = io.imread(args.image)
        source = args.image
    else:
        ap.error("give an image path, or --photo with one of the project's photographs")

    c = gs.chroma(img)
    print(f"{source}  {img.shape[1]}x{img.shape[0]}  mean chroma {c:.1f}")
    if c < 10:
        print("  that is nearly monochrome — expect every conversion to agree exactly")

    reference = gs.gray_bt601(img)
    print(f"\n{'conversion':26s} {'levels from BT.601':>19s} {'contrast kept':>14s} "
          f"{'worst edge':>11s}")
    outputs, worst = {}, {}
    for name, fn in gs.METHODS.items():
        out = fn(img)
        outputs[name] = out
        diff = float(np.mean(np.abs(out.astype(np.float64) - reference.astype(np.float64))))
        kept = gs.contrast_retained(out, img)
        w = _worst_edge_retention(out, img)
        worst[name] = w
        print(f"{name:26s} {diff:19.2f} {kept:14.4f} {w:11.3f}")

    spread = max(
        float(np.mean(np.abs(outputs[a].astype(np.float64) - outputs[b].astype(np.float64))))
        for a in outputs for b in outputs
    )
    print(f"\nWidest disagreement between any two conversions: {spread:.2f} grey levels.")
    if spread < 8:
        print("  For this image the choice does not matter. Use the default.")
    else:
        print("  This image is saturated enough that the choice is visible — and it is "
              "usually `Value` doing it, since max(R,G,B) is not a luminance.")

    # ------------------------------------------------------------------ #
    # where the projection actually costs something
    # ------------------------------------------------------------------ #
    if args.blind_spot:
        name = args.method or "BT.601 (OpenCV default)"
        gray = outputs[name]
        ratio, strong = _retention_map(gray, img)
        damaged = strong & (ratio < DAMAGED)

        print(f"\n--- blind spot for {name} ---")
        print(f"strongest colour edges: {int(strong.sum()):,} pixels "
              f"(top {100 - gs.EDGE_PERCENTILE:g}% of colour gradient)")
        print(f"worst retention: {float(ratio[strong].min()):.3f}   "
              f"1st percentile: {float(np.percentile(ratio[strong], 1)):.3f}")
        print(f"pixels keeping under {DAMAGED:g} of their colour contrast: "
              f"{int(damaged.sum()):,} ({100 * damaged.sum() / max(strong.sum(), 1):.2f}%)")

        if damaged.any():
            ys, xs = np.nonzero(damaged)
            print(f"  concentrated around ({int(np.median(xs))}, {int(np.median(ys))}); "
                  "look there for two different hues at the same brightness")
            best = max(gs.METHODS, key=lambda m: worst[m])
            print(f"  {best} keeps the most at that edge ({worst[best]:.3f} "
                  f"against {worst[name]:.3f})")
        else:
            print("  none — every strong colour edge in this image survives the "
                  "projection, which is the usual outcome for a photograph.")

        overlay = img.copy()
        overlay[damaged] = (255, 40, 40)
        overlay[strong & ~damaged] = (
            0.6 * overlay[strong & ~damaged] + 0.4 * np.array([40, 200, 40])).astype(np.uint8)
        figures.grid(
            [("photograph", img),
             (f"{name}", ensure_rgb(gray)),
             (f"strong colour edges (green)\nlosing over {1 - DAMAGED:.0%} (red)", overlay)],
            PROJECT_DIR / "results" / "blind_spot.png", ncols=3,
            suptitle=f"Where {name} discards colour contrast in {Path(source).name}",
        )
        print(f"wrote {PROJECT_DIR / 'results' / 'blind_spot.png'}")

    if args.method:
        out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "gray.png"
        io.imwrite(out_path, outputs[args.method])
        print(f"\nwrote {out_path}")
    else:
        out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "conversions.png"
        figures.grid(
            [("photograph", img)] + [(n, ensure_rgb(o)) for n, o in outputs.items()],
            out_path, ncols=4,
            suptitle=f"Six conversions of {Path(source).name} — spread {spread:.2f} grey levels",
        )
        print(f"\nwrote {out_path}")
    return 0


def _retention_map(gray: np.ndarray, colour: np.ndarray):
    """Per-pixel fraction of colour contrast kept, and which pixels are strong edges."""
    ref = gs._colour_gradient(to_float(colour))
    strong = ref >= float(np.percentile(ref, gs.EDGE_PERCENTILE))
    g = to_float(gray)
    mag = np.sqrt(cv2.Sobel(g, cv2.CV_32F, 1, 0, 3) ** 2
                  + cv2.Sobel(g, cv2.CV_32F, 0, 1, 3) ** 2)
    ratio = mag / np.maximum(ref, gs.EPS) * np.sqrt(3.0)
    return ratio, strong


def _worst_edge_retention(gray: np.ndarray, colour: np.ndarray) -> float:
    ratio, strong = _retention_map(gray, colour)
    return float(np.percentile(ratio[strong], 1)) if strong.any() else 1.0


if __name__ == "__main__":
    sys.exit(main())
