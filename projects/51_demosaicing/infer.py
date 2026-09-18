"""Mosaic an image of your own and recover it six ways — with real PSNR.

    python infer.py photo.jpg
    python infer.py photo.jpg --pattern GRBG --noise 5
    python infer.py --photo flounder_on_gravel --method "VNG (gradient)" --out out.png

This is one of the few places in this repository where an arbitrary input gets a
**real** score: the mosaic is made from your image, so your image *is* the ground
truth the methods are trying to recover.

Two numbers are printed beside the usual PSNR, and they are the interesting ones:

* your image's **saturated-edge share** — the fraction of pixels that are both an
  edge and far from grey. That is where demosaicing error lives, and it predicts
  how far the methods will separate before any of them runs.
* **PSNR on edge pixels alone**, which on this project's twelve photographs is
  2.1 to 2.8 dB below the whole-image figure for every method. A whole-image
  average is dominated by the smooth regions where the problem does not exist.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import numpy as np  # noqa: E402

from shared import figures, io, synth  # noqa: E402
from shared.io import ensure_rgb  # noqa: E402
from shared.metrics import psnr  # noqa: E402
from shared.report import init_console  # noqa: E402

import demosaicing as dm  # noqa: E402

#: Saturated-edge share above which the methods will clearly separate. The
#: control in this project's pool sits at 0.0% and the hardest at 71.8%.
SEPARATES = 5.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", default=None)
    ap.add_argument("--photo", default=None, choices=list(dm.IMAGES))
    ap.add_argument("--pattern", default="RGGB", choices=list(dm.PATTERNS))
    ap.add_argument("--noise", type=float, default=0.0,
                    help="sensor noise sigma, added to the raw mosaic")
    ap.add_argument("--method", default=None, choices=list(dm.METHODS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    init_console()
    if args.photo:
        truth = dm.load_scene(args.photo)
        source = args.photo
    elif args.image:
        truth = io.imread(args.image)
        source = args.image
    else:
        ap.error("give an image path, or --photo with one of the project's photographs")

    share = dm.saturated_edge_share(truth)
    print(f"{source}  {truth.shape[1]}x{truth.shape[0]}  "
          f"saturated edges {share:.1f}% of pixels")
    if share < SEPARATES:
        print(f"  under {SEPARATES:g}% — there is little for the methods to disagree "
              "about here. Expect them to cluster, and read the edge column rather "
              "than the whole-image one.")
    else:
        print("  enough saturated edge for the cross-channel methods to show their "
              "advantage clearly.")

    raw = dm.mosaic(truth, args.pattern)
    if args.noise > 0:
        raw = synth.gaussian_noise(raw, sigma=args.noise, seed=0)
        print(f"  sensor noise sigma {args.noise:g} added to the raw mosaic, before any "
              "colour exists")
    print(f"  {args.pattern} mosaic: two thirds of the colour is now missing")

    edges = dm.edge_mask(truth)
    print(f"\n{'method':24s} {'PSNR':>8s} {'on edges':>9s} {'penalty':>8s} "
          f"{'fringing':>9s} {'ms':>8s}")

    panels = [("ground truth", truth), (f"{args.pattern} mosaic", ensure_rgb(raw))]
    results = {}
    for name, fn in dm.METHODS.items():
        t0 = time.perf_counter()
        out = fn(raw, args.pattern)
        ms = (time.perf_counter() - t0) * 1000
        whole = psnr(out, truth)
        on_edges = dm.psnr_on_mask(out, truth, edges)
        fringing = dm.colour_fringing(out, truth, edges)
        results[name] = (out, whole, on_edges, fringing)
        print(f"{name:24s} {whole:8.2f} {on_edges:9.2f} {whole - on_edges:+8.2f} "
              f"{fringing:9.2f} {ms:8.2f}")
        panels.append((f"{name}\n{whole:.2f} dB", ensure_rgb(out)))

    # ------------------------------------------------------------------ #
    # what to make of it
    # ------------------------------------------------------------------ #
    print()
    best = max(results, key=lambda n: results[n][1])
    worst = min(results, key=lambda n: results[n][1])
    print(f"best: {best} at {results[best][1]:.2f} dB; "
          f"worst: {worst} at {results[worst][1]:.2f} — "
          f"{results[best][1] - results[worst][1]:.2f} dB of interpolation")

    penalties = {n: v[1] - v[2] for n, v in results.items()}
    print(f"\nevery method scores worse on edge pixels, by "
          f"{min(penalties.values()):.2f} to {max(penalties.values()):.2f} dB.")
    print("  that is the failure the whole-image average hides: most of a photograph "
          "is smooth, and interpolating a missing colour across smooth ground is "
          "trivial.")

    cross = results["Malvar (cross-channel)"]
    bilinear = results["Bilinear (OpenCV)"]
    print(f"\nusing green to guide red and blue is worth "
          f"{cross[1] - bilinear[1]:+.2f} dB here, and takes colour fringing from "
          f"{bilinear[3]:.2f} to {cross[3]:.2f}")

    ea = results["Edge-aware"]
    if abs(ea[2] - bilinear[2]) < 0.5:
        print(f"\nOpenCV's edge-aware flag scores {ea[2]:.2f} on edges against its "
              f"bilinear's {bilinear[2]:.2f} — no help on the pixels it is named for, "
              "which matches what this project finds across twelve photographs.")

    if args.method:
        out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "demosaiced.png"
        io.imwrite(out_path, results[args.method][0])
    else:
        out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "methods.png"
        figures.grid(panels, out_path, ncols=4,
                     suptitle=f"Six reconstructions of {Path(source).name} "
                              f"from a {args.pattern} mosaic")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
