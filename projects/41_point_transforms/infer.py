"""Apply any point transform to your own image, and see what it costs first.

    python infer.py photo.jpg
    python infer.py photo.jpg --transform "Gamma 0.5 (brighten)" --out out.png
    python infer.py photo.jpg --gamma 0.6

The cost of a point transform is a property of its 256-entry table, not of your
picture, so it is printed **before** the image is touched:

* **levels surviving** — how many of the 256 input values map somewhere distinct;
* **longest run** — the most input levels collapsed onto one output. This is
  detail loss, and it makes the result smoother;
* **largest gap** — the biggest jump between the outputs of two *adjacent*
  inputs. This is banding, and it makes a gradient into steps.

The last two are opposite failures from opposite halves of the same curve, and a
single "information loss" number cannot tell them apart: log and inverse log lose
exactly the same 131 levels, one by banding and one by smearing.

Then your image is measured, because a curve that collapses levels your
photograph does not contain costs you nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import numpy as np  # noqa: E402

from shared import figures, io  # noqa: E402
from shared.io import ensure_rgb, to_gray  # noqa: E402
from shared.metrics import entropy, rms_contrast  # noqa: E402
from shared.report import init_console  # noqa: E402

import point_transforms as pt  # noqa: E402

#: A gap this wide between adjacent output levels is visible as banding on a
#: smooth gradient. Gamma 0.5 reaches 16.
BANDING_GAP = 8

#: A run this long merges enough neighbouring levels to flatten visible texture.
SMEARING_RUN = 8


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", default=None)
    ap.add_argument("--photo", default=None, choices=list(pt.IMAGES))
    ap.add_argument("--transform", default=None, choices=list(pt.LUTS))
    ap.add_argument("--gamma", type=float, default=None,
                    help="build a power-law table with this exponent")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    init_console()
    if args.photo:
        img = pt.load_scene(args.photo)
        source = args.photo
    elif args.image:
        img = io.imread(args.image)
        source = args.image
    else:
        ap.error("give an image path, or --photo with one of the project's photographs")

    gray = to_gray(img)
    present = int(len(np.unique(gray)))
    print(f"{source}  {img.shape[1]}x{img.shape[0]}  mean {gray.mean():.1f}  "
          f"{entropy(gray):.3f} bits")
    print(f"your image uses {present} of the 256 possible levels")
    if present < 200:
        print(f"  {256 - present} levels are absent already, so a curve that collapses "
              "them costs you nothing — the table's figures below are an upper bound")

    tables = dict(pt.LUTS)
    if args.gamma is not None:
        tables[f"Gamma {args.gamma:g} (custom)"] = lambda g=args.gamma: pt.lut_power(g)

    print(f"\n--- what each table costs, before your image is touched ---")
    print(f"{'transform':24s} {'levels':>7s} {'run':>5s} {'gap':>5s}  what it does to a gradient")
    costs = {}
    for name, build in tables.items():
        lut = build()
        run = pt.max_run_length(lut)
        gap = pt.max_output_gap(lut)
        costs[name] = (pt.levels_surviving(lut), run, gap)

        if gap >= BANDING_GAP and run >= SMEARING_RUN:
            verdict = "bands AND smears"
        elif gap >= BANDING_GAP:
            verdict = "bands (steps in smooth areas)"
        elif run >= SMEARING_RUN:
            verdict = "smears (flattens texture)"
        else:
            verdict = "neither, at this bit depth"
        print(f"{name:24s} {costs[name][0]:7d} {run:5d} {gap:5d}  {verdict}")

    print(f"\n--- and what it does to YOUR image ---")
    print(f"{'transform':24s} {'levels used':>12s} {'entropy':>9s} {'contrast':>9s}")
    panels, results = [("as given", img)], {}
    for name, build in tables.items():
        out = pt.apply_lut(img, build())
        results[name] = out
        used = int(len(np.unique(to_gray(out))))
        print(f"{name:24s} {used:12d} {entropy(to_gray(out)):9.3f} "
              f"{rms_contrast(to_gray(out)):9.4f}")
        panels.append((name, ensure_rgb(out)))

    # ------------------------------------------------------------------ #
    # what to make of it
    # ------------------------------------------------------------------ #
    bands = [n for n, (_, run, gap) in costs.items() if gap >= BANDING_GAP]
    smears = [n for n, (_, run, gap) in costs.items() if run >= SMEARING_RUN]
    print(f"\n{len(bands)} of {len(costs)} curves will band on a smooth gradient; "
          f"{len(smears)} will flatten texture.")
    both = set(bands) & set(smears)
    if both:
        print(f"  {', '.join(sorted(both))} do both — those are the quantising curves, "
              "where the loss is the point rather than a side effect.")

    if img.ndim == 3:
        posterised = int(len(np.unique(to_gray(results["Posterise (6 levels)"]))))
        if posterised > 6:
            print(f"\nNote: posterising to 6 levels leaves {posterised} distinct *greys*, "
                  "not 6. The table is applied to each colour channel independently, and "
                  "converting three separately-quantised channels to grey mixes them back "
                  "into intermediate values. The 6 is a per-channel figure.")

    lossless = [n for n, (levels, _, _) in costs.items() if levels == 256]
    print(f"\n{len(lossless)} curve(s) lose nothing at all: {', '.join(lossless)}. "
          "Every other mapping of 8-bit data onto 8-bit data throws something away, "
          "and the table says exactly how much before you run it.")

    if args.transform or args.gamma is not None:
        chosen = args.transform or f"Gamma {args.gamma:g} (custom)"
        out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "transformed.png"
        io.imwrite(out_path, results[chosen])
    else:
        out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "curves.png"
        figures.grid(panels, out_path, ncols=4,
                     suptitle=f"Every point transform applied to {Path(source).name}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
