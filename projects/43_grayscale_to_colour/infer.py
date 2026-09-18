"""Colourise a grayscale image — and see the ceiling before you see the results.

    python infer.py --image woman_in_a_red_top
    python infer.py --image otter_on_a_log --scribbles 100
    python infer.py --image seal_on_grey_ice --reference glacier_cave_mouth
    python infer.py photo.jpg

The first thing printed is the photograph's **greyscale ambiguity**: how much
chroma variation survives inside one luminance level. On this project's twelve it
predicts what even an oracle cannot recover, at r = 0.995 — so it tells you the
ceiling before any method runs.

On one of the project's photographs there is an exact truth and every method gets
a real score, including the **do-nothing control** that returns the grey image. On
your own photograph there is no truth, so what you get is the outputs and how
much colour each invented.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import numpy as np  # noqa: E402

from shared import figures, io  # noqa: E402
from shared.report import init_console  # noqa: E402

import colourise as co  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default=None,
                    help="your own image; it is converted to grey and colourised")
    ap.add_argument("--image", default=None, choices=list(co.IMAGES),
                    help="one of the project's twelve, which has an exact truth")
    ap.add_argument("--reference", default=None, choices=list(co.IMAGES),
                    help="the reference Welsh transfer is given")
    ap.add_argument("--scribbles", type=int, default=co.N_SCRIBBLES)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    init_console()

    if args.image:
        truth = co.load(args.image)
        name = args.image
        has_truth = True
    elif args.path:
        truth = io.imread(args.path)
        name = Path(args.path).name
        has_truth = True  # we made the grey ourselves, so the original is the truth
    else:
        ap.error("give an image path or --image with one of the project's twelve")

    grey = co.greyscale(truth)
    reference = co.load(args.reference) if args.reference else \
        co.load(co.reference_for(args.image)) if args.image else co.load(co.IMAGES[0])
    reference_name = (args.reference or (co.reference_for(args.image) if args.image
                                         else co.IMAGES[0]))

    amb = co.ambiguity(truth)
    print(f"{name}  {truth.shape[1]}x{truth.shape[0]}")
    print(f"  greyscale ambiguity {amb:.2f} Lab units")
    fit = co.axis_predicts_the_ceiling()
    predicted = fit["slope"] * amb + fit["intercept"]
    print(f"  on this project's twelve, ambiguity predicts the oracle's residual at "
          f"r = {fit['pearson_r']:.3f}")
    print(f"  so the ceiling here is about {predicted:.2f} chroma error — no method "
          "that works from luminance alone should be expected to beat it")
    if amb > 12:
        print("  that is a high-ambiguity photograph: the same grey means several "
              "different colours in different places")

    print(f"\n  Welsh transfer's reference: {reference_name}")
    print(f"\n{'method':32s} {'chroma error':>13s} {'PSNR (dB)':>10s} "
          f"{'colourfulness':>14s}")

    panels = [("the original (truth)", truth), ("the input: no chroma", grey)]
    results = {}
    for method in co.METHODS:
        kw = {"n_scribbles": args.scribbles} if "Levin" in method else {}
        prediction = co.METHODS[method](grey, truth=truth, reference=reference, **kw)
        error = co.chroma_error(prediction, truth) if has_truth else float("nan")
        psnr = co.rgb_psnr(prediction, truth) if has_truth else float("nan")
        vivid = co.colourfulness(prediction)
        results[method] = (prediction, error, vivid)
        print(f"{method:32s} {error:13.3f} {psnr:10.3f} {vivid:14.2f}")
        if method != "Do nothing (grey, control)":
            panels.append((f"{method}\n{error:.1f}" if has_truth else method,
                           prediction))

    # ------------------------------------------------------------------ #
    control = results["Do nothing (grey, control)"][1]
    worse = [m for m, v in results.items()
             if m != "Do nothing (grey, control)" and v[1] > control]
    print(f"\nreturning the grey image scores {control:.3f}")
    if worse:
        print(f"  and beats {len(worse)} method(s) here: {', '.join(worse)}")
        for m in worse:
            vivid = results[m][2]
            why = ("a lot of colour, and the wrong colour" if vivid > 25
                   else "not much colour, and still the wrong colour")
            print(f"    {m:32s} {results[m][1]:7.3f} at colourfulness "
                  f"{vivid:.1f} — {why}")

    oracles = [m for m in co.METHODS if m in co.ORACLES]
    best = min(results, key=lambda m: results[m][1])
    print(f"\nbest here: {best} ({results[best][1]:.3f})"
          + ("  — but it is an oracle, given the truth in some form"
             if best in oracles else ""))
    honest = min((m for m in results if m not in oracles
                  and m != "Do nothing (grey, control)"),
                 key=lambda m: results[m][1])
    print(f"best that is not given the answer: {honest} ({results[honest][1]:.3f})")
    if results[honest][1] > control:
        print("  which is worse than doing nothing.")

    levin = results["Levin scribbles (40)"][1]
    lookup = results["Luminance lookup (oracle)"][1]
    if levin < lookup:
        print(f"\n{args.scribbles} scribbles ({levin:.2f}) beat knowing this image's "
              f"entire true luminance-to-colour mapping ({lookup:.2f})")
        print("  where the colour is beats what the colour is")

    out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "colourised.png"
    figures.grid(panels, out_path, ncols=4,
                 suptitle=f"Five colourisations of {name} (ambiguity {amb:.1f})")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
