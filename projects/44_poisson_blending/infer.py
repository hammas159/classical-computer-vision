"""Blend a region of one image of yours into another, five ways.

    python infer.py target.jpg source.jpg --x 300 --y 200
    python infer.py target.jpg source.jpg --method "Poisson (mixed gradients)"
    python infer.py --pair 0 --offset 0.5

The number to read is **how far each method moved your pasted pixels**. Poisson
blending works by changing them — on this project's pairs it moves them 51 grey
levels where alpha feathering moves them 4.5 — so a method that barely moved them
has not reconciled anything, whatever its seam score says.

Seam visibility is reported too, with the warning that it can be gamed: both
feathering and mixed gradients lower it by attenuating the boundary rather than
by making the two sides agree. There is no ground truth for a composite, so these
two numbers together, plus the picture, are what there is.
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
from shared.report import init_console  # noqa: E402

import poisson_blending as pb  # noqa: E402

#: Seam visibility above which the boundary is still plainly an edge.
VISIBLE_SEAM = 1.5

#: Pixels moved below which a method has not actually reconciled the two images,
#: whatever its seam score. Alpha feather sits at 4.5 on this project's pairs.
NOT_RECONCILED = 10.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", nargs="?", default=None)
    ap.add_argument("source", nargs="?", default=None)
    ap.add_argument("--pair", type=int, default=None,
                    help=f"use one of the project's {len(pb.PAIRS)} pairs (0-based)")
    ap.add_argument("--size", type=int, default=120)
    ap.add_argument("--x", type=int, default=None)
    ap.add_argument("--y", type=int, default=None)
    ap.add_argument("--offset", type=float, default=0.25)
    ap.add_argument("--method", default=None, choices=list(pb.METHODS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    init_console()
    if args.pair is not None:
        if not 0 <= args.pair < len(pb.PAIRS):
            ap.error(f"--pair must be 0..{len(pb.PAIRS) - 1}")
        names = pb.PAIRS[args.pair]
        target, source, mask, centre = pb.make_case(*names, size=args.size,
                                                    brightness_offset=args.offset)
        label = f"{names[0]} + {names[1]}"
    elif args.target and args.source:
        target = io.imread(args.target)
        source_full = io.imread(args.source)
        size = min(args.size, min(source_full.shape[:2]),
                   min(target.shape[:2]) - 2)
        sy = (source_full.shape[0] - size) // 2
        sx = (source_full.shape[1] - size) // 2
        source = source_full[sy:sy + size, sx:sx + size].copy()
        mask = np.zeros((size, size), np.uint8)
        cv2.circle(mask, (size // 2, size // 2), size // 2 - 2, 255, -1)
        centre = (args.x if args.x is not None else target.shape[1] // 2,
                  args.y if args.y is not None else target.shape[0] // 2)
        label = f"{Path(args.target).name} + {Path(args.source).name}"
    else:
        ap.error("give a target and a source image, or --pair")

    target_tone = pb.tonal_range(target)
    source_tone = pb.tonal_range(source)
    print(f"{label}")
    print(f"  target {target.shape[1]}x{target.shape[0]}, tone {target_tone:.0f}")
    print(f"  source patch {source.shape[1]}x{source.shape[0]}, tone {source_tone:.0f}")
    print(f"  tonal distance {abs(target_tone - source_tone):.0f} levels" +
          ("  — a wide mismatch, which is where the gradient domain earns most"
           if abs(target_tone - source_tone) > 60 else
           "  — a narrow mismatch, so copy-paste will not look as bad as usual"))

    print(f"\n{'method':32s} {'seam':>8s} {'pixels moved':>14s} {'ms':>9s}")
    panels, results = [("target", target)], {}
    for name, fn in pb.METHODS.items():
        import time
        t0 = time.perf_counter()
        out = fn(target, source, mask, centre)
        ms = (time.perf_counter() - t0) * 1000
        seam = pb.seam_visibility(out, mask, centre, target)
        moved = pb.pixel_fidelity(out, source, mask, centre, target)
        results[name] = (out, seam, moved)
        print(f"{name:32s} {seam:8.3f} {moved:14.2f} {ms:9.2f}")
        panels.append((f"{name}\nseam {seam:.3f}", out))

    # ------------------------------------------------------------------ #
    # what to make of it
    # ------------------------------------------------------------------ #
    control_seam = results["Copy-paste (control)"][1]
    print(f"\ncopy-paste leaves a seam {control_seam:.2f}x its surroundings.")

    still_visible = [n for n, (_, seam, _) in results.items()
                     if n != "Copy-paste (control)" and seam > VISIBLE_SEAM]
    if still_visible:
        print(f"  {len(still_visible)} method(s) still leave a visible edge "
              f"(over {VISIBLE_SEAM:g}): {', '.join(still_visible)}. "
              "That usually means the region straddles strong target structure the "
              "solver has to bend around.")

    untouched = [n for n, (_, _, moved) in results.items() if moved < NOT_RECONCILED]
    print(f"\n{len(untouched)} of {len(results)} methods moved your pasted pixels less "
          f"than {NOT_RECONCILED:g} levels: {', '.join(untouched)}.")
    print("  Those have not reconciled the two images — they have hidden the join. "
          "Poisson blending works by changing the region, so a low number here is a "
          "warning and not a virtue.")

    poisson = results["Poisson (OpenCV)"]
    feather = results["Alpha feather"]
    print(f"\nPoisson moved them {poisson[2]:.1f} levels against feathering's "
          f"{feather[2]:.1f}" +
          (f" — {poisson[2] / max(feather[2], 1e-9):.0f}x more"
           if feather[2] > 0 else ""))
    if poisson[1] > feather[1]:
        print("  and feathering scores the better seam here. That is the metric's bias, "
              "not a verdict: compare the two pictures rather than the two numbers.")

    mixed = results["Poisson (mixed gradients)"]
    if mixed[1] == min(v[1] for v in results.values()):
        print(f"\nmixed gradients has the lowest seam ({mixed[1]:.3f}) — check the picture "
              "before believing it. It keeps whichever gradient is stronger, so the "
              "target's texture bleeds through and the paste can end up transparent.")

    if args.method:
        out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "blended.png"
        io.imwrite(out_path, results[args.method][0])
    else:
        out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "blends.png"
        figures.grid(panels, out_path, ncols=3,
                     suptitle=f"Five composites of {label}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
