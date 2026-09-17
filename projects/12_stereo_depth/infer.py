"""Compute a disparity map from your own stereo pair.

    python infer.py left.png right.png
    python infer.py left.png right.png --method "Semi-global, 8-path (HH)"
    python infer.py photo.jpg --simulate       # make a pair and score it

With a real pair there is no ground truth, so **no accuracy is printed**. What
is printed is what needs no reference: how much of the frame the matcher
answered, where it declined, and how much of the disparity range it used.

The density number is the useful one. A matcher that answers 40% of the frame
has not failed quietly — it has told you it is unsure, and that is worth more
than a confident wrong answer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import numpy as np  # noqa: E402

from shared import io, synth  # noqa: E402

import stereo as st  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", help="left and right views, or one image with --simulate")
    ap.add_argument("--method", default="Semi-global, 8-path (HH)", choices=list(st.METHODS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--simulate", action="store_true",
                    help="build a stereo pair from a single photograph and score it")
    args = ap.parse_args()

    truth = valid = None
    if args.simulate:
        if len(args.images) != 1:
            print("--simulate takes exactly one image", file=sys.stderr)
            return 2
        left, right, truth, valid = synth.stereo_pair(io.imread(args.images[0]), max_disparity=48)
        print("simulated a stereo pair with a 4-layer depth map, max disparity 48 px")
    else:
        if len(args.images) != 2:
            print("give a left and a right image, or one image with --simulate",
                  file=sys.stderr)
            return 2
        left, right = (io.imread(p) for p in args.images)
        if left.shape != right.shape:
            print(f"the views are different sizes: {left.shape} vs {right.shape}. "
                  "A stereo pair must be rectified to the same size first.", file=sys.stderr)
            return 2

    disparity = st.METHODS[args.method](left, right)
    answered = np.isfinite(disparity)
    print(f"\nmatcher   : {args.method}")
    print(f"density   : {100 * answered.mean():.1f}% of the frame answered")
    if answered.any():
        d = disparity[answered]
        print(f"disparity : {d.min():.1f} to {d.max():.1f} px "
              f"(search range 0 to {st.MAX_DISPARITY})")
        if d.max() > st.MAX_DISPARITY - 2:
            print("  -> the maximum is at the edge of the search range. Anything "
                  "nearer than this is being truncated, not measured.")

    if truth is not None:
        s = st.score_disparity(disparity, truth, valid)
        print(f"\nbad 2px, of what it answered : {100 * s['bad2_answered']:.2f}%")
        print(f"bad 2px, counting refusals   : {100 * s['bad2_all']:.2f}%")
        print(f"mean absolute error          : {s['mae_px']:.3f} px")
        regions = st.error_regions(truth, valid, left)
        print("\nby region:")
        for name, mask in regions.items():
            r = st.score_disparity(disparity, truth, mask)
            print(f"  {name:16s} {100 * r['bad2_all']:5.1f}% bad "
                  f"({100 * float(mask.mean()):4.1f}% of the frame)")
    else:
        print("\nNo accuracy is reported. This is a real pair and nobody recorded "
              "its true disparity, so any number here would be invented. Run with "
              "--simulate on a single photograph to see the method scored.")

    vmax = float(np.nanmax(disparity)) if answered.any() else 1.0
    from run import colourise

    out = Path(args.out) if args.out else PROJECT_DIR / "results" / "disparity.png"
    io.imwrite(out, colourise(disparity, vmax))
    print(f"\nwrote {out}  (black = not answered)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
