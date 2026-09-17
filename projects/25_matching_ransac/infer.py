"""Match two of your own images and estimate the homography between them.

    python infer.py a.jpg b.jpg
    python infer.py photo.jpg --simulate               # warp by a known H, then score
    python infer.py a.jpg b.jpg --descriptor ORB --estimator LMEDS --out matches.png

With two real photographs the true homography is unknown, so **no reprojection
error is printed**. What is printed is each estimator's inlier count and inlier
*ratio* — and that number is exactly the self-report this project exists to warn
about: an estimator that keeps four points and fits them perfectly reports a
flawless inlier ratio and an arbitrarily wrong transform.

`--simulate` applies a known homography to one image, so every match can be
labelled before any estimator runs and the errors mean something.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import numpy as np  # noqa: E402

from shared import io, synth  # noqa: E402
from shared.io import to_gray  # noqa: E402
from shared.metrics import entropy  # noqa: E402
from shared.report import init_console  # noqa: E402

import matching as mt  # noqa: E402

#: Inlier ratio below which a homography estimate is not worth trusting, whatever
#: the estimator reports. Four points determine a homography, so a handful of
#: agreeing matches proves nothing.
MIN_INLIER_RATIO = 0.25


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("first")
    ap.add_argument("second", nargs="?", default=None)
    ap.add_argument("--descriptor", default="SIFT", choices=list(mt.DESCRIPTORS))
    ap.add_argument("--estimator", default=None, choices=list(mt.ESTIMATORS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--simulate", action="store_true",
                    help="warp the first image by a known homography instead of "
                         "reading a second, so the errors can be scored")
    ap.add_argument("--outliers", type=float, default=0.0,
                    help="with --simulate, the fraction of matches to spoil")
    args = ap.parse_args()

    init_console()
    a = io.imread(args.first)
    print(f"{args.first}  {a.shape[1]}x{a.shape[0]}  "
          f"{entropy(to_gray(a)):.2f} bits of entropy")

    H = None
    if args.simulate:
        H = synth.random_homography(a.shape, jitter=0.08, seed=0)
        b = synth.warp_by_homography(a, H)
        print(f"warped by a known homography; {args.outliers:.0%} of the matches "
              "will be replaced with random ones")
    else:
        if args.second is None:
            ap.error("give a second image, or pass --simulate")
        b = io.imread(args.second)
        print(f"{args.second}  {b.shape[1]}x{b.shape[0]}")

    kp_a, desc_a, norm = mt.DESCRIPTORS[args.descriptor](to_gray(a))
    kp_b, desc_b, _ = mt.DESCRIPTORS[args.descriptor](to_gray(b))
    if desc_a is None or desc_b is None:
        print("no descriptors found in one of the images")
        return 1

    matches = mt.match_ratio(desc_a, desc_b, norm)
    print(f"{args.descriptor}: {len(kp_a)} and {len(kp_b)} keypoints, "
          f"{len(matches)} matches after the 0.75 ratio test")

    if H is not None:
        src, dst, truth = mt.label_matches(kp_a, kp_b, matches, H)
        src, dst, truth = mt.inject_outliers(src, dst, truth, args.outliers,
                                             a.shape, seed=0)
        print(f"of {len(src)} matches, {int(truth.sum())} are true inliers "
              f"({truth.mean():.1%})")
    else:
        src = np.float32([kp_a[m.queryIdx].pt for m in matches])
        dst = np.float32([kp_b[m.trainIdx].pt for m in matches])
        truth = None

    names = [args.estimator] if args.estimator else list(mt.ESTIMATORS)
    header = f"\n{'estimator':26s} {'inliers':>9s} {'ratio':>8s} {'ms':>8s}"
    if truth is not None:
        header += f" {'error px':>10s}"
    print(header)

    results = {}
    for name in names:
        t0 = time.perf_counter()
        Hest, mask = mt.ESTIMATORS[name](src, dst)
        ms = (time.perf_counter() - t0) * 1000
        kept = int(mask.sum()) if mask is not None else len(src)
        ratio = kept / max(len(src), 1)
        line = f"{name:26s} {kept:9d} {ratio:8.1%} {ms:8.1f}"
        if truth is not None:
            err = (float("nan") if Hest is None or truth.sum() < 4
                   else mt.reprojection_error(Hest, src[truth], dst[truth]))
            results[name] = err
            line += f" {err:10.3f}"
        print(line)

    if truth is not None:
        best = min(results, key=lambda n: results[n])
        print(f"\nbest: {best} at {results[best]:.3f} px against the TRUE "
              "homography, on the TRUE correspondences.")
        if args.outliers >= 0.5:
            print("  At 50% outliers LMEDS is at its theoretical breakdown point; "
                  "above it the median it minimises is itself an outlier.")
    else:
        print("\nNo reprojection error is reported. The homography between two "
              "real photographs is unknown, so any error here would be measured "
              "against the estimate itself.")
        best = max(names, key=lambda n: (mt.ESTIMATORS[n](src, dst)[1] or
                                         np.ones(len(src), bool)).sum())
        print("The inlier ratios above are each estimator's own opinion of its "
              "own answer. Treat a high one as necessary, never as sufficient: "
              "four agreeing points determine a homography and prove nothing.")
        ratios = {}
        for n in names:
            _, m = mt.ESTIMATORS[n](src, dst)
            ratios[n] = (int(m.sum()) if m is not None else len(src)) / max(len(src), 1)
        weak = [n for n, r in ratios.items() if r < MIN_INLIER_RATIO]
        if weak:
            print(f"  Below {MIN_INLIER_RATIO:.0%} inliers, and therefore not worth "
                  f"trusting at all: {', '.join(weak)}")
        print("Run with --simulate to see the estimators scored against a known "
              "homography, with an outlier fraction you choose.")

    out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "matches.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if H is not None:
        panels = [mt.estimate_and_draw(a, b, H, n, args.descriptor,
                                       args.outliers, seed=0)[2] for n in names]
        io.imwrite(out_path, panels[0] if args.estimator else np.hstack(panels))
    else:
        io.imwrite(out_path, np.hstack([a, b[:a.shape[0], :a.shape[1]]]))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
