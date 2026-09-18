"""Estimate F on one stereo pair — and compare what each method reports to the truth.

    python infer.py --view 7                # every estimator on one pair
    python infer.py --view 7 --board-only   # the degenerate case, live
    python infer.py --view 7 --detector AKAZE
    python infer.py --view 7 --threshold 8  # watch the inlier rate improve

Two columns matter and they are printed side by side:

* **its own error** — how well the matrix explains the correspondences it was
  fitted to, plus RANSAC's inlier rate. This is what a pipeline logs.
* **board error** — how well it explains 702 chessboard corners it has never
  seen, from a calibration that shares no measurement with it.

`--board-only` restricts the correspondences to the chessboard, which is a single
plane and therefore cannot determine F at all. Watch the first column stay
healthy while the second falls apart.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from shared import figures  # noqa: E402
from shared.io import ensure_rgb  # noqa: E402
from shared.report import init_console  # noqa: E402

import epipolar as ep  # noqa: E402

N_LINES = 9


def _lines_drawn(image, F, points, colour):
    out = ensure_rgb(image).copy()
    if F is not None:
        for (a, b) in ep.epipolar_lines(F, points, image.shape):
            cv2.line(out, a, b, colour, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--view", type=int, default=7)
    ap.add_argument("--detector", default="SIFT", choices=["SIFT", "ORB", "AKAZE"])
    ap.add_argument("--threshold", type=float, default=1.0,
                    help="RANSAC / LMedS threshold in pixels")
    ap.add_argument("--board-only", action="store_true",
                    help="use only chessboard matches: one plane, which cannot "
                         "determine F")
    ap.add_argument("--keep-board", action="store_true",
                    help="do not exclude the board (the default excludes it)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    init_console()
    if not ep.assets_available():
        ap.error("The calibration images are not cached. Run "
                 "`python tools/fetch_assets.py --set calibration` first.")

    pairs = ep.usable_pairs()
    if args.view not in pairs:
        ap.error(f"view {args.view} has no usable pair; available: {pairs}")

    truth = ep.calibration_truth()
    left, right = ep.undistorted_pair(args.view)

    if args.board_only:
        p1, p2 = ep.match_features(args.view, args.detector)
        keep = ep._on_board(args.view, p1, p2)
        p1, p2 = p1[keep], p2[keep]
        source = "chessboard matches only (one plane — degenerate)"
    else:
        p1, p2 = ep.correspondences(args.view, args.detector,
                                    exclude_board=not args.keep_board)
        source = ("all matches" if args.keep_board else "matches with the board removed")

    share = ep.board_share(args.view, args.detector)
    print(f"view {args.view}, {args.detector}, {left.shape[1]}x{left.shape[0]}")
    print(f"  {share['matches']} matches, {share['on_board']} on the board "
          f"({share['share_on_board']:.1%})")
    print(f"  using: {source} — {len(p1)} correspondences")
    if len(p1) < 8:
        ap.error(f"only {len(p1)} correspondences; F needs at least 8")
    if args.board_only:
        print("  NOTE: points on a single plane do not determine F. Any "
              "homography-compatible matrix fits them exactly, and the inlier rate "
              "below will not notice.")

    print(f"\n  truth F (from the chessboard calibration alone) puts all "
          f"{len(truth['board_left'])} board corners "
          f"{ep.board_error(truth['F']):.4f} px from their lines")

    print(f"\n{'estimator':50s} {'own error':>10s} {'inliers':>9s} "
          f"{'board error':>12s}")

    results = {}
    for name, fn in ep.ESTIMATORS.items():
        try:
            F, mask = fn(p1, p2, args.threshold)
        except Exception:  # noqa: BLE001
            F, mask = None, None
        if F is None or not np.all(np.isfinite(F)):
            print(f"{name:50s} {'failed':>10s}")
            continue
        own = ep.symmetric_epipolar_distance(F, p1, p2)
        board = ep.board_error(F)
        rate = f"{mask.mean():9.3f}" if mask is not None else f"{'—':>9s}"
        results[name] = (F, own, board)
        print(f"{name:50s} {own:10.3f} {rate} {board:12.3f}")

    print(f"{'Calibration (truth)':50s} {'—':>10s} {'—':>9s} "
          f"{ep.board_error(truth['F']):12.3f}")

    # ------------------------------------------------------------------ #
    best = min(results, key=lambda n: results[n][2])
    best_own = min(results, key=lambda n: results[n][1])
    print(f"\nbest by board error (the honest one): {best} "
          f"({results[best][2]:.3f} px)")
    print(f"best by its own error (what a log would show): {best_own} "
          f"({results[best_own][1]:.3f} px, board {results[best_own][2]:.3f} px)")
    if best != best_own:
        print("  — they disagree. Fitting its own correspondences well is not the "
              "same as being right.")

    control = "Assume rectified + offset (1 parameter, control)"
    if control in results:
        beaten = [n for n, v in results.items()
                  if "control" not in n and v[2] > results[control][2]]
        if beaten:
            print(f"\nthe one-parameter control ({results[control][2]:.3f} px) beats "
                  f"{len(beaten)} of the estimators here: {', '.join(beaten)}")
            print("  it is a median of the vertical disparity. This rig is nearly "
                  "rectified, so one number captures nearly all of the geometry.")

    if args.board_only and "RANSAC" in results:
        _, mask = ep.opencv_ransac(p1, p2, args.threshold)
        if mask is not None:
            print(f"\nRANSAC called {mask.mean():.1%} of these coplanar "
                  f"correspondences inliers, and its board error is "
                  f"{results['RANSAC'][2]:.3f} px. The inlier rate is not a quality "
                  "measure; it is a report about the threshold.")

    # ------------------------------------------------------------------ #
    idx = np.linspace(0, len(p1) - 1, min(N_LINES, len(p1))).round().astype(int)
    drawn = p1[idx]
    panels = [("truth (chessboard calibration)",
               _lines_drawn(right, truth["F"], drawn, (60, 200, 60)))]
    for name in ("8-point, raw pixels", "8-point, normalised (Hartley)", "RANSAC",
                 control):
        if name in results:
            panels.append((f"{name}\n{results[name][2]:.2f} px",
                           _lines_drawn(right, results[name][0], drawn, (220, 60, 60))))

    out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "epipolar.png"
    figures.grid(panels, out_path, ncols=3,
                 suptitle=f"Epipolar lines in right{args.view:02d} — {source}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
