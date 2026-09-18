"""Calibrate from a subset of the views — and see the reported number lie.

    python infer.py                              # all thirteen views, default model
    python infer.py --views 3                    # the overfitting case, live
    python infer.py --model "Everything (14 coefficients, control)"
    python infer.py --image my_chessboard.jpg    # undistort your own 9x6 board photo

Three numbers are printed for every run, and the whole point of this project is
that they disagree:

* **RMS fitted** — what `cv2.calibrateCamera` returns and what gets quoted.
* **RMS held out** — the same measure on views the calibration never saw.
* **Straightness** — how far a board row sits off a straight line after
  undistortion. `calibrateCamera` never optimises this, so it is the only one of
  the three that is outside the objective.

With `--views 3` the first number is the best in the project and the second is the
worst.
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

import calibration as cal  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--views", type=int, default=None,
                    help="how many views to calibrate on (default: all usable)")
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--model", default="k1, k2, p1, p2 (OpenCV default)",
                    choices=list(cal.MODELS))
    ap.add_argument("--show", type=int, default=None,
                    help="which view to undistort (default: the first held-out one)")
    ap.add_argument("--image", default=None,
                    help="undistort your own photo of a 9x6 board with this calibration")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    init_console()
    if not cal.assets_available():
        ap.error("The calibration images are not cached. Run "
                 "`python tools/fetch_assets.py --set calibration` first.")

    usable = cal.usable_views(args.side)
    n = args.views or len(usable)
    if n < 3:
        ap.error("calibration needs at least 3 views")
    rng = np.random.default_rng(args.seed)
    order = list(usable)
    rng.shuffle(order)
    fit, held = sorted(order[:n]), sorted(order[n:])

    print(f"{args.side} camera, board {cal.BOARD[0]}x{cal.BOARD[1]}, "
          f"{len(usable)} usable views of {len(cal.VIEW_IDS)}")
    print(f"  calibrating on {len(fit)}: {fit}")
    print(f"  holding out {len(held)}: {held or 'nothing'}")
    print(f"  model: {args.model} ({cal.coefficient_count(args.model)} coefficients)")

    result = cal.calibrate(fit, args.side, cal.MODELS[args.model])
    K, dist = result["K"], result["dist"]
    w, h = cal.image_size(args.side)

    held_rms = cal.reprojection_error(K, dist, held, args.side) if held else float("nan")
    straight_after = cal.line_straightness(K, dist, usable, args.side)
    straight_before = cal.distorted_straightness(usable, args.side)

    print(f"\n  RMS fitted    {result['rms']:.4f} px   <- the number that gets quoted")
    if held:
        print(f"  RMS held out  {held_rms:.4f} px   <- the number that matters")
        if held_rms > result["rms"]:
            print(f"                {held_rms / result['rms']:.1f}x worse on views it "
                  "never saw")
    print(f"  straightness  {straight_after:.4f} px  (was {straight_before:.4f} px "
          f"before undistortion, {straight_before / max(straight_after, 1e-9):.1f}x "
          "better)")

    print(f"\n  fx {K[0, 0]:8.2f}   fy {K[1, 1]:8.2f}")
    print(f"  cx {K[0, 2]:8.2f}   cy {K[1, 2]:8.2f}   "
          f"(image centre is {w / 2:.0f}, {h / 2:.0f})")
    offset = float(np.hypot(K[0, 2] - w / 2, K[1, 2] - h / 2))
    print(f"  principal point sits {offset:.1f} px from the image centre")
    if offset > 0.08 * w:
        print("    — that is a long way for a normal lens. A model with enough free "
              "parameters can imitate a shifted centre, and the reported RMS will "
              "not complain.")
    coefficients = ", ".join(f"{v:+.5f}" for v in dist.ravel()[:5])
    print(f"  distortion    {coefficients}"
          + (" ..." if dist.size > 5 else ""))

    # ------------------------------------------------------------------ #
    if args.image:
        raw = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            ap.error(f"could not read {args.image}")
        if raw.shape[1] != w or raw.shape[0] != h:
            print(f"\n  note: your image is {raw.shape[1]}x{raw.shape[0]} and this "
                  f"calibration is for {w}x{h}. The intrinsics do not transfer between "
                  "sensor sizes; what follows is an illustration, not a calibration "
                  "of your camera.")
        newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (raw.shape[1], raw.shape[0]),
                                                0.0, (raw.shape[1], raw.shape[0]))
        fixed = cv2.undistort(raw, K, dist, None, newK)
        panels = [("your image", ensure_rgb(raw)), ("undistorted", ensure_rgb(fixed))]
        found, corners = cv2.findChessboardCorners(raw, cal.BOARD, None)
        if found:
            refined = cv2.cornerSubPix(raw, corners, (11, 11), (-1, -1),
                                       cal.CORNER_CRITERIA)
            before = cal._straightness_of(refined) if hasattr(cal, "_straightness_of") \
                else None
            del before
            print(f"\n  a {cal.BOARD[0]}x{cal.BOARD[1]} board was found in your image")
            del refined
        else:
            print(f"\n  no {cal.BOARD[0]}x{cal.BOARD[1]} board found in your image, so "
                  "there is nothing to measure straightness on — only the picture.")
        suptitle = f"{args.image} undistorted with a {len(fit)}-view calibration"
    else:
        show = args.show if args.show is not None else (held[0] if held else fit[0])
        raw = cal.load_view(args.side, show)
        fixed = cal.undistorted_view(show, K, dist, args.side)
        per_view = cal.line_straightness(K, dist, [show], args.side)
        panels = [
            (f"{args.side}{show:02d} as shot", ensure_rgb(raw)),
            (f"undistorted\n{per_view:.3f} px off straight", ensure_rgb(fixed)),
            ("difference", ensure_rgb(cv2.absdiff(raw, fixed))),
        ]
        seen = "held out" if show in held else "fitted"
        print(f"\n  showing {args.side}{show:02d} ({seen}): "
              f"{per_view:.4f} px off straight after undistortion")
        suptitle = (f"{args.side}{show:02d}, undistorted with a {len(fit)}-view "
                    f"'{args.model}' calibration")

    out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "undistorted.png"
    figures.grid(panels, out_path, ncols=3, suptitle=suptitle)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
