"""Run every keypoint detector on your own image at once.

    python infer.py photo.jpg
    python infer.py photo.jpg --rotate 30        # score repeatability against a known warp
    python infer.py photo.jpg --scale 0.6
    python infer.py photo.jpg --noise 15
    python infer.py photo.jpg --detector Harris --out corners.png

On a single image there is no second view, so there is no repeatability to
measure and **none is printed**. What is printed instead is how many keypoints
each detector found, how long it took, and how *evenly spread* they are — a
detector that piles every keypoint into one textured corner of the frame is
useless for homography estimation no matter how many it finds.

Pass `--rotate`, `--scale` or `--noise` to apply a known transform; the
repeatability against that transform is then real and is printed.
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
from shared.metrics import repeatability  # noqa: E402
from shared.report import init_console  # noqa: E402

import keypoints as kp  # noqa: E402

#: Grid the spread measure divides the frame into.
SPREAD_GRID = 8


def coverage(pts: np.ndarray, shape: tuple[int, int]) -> float:
    """Percentage of an 8x8 grid of cells that contain at least one keypoint.

    Count alone is misleading: 2000 keypoints all inside one patch of gravel
    constrain a homography no better than the 20 that happen to span the frame.
    """
    if len(pts) == 0:
        return 0.0
    h, w = shape[:2]
    xs = np.clip((pts[:, 0] / w * SPREAD_GRID).astype(int), 0, SPREAD_GRID - 1)
    ys = np.clip((pts[:, 1] / h * SPREAD_GRID).astype(int), 0, SPREAD_GRID - 1)
    return float(len(set(zip(xs.tolist(), ys.tolist()))) / (SPREAD_GRID ** 2)) * 100


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--detector", default=None, choices=list(kp.DETECTORS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--rotate", type=float, default=None, help="degrees")
    ap.add_argument("--scale", type=float, default=None, help="factor, e.g. 0.6")
    ap.add_argument("--noise", type=float, default=None, help="gaussian sigma")
    args = ap.parse_args()

    init_console()
    img = io.imread(args.image)
    gray = to_gray(img)
    print(f"{args.image}  {img.shape[1]}x{img.shape[0]}")

    H, warped = None, None
    if args.rotate is not None:
        H = kp.homography_rotation(gray.shape, args.rotate)
        warped = kp.apply_homography(img, H)
        print(f"rotated {args.rotate:g} degrees")
        if abs(args.rotate) % 90 == 0:
            print("  NOTE: a right-angle rotation is an exact transpose of the "
                  "pixel grid, so it tests arithmetic rather than invariance. "
                  "Try 30 for a real answer.")
    elif args.scale is not None:
        H = kp.homography_scale(gray.shape, args.scale)
        warped = kp.apply_homography(img, H)
        print(f"scaled by {args.scale:g}")
    elif args.noise is not None:
        H = np.eye(3)
        warped = synth.gaussian_noise(img, sigma=args.noise, seed=0)
        print(f"added gaussian noise, sigma {args.noise:g}")

    wgray = to_gray(warped) if warped is not None else None

    names = [args.detector] if args.detector else list(kp.DETECTORS)
    header = f"\n{'detector':14s} {'keypoints':>10s} {'coverage':>9s} {'ms':>8s}"
    if H is not None:
        header += f" {'repeatability':>14s}"
    print(header)

    found = {}
    for name in names:
        fn = kp.DETECTORS[name]
        t0 = time.perf_counter()
        pts = fn(gray)
        ms = (time.perf_counter() - t0) * 1000
        found[name] = pts
        line = (f"{name:14s} {len(pts):10d} {coverage(pts, gray.shape):8.1f}% "
                f"{ms:8.2f}")
        if H is not None:
            rep = repeatability(pts, fn(wgray), H, kp.MATCH_THRESHOLD, shape=wgray.shape)
            line += f" {rep:14.4f}"
        print(line)

    if H is None:
        best_cov = max(found, key=lambda n: coverage(found[n], gray.shape))
        print("\nNo repeatability is reported. One image is one view, and "
              "repeatability needs two related by a known transform.")
        print(f"Best frame coverage here: {best_cov}, reaching "
              f"{coverage(found[best_cov], gray.shape):.0f}% of the frame.")
        print("Coverage matters more than count — keypoints piled into one "
              "textured patch constrain a homography no better than a handful "
              "that span the image.")
        print("Pass --rotate 30, --scale 0.6 or --noise 15 to measure "
              "repeatability against a known transform.")
    else:
        scored = {n: repeatability(found[n], kp.DETECTORS[n](wgray), H,
                                   kp.MATCH_THRESHOLD, shape=wgray.shape)
                  for n in names}
        best = max(scored, key=lambda n: scored[n])
        print(f"\nmost repeatable here: {best} at {scored[best]:.4f}")
        print("Repeatability counts only keypoints the transform kept in frame; "
              "one carried outside the picture cannot be redetected by anything.")

    out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "keypoints.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panels = [kp.draw_keypoints(img, found[n]) for n in names]
    io.imwrite(out_path, panels[0] if args.detector else np.hstack(panels))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
