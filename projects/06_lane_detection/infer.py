"""Find the lane in one photograph, and say what each step was worth.

    python infer.py --image lane_test5
    python infer.py --image lane_test1 --pixel-roi
    python infer.py dashcam.jpg

Every run prints the two controls alongside the five pipelines, because on this
data the gap between a pipeline and "fit a line to the bright pixels in the
trapezoid" is often smaller than the gap between two pipelines.

`--pixel-roi` re-runs the same pipeline with the region of interest written as
the pixel literal a tutorial hard-codes for a 960x540 frame. On a 960x540 frame
nothing changes. On anything else it still returns two lines, and they are the
wrong two.
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

import lanes as ln  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default=None,
                    help="your own forward-facing photograph of a road")
    ap.add_argument("--image", default=None,
                    help="one of the project's fourteen, e.g. lane_test5")
    ap.add_argument("--pixel-roi", action="store_true",
                    help="also run with the ROI written as a 960x540 pixel literal")
    ap.add_argument("--yaw", type=float, default=40.0,
                    help="pixels of camera yaw for the repeatability check")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    init_console()

    if args.image:
        image, name = ln.load(args.image), args.image
    elif args.path:
        image, name = io.imread(args.path), Path(args.path).name
    else:
        ap.error("give an image path, or --image with one of the project's fourteen")

    h, w = image.shape[:2]
    print(f"{name}  {w}x{h}")
    region = ln.roi_mask(image.shape) > 0
    g = ln.to_gray(image)
    inside = g[region].astype(np.float32)
    contrast = float(np.percentile(inside, 98) - np.median(inside))
    shadow = float((inside < 0.6 * np.median(inside)).mean())
    print(f"  the region of interest covers {100 * region.mean():.1f}% of the frame")
    print(f"  paint contrast inside it {contrast:.0f} grey levels, "
          f"{100 * shadow:.1f}% of it in shadow")
    if contrast < 60:
        print("  — low contrast: expect the pipelines to disagree with each other")

    # ------------------------------------------------------------------ #
    control = ln.detect_bright_in_roi(image)
    print(f"\n{'method':36s} {'lane width':>11s} {'vp x':>7s} {'geometry':>10s} "
          f"{'vs ROI':>8s} {'yaw err':>9s}")
    panels = [("the region of interest", ln.draw_roi(image))]
    for detector, fn in ln.DETECTORS.items():
        lanes = fn(image)
        ok = ln.plausible(lanes, image.shape)
        width = ln.lane_width(lanes, image.shape)
        vp = ln.vanishing_point(lanes)
        gaps = [abs(a - b) / w * 100 for a, b in zip(ln.bottom_endpoints(control),
                                                     ln.bottom_endpoints(lanes))
                if a is not None and b is not None]
        # this photograph only -- the set-wide median would be a different
        # number reported in a per-image column, which is how a table starts
        # lying
        yaw = (ln.yaw_repeatability(detector, shift_px=args.yaw, names=[args.image])
               if args.image else None)
        print(f"{detector:36s} {('%.2fw' % width) if width else '—':>11s} "
              f"{('%.0f' % vp[0]) if vp else '—':>7s} "
              f"{'ok' if ok else 'IMPLAUSIBLE':>10s} "
              f"{('%.2f%%' % np.median(gaps)) if gaps else '—':>8s} "
              f"{('%.2f%%' % yaw['median_error_pct']) if yaw else '—':>9s}")
        panels.append((f"{detector}\n" + ("" if ok else "implausible"),
                       ln.draw(image, lanes,
                               colour=(60, 220, 90) if ok else (235, 70, 70))))

    # ------------------------------------------------------------------ #
    # only pipelines that returned a lane at all: a gap measured from the one
    # line a half-failed pipeline did return is not a comparison
    real_gaps = {}
    for detector in ln.REAL_DETECTORS:
        lanes = ln.DETECTORS[detector](image)
        if not ln.plausible(lanes, image.shape):
            continue
        gaps = [abs(a - b) / w * 100 for a, b in zip(ln.bottom_endpoints(control),
                                                     ln.bottom_endpoints(lanes))
                if a is not None and b is not None]
        if gaps:
            real_gaps[detector] = float(np.median(gaps))
    if real_gaps:
        closest = min(real_gaps, key=real_gaps.get)
        gap = real_gaps[closest]
        print(f"\n{closest} lands {gap:.2f}% of the frame width from 'bright pixels "
              "in the trapezoid',")
        if gap < 1.5:
            print("  which ran no colour test, no edge detector and no Hough "
                  "transform. On this frame")
            print("  the four steps after the region of interest bought nothing "
                  "measurable.")
        else:
            print("  which ran no colour test, no edge detector and no Hough "
                  "transform. Here they")
            print("  genuinely part company, so the four steps after the region of "
                  "interest are doing work.")

        fixed_gap = [abs(a - b) / w * 100
                     for a, b in zip(ln.bottom_endpoints(control),
                                     ln.bottom_endpoints(ln.detect_fixed(image)))
                     if a is not None and b is not None]
        if fixed_gap and float(np.median(fixed_gap)) < 1.5:
            print(f"  And the fixed guess -- the same two lines for every frame -- is "
                  f"{np.median(fixed_gap):.2f}% away.")
            print("  On an easy frame this comparison has almost no information in it.")

    failed = [d for d in ln.REAL_DETECTORS
              if not ln.plausible(ln.DETECTORS[d](image), image.shape)]
    if failed:
        print(f"\n{len(failed)} of {len(ln.REAL_DETECTORS)} returned something that is "
              f"not a lane: {', '.join(failed)}")
        print("  — caught by geometry alone: no annotation exists for this photograph.")

    # ------------------------------------------------------------------ #
    if args.pixel_roi:
        lanes = ln.detect_pixel_roi(image)
        ok = ln.plausible(lanes, image.shape)
        covers = float((ln.roi_mask_pixels(image.shape) > 0).mean())
        print(f"\nwith the ROI as a 960x540 pixel literal it covers "
              f"{100 * covers:.1f}% of this frame (against "
              f"{100 * region.mean():.1f}% as fractions)")
        print(f"  and it returns {'two lines' if all(l is not None for l in lanes) else 'nothing'}, "
              f"which the geometry check calls {'plausible' if ok else 'implausible'}")
        if (w, h) != (960, 540) and all(l is not None for l in lanes):
            print("  — this is the silent failure: a wrong region still produces an answer")
        panels.append(("ROI as a pixel literal\n" + ("" if ok else "implausible"),
                       ln.draw(ln.draw_roi(image, pixels=True), lanes,
                               colour=(60, 220, 90) if ok else (235, 70, 70))))

    out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "detected.png"
    figures.grid(panels, out_path, ncols=4,
                 suptitle=f"{name} — green where the answer satisfies road geometry, "
                          "red where it does not")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
