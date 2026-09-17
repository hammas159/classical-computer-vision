"""Run Canny on your own image and find out which parameter actually matters.

    python infer.py photo.jpg
    python infer.py photo.jpg --sigma 2.0 --low 50 --ratio 2.0 --out edges.png
    python infer.py --annotated deer_and_fawn     # score against human boundaries

On your own photograph there is no annotation, so **no F-measure is printed**.
What is printed instead is how much the edge map *changes* as each parameter
moves — measured as the IoU between edge maps — because a parameter that barely
changes the output cannot be worth choosing carefully.

On this project's generated scene that measurement says smoothing explains 30%
of the variance and the high:low ratio explains 0.03%. `--annotated` takes one
of the twelve photographs that has human boundaries and scores the settings
properly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from shared import bsds, io  # noqa: E402
from shared.io import to_gray  # noqa: E402
from shared.report import init_console  # noqa: E402

import canny_sensitivity as cs  # noqa: E402

#: Sensitivity below which a parameter is not worth tuning on this image.
NEGLIGIBLE = 0.05


def edge_iou(a: np.ndarray, b: np.ndarray) -> float:
    """How much two edge maps agree, with the same tolerance the scoring uses."""
    k = np.ones((2 * cs.TOLERANCE + 1,) * 2, np.uint8)
    pa, pb = (a > 0), (b > 0)
    near_a = cv2.dilate(pa.astype(np.uint8), k) > 0
    near_b = cv2.dilate(pb.astype(np.uint8), k) > 0
    inter = float((pa & near_b).sum() + (pb & near_a).sum())
    union = float(pa.sum() + pb.sum())
    return inter / max(union, 1.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", default=None)
    ap.add_argument("--annotated", default=None, choices=list(cs.IMAGES))
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--low", type=int, default=50)
    ap.add_argument("--ratio", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    init_console()
    if args.annotated:
        gray = to_gray(cs.load_scene(args.annotated))
        target = bsds.consensus_boundaries(args.annotated)
        ceiling = cs.photo_human_ceiling(images=(args.annotated,))
        print(f"{args.annotated}  {gray.shape[1]}x{gray.shape[0]}")
        print(f"human ceiling on this image: F {ceiling['best']:.3f} "
              f"(best annotator against the others' consensus)")
    elif args.image:
        gray = to_gray(io.imread(args.image))
        target, ceiling = None, None
        print(f"{args.image}  {gray.shape[1]}x{gray.shape[0]}")
    else:
        ap.error("give an image path, or --annotated with one of the project's photographs")

    th, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    density = float((cv2.Canny(gray, 0.5 * th, th) > 0).mean() * 100)
    print(f"edge density {density:.1f}% "
          "(this project's pool spans 1.6% to 36.7%, and F falls as it rises)")

    if args.sigma is not None:
        edges = cs.canny(gray, args.sigma, args.low, args.ratio)
        print(f"\nsigma {args.sigma:g}, low {args.low}, ratio {args.ratio:g}: "
              f"{100 * float((edges > 0).mean()):.2f}% of pixels are edges")
        if target is not None:
            s = bsds.boundary_f_measure(edges, target, cs.TOLERANCE)
            print(f"  F {s['f']:.3f}, precision {s['precision']:.3f}, "
                  f"recall {s['recall']:.3f}")
        out = io.ensure_rgb(edges)
    else:
        base = (1.4, args.low, args.ratio)
        print(f"\n{'parameter':12s} {'values swept':>28s} {'output changes by':>18s}")
        sensitivity = {}
        for param, values in (("sigma", cs.SIGMAS), ("low", cs.LOW_THRESHOLDS),
                              ("ratio", cs.RATIOS)):
            maps = []
            for v in values:
                s, lo, ra = base
                if param == "sigma":
                    s = v
                elif param == "low":
                    lo = v
                else:
                    ra = v
                maps.append(cs.canny(gray, s, int(lo), ra))
            # 1 - the mean agreement between consecutive settings
            changes = 1.0 - float(np.mean([edge_iou(maps[i], maps[i + 1])
                                           for i in range(len(maps) - 1)]))
            sensitivity[param] = changes
            shown = ", ".join(str(v) for v in values)
            print(f"{param:12s} {shown[:28]:>28s} {changes:17.3f}")

        ranked = sorted(sensitivity, key=sensitivity.get, reverse=True)
        print(f"\nOn this image the parameters matter in the order: "
              f"{' > '.join(ranked)}.")
        ignorable = [p for p, v in sensitivity.items() if v < NEGLIGIBLE]
        if ignorable:
            print(f"  {', '.join(ignorable)} changes the output by less than "
                  f"{NEGLIGIBLE:g} — leaving it at the default costs nothing here.")
        if ranked[0] == "sigma":
            print("  Smoothing first, as on this project's generated scene, where "
                  "it explains 30% of the variance against the ratio's 0.03%.")

        if target is not None:
            print(f"\n{'setting':32s} {'F':>7s} {'prec':>7s} {'recall':>7s}")
            for label, s, lo, ra in [
                ("textbook (s1.4, 50, 3.0)", 1.4, 50, 3.0),
                ("best on shapes (s1.0, 50, 3.0)", 1.0, 50, 3.0),
                ("best on photos (s2.0, 50, 2.0)", 2.0, 50, 2.0),
                ("no smoothing (s0, 50, 3.0)", 0.0, 50, 3.0),
            ]:
                sc = bsds.boundary_f_measure(cs.canny(gray, s, lo, ra), target,
                                             cs.TOLERANCE)
                print(f"{label:32s} {sc['f']:7.3f} {sc['precision']:7.3f} "
                      f"{sc['recall']:7.3f}")
            print("\nThe 'best on shapes' row is what a synthetic benchmark would "
                  "recommend. On this project's twelve photographs it costs 19%.")
        else:
            print("\nNo F-measure is reported. There is no annotation for this "
                  "photograph, and on the twelve that do have one, two people "
                  "only agree at F 0.92 — there is no single right edge map.")

        out = np.hstack([io.ensure_rgb(cs.canny(gray, s, args.low, args.ratio))
                         for s in (0.0, 1.0, 2.0, 4.0)])

    out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "edges.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    io.imwrite(out_path, out)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
