"""Count and measure objects in your own photo — inference from the command line.

    python infer.py my_coins.jpg
    python infer.py my_coins.jpg --reference-mm 24.25
    python infer.py my_coins.jpg --method "Hough circles" --out labelled.png
    python infer.py my_coins.jpg --all-methods
    python infer.py my_coins.jpg --no-flatten          # skip the illumination top-hat
    python infer.py my_coins.jpg --csv coins.csv

Two numbers come out, and they have very different standing:

* **The count** needs no reference at all. It is right or wrong.
* **The millimetres** rest entirely on `--reference-mm`, the assumed diameter of
  the largest object. That error propagates to every measurement **1:1**, and the
  output stays perfectly self-consistent while being uniformly wrong — so the
  uncertainty on every millimetre printed is the uncertainty on that one number.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import coins as cn  # noqa: E402
from shared.bench import timeit  # noqa: E402
from shared.io import imread, imwrite  # noqa: E402
from shared.report import init_console  # noqa: E402

MAX_SIDE = 1400


def load(path: str) -> np.ndarray:
    img = imread(path)
    if max(img.shape[:2]) > MAX_SIDE:
        s = MAX_SIDE / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def run(img, name, flatten, min_distance):
    fn = cn.METHODS[name]
    if name.startswith("Watershed (local"):
        return timeit(lambda: fn(img, min_distance=min_distance, flatten=flatten), runs=1, warmup=0)
    if name.startswith("Watershed (global"):
        return timeit(lambda: fn(img, flatten=flatten), runs=1, warmup=0)
    return timeit(lambda: fn(img), runs=1, warmup=0)


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("image", help="photo of objects on a plain background")
    ap.add_argument("--method", default="Hough circles", choices=list(cn.METHODS))
    ap.add_argument("--reference-mm", type=float, default=cn.REFERENCE_DIAMETER_MM,
                    help="true diameter of the LARGEST object, in mm")
    ap.add_argument("--min-distance", type=int, default=12, help="seed neighbourhood radius (px)")
    ap.add_argument("--no-flatten", action="store_true", help="skip the illumination top-hat")
    ap.add_argument("--out", default="labelled.png")
    ap.add_argument("--csv", help="write the per-object table here")
    ap.add_argument("--all-methods", action="store_true")
    args = ap.parse_args()

    if not Path(args.image).exists():
        print(f"error: no such file: {args.image}")
        return 2

    img = load(args.image)
    flatten = not args.no_flatten
    print(f"input     : {args.image}  {img.shape[1]}x{img.shape[0]}")
    print(f"reference : largest object assumed to be {args.reference_mm:.2f} mm")

    if args.all_methods:
        print(f"\n{'Method':<28}{'Count':>7}{'Smallest':>10}{'CV':>8}{'Implaus.':>10}{'ms':>8}")
        print("-" * 71)
        for name in cn.METHODS:
            lab, t = run(img, name, flatten, args.min_distance)
            pr = cn.region_properties(lab)
            d = cn.measure_mm(pr, cn.calibrate_mm_per_px(pr, args.reference_mm))
            bad = sum(1 for v in d if v < cn.PLAUSIBLE_MIN_FRACTION * args.reference_mm)
            cv_ = float(np.std(d) / np.mean(d)) if d else float("nan")
            small = f"{min(d):.1f} mm" if d else "-"
            print(f"{name:<28}{len(pr):>7}{small:>10}{cv_:>8.3f}{bad:>10}{t.median_ms:>8.1f}")
        print(
            "\nThere is no ground-truth count for your image, so no error column.\n"
            "The 'Implaus.' column needs no truth: it counts regions measuring under\n"
            f"{cn.PLAUSIBLE_MIN_FRACTION:.0%} of the reference, which is too small to be an object\n"
            "and is therefore a broken region. A method with the same count as the\n"
            "others but a non-zero count here segmented worse, not differently."
        )
        return 0

    labels, timing = run(img, args.method, flatten, args.min_distance)
    props = cn.region_properties(labels)
    mm_per_px = cn.calibrate_mm_per_px(props, args.reference_mm)
    mm = cn.measure_mm(props, mm_per_px)

    print(f"method    : {args.method}   {timing.median_ms:.1f} ms")
    print(f"found     : {len(props)} objects")
    if not props:
        print(
            "\nnote: nothing found. The most common cause is a background that is not\n"
            "      uniformly darker (or lighter) than the objects. Check the mask by\n"
            "      running the UI, and try --no-flatten if your lighting is already even."
        )
        return 0

    print(f"scale     : {mm_per_px:.5f} mm/px")
    print(f"diameters : {min(mm):.2f} - {max(mm):.2f} mm, mean {np.mean(mm):.2f}, "
          f"CV {np.std(mm) / np.mean(mm):.3f}")

    rng = np.random.default_rng(0)
    colour = np.zeros_like(img)
    for i in range(1, int(labels.max()) + 1):
        colour[labels == i] = rng.integers(70, 255, 3)
    blend = cv2.addWeighted(img, 0.55, colour, 0.45, 0)
    ordered = sorted(props, key=lambda q: q["centroid"][1])
    for j, p in enumerate(ordered, start=1):
        cx, cy = map(int, p["centroid"])
        cv2.putText(blend, str(j), (cx - 8, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    imwrite(args.out, blend)
    print(f"wrote     : {args.out}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["index", "centroid_x", "centroid_y", "area_px", "diameter_px", "diameter_mm"])
            for j, p in enumerate(ordered, start=1):
                w.writerow([j, round(p["centroid"][0], 1), round(p["centroid"][1], 1),
                            p["area_px"], round(p["diameter_px"], 2),
                            round(p["diameter_px"] * mm_per_px, 2)])
        print(f"wrote     : {args.csv}")

    # ------------------------------------------------------------------ #
    # notes, each from a measured result
    # ------------------------------------------------------------------ #
    floor = cn.PLAUSIBLE_MIN_FRACTION * args.reference_mm
    bad = [d for d in mm if d < floor]
    print()
    if bad:
        print(
            f"note  : {len(bad)} region(s) measure under {floor:.1f} mm, which is too small to\n"
            f"        be an object next to a {args.reference_mm:.1f} mm reference. A count can be\n"
            "        exactly right while regions are broken -- measured on the sample plate,\n"
            "        watershed counts 24/24 and still produces a basin implying a 5.75 mm\n"
            '        coin. Try --method "Hough circles", which fits a shape and cannot\n'
            "        produce a sliver."
        )
    else:
        print(
            "note  : every region is a plausible size, so the segmentation and the count\n"
            "        agree with each other. That is necessary, not sufficient -- it does\n"
            "        not check that the count is right, only that nothing is a fragment."
        )

    print(
        f"\nnote  : every millimetre above is {args.reference_mm:.2f} mm divided by "
        f"{max(p['diameter_px'] for p in props):.1f} px.\n"
        "        Calibration error propagates 1:1 -- a 5% mistake in --reference-mm is a\n"
        "        5% mistake in all of them, and nothing in the output can reveal it,\n"
        "        because the numbers stay perfectly consistent with each other."
    )
    if args.method == "Hough circles":
        print(
            "\nnote  : Hough fits CIRCLES. On this image that prior is what makes it the\n"
            "        most reliable measurer -- and it would find nothing at all on objects\n"
            "        that are not round. It is not a general object counter."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
