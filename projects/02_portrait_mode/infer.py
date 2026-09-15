"""Apply portrait mode to your own photo — inference from the command line.

    python infer.py my_portrait.jpg
    python infer.py my_portrait.jpg --matte "GrabCut (centre rect)" --kernel Disc
    python infer.py my_portrait.jpg --all-mattes
    python infer.py my_portrait.jpg --radius 25 --save-stages

No ground truth exists for a photo you supply, so no IoU is reported. What *is*
reported is everything measurable without it: which methods found a subject, how
much of the frame each called subject, the aperture's shape statistics, and the
time each stage took.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import portrait_mode as pm  # noqa: E402
from shared.bench import timeit  # noqa: E402
from shared.io import imread, imwrite  # noqa: E402
from shared.report import init_console  # noqa: E402

MAX_SIDE = 1200


def load(path: str) -> np.ndarray:
    """Read an image and cap its size.

    GrabCut is O(pixels) with a large constant — over a second on a 1200 px
    image — so an uncapped phone photo would take the better part of a minute
    for no gain in matte quality.
    """
    img = imread(path)
    if max(img.shape[:2]) > MAX_SIDE:
        scale = MAX_SIDE / max(img.shape[:2])
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("image", help="path to a portrait photo")
    ap.add_argument("--matte", default="Haar + GrabCut", choices=list(pm.MATTES))
    ap.add_argument("--kernel", default="Disc (circular aperture)", choices=list(pm.BOKEH_KERNELS))
    ap.add_argument("--radius", type=int, default=15, help="blur radius in px (3-35)")
    ap.add_argument(
        "--compositor", default="Masked (normalised convolution)", choices=list(pm.COMPOSITORS)
    )
    ap.add_argument("--out", default="portrait.png")
    ap.add_argument("--save-stages", action="store_true", help="also write the matte")
    ap.add_argument("--all-mattes", action="store_true",
                    help="run all six and report which found a subject, and how fast")
    args = ap.parse_args()

    if not Path(args.image).exists():
        print(f"error: no such file: {args.image}")
        return 2

    img = load(args.image)
    total_px = img.shape[0] * img.shape[1]
    print(f"input : {args.image}  {img.shape[1]}x{img.shape[0]}")

    if args.all_mattes:
        # Load the Haar cascade before timing anything. It costs a few hundred
        # milliseconds once, and without this the first face-based method in the
        # dict is charged for all of it -- which made the *fastest* method look
        # like the slowest.
        pm.matte_face_rect(img)

        print(f"\n{'Matting method':<28} {'Found':<7} {'Subject %':>10} {'Time (ms)':>10}")
        print("-" * 60)
        for name, fn in pm.MATTES.items():
            matte, timing = timeit(lambda f=fn: f(img), runs=1, warmup=1)
            if matte is None:
                print(f"{name:<28} {'no':<7} {'-':>10} {timing.median_ms:>10.1f}")
                continue
            pct = 100.0 * float((matte > 0).sum()) / total_px
            print(f"{name:<28} {'yes':<7} {pct:>9.1f}% {timing.median_ms:>10.1f}")
        print(
            "\nNo accuracy column: a photo you supplied has no known matte, so "
            "there is nothing to score against.\n'Subject %' is a sanity check only "
            "— a method claiming 90% of the frame has failed.\nRun `python run.py` "
            "for the measured comparison on generated portraits."
        )
        return 0

    (matte, out), timing = timeit(
        lambda: pm.portrait(img, args.matte, args.kernel, args.radius, args.compositor),
        runs=1, warmup=0,
    )

    if matte is None:
        print(f"\n{args.matte} found no subject in this image.")
        print("Methods that do not need a detected face:")
        for name in ("GrabCut (centre rect)", "Skin colour (YCrCb)", "Watershed + markers"):
            print(f'    --matte "{name}"')
        return 1

    kernel = pm.BOKEH_KERNELS[args.kernel](args.radius)
    profile = pm.highlight_profile(kernel)
    subject_pct = 100.0 * float((matte > 0).sum()) / total_px
    imwrite(args.out, out)

    print(f"matte     : {args.matte}")
    print(f"aperture  : {args.kernel}, radius {args.radius} px")
    print(f"compositor: {args.compositor}")
    print(f"subject   : {subject_pct:.1f}% of the frame")
    print(f"bokeh     : peak/mean {profile['peak_to_mean']:.2f}  "
          f"(1.00 = a flat disc, like a real lens)")
    print(f"            rim energy {profile['edge_sharpness']:.2f}")
    print(f"time      : {timing.median_ms:.0f} ms")
    print(f"wrote     : {args.out}")

    if args.save_stages:
        stem = Path(args.out).with_suffix("")
        imwrite(f"{stem}_matte.png", matte)
        print(f"wrote     : {stem}_matte.png")

    if args.compositor.startswith("Naive"):
        print(
            "\nnote: naive compositing blurs across the subject boundary, which "
            "smears subject\n      colour outward. Measured at 6x the halo error "
            "of the masked version."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
