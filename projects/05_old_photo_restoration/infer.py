"""Restore your own damaged photo — inference from the command line.

    python infer.py my_old_photo.jpg
    python infer.py my_old_photo.jpg --out restored.png
    python infer.py my_old_photo.jpg --mask my_mask.png      # you painted the damage
    python infer.py my_old_photo.jpg --detector "Top-hat + black-hat"
    python infer.py my_old_photo.jpg --all-methods
    python infer.py my_old_photo.jpg --no-fade               # inpaint only
    python infer.py my_old_photo.jpg --save-mask

No undamaged original exists for a photo you supply, so **no PSNR is reported** —
there is nothing to compare against. What *is* reported needs no reference: how
much of the image the detector flagged, the colour cast, the chroma and the
contrast before and after.

If the result is poor, the detector is the first place to look, not the
inpainting method. On the test set, using a detected mask instead of the true one
costs 14.0 dB, while the gap between the best and worst inpainting method is
1.1 dB. Pass `--mask` with a painted mask to remove that variable entirely.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import restoration as rs  # noqa: E402
from shared.bench import timeit  # noqa: E402
from shared.io import imread, imwrite, to_gray  # noqa: E402
from shared.metrics import entropy, rms_contrast  # noqa: E402
from shared.report import init_console  # noqa: E402

MAX_SIDE = 1400


def load(path: str) -> np.ndarray:
    img = imread(path)
    if max(img.shape[:2]) > MAX_SIDE:
        s = MAX_SIDE / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("image", help="path to a damaged or faded photo")
    ap.add_argument("--method", default="Telea (fast marching)", choices=list(rs.METHODS))
    ap.add_argument("--detector", default="Median residual", choices=list(rs.DETECTORS))
    ap.add_argument("--fade", default="Stretch + saturate", choices=list(rs.FADE_METHODS))
    ap.add_argument("--mask", help="your own damage mask (white = damaged); skips detection")
    ap.add_argument("--out", default="restored.png")
    ap.add_argument("--all-methods", action="store_true", help="run all four and compare")
    ap.add_argument("--no-fade", action="store_true", help="inpaint only, leave the tone alone")
    ap.add_argument("--save-mask", action="store_true", help="also write the damage mask used")
    args = ap.parse_args()

    if not Path(args.image).exists():
        print(f"error: no such file: {args.image}")
        return 2

    img = load(args.image)
    print(f"input   : {args.image}  {img.shape[1]}x{img.shape[0]}")
    print(
        f"cast    : {rs.colour_cast(img):.2f} deg from neutral   "
        f"chroma: {rs.saturation_of(img):.1f}   contrast: {rms_contrast(img):.4f}"
    )

    if args.mask:
        if not Path(args.mask).exists():
            print(f"error: no such mask file: {args.mask}")
            return 2
        mask = to_gray(imread(args.mask))
        if mask.shape != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8) * 255
        mask_source = f"your mask ({args.mask})"
    else:
        mask = rs.DETECTORS[args.detector](img)
        mask_source = args.detector

    flagged = float((mask > 0).mean())
    print(f"mask    : {mask_source} — flagged {flagged:.2%} of pixels")

    if args.all_methods:
        print(f"\n{'Method':<26}{'Contrast':>10}{'Entropy':>9}{'Time (ms)':>11}")
        print("-" * 56)
        print(f"{'(input)':<26}{rms_contrast(img):>10.4f}{entropy(img):>9.2f}{'-':>11}")
        for name, fn in rs.METHODS.items():
            out, t = timeit(lambda f=fn: f(img, mask), runs=1, warmup=0)
            print(f"{name:<26}{rms_contrast(out):>10.4f}{entropy(out):>9.2f}{t.median_ms:>11.1f}")
        print(
            "\nNo PSNR column: a photo you supplied has no undamaged original to\n"
            "compare against, and neither contrast nor entropy is a ranking -- a\n"
            "method that invents texture scores well on both. Run `python run.py`\n"
            "for the measured comparison on generated damage, where there is a truth."
        )
        return 0

    inpainted, timing = timeit(lambda: rs.METHODS[args.method](img, mask), runs=1, warmup=0)
    out = inpainted if args.no_fade else rs.FADE_METHODS[args.fade](inpainted)
    imwrite(args.out, out)

    print(f"inpaint : {args.method}   {timing.median_ms:.1f} ms")
    print(f"fade    : {'(skipped)' if args.no_fade else args.fade}")
    print(f"cast    : {rs.colour_cast(img):.2f} -> {rs.colour_cast(out):.2f} deg")
    print(f"chroma  : {rs.saturation_of(img):.1f} -> {rs.saturation_of(out):.1f}")
    print(f"contrast: {rms_contrast(img):.4f} -> {rms_contrast(out):.4f}")
    print(f"entropy : {entropy(img):.2f} -> {entropy(out):.2f} bits")
    print(f"wrote   : {args.out}")

    if args.save_mask:
        stem = Path(args.out).with_suffix("")
        imwrite(f"{stem}_mask.png", mask)
        print(f"wrote   : {stem}_mask.png")

    # ------------------------------------------------------------------ #
    # warnings that are actually actionable, each from a measured result
    # ------------------------------------------------------------------ #
    if not args.mask and flagged > 0.25:
        print(
            f"\nnote: the detector flagged {flagged:.1%} of the image as damaged. That is\n"
            "      far more than a scratched print normally carries, so it is probably\n"
            "      responding to texture rather than damage. Over-flagging is the cheaper\n"
            "      error -- a healthy pixel gets replaced by its healthy neighbours -- but\n"
            "      at this level it will visibly soften the picture. Try --detector\n"
            '      "Intensity threshold" (higher precision, lower recall) or paint a mask.'
        )
    if not args.mask and flagged < 0.005:
        print(
            f"\nnote: the detector flagged only {flagged:.2%} of the image, so almost\n"
            "      nothing was inpainted. If there is visible damage, it is probably\n"
            f"      wider than the {rs.MEDIAN_RESIDUAL_KSIZE} px median window this detector uses: a median\n"
            "      filter only rejects a MINORITY of outliers, and once the damage fills\n"
            "      half the window the residual it looks for goes to zero. Painting a\n"
            "      mask and passing --mask is the reliable fix."
        )
    if rs.colour_cast(img) > 6.0 and args.no_fade:
        print(
            f"\nnote: this image sits {rs.colour_cast(img):.1f} degrees off neutral, which is the\n"
            "      signature of a yellowed print -- and --no-fade left it there.\n"
            "      Inpainting cannot fix it: it only touches pixels in the mask."
        )
    if rs.saturation_of(out) > 1.6 * rs.saturation_of(img) + 5:
        print(
            "\nnote: the correction more than doubled the chroma. The saturation factor\n"
            f"      ({rs.SATURATION_FACTOR}) was measured on prints faded by a known amount; a photo\n"
            "      that was never that desaturated will come out garish. Use\n"
            '      --fade "Per-channel stretch" to skip the saturation step.'
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
