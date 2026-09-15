"""Brighten your own dark photo — inference from the command line.

    python infer.py my_dark_photo.jpg
    python infer.py my_dark_photo.jpg --method CLAHE --out bright.png
    python infer.py my_dark_photo.jpg --all-methods

No original exists for a photo you supply, so no PSNR is reported. What *is*
reported is everything measurable without a reference: brightness, entropy,
contrast, and — the number that matters most for low light — how much the method
amplified the noise that was hiding in the shadows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import low_light as ll  # noqa: E402
from shared.bench import timeit  # noqa: E402
from shared.io import imread, imwrite  # noqa: E402
from shared.metrics import entropy, estimate_noise_sigma, mean_brightness, rms_contrast  # noqa: E402
from shared.report import init_console  # noqa: E402

MAX_SIDE = 1600


def load(path: str) -> np.ndarray:
    img = imread(path)
    if max(img.shape[:2]) > MAX_SIDE:
        scale = MAX_SIDE / max(img.shape[:2])
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def describe(img: np.ndarray) -> dict[str, float]:
    return {
        "brightness": mean_brightness(img),
        "entropy": entropy(img),
        "contrast": rms_contrast(img),
        "noise": estimate_noise_sigma(img),
    }


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("image", help="path to a dark photo")
    ap.add_argument("--method", default="Gamma (auto-estimated)", choices=list(ll.METHODS))
    ap.add_argument("--out", default="enhanced.png")
    ap.add_argument("--all-methods", action="store_true",
                    help="run all eight and compare them on this image")
    args = ap.parse_args()

    if not Path(args.image).exists():
        print(f"error: no such file: {args.image}")
        return 2

    dark = load(args.image)
    before = describe(dark)
    print(f"input   : {args.image}  {dark.shape[1]}x{dark.shape[0]}")

    if args.all_methods:
        print(
            f"\n{'Method':<26}{'Brightness':>11}{'Entropy':>9}{'Contrast':>10}"
            f"{'Noise x':>9}{'Time (ms)':>11}"
        )
        print("-" * 76)
        print(
            f"{'(input)':<26}{before['brightness']:>11.3f}{before['entropy']:>9.2f}"
            f"{before['contrast']:>10.3f}{'-':>9}{'-':>11}"
        )
        for name, fn in ll.METHODS.items():
            out, timing = timeit(lambda f=fn: f(dark), runs=1, warmup=0)
            a = describe(out)
            gain = a["noise"] / max(before["noise"], 1e-6)
            print(
                f"{name:<26}{a['brightness']:>11.3f}{a['entropy']:>9.2f}"
                f"{a['contrast']:>10.3f}{gain:>8.2f}x{timing.median_ms:>11.1f}"
            )
        print(
            "\nNo PSNR column: a photo you supplied has no original to compare against.\n"
            "Watch the 'Noise x' column -- brightening is a multiplication, so it\n"
            "multiplies whatever noise was hiding in the shadows. Run `python run.py`\n"
            "for the measured comparison against known originals."
        )
        return 0

    out, timing = timeit(lambda: ll.METHODS[args.method](dark), runs=1, warmup=0)
    after = describe(out)
    imwrite(args.out, out)

    print(f"method  : {args.method}")
    if args.method.startswith("Gamma (auto"):
        estimated = 1.0 / ll.estimate_gamma(dark)
        print(f"estimated gamma : {estimated:.2f}   "
              "(the darkening this photo appears to have suffered)")
    print(f"brightness      : {before['brightness']:.3f} -> {after['brightness']:.3f}")
    print(f"entropy         : {before['entropy']:.2f} -> {after['entropy']:.2f} bits")
    print(f"contrast        : {before['contrast']:.3f} -> {after['contrast']:.3f}")
    gain = after["noise"] / max(before["noise"], 1e-6)
    print(f"noise           : {before['noise']:.2f} -> {after['noise']:.2f} sigma  "
          f"({gain:.2f}x amplification)")
    print(f"time            : {timing.median_ms:.1f} ms")
    print(f"wrote   : {args.out}")

    if gain > 4.0:
        print(
            f"\nnote: this method amplified the noise {gain:.1f}x. On a genuinely noisy\n"
            "      photo that is often worse than leaving it dark. Try CLAHE or the\n"
            "      gamma methods, which measured 2.7x on the test set."
        )
    if args.method.startswith("Gamma (auto"):
        print(
            "\nnote: the auto method assumes a well-exposed photo averages mid-grey.\n"
            "      On a scene that is genuinely dark -- a night shot, a dark subject --\n"
            "      that assumption is wrong and it will over-brighten. Measured error\n"
            "      tracks the gap between the true exposure and that assumption."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
