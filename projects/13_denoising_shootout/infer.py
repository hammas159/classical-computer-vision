"""Denoise your own photo — inference from the command line.

    python infer.py noisy.jpg
    python infer.py noisy.jpg --method Median --out clean.png
    python infer.py noisy.jpg --all-methods
    python infer.py noisy.jpg --noise salt_pepper        # tune for impulse noise
    python infer.py noisy.jpg --estimate                 # just measure the noise

No clean original exists for a photo you supply, so **no PSNR is reported**. What
*is* reported needs no reference: an estimate of the noise sigma, and how much
each filter changed the image.

The one genuinely useful thing this tool does without a ground truth is **tell
you which filter to use**, by estimating the noise level and naming the filter
that won at that level on the measured benchmark. That is a recommendation from
data rather than from folklore -- and the folklore ("median kills salt and
pepper") turns out to be right, while the part nobody says ("and median is the
*worst* filter on Gaussian noise") turns out to be right too.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import denoising as dn  # noqa: E402
from shared.bench import timeit  # noqa: E402
from shared.io import imread, imwrite  # noqa: E402
from shared.metrics import estimate_noise_sigma, psnr  # noqa: E402
from shared.report import init_console  # noqa: E402

MAX_SIDE = 1400

#: Which filter won at each noise model on the measured benchmark, and by how
#: much over the runner-up. Copied from results/tables.md.
WINNERS = {
    "gaussian": ("Bilateral", 29.792, 1.174),
    "salt_pepper": ("Median", 33.688, 6.796),
    "poisson": ("Non-local means", 28.860, 0.307),
}


def impulse_fraction(img: np.ndarray) -> float:
    """Fraction of **isolated** pixels pinned at 0 or 255 — the signature of impulse noise.

    "Isolated" is doing all the work. Counting extreme pixels alone does not
    separate the noise models, because it is dominated by what the photograph
    already contains: `astronaut` has **11.2%** of its pixels at 0 or 255 with no
    noise at all (large black regions), while Poisson noise at lambda 30 clips a
    further 1.7%. A threshold on the raw fraction misclassifies both.

    Requiring the extreme pixel to also *disagree with its own 5x5 median*
    separates isolated speckle from solid dark regions, and the separation is
    clean — measured over six images:

    ======================  ==================
    noise                   isolated extremes
    ======================  ==================
    none                    <= 0.004%
    Gaussian sigma=25       <= 0.005%
    Poisson lambda=30       <= 0.014%
    **Salt & pepper 6%**    **4.28% - 5.98%**
    ======================  ==================

    A 300x margin, which is what a detection threshold should look like.
    """
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    extreme = (g == 0) | (g == 255)
    return float((extreme & (cv2.absdiff(g, cv2.medianBlur(g, 5)) > 60)).mean())


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("image", help="a noisy photo")
    ap.add_argument("--method", choices=list(dn.METHODS), help="default: the recommended one")
    ap.add_argument("--noise", choices=list(dn.TUNED), help="override the detected noise model")
    ap.add_argument("--out", default="denoised.png")
    ap.add_argument("--all-methods", action="store_true")
    ap.add_argument("--estimate", action="store_true", help="measure the noise and stop")
    ap.add_argument("--default-params", action="store_true", help="skip the tuned parameters")
    args = ap.parse_args()

    if not Path(args.image).exists():
        print(f"error: no such file: {args.image}")
        return 2

    img = imread(args.image)
    if max(img.shape[:2]) > MAX_SIDE:
        s = MAX_SIDE / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    sigma = estimate_noise_sigma(img)
    impulse = impulse_fraction(img)
    # 0.5% sits three orders of magnitude above every non-impulse reading
    # and an order of magnitude below every impulse one -- see impulse_fraction
    detected = "salt_pepper" if impulse > 0.005 else "gaussian"
    kind = args.noise or detected

    print(f"input    : {args.image}  {img.shape[1]}x{img.shape[0]}")
    print(f"noise    : estimated sigma {sigma:.2f}   isolated extreme pixels: {impulse:.3%}")
    print(f"model    : {kind}" + ("" if args.noise else f"  (detected from the impulse fraction)"))

    if args.estimate:
        print(
            "\nnote     : sigma is estimated from the median absolute deviation of a\n"
            "           Laplacian response, which assumes the noise is additive and white.\n"
            "           On a salt-and-pepper image it over-reads, because a handful of\n"
            "           pixels at the extremes dominate the statistic -- which is why the\n"
            "           impulse fraction is reported next to it rather than instead of it."
        )
        return 0

    winner, winner_db, margin = WINNERS[kind]
    method = args.method or winner

    if args.all_methods:
        print(f"\n{'Filter':<24}{'Change (dB)':>13}{'Time (ms)':>12}")
        print("-" * 49)
        for name in dn.METHODS:
            if args.default_params or name.startswith("Do nothing"):
                out, t = timeit(lambda n=name: dn.METHODS[n](img), runs=1, warmup=0)
            else:
                out, t = timeit(lambda n=name: dn.tuned_call(n, img, kind), runs=1, warmup=0)
            print(f"{name:<24}{psnr(out, img):>13.2f}{t.median_ms:>12.1f}")
        print(
            "\nThe 'Change' column is PSNR against the NOISY INPUT, not against a clean\n"
            "original -- there isn't one. A low number means the filter changed the image\n"
            "a lot, which is neither good nor bad on its own: 'do nothing' scores infinity\n"
            "and is usually wrong. Use it to see how aggressive each filter is being, and\n"
            f"the benchmark below to choose. On {kind} noise, {winner} won by {margin:.2f} dB."
        )
        return 0

    if args.default_params:
        out, timing = timeit(lambda: dn.METHODS[method](img), runs=1, warmup=0)
        param = "default"
    else:
        out, timing = timeit(lambda: dn.tuned_call(method, img, kind), runs=1, warmup=0)
        spec = dn.TUNED.get(kind, {}).get(method)
        param = f"{spec[0]}={spec[1]:g}" if spec else "default"

    imwrite(args.out, out)
    print(f"\nfilter   : {method}  ({param})   {timing.median_ms:.1f} ms")
    print(f"changed  : {psnr(out, img):.2f} dB against the input")
    print(f"wrote    : {args.out}")

    print()
    if method == winner:
        print(
            f"verdict  : {method} is the measured winner on {kind} noise "
            f"({winner_db:.2f} dB,\n"
            f"           {margin:.2f} dB clear of the runner-up on the six-image benchmark)."
        )
    else:
        print(
            f"note     : on {kind} noise the measured winner is {winner} "
            f"({winner_db:.2f} dB,\n"
            f"           {margin:.2f} dB clear of second place). You picked {method}.\n"
            f"           Run with --all-methods to compare them on this image."
        )

    if kind == "salt_pepper" and method in ("Bilateral", "Non-local means", "Gaussian", "Box"):
        print(
            "\nwarning  : averaging filters are the wrong family for impulse noise. A mean\n"
            "           is dragged by an outlier and a median is not, which is why Median\n"
            "           beats Bilateral by 8.7 dB on this noise model -- the largest gap\n"
            "           anywhere in this project."
        )
    if kind == "gaussian" and method == "Median":
        print(
            "\nnote     : Median is the WORST of the six filters on Gaussian noise\n"
            "           (27.69 dB against Bilateral's 29.79). Its advantage is rejecting\n"
            "           outliers, and Gaussian noise has none -- every pixel is slightly\n"
            "           wrong rather than a few being completely wrong."
        )
    if sigma < 10 and not args.default_params:
        print(
            f"\nnote     : estimated sigma is {sigma:.1f}. Measured on the benchmark, below\n"
            "           about sigma 10 every filter here scores WORSE than leaving the image\n"
            "           alone -- their blurring costs more than the noise does. Check the\n"
            "           output against the input before keeping it."
        )
    if method == "Non-local means":
        print(
            f"\nnote     : non-local means is {timing.median_ms:.0f} ms here. On the benchmark it is\n"
            "           1,830x a Gaussian blur, and it wins outright on exactly one of the\n"
            "           three noise models (Poisson, by 0.31 dB)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
