"""Dehaze your own photo — inference from the command line.

    python infer.py my_hazy_photo.jpg
    python infer.py my_hazy_photo.jpg --method "Dark channel prior" --out clear.png
    python infer.py my_hazy_photo.jpg --all-methods
    python infer.py my_hazy_photo.jpg --save-transmission

No clear original exists for a photo you supply, so no PSNR is reported. What
*is* reported needs no reference: the estimated airlight, the recovered
transmission range, and the contrast gained.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import dehazing as dz  # noqa: E402
from shared.bench import timeit  # noqa: E402
from shared.io import imread, imwrite, to_uint8  # noqa: E402
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
    ap.add_argument("image", help="path to a hazy photo")
    ap.add_argument("--method", default="DCP + guided refine", choices=list(dz.METHODS))
    ap.add_argument("--out", default="dehazed.png")
    ap.add_argument("--all-methods", action="store_true", help="run all five and compare")
    ap.add_argument("--save-transmission", action="store_true",
                    help="also write the estimated transmission map")
    ap.add_argument("--omega", type=float, default=dz.DEFAULT_OMEGA,
                    help="how much haze to remove, 0-1 (lower leaves more)")
    args = ap.parse_args()

    if not Path(args.image).exists():
        print(f"error: no such file: {args.image}")
        return 2

    hazy = load(args.image)
    print(f"input   : {args.image}  {hazy.shape[1]}x{hazy.shape[0]}")
    print(f"contrast: {rms_contrast(hazy):.4f}   entropy: {entropy(hazy):.2f} bits")

    if args.all_methods:
        print(f"\n{'Method':<26}{'Contrast':>10}{'Entropy':>9}{'Time (ms)':>11}")
        print("-" * 56)
        print(f"{'(input)':<26}{rms_contrast(hazy):>10.4f}{entropy(hazy):>9.2f}{'-':>11}")
        for name, fn in dz.METHODS.items():
            out, t = timeit(lambda f=fn: f(hazy), runs=1, warmup=0)
            print(f"{name:<26}{rms_contrast(out):>10.4f}{entropy(out):>9.2f}{t.median_ms:>11.1f}")
        print(
            "\nNo PSNR column: a photo you supplied has no clear original to compare\n"
            "against. And note that contrast is NOT a ranking -- the contrast-only\n"
            "control beats the physical methods on that column while scoring 6.6 dB\n"
            "worse against a known truth. Run `python run.py` for the measured\n"
            "comparison on generated haze."
        )
        return 0

    a = dz.estimate_airlight(hazy)
    t = dz.refine_transmission_guided(
        hazy, dz.transmission_dcp(hazy, a, omega=args.omega)
    )
    out, timing = timeit(lambda: dz.METHODS[args.method](hazy), runs=1, warmup=0)
    imwrite(args.out, out)

    print(f"method  : {args.method}")
    print(f"airlight: R {a[0]:.3f}  G {a[1]:.3f}  B {a[2]:.3f}   (mean {float(np.mean(a)):.3f})")
    print(f"transmission: range [{t.min():.3f}, {t.max():.3f}]  mean {t.mean():.3f}")
    print(f"contrast: {rms_contrast(hazy):.4f} -> {rms_contrast(out):.4f}")
    print(f"entropy : {entropy(hazy):.2f} -> {entropy(out):.2f} bits")
    print(f"time    : {timing.median_ms:.1f} ms")
    print(f"wrote   : {args.out}")

    if args.save_transmission:
        stem = Path(args.out).with_suffix("")
        imwrite(f"{stem}_transmission.png", to_uint8(np.clip(t, 0, 1)))
        print(f"wrote   : {stem}_transmission.png")

    if float(np.max(a)) > 0.99:
        print(
            "\nnote: the airlight estimate is saturated, which usually means a bright\n"
            "      object (a light, a white wall) was mistaken for sky. Measured on the\n"
            "      test set, correcting this makes the OUTPUT worse by ~1 dB -- the\n"
            "      airlight and transmission errors partially cancel. See the README."
        )
    if t.min() < 0.1:
        print(
            f"\nnote: minimum transmission is {t.min():.3f}. Below about 0.1 the recovery\n"
            "      J = (I-A)/t + A divides by a near-zero number and amplifies noise, so\n"
            "      the deepest haze in this photo is at the limit of what is recoverable."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
