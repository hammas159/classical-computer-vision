"""Sharpen your own image with every method at once.

    python infer.py photo.jpg
    python infer.py photo.jpg --method "Unsharp mask" --out sharp.png
    python infer.py photo.jpg --amount 2.0
    python infer.py photo.jpg --simulate     # blur it first, so PSNR means something

With a real photograph there is no un-blurred original, so **no PSNR is
printed**. What is printed instead is the pair of numbers this project exists to
separate: **acutance**, which every sharpener raises, and **overshoot**, the
fraction of pixels pushed outside their neighbourhood's range — the halo.

Acutance alone cannot tell a sharpener from a sign error: on a sharp photograph
the deliberately wrong-signed Laplacian raises acutance too, while scoring SSIM
0.149 against the correct pairing's 0.644. Overshoot is what says how much of
the gain was paid for in haloing. Run with `--simulate` to blur the image by a known kernel and
get the methods scored against the truth, including the deconvolution oracle.
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from shared import io  # noqa: E402
from shared.metrics import psnr, ssim  # noqa: E402
from shared.report import init_console  # noqa: E402

import sharpening as sh  # noqa: E402

#: Overshoot above this is what people call "oversharpened" — visible rims.
HALO_WARN = 0.15


def apply(fn, img, amount: float):
    """Call a method, passing ``amount`` only if that is what it takes.

    `sharpen_high_boost`'s second parameter is ``boost``, which is a *different
    quantity* -- the fraction of the original kept, where 1.0 means a pure
    high-pass and a nearly black result. Feeding the sharpening amount into it
    positionally turned the high-boost row into noise. Methods are matched by
    parameter name rather than position.
    """
    params = list(inspect.signature(fn).parameters)
    if len(params) > 1 and params[1] == "amount":
        return fn(img, amount)
    return fn(img)


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--method", default=None, choices=list(sh.METHODS))
    ap.add_argument("--amount", type=float, default=1.0,
                    help="sharpening strength, for the methods that take one. "
                         "High-boost is controlled by a boost factor instead "
                         "and keeps its textbook A = 1.5.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--simulate", action="store_true",
                    help="blur the image by a known kernel first, so every method "
                         "can be scored against the true original")
    ap.add_argument("--sigma", type=float, default=1.5,
                    help="with --simulate, the blur to apply and then undo")
    args = ap.parse_args()

    img = io.imread(args.image)
    print(f"{args.image}  {img.shape[1]}x{img.shape[0]}")

    truth = None
    if args.simulate:
        truth = img
        img = cv2.GaussianBlur(truth, (0, 0), args.sigma,
                               borderType=cv2.BORDER_REFLECT)
        print(f"blurred by sigma {args.sigma:g} — the input is now "
              f"{psnr(img, truth):.2f} dB from the truth")

    base_acut = sh.acutance(img)
    print(f"input acutance: {base_acut:.4f}")

    names = [args.method] if args.method else list(sh.METHODS)
    header = f"\n{'method':26s} {'acutance':>9s} {'vs input':>9s} {'overshoot':>10s}"
    if truth is not None:
        header += f" {'PSNR':>8s} {'PSNRm':>8s} {'SSIM':>7s}"
    print(header)

    outs, rows = {}, []
    for name in names:
        out = apply(sh.METHODS[name], img, args.amount)
        outs[name] = out
        a, o = sh.acutance(out), sh.overshoot(out, img)
        line = f"{name:26s} {a:9.4f} {a / max(base_acut, sh.EPS):8.2f}x {o:10.4f}"
        row = {"method": name, "acutance": a, "overshoot": o}
        if truth is not None:
            p = psnr(out, truth)
            pm = psnr(sh.match_mean(out, truth), truth)
            s = ssim(out, truth)
            line += f" {p:8.2f} {pm:8.2f} {s:7.4f}"
            row |= {"psnr": p, "psnr_matched": pm, "ssim": s}
        rows.append(row)
        print(line)

    if truth is not None:
        ksize = int(max(3, round(args.sigma * 6) | 1))
        k = cv2.getGaussianKernel(ksize, args.sigma)
        rec = sh.deconvolve_oracle_best(img, (k @ k.T).astype(np.float32), truth)
        outs[sh.ORACLE_NAME] = rec
        print(f"{sh.ORACLE_NAME:26s} {sh.acutance(rec):9.4f} "
              f"{sh.acutance(rec) / max(base_acut, sh.EPS):8.2f}x "
              f"{sh.overshoot(rec, img):10.4f} {psnr(rec, truth):8.2f} "
              f"{psnr(sh.match_mean(rec, truth), truth):8.2f} {ssim(rec, truth):7.4f}")

        best = max(rows, key=lambda r: r["psnr_matched"])
        print(f"\nbest real method by matched PSNR: {best['method']} "
              f"at {best['psnr_matched']:.2f} dB, against the oracle's "
              f"{psnr(sh.match_mean(rec, truth), truth):.2f} dB and the "
              f"blurred input's {psnr(img, truth):.2f} dB.")
        print("PSNR and PSNRm differ only for formulas that change brightness; "
              "High-boost is the one that does.")
    else:
        haloed = [r for r in rows if r["overshoot"] > HALO_WARN]
        print("\nNo PSNR is reported. There is no un-blurred original of a real "
              "photograph, so any fidelity number here would be invented.")
        if haloed:
            print("Visible haloing (overshoot > "
                  f"{HALO_WARN:g}): " + ", ".join(
                      f"{r['method']} ({r['overshoot']:.3f})" for r in haloed))
        risen = [r["method"] for r in rows
                 if r["acutance"] > base_acut and not r["method"].startswith("Do ")]
        wrong = "Laplacian (WRONG sign)"
        if wrong in risen:
            print(f"\n{wrong} RAISED acutance on this image "
                  "— and it is a bug, not a method. Acutance going up is not "
                  "evidence that a sharpener is working.")
        print("Run with --simulate to see every method scored against a truth.")

    out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "sharpened.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.method:
        io.imwrite(out_path, outs[args.method])
    else:
        h = min(o.shape[0] for o in outs.values())
        io.imwrite(out_path, np.hstack([o[:h] for o in outs.values()]))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
