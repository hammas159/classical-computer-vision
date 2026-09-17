"""Deblur your own image with every method at once.

    python infer.py blurred.jpg
    python infer.py blurred.jpg --angle 30 --length 15
    python infer.py sharp.jpg --simulate              # blur it first, then score properly
    python infer.py blurred.jpg --method Richardson-Lucy --out fixed.png

On a real blurred photograph there is no sharp original, so **no PSNR is
printed** — any number there would be invented. The blur direction *is*
estimated, from the log power spectrum, and that estimate is real.

Acutance is printed for each method, with a warning attached: on this project's
twelve photographs the inverse filter raises acutance while scoring **17 dB
below doing nothing**, because static has a very high gradient magnitude. A
sharpness number going up is not evidence that the picture got better.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from shared import io, synth  # noqa: E402
from shared.io import to_float, to_gray  # noqa: E402
from shared.metrics import psnr, ssim  # noqa: E402
from shared.report import init_console  # noqa: E402

import deblurring as db  # noqa: E402


def acutance(img: np.ndarray) -> float:
    g = to_float(to_gray(img))
    dx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.hypot(dx, dy)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--method", default=None, choices=list(db.METHODS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--angle", type=float, default=None,
                    help="motion-blur angle in degrees; estimated from the image if omitted")
    ap.add_argument("--length", type=int, default=15,
                    help="motion-blur length in pixels. NOT estimated — see the README")
    ap.add_argument("--simulate", action="store_true",
                    help="treat the input as sharp, blur it by a known kernel, "
                         "and score every method against it")
    ap.add_argument("--noise", type=float, default=3.0,
                    help="with --simulate, the noise added after blurring")
    args = ap.parse_args()

    init_console()
    img = io.imread(args.image)
    print(f"{args.image}  {img.shape[1]}x{img.shape[0]}")

    truth = None
    if args.simulate:
        truth = to_gray(img)
        angle = args.angle if args.angle is not None else 30.0
        psf = synth.motion_blur_kernel(args.length, angle)
        blurred = synth.apply_kernel(img, psf)
        if args.noise > 0:
            blurred = synth.gaussian_noise(blurred, sigma=args.noise, seed=0)
        print(f"blurred by a {args.length} px motion kernel at {angle:g} degrees, "
              f"noise sigma {args.noise:g} — the input is now "
              f"{psnr(to_gray(blurred), truth):.2f} dB from the truth")
    else:
        blurred = img

    estimated = db.estimate_motion_angle(blurred)
    if args.angle is None:
        angle = estimated
        print(f"estimated motion angle: {angle:.1f} degrees "
              "(from the log power spectrum)")
        print(f"  blur LENGTH is not estimated — using --length {args.length}. "
              "A wrong length costs more than a wrong angle.")
    else:
        angle = args.angle
        print(f"using the given angle {angle:g} degrees "
              f"(the estimator said {estimated:.1f})")

    psf = synth.motion_blur_kernel(args.length, angle)
    base_acut = acutance(blurred)
    print(f"input acutance: {base_acut:.4f}")

    names = [args.method] if args.method else list(db.METHODS)
    header = f"\n{'method':22s} {'knows psf':>10s} {'acutance':>9s} {'ms':>8s}"
    if truth is not None:
        header += f" {'PSNR':>8s} {'SSIM':>7s}"
    print(header)

    outs = {}
    for name in names:
        t0 = time.perf_counter()
        out = db.METHODS[name](blurred, psf)
        ms = (time.perf_counter() - t0) * 1000
        outs[name] = out
        knows = "yes" if name in db.KNOWS_KERNEL else "no"
        line = f"{name:22s} {knows:>10s} {acutance(out):9.4f} {ms:8.1f}"
        if truth is not None:
            line += f" {psnr(out, truth):8.2f} {ssim(out, truth):7.4f}"
        print(line)

    if truth is not None:
        control = psnr(to_gray(blurred), truth)
        best = max(((psnr(o, truth), n) for n, o in outs.items()))
        print(f"\ndoing nothing scores {control:.2f} dB. "
              f"Best here: {best[1]} at {best[0]:.2f} dB "
              f"({best[0] - control:+.2f}).")
        losers = [n for n, o in outs.items()
                  if psnr(o, truth) < control and not n.startswith("Do nothing")]
        if losers:
            print("Worse than leaving the image alone: " + ", ".join(losers))
    else:
        print("\nNo PSNR is reported. There is no sharp original of a real "
              "blurred photograph, so any fidelity number here would be invented.")
        risen = [n for n, o in outs.items() if acutance(o) > base_acut * 1.5]
        if "Inverse filter" in risen:
            print("Note: the inverse filter's acutance went UP. On this "
                  "project's photographs it does that while scoring 17 dB below "
                  "doing nothing — static has a very high gradient magnitude. "
                  "Look at the picture, not the number.")
        print("Richardson-Lucy was the only method that clearly beat doing "
              "nothing here, and it diverges if run too long; 30 iterations is "
              "the default for that reason.")

    out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "deblurred.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panels = [io.ensure_rgb(outs[n]) for n in names]
    io.imwrite(out_path, panels[0] if args.method else np.hstack(panels))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
