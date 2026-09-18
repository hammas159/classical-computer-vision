"""Denoise an image of your own, and check the premise before trusting it.

    python infer.py photo.jpg
    python infer.py photo.jpg --sigma 25          # add known noise, so PSNR is real
    python infer.py photo.jpg --method "Bilateral (spatial)" --out clean.png

Wavelet denoising rests on your image being **sparse** in the wavelet basis: a
few large coefficients carrying the signal while noise spreads evenly. That is
measurable in advance and it is measured here first, because on this project's
twelve photographs it ranges from 0.688 to 0.989 and the method has much less to
work with at the bottom of that range.

Without a clean reference no PSNR can be printed, so what is printed instead is
the estimated noise, your image's sparsity, and how far each method moved the
pixels. The noise estimate is biased — it over-reads by 20% on a quiet image and
under-reads by 39% on a loud one — and that is stated rather than hidden, because
the wavelet thresholds are proportional to it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import numpy as np  # noqa: E402

from shared import figures, io, synth  # noqa: E402
from shared.io import ensure_rgb, to_gray  # noqa: E402
from shared.metrics import psnr, ssim  # noqa: E402
from shared.report import init_console  # noqa: E402

import wavelet_denoising as wd  # noqa: E402

#: Sparsity below which the wavelet basis has little to separate. The densest
#: photograph in this project's pool sits at 0.688.
LOW_SPARSITY = 0.75

#: Estimated noise below which denoising is likely to cost more than it saves —
#: at sigma 5 five of seven methods here score worse than doing nothing.
BARELY_NOISY = 7.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", default=None)
    ap.add_argument("--photo", default=None, choices=list(wd.IMAGES))
    ap.add_argument("--sigma", type=float, default=None,
                    help="add Gaussian noise of this sigma, so PSNR can be reported")
    ap.add_argument("--method", default=None, choices=list(wd.METHODS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    init_console()
    if args.photo:
        clean = wd.load_scene(args.photo)
        source = args.photo
    elif args.image:
        clean = io.imread(args.image)
        source = args.image
    else:
        ap.error("give an image path, or --photo with one of the project's photographs")

    if args.sigma:
        noisy = synth.gaussian_noise(clean, sigma=args.sigma, seed=0)
        truth = clean
        print(f"{source}: noise added at sigma {args.sigma:g}, so PSNR below is real")
    else:
        noisy = clean
        truth = None
        print(f"{source}: taken as-is, so there is no clean reference")

    # ------------------------------------------------------------------ #
    # the premise, before anything that depends on it
    # ------------------------------------------------------------------ #
    sparsity = wd.image_sparsity(noisy)
    estimated = wd.estimate_noise(noisy)
    print(f"  {noisy.shape[1]}x{noisy.shape[0]}, wavelet sparsity {sparsity:.3f}, "
          f"estimated noise sigma {estimated:.2f}")

    if sparsity < LOW_SPARSITY:
        print(f"  that sparsity is below {LOW_SPARSITY:g} — this image is close to "
              "wavelet-incompressible, and thresholding has little to separate. "
              "Expect the spatial filters to win by more than usual.")
    else:
        print("  sparse enough for the premise to hold; white noise scores 0.44 for "
              "comparison.")

    if estimated < BARELY_NOISY:
        print(f"  and the estimate is under {BARELY_NOISY:g}, so there may be nothing "
              "worth removing. On this project's photographs at sigma 5, five of seven "
              "methods scored WORSE than leaving the image alone.")
    if args.sigma:
        ratio = estimated / args.sigma
        print(f"  the estimate is {100 * ratio:.0f}% of the true sigma — it over-reads on "
              "quiet images and under-reads on loud ones, and the wavelet thresholds "
              "are proportional to it.")

    # ------------------------------------------------------------------ #
    # the methods
    # ------------------------------------------------------------------ #
    header = f"\n{'method':30s} {'moved':>8s} {'sparsity after':>15s}"
    if truth is not None:
        header += f" {'PSNR':>8s} {'SSIM':>8s}"
    print(header)

    panels = [("input", noisy)]
    results, scores = {}, {}
    for name, fn in wd.METHODS.items():
        out = fn(noisy)
        results[name] = out
        moved = float(np.abs(out.astype(np.float64) - noisy.astype(np.float64)).mean())
        line = f"{name:30s} {moved:8.2f} {wd.image_sparsity(out):15.3f}"
        if truth is not None:
            value = psnr(out, truth)
            scores[name] = value
            line += f" {value:8.2f} {ssim(to_gray(out), to_gray(truth)):8.4f}"
        print(line)
        panels.append((name, ensure_rgb(out)))

    # ------------------------------------------------------------------ #
    # what to make of it
    # ------------------------------------------------------------------ #
    print()
    if truth is not None:
        control = scores["Do nothing (control)"]
        best = max(scores, key=scores.get)
        harmful = [n for n, v in scores.items()
                   if n != "Do nothing (control)" and v < control]
        print(f"best: {best} at {scores[best]:.2f} dB "
              f"({scores[best] - control:+.2f} over doing nothing)")
        if harmful:
            print(f"  {len(harmful)} method(s) scored BELOW doing nothing: "
                  f"{', '.join(n.split(' (')[0] for n in harmful)}")

        wavelets = {n: v for n, v in scores.items() if n.startswith("Wavelet")}
        spatial = {n: v for n, v in scores.items() if "spatial" in n}
        bw, bs = max(wavelets, key=wavelets.get), max(spatial, key=spatial.get)
        print(f"  best wavelet {bw.replace('Wavelet ', '')} {wavelets[bw]:.2f} dB, "
              f"best spatial {bs.split(' (')[0]} {spatial[bs]:.2f} dB "
              f"({spatial[bs] - wavelets[bw]:+.2f})")
    else:
        moved = {n: float(np.abs(results[n].astype(np.float64)
                                 - noisy.astype(np.float64)).mean())
                 for n in results}
        busiest = max(moved, key=moved.get)
        print("No PSNR is printed — there is no clean reference for your image.")
        print(f"  {busiest.split(' (')[0]} changed it most ({moved[busiest]:.2f} levels "
              "per pixel). With no truth to check against, a large change is a warning "
              "and not evidence: run with --sigma on a clean photograph to get a real "
              "number, or compare the pictures.")

    if args.method:
        out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "denoised.png"
        io.imwrite(out_path, results[args.method])
    else:
        out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "methods.png"
        figures.grid(panels, out_path, ncols=3,
                     suptitle=f"Every denoiser on {Path(source).name}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
