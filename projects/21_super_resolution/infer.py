"""Upscale your own image with every method at once.

    python infer.py photo.jpg --scale 4
    python infer.py photo.jpg --simulate --scale 4     # shrink it first, then score
    python infer.py photo.jpg --method Back-projection --out big.png

Upscaling a photograph you already have gives no ground truth, so **no PSNR is
printed**. What is printed is each method's high-frequency energy, with the
warning attached: nearest-neighbour scores the *highest* of any method here
while being the worst by PSNR, because a blocky staircase is high-frequency
energy too.

`--simulate` downsamples the image correctly first — blur then decimate, not
`cv2.resize` — so the methods and the band-limited reference can be scored
against a real original. That distinction is worth 2.12 dB; see the README.
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
from shared.metrics import psnr, ssim  # noqa: E402
from shared.report import init_console  # noqa: E402

import super_resolution as sr  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--method", default=None, choices=list(sr.METHODS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--simulate", action="store_true",
                    help="treat the input as the high-resolution truth, shrink "
                         "it correctly, and score every method against it")
    args = ap.parse_args()

    init_console()
    img = io.imread(args.image)
    scale = args.scale

    truth = None
    if args.simulate:
        h = (img.shape[0] // scale) * scale
        w = (img.shape[1] // scale) * scale
        truth = img[:h, :w]
        lr = synth.downsample_for_sr(truth, scale=scale)
        print(f"{args.image}  {w}x{h} -> {lr.shape[1]}x{lr.shape[0]} "
              "(blurred then decimated, NOT cv2.resize)")
    else:
        lr = img
        print(f"{args.image}  {lr.shape[1]}x{lr.shape[0]} -> "
              f"{lr.shape[1] * scale}x{lr.shape[0] * scale}")

    names = [args.method] if args.method else list(sr.METHODS)
    header = f"\n{'method':18s} {'high-freq':>10s} {'ms':>8s}"
    if truth is not None:
        header += f" {'PSNR':>8s} {'SSIM':>7s}"
    print(header)

    outs = {}
    for name in names:
        t0 = time.perf_counter()
        out = sr.METHODS[name](lr, scale)
        ms = (time.perf_counter() - t0) * 1000
        if truth is not None:
            out = out[:truth.shape[0], :truth.shape[1]]
        outs[name] = out
        line = f"{name:18s} {sr.frequency_content(out):10.4f} {ms:8.1f}"
        if truth is not None:
            line += f" {psnr(out, truth):8.2f} {ssim(out, truth):7.4f}"
        print(line)

    if truth is not None:
        ref = sr.oracle_band_limited(truth, scale)
        print(f"{sr.ORACLE_NAME:18s} {sr.frequency_content(ref):10.4f} "
              f"{'-':>8s} {psnr(ref, truth):8.2f} {ssim(ref, truth):7.4f}")
        best = max(((psnr(o, truth), n) for n, o in outs.items()))
        gap = psnr(ref, truth) - best[0]
        print(f"\nbest: {best[1]} at {best[0]:.2f} dB, "
              f"{abs(gap):.2f} dB {'below' if gap > 0 else 'ABOVE'} the "
              "band-limited reference.")
        if gap < 0:
            print("  Above the reference is legitimate: a Gaussian anti-alias "
                  "filter only attenuates the frequencies above Nyquist, and a "
                  "method that models the degradation can partly invert that.")
        classics = [n for n in outs if n in
                    ("Nearest", "Bilinear", "Bicubic", "Lanczos-4", "Edge-directed")]
        if len(classics) > 1:
            vals = [psnr(outs[n], truth) for n in classics]
            print(f"  The whole nearest-to-Lanczos spread here: "
                  f"{max(vals) - min(vals):.2f} dB.")
    else:
        hf = {n: sr.frequency_content(o) for n, o in outs.items()}
        print("\nNo PSNR is reported. There is no high-resolution original of an "
              "image you are upscaling, so any fidelity number would be invented.")
        if len(hf) > 1 and max(hf, key=hf.get) == "Nearest":
            print("  Nearest has the highest high-frequency energy here — as it "
                  "does on every one of this project's twelve photographs, where "
                  "it is also the WORST method by PSNR. That energy is the blocky "
                  "staircase, not recovered detail.")
        print("Run with --simulate to see the methods scored against a truth.")

    out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "upscaled.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.method:
        io.imwrite(out_path, outs[args.method])
    else:
        h = min(o.shape[0] for o in outs.values())
        w = min(o.shape[1] for o in outs.values())
        io.imwrite(out_path, np.hstack([o[:h, :w] for o in outs.values()]))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
