"""Reconstruct a high-resolution image from a burst of your own frames.

    python infer.py frame1.png frame2.png frame3.png ...
    python infer.py --photo mayan_stone_carving --frames 16
    python infer.py burst/*.png --registration ECC --scale 2

Your frames are registered to the first one and fused. There is no ground truth
for a real burst, so **no PSNR is printed** — what is printed instead is the
measured offset of each frame and whether it is sub-pixel.

That is the number that decides whether any of this is worth running. Frames
offset by whole pixels land the sensor on exactly the same grid positions and
carry no samples the first frame does not already have: on this project's
photographs that costs 0.81 dB, most of the fusion gain. A burst shot on a tripod
is precisely that case.

With `--photo` the frames are simulated from a known image with known offsets, so
the PSNR against the truth *is* printed and the reconstruction can be checked.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import numpy as np  # noqa: E402

from shared import figures, io  # noqa: E402
from shared.io import ensure_rgb  # noqa: E402
from shared.report import init_console  # noqa: E402

import multiframe_sr as sr  # noqa: E402

#: Fraction of a low-resolution pixel below which an offset counts as "whole
#: pixel" — no new sample positions, so nothing to super-resolve.
WHOLE_PIXEL = 0.15

#: Registration error above which fusion stops paying, in high-resolution pixels,
#: measured by `sweep_registration_error`.
TOLERANCE = 2.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", nargs="*", default=None)
    ap.add_argument("--photo", default=None, choices=list(sr.IMAGES))
    ap.add_argument("--frames", dest="n_frames", type=int, default=8)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--registration", default="ECC", choices=list(sr.REGISTRATION))
    ap.add_argument("--method", default="Iterative back-projection",
                    choices=list(sr.METHODS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    init_console()
    truth = None
    if args.photo:
        truth = sr._prepare(args.photo, args.scale)
        frames, true_offsets = sr.make_stack(truth, n_frames=args.n_frames,
                                             scale=args.scale, seed=0)
        source = f"{args.photo} (simulated, detail {sr.detail(truth):.0f})"
    elif args.frames:
        frames = [io.imread(f) for f in args.frames]
        true_offsets = None
        source = f"{len(frames)} frames from {Path(args.frames[0]).parent}"
        shapes = {f.shape for f in frames}
        if len(shapes) != 1:
            raise SystemExit(f"the frames must all be the same size; got {shapes}")
    else:
        ap.error("give some frame paths, or --photo with one of the project's images")

    print(f"{source}: {len(frames)} frames of "
          f"{frames[0].shape[1]}x{frames[0].shape[0]}, upscaling x{args.scale}")
    if len(frames) < 2:
        print("  one frame only — there is nothing to fuse. Back-projection will still "
              "deblur it, which on this project's photographs is worth +0.68 dB.")

    # ------------------------------------------------------------------ #
    # registration
    # ------------------------------------------------------------------ #
    register = sr.REGISTRATION[args.registration]
    offsets = [(0.0, 0.0)]
    print(f"\nregistering with {args.registration} "
          f"(mean error on this project's stacks: "
          f"{'0.065' if args.registration == 'ECC' else '0.410'} px)")
    print(f"\n{'frame':>6s} {'dx (hi-res px)':>16s} {'dy':>10s} "
          f"{'sub-pixel?':>12s}" + (f" {'true dx, dy':>18s}" if true_offsets else ""))
    for i, frame in enumerate(frames):
        if i == 0:
            estimate = (0.0, 0.0)
        else:
            estimate = register(frames[0], frame, args.scale)
            offsets.append(estimate)
        lr = (estimate[0] / args.scale, estimate[1] / args.scale)
        frac = max(abs(lr[0] - round(lr[0])), abs(lr[1] - round(lr[1])))
        line = (f"{i:6d} {estimate[0]:16.3f} {estimate[1]:10.3f} "
                f"{('yes' if frac > WHOLE_PIXEL else 'NO'):>12s}")
        if true_offsets:
            line += f" {f'{true_offsets[i][0]:.2f}, {true_offsets[i][1]:.2f}':>18s}"
        print(line)

    fractional = []
    for dx, dy in offsets[1:]:
        lr = (dx / args.scale, dy / args.scale)
        fractional.append(max(abs(lr[0] - round(lr[0])), abs(lr[1] - round(lr[1]))))
    sub_pixel = sum(1 for f in fractional if f > WHOLE_PIXEL)

    print(f"\n{sub_pixel} of {len(fractional)} frames are offset by a genuine fraction "
          "of a low-resolution pixel.")
    if sub_pixel < len(fractional) / 2:
        print("  Most of your frames land on the same sensor grid as the first, so they "
              "carry no new sample positions. Expect noise reduction and no resolution "
              "— on this project's photographs that is 0.81 dB of the gain missing.")
    else:
        print("  That is the case super-resolution is for.")

    if true_offsets:
        errors = [np.hypot(e[0] - t[0], e[1] - t[1])
                  for e, t in zip(offsets, true_offsets)]
        mean_error = float(np.mean(errors))
        print(f"\nregistration error against the truth: {mean_error:.4f} px mean, "
              f"{max(errors):.4f} worst")
        if mean_error > TOLERANCE:
            print(f"  above the {TOLERANCE:g} px where fusion stops paying — the "
                  "reconstruction will be worse than upscaling one frame")

    # ------------------------------------------------------------------ #
    # reconstruction
    # ------------------------------------------------------------------ #
    print(f"\n{'method':28s} {'':>10s}" + ("  PSNR (dB)" if truth is not None else ""))
    panels, results = [], {}
    for name, fn in sr.METHODS.items():
        out = fn(frames, offsets, scale=args.scale)
        results[name] = out
        line = f"{name:28s} {'':>10s}"
        if truth is not None:
            from shared.metrics import psnr
            line += f"  {psnr(out, truth):9.2f}"
        print(line)
        panels.append((name, ensure_rgb(out)))

    if truth is not None:
        from shared.metrics import psnr
        scores = {n: psnr(o, truth) for n, o in results.items()}
        best = max(scores, key=scores.get)
        single = scores["Single frame (bicubic)"]
        print(f"\nbest: {best} at {scores[best]:.2f} dB, "
              f"{scores[best] - single:+.2f} over upscaling one frame")
        worse = [n for n in scores if scores[n] < single]
        if worse:
            print(f"  {len(worse)} method(s) scored BELOW one frame: "
                  f"{', '.join(worse)} — averaging frames that are not aligned is a blur")
    else:
        print("\nNo PSNR is printed. There is no ground truth for your burst, and the "
              "sub-pixel count above is the measurement that is real.")

    out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "reconstructed.png"
    io.imwrite(out_path, results[args.method])
    figures.grid([("one frame, upscaled",
                   ensure_rgb(sr.sr_single_frame(frames, offsets, scale=args.scale)))]
                 + panels[1:],
                 PROJECT_DIR / "results" / "methods.png", ncols=2,
                 suptitle=f"Reconstructions from {len(frames)} frames at x{args.scale}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
