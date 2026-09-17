"""Equalise your own image with every method at once.

    python infer.py photo.jpg
    python infer.py photo.jpg --method "CLAHE (clip 2.0)" --out fixed.png
    python infer.py photo.jpg --simulate     # flatten it first, so PSNR means something

With a real photograph there is no un-degraded original, so **no PSNR is
printed**. What is printed instead are the two measurements that between them
predict whether equalisation has anything to give: **how much of the 0-255 range
the image already uses** (between its 1st and 99th percentiles) and its **mean
luminance**. Counter-intuitively it is the *bright, wide-range* photographs that
gain from equalising, and the dark compressed ones where every method loses.

Entropy and RMS contrast are printed too, and labelled for what they are. Across
this project's eleven photographs, entropy's favourite method is the *worst* one
in the table, and it cannot see a 9 dB improvement. A number going up here is not
evidence that the picture got better.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import numpy as np  # noqa: E402

from shared import io  # noqa: E402
from shared.io import to_gray  # noqa: E402
from shared.metrics import entropy, estimate_noise_sigma, psnr, rms_contrast, ssim  # noqa: E402
from shared.report import init_console  # noqa: E402

import histogram_eq as he  # noqa: E402

#: Where this project's eleven photographs split. Images ABOVE these values are
#: the ones equalisation helped; below them, every method scored at or under
#: doing nothing. The direction is the opposite of the intuition -- a *dark,
#: narrow-range* photograph looks like the one that needs equalising, and is the
#: one where it backfires, because stretching a compressed range stretches its
#: noise by the same factor.
#:
#: These are reported as a tendency, not a rule: the correlation is +0.70 with
#: tone range and +0.73 with mean luma over eleven images. Two of the eleven sit
#: on the wrong side of both.
TONE_SPLIT = 75.0
LUMA_SPLIT = 100.0


def tone_range(img: np.ndarray) -> float:
    """Percentage of 0-255 the image occupies between its 1st and 99th percentiles."""
    lo, hi = np.percentile(to_gray(img), (1, 99))
    return float(hi - lo) / 255.0 * 100


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--method", default=None, choices=list(he.METHODS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--simulate", action="store_true",
                    help="compress the tone range by a known amount first, so "
                         "every method can be scored against the true original")
    ap.add_argument("--kind", default="low_contrast", choices=["low_contrast", "gamma"])
    ap.add_argument("--noise", type=float, default=3.0,
                    help="with --simulate, the noise added alongside the degradation")
    args = ap.parse_args()

    init_console()
    img = io.imread(args.image)
    print(f"{args.image}  {img.shape[1]}x{img.shape[0]}")

    truth = None
    if args.simulate:
        truth = img
        img = he.degrade(truth, kind=args.kind, noise_sigma=args.noise, seed=0)
        print(f"degraded ({args.kind}, noise {args.noise:g}) — the input is now "
              f"{psnr(img, truth):.2f} dB from the truth")

    base_tone = tone_range(img)
    print(f"input: tone range {base_tone:.1f}%, mean luma {to_gray(img).mean():.1f}, "
          f"entropy {entropy(img):.3f} bits")

    names = [args.method] if args.method else list(he.METHODS)
    header = (f"\n{'method':24s} {'tone %':>7s} {'entropy':>8s} {'contrast':>9s} "
              f"{'noise':>7s}")
    if truth is not None:
        header += f" {'PSNR':>8s} {'SSIM':>7s}"
    print(header)

    outs = {}
    for name in names:
        out = he.METHODS[name](img)
        outs[name] = out
        line = (f"{name:24s} {tone_range(out):6.1f}% {entropy(out):8.3f} "
                f"{rms_contrast(out):9.4f} {estimate_noise_sigma(out):7.2f}")
        if truth is not None:
            line += f" {psnr(out, truth):8.2f} {ssim(out, truth):7.4f}"
        print(line)

    if truth is not None:
        rec = he.eq_match_oracle(img, truth)
        outs[he.ORACLE_NAME] = rec
        print(f"{he.ORACLE_NAME:24s} {tone_range(rec):6.1f}% {entropy(rec):8.3f} "
              f"{rms_contrast(rec):9.4f} {estimate_noise_sigma(rec):7.2f} "
              f"{psnr(rec, truth):8.2f} {ssim(rec, truth):7.4f}")

        control = psnr(img, truth)
        best = max(((psnr(o, truth), n) for n, o in outs.items()
                    if n != he.ORACLE_NAME and not n.startswith("Do nothing")))
        print(f"\nbest real method: {best[1]} at {best[0]:.2f} dB "
              f"({best[0] - control:+.2f} against doing nothing), "
              f"oracle {psnr(rec, truth):.2f} dB.")
        if best[0] <= control:
            print("Nothing beat leaving the image alone — which happens on 5 of "
                  "this project's 11 photographs.")
        print("The oracle is the same family of operation handed the right "
              "target, so the gap above it is curve choice, not a limit of "
              "tone curves.")
    else:
        print("\nNo PSNR is reported. There is no un-degraded original of a real "
              "photograph, so any fidelity number here would be invented.")
        luma = float(to_gray(img).mean())
        if base_tone < TONE_SPLIT and luma < LUMA_SPLIT:
            print(f"  -> tone range {base_tone:.1f}%, mean luma {luma:.0f}: both "
                  "below where this project's photographs split. On images like "
                  "this **every method scored at or below doing nothing** — "
                  "stretching a compressed range stretches its noise too.")
        elif base_tone >= TONE_SPLIT and luma >= LUMA_SPLIT:
            print(f"  -> tone range {base_tone:.1f}%, mean luma {luma:.0f}: both "
                  "above the split. This is the side where equalisation helped, "
                  "by up to +3.9 dB. CLAHE was the method that won most often.")
        else:
            print(f"  -> tone range {base_tone:.1f}%, mean luma {luma:.0f}: the two "
                  "indicators disagree, which happened on 2 of this project's 11 "
                  "photographs. Try CLAHE and compare by eye.")
        print("  (a tendency, not a rule: +0.70 correlation with tone range and "
              "+0.73 with mean luma, over eleven images)")
        print("Entropy and contrast are printed above because they are what you "
              "have without a reference — not because they are trustworthy. "
              "Run with --simulate to see how badly they rank these methods.")

    out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "equalised.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.method:
        io.imwrite(out_path, outs[args.method])
    else:
        io.imwrite(out_path, np.hstack(list(outs.values())))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
