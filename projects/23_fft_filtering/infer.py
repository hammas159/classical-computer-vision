"""Inspect and filter your own image in the frequency domain.

    python infer.py photo.jpg                     # spectrum, detected peaks, low-pass
    python infer.py photo.jpg --notch --out clean.png
    python infer.py photo.jpg --simulate          # add known interference, then score

On a real photograph there is no clean original, so **no PSNR is printed**. What
is printed instead is the measurement that actually decides whether a notch is
worth applying: the strongest peaks in the spectrum away from DC, and **how far
above the local background they stand**.

A photograph with no periodic interference has no such peaks — its spectrum
falls off smoothly — so a peak standing well clear of its surroundings is the
evidence, and the prominence is the number to read.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from shared import io  # noqa: E402
from shared.io import to_gray  # noqa: E402
from shared.metrics import psnr  # noqa: E402
from shared.report import init_console  # noqa: E402

import fft_filtering as ff  # noqa: E402

#: Prominence above the local spectral background, in dB, below which a peak is
#: just the image's own structure rather than interference worth notching.
PROMINENCE_DB = 6.0


def peak_prominence(img: np.ndarray, peak: tuple[int, int], radius: int = 20) -> float:
    """How far a spectral peak stands above its own neighbourhood, in dB.

    A natural image's spectrum falls off smoothly, so any peak is a local
    excess. Comparing the peak to an annulus around it — rather than to the
    global mean, which is dominated by DC — is what makes the number mean
    "interference" rather than "low frequency".
    """
    spectrum = np.abs(ff.fft(img))
    y, x = peak
    h, w = spectrum.shape
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot(yy - y, xx - x)
    ring = (d > radius * 0.5) & (d <= radius)
    if not ring.any():
        return 0.0
    return float(20 * np.log10(max(spectrum[y, x], 1e-9)
                               / max(float(np.median(spectrum[ring])), 1e-9)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--out", default=None)
    ap.add_argument("--notch", action="store_true", help="apply the blind notch")
    ap.add_argument("--lowpass", type=float, default=None,
                    help="cutoff radius for a Gaussian low-pass")
    ap.add_argument("--simulate", action="store_true",
                    help="add periodic interference at known frequencies first")
    ap.add_argument("--fx", type=int, default=40)
    ap.add_argument("--fy", type=int, default=25)
    args = ap.parse_args()

    init_console()
    gray = to_gray(io.imread(args.image))
    print(f"{args.image}  {gray.shape[1]}x{gray.shape[0]}")

    truth, true_peaks = None, None
    if args.simulate:
        truth = gray
        gray, true_peaks = ff.add_periodic_noise(truth, args.fx, args.fy, 0.25)
        print(f"added periodic interference at ({args.fx}, {args.fy}) — the input "
              f"is now {psnr(gray, truth):.2f} dB from the original")

    peaks = ff.detect_noise_peaks(gray)
    print(f"\n{'peak (y, x)':>16s} {'prominence':>12s}")
    prominences = []
    for p in peaks:
        prom = peak_prominence(gray, p)
        prominences.append(prom)
        print(f"{str(tuple(int(v) for v in p)):>16s} {prom:11.1f} dB")

    if true_peaks is not None:
        err = min(np.hypot(*(np.array(p) - np.array(t)))
                  for p in peaks for t in true_peaks)
        print(f"  true peaks: {[tuple(int(v) for v in t) for t in true_peaks]}, "
              f"closest detection is {err:.1f} px away")

    strong = [p for p, q in zip(peaks, prominences) if q >= PROMINENCE_DB]
    if strong:
        print(f"\n{len(strong)} peak(s) stand more than {PROMINENCE_DB:g} dB above "
              "their surroundings — that is periodic interference, and a notch "
              "will remove it without touching anything else.")
    else:
        print(f"\nNo peak stands more than {PROMINENCE_DB:g} dB above its "
              "surroundings. This image probably has no periodic interference; "
              "a notch would remove some of the picture instead.")

    outs = {"input": gray}
    if args.notch or args.simulate:
        outs["notch (blind)"] = ff.apply_mask(gray, ff.mask_notch(gray.shape, peaks))
        outs["median 5x5 (spatial)"] = cv2.medianBlur(gray, 5)
    if args.lowpass:
        outs[f"Gaussian low-pass r={args.lowpass:g}"] = ff.apply_mask(
            gray, ff.mask_gaussian(gray.shape, args.lowpass))

    if truth is not None:
        print(f"\n{'method':26s} {'PSNR':>8s}")
        for name, out in outs.items():
            print(f"{name:26s} {psnr(out, truth):8.2f}")
        oracle = ff.apply_mask(gray, ff.mask_notch(gray.shape, true_peaks))
        print(f"{'notch (true peaks)':26s} {psnr(oracle, truth):8.2f}")
    else:
        print("\nNo PSNR is reported. There is no clean original of a real "
              "photograph, so any fidelity number here would be invented. The "
              "prominence above is the measurement that is real.")

    out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "filtered.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panels = [io.ensure_rgb(ff.spectrum_image(ff.fft(gray)))] + [
        io.ensure_rgb(o) for o in outs.values()]
    io.imwrite(out_path, np.hstack(panels))
    print(f"\nwrote {out_path}  (spectrum first, then {', '.join(outs)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
