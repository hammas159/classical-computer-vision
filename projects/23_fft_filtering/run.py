"""Run the frequency-domain filtering comparison and write results + figures.

    python run.py

Regenerates `results/results.json`, `results/tables.md` and every figure in
`docs/images/`. The README's numbers are copied from those files rather than
typed, so this script is the single source of truth for every claim made.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from shared import figures  # noqa: E402
from shared.io import ensure_rgb, to_float, to_gray  # noqa: E402
from shared.metrics import psnr  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import fft_filtering as ff  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"

#: Cutoff radius the headline comparison runs at.
GALLERY_CUTOFF = 40.0

#: Cutoff the ringing measurement runs at. Low enough that a hard truncation
#: genuinely rings; at 40 every filter's profile is smooth and the question
#: cannot be asked.
RINGING_CUTOFF = 20.0

#: The periodic interference added for the notch experiment.
NOISE_FX, NOISE_FY, NOISE_AMP = 40, 25, 0.25

MASKS = {
    "Ideal": lambda shape, c: ff.mask_ideal(shape, c),
    "Butterworth (n=2)": lambda shape, c: ff.mask_butterworth(shape, c, 2),
    "Butterworth (n=8)": lambda shape, c: ff.mask_butterworth(shape, c, 8),
    "Gaussian": lambda shape, c: ff.mask_gaussian(shape, c),
}


def detail_of(img: np.ndarray) -> float:
    g = to_float(to_gray(img))
    dx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.hypot(dx, dy)) * 1000)


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    print(f"Low-passing {len(ff.IMAGES)} photographs at cutoff {GALLERY_CUTOFF:g} ...")
    lowpass = ff.evaluate_lowpass(cutoff=GALLERY_CUTOFF, images=ff.IMAGES, runs=args.runs)
    print("Sweeping the Butterworth order ...")
    order_rows = ff.sweep_butterworth_order(images=ff.IMAGES)
    print("Sweeping the cutoff ...")
    cutoff_rows = ff.sweep_cutoff(images=ff.IMAGES)
    print("Removing periodic interference with a notch filter ...")
    notch_rows, notch_extra = ff.evaluate_notch(images=ff.IMAGES, fx=NOISE_FX,
                                                fy=NOISE_FY, amplitude=NOISE_AMP)
    print("Correcting uneven illumination ...")
    homo_rows = ff.evaluate_homomorphic(images=ff.IMAGES)
    print("Measuring ringing on a step edge ...")
    ringing = [dict(filter=name, **ff.ringing_on_step(fn, RINGING_CUTOFF))
               for name, fn in MASKS.items()]

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    # The notch experiment is the headline, because it is the one thing the
    # frequency domain does that the spatial domain cannot: periodic
    # interference is a handful of isolated spikes in the spectrum and a
    # spatially unbounded mess in the image.
    columns = ["periodic noise added", "Median filter (spatial)",
               "Notch, blind peaks", "Notch, true peaks (oracle)"]
    DETAIL_BANDS = [("smooth", 0, 200), ("moderate", 200, 290),
                    ("detailed", 290, 400), ("fine", 400, 10000)]
    candidates: dict[str, list[dict]] = {}
    for name in ff.IMAGES:
        clean = to_gray(ff.load_scene(name))
        detail = detail_of(clean)
        band = next(b for b, a, z in DETAIL_BANDS if a <= detail < z)

        noisy, peaks = ff.add_periodic_noise(clean, NOISE_FX, NOISE_FY, NOISE_AMP)
        med = cv2.medianBlur(noisy, 5)
        blind = ff.apply_mask(noisy, ff.mask_notch(
            clean.shape, ff.detect_noise_peaks(noisy)))
        oracle = ff.apply_mask(noisy, ff.mask_notch(clean.shape, peaks))
        outs = [noisy, med, blind, oracle]
        scores = [psnr(o, clean) for o in outs]
        print(f"scene candidate {name:22s} detail {detail:6.1f}  [{band:9s}]  "
              f"noisy {scores[0]:5.2f} dB -> notch {scores[2]:5.2f} dB "
              f"(median only {scores[1]:5.2f})")
        candidates.setdefault(band, []).append({
            "label": f"{name.replace('_', ' ')}\ndetail {detail:.0f}",
            "subject": name,
            "detail": detail,
            "images": [ensure_rgb(clean)] + [ensure_rgb(o) for o in outs],
            "notes": ["original"] + [f"{s:.1f} dB" for s in scores],
            "scores": scores,
            "score": float(np.std(scores)),
        })

    chosen, used = [], set()
    for band, _, _ in DETAIL_BANDS:
        pool = sorted(candidates.get(band, []), key=lambda r: -r["score"])
        pick = next((r for r in pool if r["subject"] not in used), None)
        if pick:
            used.add(pick["subject"])
            chosen.append(pick)
    chosen = chosen[:4]
    assert len({r["subject"] for r in chosen}) == len(chosen), "a subject repeats"

    figures.gallery(
        ["original"] + columns,
        [(r["label"], r["images"]) for r in chosen],
        IMAGES / "compare_notch.png",
        cell_notes=[r["notes"] for r in chosen],
        suptitle=(
            "Periodic interference: a handful of spikes in the spectrum and an "
            "unbounded mess in the image. Cells are PSNR against the original — "
            "the blind notch finds the spikes itself."
        ),
    )
    gallery_table = markdown_table(
        [dict([("Sr", i), ("Scene", r["label"].replace("\n", " · "))]
              + [(c, f"{s:.1f} dB") for c, s in zip(columns, r["scores"])])
         for i, r in enumerate(chosen, start=1)],
        [("Sr", "Sr"), ("Scene", "Scene")] + [(c, c) for c in columns],
    )
    print(f"\nfront-on comparison: {len(chosen)} scenes x {len(columns)} methods")
    print("\n--- notch filtering ---\n" + gallery_table)

    # ------------------------------------------------------------------ #
    # the signature figure: the filter's profile, and the ringing it causes
    # ------------------------------------------------------------------ #
    # The transfer function IS the method here, exactly as the transfer curve is
    # in project 17. A radial slice through each mask says everything about what
    # the filter will do, including whether it will ring.
    size = ff.STEP_SIZE
    radii = np.arange(size // 2)
    profiles = {}
    for name, fn in MASKS.items():
        mask = fn((size, size), RINGING_CUTOFF)
        profiles[name] = mask[size // 2, size // 2:].tolist()
    figures.lines(
        radii.tolist(),
        profiles,
        IMAGES / "filter_profiles.png",
        xlabel="radius in the spectrum (px)",
        ylabel="gain",
        title=(f"The transfer function IS the method — radial slice at cutoff "
               f"{RINGING_CUTOFF:g}. The sharper the corner, the worse the ringing."),
    )

    step = to_float(ff.step_edge())
    step_series = {"original step": step[size // 2].tolist()}
    for name, fn in MASKS.items():
        spec = np.fft.fftshift(np.fft.fft2(step))
        out = np.real(np.fft.ifft2(np.fft.ifftshift(spec * fn(step.shape, RINGING_CUTOFF))))
        step_series[name] = out[size // 2].tolist()
    lo, hi = size // 2 - 40, size // 2 + 40
    figures.lines(
        list(range(lo, hi)),
        {k: v[lo:hi] for k, v in step_series.items()},
        IMAGES / "step_ringing.png",
        xlabel="column (pixels)",
        ylabel="intensity",
        title="Gibbs ringing on a step edge — where it can actually be measured",
        dashed={"original step"},
    )

    figures.lines(
        [r["order"] for r in order_rows],
        {"PSNR (dB)": [r["psnr_db"] for r in order_rows],
         "ringing score x 100": [r["ringing"] * 100 for r in order_rows]},
        IMAGES / "butterworth_order.png",
        xlabel="Butterworth order",
        ylabel="score",
        title="Butterworth order: the knob between Gaussian and ideal",
    )

    figures.lines(
        [r["cutoff"] for r in cutoff_rows],
        {k: [r[k] for r in cutoff_rows] for k in cutoff_rows[0] if k != "cutoff"},
        IMAGES / "cutoff_sweep.png",
        xlabel="cutoff radius (px)",
        ylabel="PSNR (dB)",
        title="How much of the picture lives above each radius",
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS,
        "23_fft_filtering",
        {
            "images": list(ff.IMAGES),
            "gallery_cutoff": GALLERY_CUTOFF,
            "ringing_cutoff": RINGING_CUTOFF,
            "periodic_noise": {"fx": NOISE_FX, "fy": NOISE_FY, "amplitude": NOISE_AMP},
            "lowpass": lowpass,
            "butterworth_order": order_rows,
            "cutoff_sweep": cutoff_rows,
            "notch": notch_rows,
            "notch_extra": notch_extra,
            "homomorphic": homo_rows,
            "ringing_on_step": ringing,
        },
    )

    lowpass_table = markdown_table(
        lowpass,
        [("Filter", "filter"), ("PSNR (dB)", "psnr_db"), ("SSIM", "ssim"),
         ("Ringing score", "ringing"), ("Time (ms)", "median_ms")],
    )
    ringing_table = markdown_table(
        ringing,
        [("Filter", "filter"), ("Oscillations", "oscillations"), ("Overshoot", "overshoot")],
    )
    notch_table = markdown_table(
        notch_rows, [("Method", "method"), ("PSNR (dB)", "psnr_db")],
    )
    homo_table = markdown_table(
        homo_rows,
        [("Method", "method"), ("PSNR (dB)", "psnr_db"),
         ("PSNR matched (dB)", "psnr_matched_db")],
    )
    write_tables(
        RESULTS,
        [
            (f"Low-pass filters at cutoff {GALLERY_CUTOFF:g}", lowpass_table),
            (f"Ringing on a step edge at cutoff {RINGING_CUTOFF:g}", ringing_table),
            ("Periodic interference removed", notch_table),
            ("Uneven illumination corrected", homo_table),
            ("Four scenes down the rows", gallery_table),
        ],
    )
    print("\n" + lowpass_table + "\n\n" + ringing_table + "\n\n" + notch_table
          + "\n\n" + homo_table)

    noisy = next(r for r in notch_rows if r["method"].startswith("periodic")
                 or r["method"].startswith("Noisy"))
    median = next(r for r in notch_rows if "Median" in r["method"])
    blind = next(r for r in notch_rows if "blind" in r["method"])
    oracle = next(r for r in notch_rows if "oracle" in r["method"])
    homo = next(r for r in homo_rows if r["method"] == "Homomorphic")
    uneven = next(r for r in homo_rows if r["method"] == "Uneven input")
    best_lp = max(lowpass, key=lambda r: r["psnr_db"])

    print("\n--- HEADLINE NUMBERS ---")
    print(f"notch: noisy {noisy['psnr_db']:.2f} dB -> blind {blind['psnr_db']:.2f} dB "
          f"(+{blind['psnr_db'] - noisy['psnr_db']:.2f}); the spatial median only "
          f"reaches {median['psnr_db']:.2f}")
    print(f"  blind vs oracle  : {blind['psnr_db'] - oracle['psnr_db']:+.3f} dB, "
          f"peak localisation error {notch_extra['peak_localisation_error_px']:.1f} px")
    print(f"ringing on a step : "
          + ", ".join(f"{r['filter']} {r['overshoot']:.4f}" for r in ringing))
    print(f"best low-pass     : {best_lp['filter']} @ {best_lp['psnr_db']:.2f} dB")
    print(f"homomorphic       : {homo['psnr_db']:.2f} dB raw -> "
          f"{homo['psnr_matched_db']:.2f} matched "
          f"(+{homo['psnr_matched_db'] - homo['psnr_db']:.2f} from brightness alone), "
          f"against a {uneven['psnr_db']:.2f} dB input")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
