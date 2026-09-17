"""Run the deblurring comparison and write results + figures.

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

import numpy as np  # noqa: E402

from shared import figures  # noqa: E402
from shared.io import ensure_rgb, to_gray  # noqa: E402
from shared.metrics import psnr, ssim  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import deblurring as db  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"

#: The noise the headline comparison runs at. Without noise even the naive
#: inverse filter works, so a noiseless comparison has nothing to say.
GALLERY_NOISE = 3.0

#: Angles the blind estimator is scored at. Deliberately includes values that
#: are not multiples of 45, because the previous estimator was exact at those
#: and nowhere else.
BLIND_ANGLES = (0.0, 15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 135.0, 160.0)


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    print(f"Deblurring {len(db.IMAGES)} photographs with {len(db.METHODS)} methods ...")
    rows, blurred_stats = db.evaluate_methods(kind="motion", noise_sigma=GALLERY_NOISE,
                                              images=db.IMAGES, runs=args.runs)
    defocus_rows, defocus_stats = db.evaluate_methods(kind="defocus",
                                                     noise_sigma=GALLERY_NOISE,
                                                     images=db.IMAGES, runs=1)

    print("Sweeping Richardson-Lucy iterations ...")
    rl_rows = db.sweep_rl_iterations(images=db.IMAGES)
    print("Sweeping Wiener's noise-to-signal ratio ...")
    nsr_rows = db.sweep_nsr(images=db.IMAGES)
    print("Sweeping the noise ...")
    noise_rows = db.sweep_noise(images=db.IMAGES)
    print("Estimating the blur direction blind ...")
    blind_rows = db.evaluate_blind_angle(images=db.IMAGES, angles=BLIND_ANGLES)

    names = list(db.METHODS)

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    # Rows cross the axis the pool was built on -- detail density -- because
    # deblurring is judged on how much fine structure it puts back, and an image
    # with none cannot separate the methods. A subject-uniqueness rule is
    # applied on top so the four rows are four different pictures.
    DETAIL_BANDS = [("smooth", 0, 200), ("moderate", 200, 290),
                    ("detailed", 290, 400), ("fine", 400, 10000)]
    candidates: dict[str, list[dict]] = {}
    for i, name in enumerate(db.IMAGES):
        clean, blurred, psf = db.make_blurred(name, "motion", GALLERY_NOISE, seed=i)
        detail = db.detail_of(clean)
        band = next(b for b, a, z in DETAIL_BANDS if a <= detail < z)

        outs = [db.METHODS[m](blurred, psf) for m in names]
        scores = [psnr(o, clean) for o in outs]
        base = psnr(to_gray(blurred), clean)
        print(f"scene candidate {name:20s} detail {detail:6.1f}  [{band:8s}]  "
              f"blurred {base:5.2f} dB -> best {max(scores):5.2f} dB "
              f"(gain {max(scores) - base:+.2f})")
        candidates.setdefault(band, []).append({
            "label": f"{name.replace('_', ' ')}\ndetail {detail:.0f}",
            "subject": name,
            "detail": detail,
            "images": [ensure_rgb(clean), ensure_rgb(to_gray(blurred))]
            + [ensure_rgb(o) for o in outs],
            "notes": ["original", f"{base:.1f} dB"] + [f"{s:.1f} dB" for s in scores],
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
        ["original", "blurred input"] + names,
        [(r["label"], r["images"]) for r in chosen],
        IMAGES / "compare_deblurring.png",
        cell_notes=[r["notes"] for r in chosen],
        suptitle=(
            f"A known 15 px motion blur at 30 degrees plus sigma {GALLERY_NOISE:g} "
            "noise, then deblurred. Four of the six methods are handed the TRUE "
            "kernel; the last two are not. Cells are PSNR against the original."
        ),
    )
    gallery_table = markdown_table(
        [dict([("Sr", i), ("Scene", r["label"].replace("\n", " · ")),
               ("Blurred", r["notes"][1])]
              + [(m, f"{s:.1f} dB") for m, s in zip(names, r["scores"])])
         for i, r in enumerate(chosen, start=1)],
        [("Sr", "Sr"), ("Scene", "Scene"), ("Blurred", "Blurred")]
        + [(m, m) for m in names],
    )
    print(f"\nfront-on comparison: {len(chosen)} scenes x {len(names)} methods")
    print("\n--- deblurring ---\n" + gallery_table)

    # ------------------------------------------------------------------ #
    # the signature figure: the RL iteration curve, optimum then divergence
    # ------------------------------------------------------------------ #
    # Richardson-Lucy is a maximum-likelihood iteration with no regulariser, so
    # it converges toward a solution that explains the NOISE as well as the
    # signal. The curve rises, peaks and falls, and the peak is not where SSIM
    # puts it -- which is the whole problem, because at run time you have
    # neither metric.
    control = next(r for r in rows if r["method"].startswith("Do nothing"))
    figures.lines(
        [r["iterations"] for r in rl_rows],
        {
            "PSNR (dB)": [r["psnr_db"] for r in rl_rows],
            "SSIM x 40": [r["ssim"] * 40 for r in rl_rows],
            "do nothing (PSNR)": [control["psnr_db"]] * len(rl_rows),
        },
        IMAGES / "rl_iterations.png",
        xlabel="Richardson-Lucy iterations",
        ylabel="score",
        title=("Richardson-Lucy: an optimum, then divergence — and PSNR and SSIM "
               "disagree about where it is"),
        logx=True,
        dashed={"do nothing (PSNR)"},
    )

    figures.lines(
        [r["nsr"] for r in nsr_rows],
        {"PSNR (dB)": [r["psnr_db"] for r in nsr_rows],
         "SSIM x 40": [r["ssim"] * 40 for r in nsr_rows],
         "do nothing (PSNR)": [control["psnr_db"]] * len(nsr_rows)},
        IMAGES / "nsr_sweep.png",
        xlabel="Wiener noise-to-signal ratio",
        ylabel="score",
        title="Wiener's one parameter, and the two metrics' two different answers",
        logx=True,
        dashed={"do nothing (PSNR)"},
    )

    figures.lines(
        [r["noise_sigma"] for r in noise_rows],
        {m: [r[m] for r in noise_rows] for m in noise_rows[0]
         if m not in ("noise_sigma", "blurred_input")},
        IMAGES / "noise_sweep.png",
        xlabel="noise sigma added to the blurred image",
        ylabel="PSNR against the original (dB)",
        title="Deblurring is a noise-amplification problem — every method dies of it",
    )

    figures.lines(
        [r["true_angle_deg"] for r in blind_rows],
        {"absolute error (degrees)": [r["mean_abs_error_deg"] for r in blind_rows]},
        IMAGES / "blind_angle.png",
        xlabel="true motion-blur angle (degrees)",
        ylabel="mean absolute error (degrees)",
        title="Blind angle estimation from the log power spectrum",
    )

    figures.comparison_matrix(
        rows,
        [("PSNR (dB)", "psnr_db", True), ("SSIM", "ssim", True),
         ("Time (ms)", "median_ms", False)],
        IMAGES / "method_matrix.png",
        title=f"Methods x metrics on a motion blur at noise sigma {GALLERY_NOISE:g}",
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS,
        "20_deblurring",
        {
            "images": list(db.IMAGES),
            "noise_sigma": GALLERY_NOISE,
            "wiener_nsr": db.WIENER_NSR,
            "knows_kernel": sorted(db.KNOWS_KERNEL),
            "motion": rows,
            "motion_blurred_input": blurred_stats,
            "defocus": defocus_rows,
            "defocus_blurred_input": defocus_stats,
            "rl_iterations": rl_rows,
            "nsr_sweep": nsr_rows,
            "noise_sweep": noise_rows,
            "blind_angle": blind_rows,
        },
    )

    method_table = markdown_table(
        rows,
        [("Method", "method"), ("Knows the kernel", "knows_kernel"),
         ("PSNR (dB)", "psnr_db"), ("SSIM", "ssim"), ("Time (ms)", "median_ms")],
    )
    defocus_table = markdown_table(
        defocus_rows,
        [("Method", "method"), ("PSNR (dB)", "psnr_db"), ("SSIM", "ssim")],
    )
    rl_table = markdown_table(
        rl_rows,
        [("Iterations", "iterations"), ("PSNR (dB)", "psnr_db"), ("SSIM", "ssim")],
    )
    nsr_table = markdown_table(
        nsr_rows, [("NSR", "nsr"), ("PSNR (dB)", "psnr_db"), ("SSIM", "ssim")],
    )
    blind_table = markdown_table(
        blind_rows,
        [("True angle", "true_angle_deg"), ("Mean abs error", "mean_abs_error_deg")],
    )
    write_tables(
        RESULTS,
        [
            (f"Motion blur, noise sigma {GALLERY_NOISE:g}", method_table),
            ("The same methods on a defocus blur", defocus_table),
            ("Richardson-Lucy iterations swept", rl_table),
            ("Wiener noise-to-signal ratio swept", nsr_table),
            ("Blind angle estimation", blind_table),
            ("Four scenes down the rows", gallery_table),
        ],
    )
    print("\n" + method_table + "\n\n" + rl_table + "\n\n" + blind_table)

    best = max((r for r in rows if not r["method"].startswith("Do nothing")),
               key=lambda r: r["psnr_db"])
    inverse = next(r for r in rows if r["method"].startswith("Inverse"))
    rl_best_psnr = max(rl_rows, key=lambda r: r["psnr_db"])
    rl_best_ssim = max(rl_rows, key=lambda r: r["ssim"])
    nsr_best_psnr = max(nsr_rows, key=lambda r: r["psnr_db"])
    nsr_best_ssim = max(nsr_rows, key=lambda r: r["ssim"])

    print("\n--- HEADLINE NUMBERS ---")
    print(f"blurred input    : {blurred_stats['psnr']:.2f} dB, SSIM {blurred_stats['ssim']:.4f}")
    print(f"control          : {control['psnr_db']:.2f} dB")
    print(f"best method      : {best['method']} @ {best['psnr_db']:.2f} dB "
          f"({best['psnr_db'] - control['psnr_db']:+.2f} over doing nothing)")
    print(f"inverse filter   : {inverse['psnr_db']:.2f} dB "
          f"({control['psnr_db'] - inverse['psnr_db']:.1f} dB BELOW doing nothing)")
    print(f"RL optimum       : PSNR at {rl_best_psnr['iterations']} iters, "
          f"SSIM at {rl_best_ssim['iterations']} iters — they disagree by "
          f"{rl_best_psnr['iterations'] / max(rl_best_ssim['iterations'], 1):.0f}x")
    print(f"RL at 200 iters  : {rl_rows[-1]['psnr_db']:.2f} dB "
          f"({rl_best_psnr['psnr_db'] - rl_rows[-1]['psnr_db']:.2f} dB below its own peak)")
    print(f"Wiener optimum   : PSNR at nsr {nsr_best_psnr['nsr']}, "
          f"SSIM at nsr {nsr_best_ssim['nsr']}")
    print(f"blind angle      : mean {np.mean([r['mean_abs_error_deg'] for r in blind_rows]):.1f} deg error, "
          f"worst {max(r['mean_abs_error_deg'] for r in blind_rows):.1f} deg")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
