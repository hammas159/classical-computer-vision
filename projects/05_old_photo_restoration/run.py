"""Run the old-photo restoration experiments and write results + figures.

    python run.py [--thickness 3]

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

from shared import figures, io, synth  # noqa: E402
from shared.io import to_gray  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import restoration as rs  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--thickness", type=int, default=3, help="scratch width in px")
    ap.add_argument("--images", type=int, default=len(rs.IMAGES))
    args = ap.parse_args()

    images = rs.IMAGES[: args.images]
    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    print(f"Scoring {len(rs.METHODS)} inpainting methods on {len(images)} images ...")
    method_rows, damaged_stats = rs.evaluate_methods(thickness=args.thickness, images=images)

    print(f"Sweeping scratch width over {len(rs.WIDTH_LEVELS)} levels ...")
    sweep_rows = rs.sweep_thickness(images=images)

    print(f"Scoring {len(rs.DETECTORS)} damage detectors ...")
    detector_rows = rs.evaluate_detectors(thickness=args.thickness, images=images)

    print("Sweeping the median-residual window ...")
    window_rows = rs.sweep_detector_window(images=images, thickness=args.thickness)

    print(f"Scoring {len(rs.FADE_METHODS)} fade corrections ...")
    fade_rows, faded_stats = rs.evaluate_fade()

    print("Scoring the two halves together ...")
    pipeline_rows = rs.evaluate_pipeline(thickness=args.thickness)

    # ------------------------------------------------------------------ #
    # figures
    # ------------------------------------------------------------------ #
    clean = io.sample("astronaut")
    damaged, mask = synth.add_scratches(clean, thickness=args.thickness, seed=0)

    panels = [("Original (truth)", clean), ("Damaged", damaged), ("True damage mask", mask)]
    for name, fn in rs.METHODS.items():
        row = next(r for r in method_rows if r["method"] == name)
        panels.append((f"{name}\n{row['damage_psnr_db']:.1f} dB on damage", fn(damaged, mask)))
    figures.grid(
        panels,
        IMAGES / "methods.png",
        ncols=4,
        suptitle=f"Four inpainting methods, {args.thickness} px scratches, true mask supplied",
    )

    # the sweep is the experiment that separates them
    figures.lines(
        [r["thickness_px"] for r in sweep_rows],
        {name: [r[name] for r in sweep_rows] for name in rs.METHODS},
        IMAGES / "thickness_sweep.png",
        xlabel="scratch width (px)",
        ylabel="PSNR over the damaged pixels only (dB)",
        title="All four are equal on thin scratches; width is what separates them",
    )

    # detection: the mask a real pipeline has to find for itself
    det_panels = [("Damaged", damaged), ("True mask", mask)]
    for name, fn in rs.DETECTORS.items():
        row = next(r for r in detector_rows if r["detector"] == name)
        det_panels.append(
            (f"{name}\nIoU {row['mask_iou']:.3f}  R {row['recall']:.2f}", fn(damaged))
        )
    figures.grid(
        det_panels,
        IMAGES / "detection.png",
        ncols=5,
        suptitle="Finding the damage — the step a restoration demo usually skips",
    )

    # the detector's blind spot, as a curve
    figures.lines(
        [r["window_px"] for r in window_rows],
        {
            "recall": [r["recall"] for r in window_rows],
            "precision": [r["precision"] for r in window_rows],
            "IoU": [r["mask_iou"] for r in window_rows],
        },
        IMAGES / "window_sweep.png",
        xlabel="median window (px)",
        ylabel="score",
        title=(
            f"A median filter rejects a minority of outliers: below ~4x the "
            f"{args.thickness} px damage it goes blind"
        ),
    )

    # wide damage, where inpainting stops being interpolation and starts inventing
    wide_damaged, wide_mask = synth.add_scratches(clean, thickness=25, blotches=10, seed=0)
    wide_panels = [("Original", clean), ("Damaged, 25 px", wide_damaged)]
    for name, fn in rs.METHODS.items():
        wide_panels.append((name, fn(wide_damaged, wide_mask)))
    figures.grid(
        wide_panels,
        IMAGES / "wide_damage.png",
        ncols=3,
        suptitle=(
            "At 25 px the surrounding pixels no longer constrain the answer — every method invents"
        ),
    )

    # the other half: fading
    faded = synth.fade_photo(clean, seed=0)
    fade_panels = [("Original (truth)", clean), ("Faded print", faded)]
    for name, fn in rs.FADE_METHODS.items():
        if name == "None (control)":
            continue
        row = next(r for r in fade_rows if r["method"] == name)
        fade_panels.append((f"{name}\n{row['psnr_db']:.1f} dB, chroma {row['chroma']:.0f}", fn(faded)))
    figures.grid(
        fade_panels,
        IMAGES / "fade.png",
        ncols=4,
        suptitle=(
            f"Fade correction. The original's mean chroma is "
            f"{faded_stats['original_chroma']:.1f} — the number to land on, not exceed"
        ),
    )

    # four different photographs, end to end. A single before/after lets the
    # reader assume that one image was representative; these four are the same
    # four the tables average over, so the pictures and the numbers describe the
    # same thing.
    gallery_names = rs.COLOUR_IMAGES
    g_truth, g_input, g_mask, g_inpainted, g_restored = [], [], [], [], []
    for i, name in enumerate(gallery_names):
        src = io.sample(name)
        aged, _ = rs.add_damage_and_fade(src, thickness=args.thickness, seed=i)
        det, inp, res = rs.restore(aged)
        g_truth.append(src)
        g_input.append(aged)
        g_mask.append(det)
        g_inpainted.append(inp)
        g_restored.append(res)
    figures.gallery(
        list(gallery_names),
        [
            ("original", g_truth),
            ("faded + damaged", g_input),
            ("damage found", g_mask),
            ("inpainted", g_inpainted),
            ("restored", g_restored),
        ],
        IMAGES / "samples.png",
        suptitle="Four photographs, the same pipeline, nothing hand-picked",
    )

    # both degradations, the end-to-end result a user actually sees
    both, both_mask = rs.add_damage_and_fade(clean, thickness=args.thickness, seed=0)
    detected, inpainted, restored = rs.restore(both)
    figures.grid(
        [
            ("Original (truth)", clean),
            ("Faded + damaged", both),
            ("Detected damage", detected),
            ("After inpainting", inpainted),
            ("After fade correction", restored),
            ("Ceiling: true mask", rs.correct_full(rs.inpaint_telea(both, both_mask))),
        ],
        IMAGES / "pipeline.png",
        ncols=3,
        suptitle="End to end, with the detected mask — and what the true mask would have bought",
    )

    # ------------------------------------------------------------------ #
    # distributions and matrices
    # ------------------------------------------------------------------ #
    figures.histogram(
        {
            "original (damaged region)": to_gray(clean)[mask > 0],
            "damaged (damaged region)": to_gray(damaged)[mask > 0],
            "Telea restored": to_gray(rs.inpaint_telea(damaged, mask))[mask > 0],
            "Harmonic diffusion": to_gray(rs.inpaint_diffusion(damaged, mask))[mask > 0],
        },
        IMAGES / "histogram_damage.png",
        bins=128,
        title="Intensity distribution inside the damaged pixels only",
    )

    figures.histogram(
        {
            "original": to_gray(clean).ravel(),
            "faded": to_gray(faded).ravel(),
            "per-channel stretch": to_gray(rs.correct_channel_stretch(faded)).ravel(),
            "stretch + saturate": to_gray(rs.correct_full(faded)).ravel(),
        },
        IMAGES / "histogram_fade.png",
        bins=128,
        title="Fading compresses the tonal range and lifts the black point; the stretch undoes both",
    )

    ys, xs = np.nonzero(mask)
    cy, cx = int(ys[len(ys) // 2]), int(xs[len(xs) // 2])
    size = 12
    y0 = max(0, min(clean.shape[0] - size, cy - size // 2))
    x0 = max(0, min(clean.shape[1] - size, cx - size // 2))
    crop = lambda a: to_gray(a)[y0 : y0 + size, x0 : x0 + size]  # noqa: E731
    figures.value_matrix(
        [
            ("Original", crop(clean)),
            ("Damaged", crop(damaged)),
            ("Telea", crop(rs.inpaint_telea(damaged, mask))),
            ("Masked mean", crop(rs.inpaint_masked_mean(damaged, mask))),
        ],
        IMAGES / "pixel_matrix.png",
        title="A patch straddling a scratch, read as numbers",
    )

    figures.comparison_matrix(
        method_rows,
        [
            ("PSNR whole (dB)", "psnr_db", True),
            ("PSNR on damage (dB)", "damage_psnr_db", True),
            ("SSIM", "ssim", True),
            ("Time (ms)", "median_ms", False),
        ],
        IMAGES / "method_matrix.png",
        title="Methods x metrics — whole-image PSNR barely moves, damage-only PSNR does",
    )

    figures.comparison_matrix(
        detector_rows,
        [
            ("Mask IoU", "mask_iou", True),
            ("Precision", "precision", True),
            ("Recall", "recall", True),
            ("PSNR on damage (dB)", "damage_psnr_db", True),
        ],
        IMAGES / "detector_matrix.png",
        row_key="detector",
        title="Detectors — the best IoU is not the best restoration",
    )

    figures.comparison_matrix(
        fade_rows,
        [
            ("PSNR (dB)", "psnr_db", True),
            ("Cast error (deg)", "cast_error_deg", False),
            ("Chroma", "chroma", True),
            ("RMS contrast", "rms_contrast", True),
        ],
        IMAGES / "fade_matrix.png",
        title="Fade corrections — no method wins every column",
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS,
        "05_old_photo_restoration",
        {
            "thickness_px": args.thickness,
            "images": list(images),
            "colour_images": list(rs.COLOUR_IMAGES),
            "damaged_input": damaged_stats,
            "faded_input": faded_stats,
            "methods": method_rows,
            "thickness_sweep": sweep_rows,
            "detectors": detector_rows,
            "detector_window_sweep": window_rows,
            "fade_methods": fade_rows,
            "pipeline": pipeline_rows,
            "saturation_factor": rs.SATURATION_FACTOR,
            "median_residual_ksize": rs.MEDIAN_RESIDUAL_KSIZE,
        },
    )

    method_table = markdown_table(
        method_rows,
        [
            ("Method", "method"),
            ("PSNR whole (dB)", "psnr_db"),
            ("PSNR on damage (dB)", "damage_psnr_db"),
            ("SSIM", "ssim"),
            ("Time (ms)", "median_ms"),
        ],
    )
    sweep_table = markdown_table(
        sweep_rows,
        [("Width (px)", "thickness_px"), ("Damage fraction", "damage_fraction")]
        + [(n, n) for n in rs.METHODS],
    )
    det_table = markdown_table(
        detector_rows,
        [
            ("Detector", "detector"),
            ("Mask IoU", "mask_iou"),
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("Flagged", "flagged_fraction"),
            ("PSNR on damage (dB)", "damage_psnr_db"),
            ("PSNR whole (dB)", "whole_psnr_db"),
        ],
    )
    window_table = markdown_table(
        window_rows,
        [
            ("Median window (px)", "window_px"),
            ("Damage (px)", "damage_px"),
            ("Mask IoU", "mask_iou"),
            ("Recall", "recall"),
            ("Precision", "precision"),
        ],
    )
    fade_table = markdown_table(
        fade_rows,
        [
            ("Method", "method"),
            ("PSNR (dB)", "psnr_db"),
            ("SSIM", "ssim"),
            ("Cast error (deg)", "cast_error_deg"),
            ("Chroma", "chroma"),
            ("RMS contrast", "rms_contrast"),
            ("Time (ms)", "median_ms"),
        ],
    )
    pipeline_table = markdown_table(
        pipeline_rows,
        [
            ("Stage", "stage"),
            ("PSNR whole (dB)", "psnr_db"),
            ("PSNR on damage (dB)", "damage_psnr_db"),
            ("SSIM", "ssim"),
            ("Cast error (deg)", "cast_error_deg"),
        ],
    )
    write_tables(
        RESULTS,
        [
            (f"Inpainting at {args.thickness} px, true mask ({len(images)} images)", method_table),
            ("PSNR on damaged pixels vs scratch width", sweep_table),
            ("Damage detection, and restoring with the detected mask", det_table),
            (f"Median-residual detector vs its window size ({args.thickness} px damage)", window_table),
            (f"Fade correction ({len(rs.COLOUR_IMAGES)} colour images)", fade_table),
            ("Both degradations together, and the order they are undone in", pipeline_table),
        ],
    )
    print(
        "\n"
        + "\n\n".join(
            [method_table, sweep_table, det_table, window_table, fade_table, pipeline_table]
        )
    )

    best = max(method_rows, key=lambda r: r["damage_psnr_db"])
    fastest = min(method_rows, key=lambda r: r["median_ms"])
    simple = next(r for r in method_rows if r["method"] == "Iterative masked mean")
    thin, wide = sweep_rows[0], sweep_rows[-1]
    best_det = max(detector_rows, key=lambda r: r["mask_iou"])
    best_restore = max(detector_rows, key=lambda r: r["damage_psnr_db"])
    best_fade = max(fade_rows, key=lambda r: r["psnr_db"])
    recommended = next(r for r in fade_rows if r["method"] == "Stretch + saturate")
    control = next(r for r in fade_rows if r["method"] == "None (control)")
    grayworld = next(r for r in fade_rows if r["method"] == "Gray-world balance")
    narrow = next(r for r in window_rows if r["window_px"] == 5)
    chosen = next(r for r in window_rows if r["window_px"] == rs.MEDIAN_RESIDUAL_KSIZE)

    print("\n--- HEADLINE NUMBERS ---")
    print(f"damaged input    : {damaged_stats}")
    print(f"faded input      : {faded_stats}")
    print(f"best method      : {best['method']} @ {best['damage_psnr_db']} dB on damaged pixels")
    print(
        f"simple baseline  : {simple['method']} @ {simple['damage_psnr_db']} dB "
        f"({simple['damage_psnr_db'] / best['damage_psnr_db']:.1%} of the best), "
        f"{simple['median_ms']} ms"
    )
    print(f"fastest          : {fastest['method']} @ {fastest['median_ms']} ms")
    print(
        f"whole-image PSNR spread: "
        f"{min(r['psnr_db'] for r in method_rows)} to {max(r['psnr_db'] for r in method_rows)} dB "
        f"(damage-only spread: {min(r['damage_psnr_db'] for r in method_rows)} to "
        f"{max(r['damage_psnr_db'] for r in method_rows)} dB)"
    )
    print(
        f"thin damage ({thin['thickness_px']} px): "
        + ", ".join(f"{n} {thin[n]}" for n in rs.METHODS)
    )
    print(
        f"wide damage ({wide['thickness_px']} px): "
        + ", ".join(f"{n} {wide[n]}" for n in rs.METHODS)
    )
    print(
        f"best detector by IoU : {best_det['detector']} {best_det['mask_iou']}, "
        f"restoring to {best_det['damage_psnr_db']} dB"
    )
    print(
        f"best detector by dB  : {best_restore['detector']} {best_restore['mask_iou']} IoU, "
        f"{best_restore['damage_psnr_db']} dB vs {best['damage_psnr_db']} dB with the true mask "
        f"(a {best['damage_psnr_db'] - best_restore['damage_psnr_db']:.1f} dB detection tax)"
    )
    print(
        f"detector window  : {narrow['window_px']} px finds {narrow['recall']:.1%} of the damage, "
        f"{chosen['window_px']} px finds {chosen['recall']:.1%}"
    )
    print(
        f"best fade by PSNR: {best_fade['method']} @ {best_fade['psnr_db']} dB "
        f"(control {control['psnr_db']} dB)"
    )
    print(
        f"recommended fade : {recommended['method']} — chroma {recommended['chroma']} against the "
        f"original's {faded_stats['original_chroma']}, cast error "
        f"{recommended['cast_error_deg']} deg from {faded_stats['cast_err']}"
    )
    print(
        f"gray-world       : cast error {grayworld['cast_error_deg']} deg — WORSE than the "
        f"{control['cast_error_deg']} deg it started with, while looking neutral"
    )
    print("pipeline order   : " + ", ".join(f"{r['stage']} {r['psnr_db']} dB" for r in pipeline_rows))
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
