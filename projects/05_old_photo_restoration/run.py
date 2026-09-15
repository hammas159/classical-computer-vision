"""Run the old-photo restoration experiment and write results + figures.

    python run.py [--thickness 3]
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
        det_panels.append((f"{name}\nIoU {row['mask_iou']:.3f}", fn(damaged)))
    figures.grid(
        det_panels,
        IMAGES / "detection.png",
        ncols=4,
        suptitle="Finding the damage — the step a restoration demo usually skips",
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
        suptitle="At 25 px the surrounding pixels no longer constrain the answer — every method invents",
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

    ys, xs = np.nonzero(mask)
    cy, cx = int(ys[len(ys) // 2]), int(xs[len(xs) // 2])
    size = 12
    y0 = max(0, min(clean.shape[0] - size, cy - size // 2))
    x0 = max(0, min(clean.shape[1] - size, cx - size // 2))
    crop = lambda a: to_gray(a)[y0 : y0 + size, x0 : x0 + size]  # noqa: E731
    figures.value_matrix(
        [
            ("Original", crop(clean)),
            ("Damaged (255 = scratch)", crop(damaged)),
            ("Telea", crop(rs.inpaint_telea(damaged, mask))),
            ("Harmonic diffusion", crop(rs.inpaint_diffusion(damaged, mask))),
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

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS,
        "05_old_photo_restoration",
        {
            "thickness_px": args.thickness,
            "images": list(images),
            "damaged_input": damaged_stats,
            "methods": method_rows,
            "thickness_sweep": sweep_rows,
            "detectors": detector_rows,
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
            ("Mask Dice", "mask_dice"),
            ("PSNR on damage (dB)", "damage_psnr_db"),
        ],
    )
    write_tables(
        RESULTS,
        [
            (f"Inpainting at {args.thickness} px, true mask ({len(images)} images)", method_table),
            ("PSNR on damaged pixels vs scratch width", sweep_table),
            (f"Damage detection, and restoring with the detected mask", det_table),
        ],
    )
    print("\n" + method_table + "\n\n" + sweep_table + "\n\n" + det_table)

    best = max(method_rows, key=lambda r: r["damage_psnr_db"])
    fastest = min(method_rows, key=lambda r: r["median_ms"])
    thin, wide = sweep_rows[0], sweep_rows[-1]
    best_det = max(detector_rows, key=lambda r: r["mask_iou"])
    true_mask_psnr = best["damage_psnr_db"]

    print("\n--- HEADLINE NUMBERS ---")
    print(f"damaged input    : {damaged_stats}")
    print(f"best method      : {best['method']} @ {best['damage_psnr_db']} dB on damaged pixels")
    print(f"fastest          : {fastest['method']} @ {fastest['median_ms']} ms")
    print(f"whole-image PSNR spread: "
          f"{min(r['psnr_db'] for r in method_rows)} to {max(r['psnr_db'] for r in method_rows)} dB "
          f"(damage-only spread: {min(r['damage_psnr_db'] for r in method_rows)} to "
          f"{max(r['damage_psnr_db'] for r in method_rows)} dB)")
    print(f"thin damage ({thin['thickness_px']} px): " +
          ", ".join(f"{n} {thin[n]}" for n in rs.METHODS))
    print(f"wide damage ({wide['thickness_px']} px): " +
          ", ".join(f"{n} {wide[n]}" for n in rs.METHODS))
    print(f"best detector    : {best_det['detector']} mask IoU {best_det['mask_iou']}, "
          f"restoring to {best_det['damage_psnr_db']} dB vs {true_mask_psnr} dB with the true mask")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
