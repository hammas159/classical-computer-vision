"""Run the document-scanner experiment and write results + figures.

    python run.py [--scenes 30]

Everything this repo publishes as a number comes out of this script. Nothing is
hand-written into the README.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Path bootstrap must come before any repo import, and cannot itself live in the
# repo it is bootstrapping. Three lines, once per entry point.
PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from shared import figures, synth  # noqa: E402
from shared.io import to_gray  # noqa: E402
from shared.metrics import iou  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import document_scanner as ds  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"


def draw_corners(img, corners, colour=(0, 200, 0), truth=None):
    """Overlay a detected quad (green) and optionally the true quad (red)."""
    out = img.copy()
    if truth is not None:
        cv2.polylines(out, [np.int32(truth)], True, (220, 40, 40), 3)
    if corners is not None:
        cv2.polylines(out, [np.int32(corners)], True, colour, 3)
        for x, y in np.int32(corners):
            cv2.circle(out, (int(x), int(y)), 7, colour, -1)
    return out


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=30)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating {len(ds.DETECTORS)} detectors over {args.scenes} scenes ...")
    detector_rows = ds.evaluate_detectors(n_scenes=args.scenes)

    print(f"Evaluating {len(ds.BINARISERS)} binarisers over {args.scenes} scenes ...")
    binariser_rows = ds.evaluate_binarisers(n_scenes=args.scenes)

    sweep_scenes = max(4, args.scenes // 3)
    print(f"Sweeping illumination over {len(ds.ILLUM_LEVELS)} levels x {sweep_scenes} scenes ...")
    sweep_rows = ds.sweep_illumination(n_scenes=sweep_scenes)

    print(f"Comparing aspect-ratio recovery over {args.scenes} scenes ...")
    aspect_rows = ds.evaluate_aspect_recovery(n_scenes=args.scenes)

    # ------------------------------------------------------------------ #
    # figures
    # ------------------------------------------------------------------ #
    photo, page, truth = synth.document_scene(seed=0)

    panels = [("Input photo + true page (red)", draw_corners(photo, None, truth=truth))]
    for name, fn in ds.DETECTORS.items():
        corners = fn(photo)
        err = ds.corner_error(corners, truth) if corners is not None else None
        title = f"{name}\n{'FAILED' if err is None else f'{err:.1f} px error'}"
        panels.append((title, draw_corners(photo, corners, truth=truth)))
    figures.grid(
        panels,
        IMAGES / "detectors.png",
        ncols=4,
        suptitle="Page-boundary detection: green = detected, red = ground truth (scene 0)",
    )

    page_h, page_w = page.shape[:2]
    rect = ds.rectify(photo, truth, (page_w, page_h))
    gray = to_gray(rect)
    truth_text = ds.text_mask(to_gray(page))
    bin_panels = [("Rectified page (uneven light)", gray), ("True text mask", truth_text)]
    for name, fn in ds.BINARISERS.items():
        out = fn(gray)
        bin_panels.append((f"{name}\nIoU {iou(255 - out, truth_text):.3f}", out))
    figures.grid(
        bin_panels,
        IMAGES / "binarisers.png",
        ncols=3,
        suptitle="Binarisation of the rectified page, scored against the true text mask",
    )

    best_det = min(
        (r for r in detector_rows if r["corner_error_px_mean"] is not None),
        key=lambda r: r["corner_error_px_mean"],
    )
    corners, rectified, binary = ds.scan(photo, best_det["method"], "Sauvola")
    figures.grid(
        [
            ("1. Input photo", photo),
            ("2. Detected page", draw_corners(photo, corners)),
            ("3. Rectified", rectified),
            ("4. Binarised", binary),
        ],
        IMAGES / "pipeline.png",
        ncols=4,
        suptitle=f"End-to-end scan: {best_det['method']} -> homography -> Sauvola",
    )

    # the illumination sweep: where each binariser stops working, against the
    # oracle that shows whether a global threshold was available at all
    series = {name: [r[name] for r in sweep_rows] for name in ds.BINARISERS}
    series[ds.ORACLE_NAME] = [r[ds.ORACLE_NAME] for r in sweep_rows]
    figures.lines(
        [r["page_ratio"] for r in sweep_rows],
        series,
        IMAGES / "illumination_sweep.png",
        xlabel="illumination ratio across the page (1.0 = flat light)",
        ylabel="text IoU vs ground truth",
        title=f"Binarisation under worsening light ({sweep_scenes} scenes per level)",
        invert_x=True,
        dashed={ds.ORACLE_NAME},
        vlines={f"no global cut exists below {ds.INK_REFLECTANCE:.2f}": ds.INK_REFLECTANCE},
    )

    # a visual of the hardest condition, so the numbers have a picture attached
    hard_level = ds.ILLUM_LEVELS[-1]
    hard_photo, hard_page, hard_truth = synth.document_scene(seed=0, illum_min=hard_level)
    hp_h, hp_w = hard_page.shape[:2]
    hard_gray = to_gray(ds.rectify(hard_photo, hard_truth, (hp_w, hp_h)))
    hard_truth_text = ds.text_mask(to_gray(hard_page))
    hard_ratio = ds.page_illumination_ratio(
        hard_truth, (hard_photo.shape[1], hard_photo.shape[0]), hard_level
    )
    hard_panels = [("Rectified page, deep shadow", hard_gray)]
    for name, fn in ds.BINARISERS.items():
        out = fn(hard_gray)
        hard_panels.append((f"{name}\nIoU {iou(255 - out, hard_truth_text):.3f}", out))
    oracle_img, oracle_t = ds.binarise_best_global(hard_gray, hard_truth_text)
    hard_panels.append(
        (
            f"{ds.ORACLE_NAME}, t={oracle_t}\nIoU {iou(255 - oracle_img, hard_truth_text):.3f}",
            oracle_img,
        )
    )
    figures.grid(
        hard_panels,
        IMAGES / "binarisers_hard.png",
        ncols=3,
        suptitle=(
            f"Page illumination ratio {hard_ratio:.2f} — Otsu fails, "
            "but the oracle shows a global threshold was still available"
        ),
    )

    ok = [r for r in detector_rows if r["corner_error_px_mean"] is not None]
    figures.metric_bars(
        [r["method"] for r in ok],
        [r["corner_error_px_mean"] for r in ok],
        IMAGES / "corner_error.png",
        ylabel="mean corner error (px)",
        title=f"Corner error over {args.scenes} scenes (lower is better)",
        highlight_best="min",
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS,
        "01_document_scanner",
        {
            "n_scenes": args.scenes,
            "detectors": detector_rows,
            "binarisers": binariser_rows,
            "illumination_sweep": {
                "n_scenes_per_level": sweep_scenes,
                "ink_reflectance": round(ds.INK_REFLECTANCE, 4),
                "rows": sweep_rows,
            },
            "aspect_recovery": aspect_rows,
        },
    )

    det_table = markdown_table(
        detector_rows,
        [
            ("Method", "method"),
            ("Found a quad", "success_pct"),
            ("Usable (≤10 px)", "usable_pct"),
            ("Mean corner err (px)", "corner_error_px_mean"),
            ("Median (px)", "corner_error_px_median"),
            ("p90 (px)", "corner_error_px_p90"),
            ("Area IoU", "area_iou_mean"),
            ("Time (ms)", "median_ms"),
        ],
    )
    bin_table = markdown_table(
        binariser_rows,
        [
            ("Method", "method"),
            ("Text IoU (mean)", "text_iou_mean"),
            ("Text IoU (median)", "text_iou_median"),
            ("Time (ms)", "median_ms"),
        ],
    )
    sweep_table = markdown_table(
        sweep_rows,
        [("Page illum. ratio", "page_ratio")]
        + [(n, n) for n in ds.BINARISERS]
        + [(ds.ORACLE_NAME, ds.ORACLE_NAME)],
    )
    aspect_table = markdown_table(
        aspect_rows,
        [
            ("Method", "method"),
            ("True w/h", "true_ratio"),
            ("Recovered w/h", "mean_ratio"),
            ("Mean error", "rel_error_pct"),
            ("Worst error", "rel_error_pct_max"),
        ],
    )
    write_tables(
        RESULTS,
        [
            (f"Page-boundary detection ({args.scenes} scenes)", det_table),
            (
                f"Binarisation ({args.scenes} scenes, illumination ratio 0.62)",
                bin_table,
            ),
            (
                f"Binarisation vs illumination ({sweep_scenes} scenes per level, text IoU)",
                sweep_table,
            ),
            (f"Aspect-ratio recovery ({args.scenes} scenes, % error)", aspect_table),
        ],
    )

    print("\n" + det_table + "\n\n" + bin_table + "\n\n" + sweep_table + "\n\n" + aspect_table)

    best_bin = max(binariser_rows, key=lambda r: r["text_iou_mean"])
    worst_bin = min(binariser_rows, key=lambda r: r["text_iou_mean"])
    baseline = next(r for r in detector_rows if r["method"].startswith("minAreaRect"))

    # locate the page ratio at which the global threshold falls behind
    otsu_name, sauvola_name = "Otsu (global)", "Sauvola"
    crossover = next(
        (r["page_ratio"] for r in sweep_rows if r[otsu_name] < r[sauvola_name] - 0.05), None
    )
    flat, worst_light = sweep_rows[0], sweep_rows[-1]
    edge_aspect = aspect_rows[0]
    persp_aspect = aspect_rows[1]

    print("\n--- HEADLINE NUMBERS ---")
    print(f"best detector  : {best_det['method']} @ {best_det['corner_error_px_mean']} px mean, "
          f"usable {best_det['usable_rate'] * 100:.0f}%, {best_det['median_ms']} ms")
    print(f"rect baseline  : {baseline['method']} @ {baseline['corner_error_px_mean']} px mean, "
          f"usable {baseline['usable_rate'] * 100:.0f}%")
    print(f"best binariser : {best_bin['method']} @ text IoU {best_bin['text_iou_mean']}")
    print(f"worst binariser: {worst_bin['method']} @ text IoU {worst_bin['text_iou_mean']}")
    print(f"flat light     : Otsu {flat[otsu_name]}  Sauvola {flat[sauvola_name]}  "
          f"oracle {flat[ds.ORACLE_NAME]}")
    print(f"deep shadow    : Otsu {worst_light[otsu_name]}  Sauvola {worst_light[sauvola_name]}  "
          f"oracle {worst_light[ds.ORACLE_NAME]}  (page ratio {worst_light['page_ratio']})")
    print(f"Otsu falls >0.05 IoU behind Sauvola at page illumination ratio: {crossover}")
    print(f"theoretical limit for any global cut: page ratio {ds.INK_REFLECTANCE:.4f}")
    print(f"aspect: edge-length {edge_aspect['rel_error_pct']}% mean "
          f"({edge_aspect['rel_error_pct_max']}% worst) vs perspective "
          f"{persp_aspect['rel_error_pct']}% mean ({persp_aspect['rel_error_pct_max']}% worst), "
          f"{persp_aspect.get('n_degenerate_fallback')} degenerate fallbacks")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
