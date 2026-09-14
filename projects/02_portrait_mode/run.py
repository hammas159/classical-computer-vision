"""Run the portrait-mode experiment and write results + figures.

    python run.py [--scenes 12] [--radius 15]

Every number published for this project comes out of this script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from shared import figures, synth  # noqa: E402
from shared.io import to_float  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import portrait_mode as pm  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"


def mask_overlay(img: np.ndarray, mask: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Green where the matte is right, red where it is missing or spilling."""
    out = img.copy()
    pred = mask > 0
    true = truth > 0
    tint = np.zeros_like(out)
    tint[pred & true] = (0, 190, 0)        # correct
    tint[~pred & true] = (220, 40, 40)     # missed subject
    tint[pred & ~true] = (250, 170, 0)     # spilled into background
    return cv2.addWeighted(out, 0.55, tint, 0.45, 0)


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=12)
    ap.add_argument("--radius", type=int, default=15)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    print(f"Scoring {len(pm.MATTES)} matting methods over {args.scenes} scenes ...")
    matte_rows = pm.evaluate_mattes(n_scenes=args.scenes)

    stability_seeds = max(8, args.scenes * 2)
    print(f"Measuring GrabCut seed stability ({stability_seeds} seeds x 4 scenes) ...")
    stability_rows = pm.evaluate_grabcut_stability(n_scenes=4, n_seeds=stability_seeds)

    print(f"Characterising {len(pm.BOKEH_KERNELS)} bokeh kernels ...")
    bokeh_rows = pm.evaluate_bokeh(radius=args.radius)

    print(f"Measuring halo for {len(pm.COMPOSITORS)} compositing strategies ...")
    comp_rows = pm.evaluate_compositing(n_scenes=args.scenes, radius=args.radius)

    # ------------------------------------------------------------------ #
    # figures
    # ------------------------------------------------------------------ #
    scene = synth.portrait_scene(background="coffee", seed=0)
    kernel = pm.kernel_disc(args.radius)

    panels = [("Input", scene.image), ("True matte", scene.mask)]
    for name, fn in pm.MATTES.items():
        m = fn(scene.image)
        if m is None:
            panels.append((f"{name}\nNO SUBJECT FOUND", np.zeros_like(scene.mask)))
        else:
            row = next(r for r in matte_rows if r["method"] == name)
            panels.append((f"{name}\nIoU {row['iou']} · hair {row['hair_recall']}", m))
    figures.grid(panels, IMAGES / "mattes.png", ncols=4, suptitle="Six ways to cut out a subject")

    overlays = [("True matte", mask_overlay(scene.image, scene.mask, scene.mask))]
    for name, fn in pm.MATTES.items():
        m = fn(scene.image)
        if m is not None:
            overlays.append((name, mask_overlay(scene.image, m, scene.mask)))
    figures.grid(
        overlays,
        IMAGES / "matte_errors.png",
        ncols=4,
        suptitle="Green = correct · red = missed subject · orange = spilled into background",
    )

    # bokeh: how each kernel renders a point highlight
    point = np.zeros((args.radius * 6 + 1, args.radius * 6 + 1), np.float32)
    point[point.shape[0] // 2, point.shape[1] // 2] = 1.0
    bokeh_panels = []
    for name, make in pm.BOKEH_KERNELS.items():
        k = make(args.radius)
        rendered = cv2.filter2D(point, -1, k)
        rendered = rendered / rendered.max()
        prof = pm.highlight_profile(k)
        bokeh_panels.append(
            (f"{name}\npeak/mean {prof['peak_to_mean']} · rim {prof['edge_sharpness']}", rendered)
        )
    figures.grid(
        bokeh_panels,
        IMAGES / "bokeh_kernels.png",
        ncols=4,
        suptitle="How each kernel renders a single point of light (normalised)",
    )

    # compositing: the halo
    ideal = pm.composite_reference(scene, kernel)
    naive = pm.composite_naive(scene.image, scene.mask, kernel)
    masked = pm.composite_masked(scene.image, scene.mask, kernel)

    def amplified_error(a, b):
        d = np.abs(to_float(a) - to_float(b)).mean(axis=-1)
        return np.clip(d * 8.0, 0, 1)

    figures.grid(
        [
            ("Ideal (blur the true plate)", ideal),
            ("Naive: blur all, paste back", naive),
            ("Masked: normalised convolution", masked),
            ("Naive error (x8)", amplified_error(naive, ideal)),
            ("Masked error (x8)", amplified_error(masked, ideal)),
            ("Halo ring measured", pm.halo_ring(scene.mask, 12)),
        ],
        IMAGES / "compositing.png",
        ncols=3,
        suptitle="Blurring across the subject boundary smears the subject into the background",
    )

    best_matte = max((r for r in matte_rows if r["iou"] is not None), key=lambda r: r["iou"])
    mask, out = pm.portrait(scene.image, best_matte["method"], "Disc (circular aperture)", args.radius)
    figures.grid(
        [("1 · Input", scene.image), ("2 · Matte", mask), ("3 · Portrait mode", out)],
        IMAGES / "pipeline.png",
        ncols=3,
        suptitle=f"End-to-end: {best_matte['method']} -> disc bokeh r={args.radius} -> masked composite",
    )

    ok = [r for r in matte_rows if r["iou"] is not None]
    figures.metric_bars(
        [r["method"] for r in ok],
        [r["hair_recall"] for r in ok],
        IMAGES / "hair_recall.png",
        ylabel="fraction of hair pixels recovered",
        title="Hair is 2.5% of the subject — and where every method fails",
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS,
        "02_portrait_mode",
        {
            "n_scenes": args.scenes,
            "bokeh_radius_px": args.radius,
            "haar_cascade": pm.FACE_CASCADE_FILE.split("/")[-1],
            "grabcut_seed_used": pm.GRABCUT_SEED,
            "mattes": matte_rows,
            "grabcut_stability": stability_rows,
            "bokeh_kernels": bokeh_rows,
            "compositing": comp_rows,
        },
    )

    matte_table = markdown_table(
        matte_rows,
        [
            ("Method", "method"),
            ("Subject found", "detected_pct"),
            ("IoU", "iou"),
            ("Dice", "dice"),
            ("Body recall", "body_recall"),
            ("Hair recall", "hair_recall"),
            ("Background FPR", "background_fpr"),
            ("Boundary F1", "boundary_f1"),
            ("Time (ms)", "median_ms"),
        ],
    )
    stability_table = markdown_table(
        stability_rows,
        [
            ("Scene", "scene"),
            ("Seeds", "n_seeds"),
            ("IoU mean", "iou_mean"),
            ("IoU std", "iou_std"),
            ("Worst", "iou_min"),
            ("Best", "iou_max"),
            ("Spread", "iou_spread"),
        ],
    )
    bokeh_table = markdown_table(
        bokeh_rows,
        [
            ("Kernel", "kernel"),
            ("Radius (px)", "radius_px"),
            ("Peak / mean", "peak_to_mean"),
            ("Rim energy", "edge_sharpness"),
        ],
    )
    comp_table = markdown_table(
        comp_rows,
        [
            ("Strategy", "method"),
            ("Halo error (0-255)", "halo_err_0_255"),
            ("Whole-background error", "background_err_0_255"),
            ("Time (ms)", "median_ms"),
        ],
    )

    write_tables(
        RESULTS,
        [
            (
                f"Subject matting ({args.scenes} scenes, GrabCut pinned to seed "
                f"{pm.GRABCUT_SEED})",
                matte_table,
            ),
            (
                f"GrabCut seed stability ({stability_seeds} seeds on each identical image)",
                stability_table,
            ),
            (f"Bokeh kernels (radius {args.radius} px)", bokeh_table),
            (f"Compositing ({args.scenes} scenes, true matte, disc r={args.radius})", comp_table),
        ],
    )
    print(
        "\n" + matte_table + "\n\n" + stability_table + "\n\n" + bokeh_table + "\n\n" + comp_table
    )

    by_iou = max(ok, key=lambda r: r["iou"])
    by_boundary = max(ok, key=lambda r: r["boundary_f1"])
    by_hair = max((r for r in ok if r["background_fpr"] < 0.2), key=lambda r: r["hair_recall"])
    naive_row = comp_rows[0]
    masked_row = comp_rows[1]

    print("\n--- HEADLINE NUMBERS ---")
    print(f"best by IoU        : {by_iou['method']} ({by_iou['iou']}), "
          f"but recovers only {by_iou['hair_recall'] * 100:.1f}% of hair")
    print(f"best by boundary F1: {by_boundary['method']} ({by_boundary['boundary_f1']}), "
          f"IoU {by_boundary['iou']}, hair {by_boundary['hair_recall'] * 100:.1f}%")
    print(f"best hair (FPR<0.2): {by_hair['method']} at {by_hair['hair_recall'] * 100:.1f}%")
    print(f"slowest            : {max(ok, key=lambda r: r['median_ms'])['method']} "
          f"@ {max(r['median_ms'] for r in ok):.0f} ms")
    print(f"halo: naive {naive_row['halo_err_0_255']} vs masked {masked_row['halo_err_0_255']} "
          f"(x{naive_row['halo_err_0_255'] / max(masked_row['halo_err_0_255'], 1e-6):.1f} worse)")
    g = next(r for r in bokeh_rows if r["kernel"] == "Gaussian")
    d = next(r for r in bokeh_rows if r["kernel"].startswith("Disc"))
    print(f"bokeh: Gaussian peak/mean {g['peak_to_mean']} vs disc {d['peak_to_mean']}; "
          f"rim energy {g['edge_sharpness']} vs {d['edge_sharpness']}")
    worst = max(stability_rows, key=lambda r: r["iou_spread"])
    print(f"GrabCut seed noise: worst scene {worst['scene']} spans IoU "
          f"{worst['iou_min']}-{worst['iou_max']} (spread {worst['iou_spread']}, "
          f"std {worst['iou_std']}) on an image that never changed")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
