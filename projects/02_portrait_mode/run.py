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

from shared import figures, io, synth  # noqa: E402
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

    # ------------------------------------------------------------------ #
    # front-on comparison: four subjects down, every matting method across
    # ------------------------------------------------------------------ #
    # Four different subject SHAPES on four different backgrounds — a person
    # mid-stride against a crowd, a round still life, a small animal against
    # foliage, a flat insect on leaves. Laid out methods-across so a reader can
    # compare six matting methods on one subject and one method down four.
    #
    # No IoU is printed. Only the footballer has a reference matte, and inventing
    # one for the other three by running a method and calling it truth would be
    # marking the methods' own homework. The area found is reported instead,
    # which needs no annotation, and the outputs are there to be looked at.
    subject_specs = [
        ("footballer\nperson · crowd behind", scene.image),
        ("fruit bowl\nstill life · cluttered", io.real_photo("fruits")),
        ("squirrel\nanimal · foliage", io.real_photo("squirrel")),
        ("butterfly\nflat subject · leaves", io.real_photo("butterfly")),
    ]
    matte_names = list(pm.MATTES)
    rows_matte, rows_portrait, matte_notes, portrait_notes = [], [], [], []
    for label, src in subject_specs:
        m_imgs, m_notes = [src], [""]
        p_imgs, p_notes = [src], [""]
        for mname in matte_names:
            mask = pm.MATTES[mname](src)
            if mask is None:
                blank = np.zeros(src.shape[:2], np.uint8)
                m_imgs.append(blank)
                p_imgs.append(src)
                m_notes.append("NO SUBJECT")
                p_notes.append("NO SUBJECT")
                continue
            m_imgs.append(mask)
            p_imgs.append(pm.COMPOSITORS["Masked (normalised convolution)"](src, mask, kernel))
            m_notes.append(f"{float((mask > 0).mean()):.1%} of frame")
            p_notes.append("")
        rows_matte.append((label, m_imgs))
        rows_portrait.append((label, p_imgs))
        matte_notes.append(m_notes)
        portrait_notes.append(p_notes)

    figures.gallery(
        ["input"] + matte_names,
        rows_matte,
        IMAGES / "compare_mattes.png",
        cell_notes=matte_notes,
        suptitle="Six matting methods on four subjects — what each one thinks the subject is",
    )
    figures.gallery(
        ["input"] + matte_names,
        rows_portrait,
        IMAGES / "compare_portraits.png",
        cell_notes=portrait_notes,
        suptitle=f"The portrait each matte produces (disc aperture r={args.radius})",
    )
    print(f"front-on comparison: {len(subject_specs)} subjects x {len(matte_names)} mattes")

    panels = [("Input", scene.image), ("True matte", scene.mask)]
    for name, fn in pm.MATTES.items():
        m = fn(scene.image)
        if m is None:
            panels.append((f"{name}\nNO SUBJECT FOUND", np.zeros_like(scene.mask)))
        else:
            row = next(r for r in matte_rows if r["method"] == name)
            panels.append((f"{name}\nIoU {row['iou']} · fine {row['fine_recall']}", m))
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

    # ------------------------------------------------------------------ #
    # distributions and matrices
    # ------------------------------------------------------------------ #

    # 1. confusion matrix per matting method
    for name, fn in pm.MATTES.items():
        m = fn(scene.image)
        if m is None:
            continue
        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "")
        figures.confusion_matrix(
            pm.matte_confusion(m, scene.mask),
            IMAGES / f"confusion_{slug}.png",
            row_labels=["background", "subject"],
            col_labels=["background", "subject"],
            title=f"{name} — where every pixel went",
        )

    # 2. the region breakdown: body vs hair vs background, per method.
    # Built from the aggregated rows, not from one scene: scene 0 is the
    # pathological one for GrabCut, and a single-scene figure would flatly
    # contradict the table above it.
    region_rows = [
        {
            "method": r["method"],
            "body": r["body_recall"],
            "fine": r["fine_recall"],
            "background": round(1.0 - r["background_fpr"], 4),
        }
        for r in matte_rows
        if r["iou"] is not None
    ]
    figures.comparison_matrix(
        region_rows,
        [("Body", "body", True), ("Fine detail", "fine", True), ("Background", "background", True)],
        IMAGES / "region_matrix.png",
        title=(
            f"Fraction of each region recovered, mean over {args.scenes} scenes — "
            "hair is where they all fail"
        ),
    )

    # 3. the halo, as a distribution rather than two numbers
    band = pm.halo_ring(scene.mask, 12) > 0
    figures.histogram(
        {
            "Naive (blur all, paste back)": pm.halo_error_map(naive, ideal)[band],
            "Masked (normalised convolution)": pm.halo_error_map(masked, ideal)[band],
        },
        IMAGES / "halo_distribution.png",
        bins=80,
        value_range=(0, 60),
        xlabel="absolute error vs the ideal composite (0-255)",
        title="Error in the 12 px ring outside the subject",
    )

    # 4. the bokeh kernels as actual numbers
    # radius 7, not 5: a hexagon rasterised onto an 11x11 grid is visibly
    # lopsided and reads as a bug rather than as coarse sampling
    small_r = 7
    figures.value_matrix(
        [
            (name, (make(small_r) / make(small_r).max() * 100))
            for name, make in pm.BOKEH_KERNELS.items()
        ],
        IMAGES / "kernel_matrix.png",
        cmap="magma",
        vmin=0,
        vmax=100,
        title=(
            f"The four apertures as matrices (radius {small_r}, scaled to 100). "
            "A disc is flat; a Gaussian peaks in the middle"
        ),
    )

    # 5. the comparative matrix
    figures.comparison_matrix(
        [r for r in matte_rows if r["iou"] is not None],
        [
            ("IoU", "iou", True),
            ("Dice", "dice", True),
            ("Body recall", "body_recall", True),
            ("Fine detail recall", "fine_recall", True),
            ("Background FPR", "background_fpr", False),
            ("Boundary F1", "boundary_f1", True),
            ("Time (ms)", "median_ms", False),
        ],
        IMAGES / "matte_matrix.png",
        title="Matting methods x metrics — note that no column agrees with another",
    )

    ok = [r for r in matte_rows if r["iou"] is not None]
    figures.metric_bars(
        [r["method"] for r in ok],
        [r["fine_recall"] for r in ok],
        IMAGES / "fine_recall.png",
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
            ("Fine detail recall", "fine_recall"),
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
    by_fine = max((r for r in ok if r["background_fpr"] < 0.2), key=lambda r: r["fine_recall"])
    naive_row = comp_rows[0]
    masked_row = comp_rows[1]

    print("\n--- HEADLINE NUMBERS ---")
    print(f"best by IoU        : {by_iou['method']} ({by_iou['iou']}), "
          f"but recovers only {by_iou['fine_recall'] * 100:.1f}% of fine detail")
    print(f"best by boundary F1: {by_boundary['method']} ({by_boundary['boundary_f1']}), "
          f"IoU {by_boundary['iou']}, fine {by_boundary['fine_recall'] * 100:.1f}%")
    print(f"best fine (FPR<0.2): {by_fine['method']} at {by_fine['fine_recall'] * 100:.1f}%")
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
