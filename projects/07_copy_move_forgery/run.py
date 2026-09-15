"""Run the copy-move forgery experiments and write results + figures.

    python run.py [--size 96]

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

from shared import figures, io, synth  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import forgery as fg  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=96, help="pasted patch size in px")
    ap.add_argument("--images", type=int, default=len(fg.IMAGES))
    args = ap.parse_args()

    images = fg.IMAGES[: args.images]
    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    print(f"Scoring {len(fg.METHODS)} detectors on an exact copy, {len(images)} images ...")
    exact_rows = fg.evaluate_methods(size=args.size, images=images)

    print(f"Sweeping rotation over {len(fg.ANGLE_LEVELS)} angles ...")
    rotation_rows = fg.sweep_rotation(images=images)

    print(f"Sweeping scale over {len(fg.SCALE_LEVELS)} factors ...")
    scale_rows = fg.sweep_scale(images=images)

    print(f"Sweeping forgery size over {len(fg.SIZE_LEVELS)} sizes ...")
    size_rows = fg.sweep_size(images=images)

    print("Counting false alarms on untampered images ...")
    false_rows = fg.evaluate_false_alarms(images=images)

    # ------------------------------------------------------------------ #
    # figures
    # ------------------------------------------------------------------ #
    clean = io.sample("astronaut")
    f0 = synth.copy_move_forgery(clean, size=args.size, seed=0)

    panels = [("Original", clean), ("Forged", f0.image), ("Truth: both copies", f0.mask_both)]
    for name, fn in fg.METHODS.items():
        if name.startswith("Predict nothing"):
            continue
        row = next(r for r in exact_rows if r["method"] == name)
        panels.append((f"{name}\nIoU {row['iou']:.3f}", fn(f0.image)))
    figures.grid(
        panels,
        IMAGES / "methods.png",
        ncols=4,
        suptitle=f"An exact {args.size}px copy-move — the easy case every demo shows",
    )

    f15 = synth.copy_move_forgery(clean, size=args.size, angle_deg=15.0, seed=0)
    panels = [("Forged, rotated 15°", f15.image), ("Truth: both copies", f15.mask_both)]
    for name, fn in fg.METHODS.items():
        if name.startswith("Predict nothing"):
            continue
        row = next(r for r in rotation_rows if r["angle_deg"] == 15.0)
        panels.append((f"{name}\nIoU {row[name]:.3f}", fn(f15.image)))
    figures.grid(
        panels,
        IMAGES / "rotated.png",
        ncols=3,
        suptitle="The same forgery turned 15° — where block matching stops working entirely",
    )

    method_names = [n for n in fg.METHODS if not n.startswith("Predict nothing")]
    figures.lines(
        [r["angle_deg"] for r in rotation_rows],
        {n: [r[n] for r in rotation_rows] for n in method_names},
        IMAGES / "rotation_sweep.png",
        xlabel="paste rotation (degrees)",
        ylabel="mask IoU",
        title="The best method on an exact copy is the worst at every non-zero angle",
    )

    figures.lines(
        [r["scale"] for r in scale_rows],
        {n: [r[n] for r in scale_rows] for n in method_names},
        IMAGES / "scale_sweep.png",
        xlabel="paste scale factor",
        ylabel="mask IoU",
        title="Rescaling the copy: 5% is enough to end block matching",
    )

    figures.lines(
        [r["size_px"] for r in size_rows],
        {n: [r[n] for r in size_rows] for n in method_names},
        IMAGES / "size_sweep.png",
        xlabel="pasted region size (px)",
        ylabel="mask IoU",
        title="Every method has a hard floor, and it is built into the algorithm",
    )

    # the false-alarm figure: untampered images, and what each method claims
    fa_panels = []
    worst = max(false_rows, key=lambda r: r["mean_flagged"])
    for name in ("grass", "brick", "astronaut"):
        img = io.sample(name)
        fa_panels.append((f"{name} (untampered)", img))
        fa_panels.append(
            (
                f"{worst['method']}\nflags {float((fg.METHODS[worst['method']](img) > 0).mean()):.1%}",
                fg.METHODS[worst["method"]](img),
            )
        )
    figures.grid(
        fa_panels,
        IMAGES / "false_alarms.png",
        ncols=2,
        suptitle="Untampered photographs, and what the most rotation-robust method accuses them of",
    )

    figures.comparison_matrix(
        exact_rows,
        [
            ("Mask IoU", "iou", True),
            ("Precision", "precision", True),
            ("Recall", "recall", True),
            ("Pixel accuracy", "pixel_accuracy", True),
            ("Time (ms)", "median_ms", False),
        ],
        IMAGES / "method_matrix.png",
        title="Exact copy — note the control's pixel accuracy against its IoU",
    )

    # what the keypoints actually see
    pairs = fg.self_matches(f0.image, fg.make_sift(), cv2.NORM_L2, 0.6, 30)
    overlay = f0.image.copy()

    for p1, p2 in pairs:
        cv2.line(overlay, tuple(np.int32(p1)), tuple(np.int32(p2)), (255, 60, 60), 1)
        cv2.circle(overlay, tuple(np.int32(p1)), 3, (60, 255, 60), -1)
    figures.grid(
        [
            ("Forged", f0.image),
            (f"{len(pairs)} self-matches", overlay),
            ("After similarity verification", fg.detect_sift(f0.image)),
            ("Truth", f0.mask_both),
        ],
        IMAGES / "matches.png",
        ncols=4,
        suptitle="Sparse matches say WHERE and by HOW MUCH; verification says which pixels",
    )

    figures.histogram(
        {
            "whole image": io.to_gray(f0.image).ravel(),
            "the duplicated pair": io.to_gray(f0.image)[f0.mask_both > 0],
        },
        IMAGES / "histogram.png",
        bins=128,
        title="A copy-move leaves no statistical trace — the pasted pixels came from this photo",
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS,
        "07_copy_move_forgery",
        {
            "size_px": args.size,
            "images": list(images),
            "exact_copy": exact_rows,
            "rotation_sweep": rotation_rows,
            "scale_sweep": scale_rows,
            "size_sweep": size_rows,
            "false_alarms": false_rows,
            "sift_contrast_threshold": fg.SIFT_CONTRAST_THRESHOLD,
        },
    )

    exact_table = markdown_table(
        exact_rows,
        [
            ("Method", "method"),
            ("Mask IoU", "iou"),
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("Pixel accuracy", "pixel_accuracy"),
            ("Time (ms)", "median_ms"),
        ],
    )
    rotation_table = markdown_table(
        rotation_rows, [("Rotation (deg)", "angle_deg")] + [(n, n) for n in method_names]
    )
    scale_table = markdown_table(
        scale_rows, [("Scale", "scale")] + [(n, n) for n in method_names]
    )
    size_table = markdown_table(
        size_rows,
        [("Size (px)", "size_px"), ("Area fraction", "area_fraction")]
        + [(n, n) for n in method_names],
    )
    false_table = markdown_table(
        false_rows,
        [
            ("Method", "method"),
            ("Mean flagged", "mean_flagged"),
            ("Worst image", "max_flagged"),
            ("Images accused", "images_with_any_flag"),
        ],
    )
    write_tables(
        RESULTS,
        [
            (f"Exact {args.size}px copy-move ({len(images)} images)", exact_table),
            ("Mask IoU vs paste rotation", rotation_table),
            ("Mask IoU vs paste scale", scale_table),
            ("Mask IoU vs forgery size", size_table),
            ("FALSE ALARMS: fraction of pixels flagged on UNTAMPERED images", false_table),
        ],
    )
    print(
        "\n"
        + "\n\n".join([exact_table, rotation_table, scale_table, size_table, false_table])
    )

    best_exact = max(exact_rows, key=lambda r: r["iou"])
    control = next(r for r in exact_rows if r["method"].startswith("Predict nothing"))
    block = next(r for r in exact_rows if r["method"] == "Block matching")
    sim = next(r for r in exact_rows if r["method"] == "SIFT + similarity verify")
    trans = next(r for r in exact_rows if r["method"] == "SIFT + translation verify")
    blobs = next(r for r in exact_rows if r["method"] == "SIFT blobs (no verify)")
    rot15 = next(r for r in rotation_rows if r["angle_deg"] == 15.0)
    rot90 = next(r for r in rotation_rows if r["angle_deg"] == 90.0)
    rot2 = next(r for r in rotation_rows if r["angle_deg"] == 2.0)
    sc95 = next(r for r in scale_rows if r["scale"] == 0.95)
    fa_block = next(r for r in false_rows if r["method"] == "Block matching")
    fa_sim = next(r for r in false_rows if r["method"] == "SIFT + similarity verify")

    print("\n--- HEADLINE NUMBERS ---")
    print(f"best on an exact copy : {best_exact['method']} @ IoU {best_exact['iou']}")
    print(
        f"the control           : IoU {control['iou']} but pixel accuracy "
        f"{control['pixel_accuracy']} — predicting 'no forgery' is {control['pixel_accuracy']:.1%} accurate"
    )
    print(
        f"block matching        : {block['iou']} exact -> {rot2['Block matching']} at 2 deg "
        f"-> {rot15['Block matching']} at 15 deg -> {sc95['Block matching']} at 0.95x scale"
    )
    print(
        f"verifier hypothesis   : same SIFT matches, translation {trans['iou']} vs similarity "
        f"{sim['iou']} on an exact copy; at 90 deg {rot90['SIFT + translation verify']} vs "
        f"{rot90['SIFT + similarity verify']}"
    )
    print(
        f"dense verify vs blobs : {sim['iou']} vs {blobs['iou']} — same keypoints, "
        f"same matches, different way of turning them into a region"
    )
    print(
        f"FALSE ALARMS          : block matching flags {fa_block['mean_flagged']:.1%} of an "
        f"untampered image ({fa_block['images_with_any_flag']} accused); "
        f"SIFT + similarity flags {fa_sim['mean_flagged']:.1%} "
        f"(worst {fa_sim['max_flagged']:.1%}, {fa_sim['images_with_any_flag']} accused)"
    )
    floor = [r for r in size_rows if r["Block matching"] > 0.5]
    if floor:
        print(
            f"size floor            : block matching works down to {floor[0]['size_px']} px "
            f"({floor[0]['area_fraction']:.1%} of the image) and is 0.0 below it"
        )
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
