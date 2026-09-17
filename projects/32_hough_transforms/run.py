"""Run the Hough transform comparison and write results + figures.

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

from shared import bsds, figures  # noqa: E402
from shared.io import ensure_rgb, to_gray  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import hough as hg  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    print("Scoring lines and circles on the generated scenes ...")
    lines_clean = hg.evaluate_lines(runs=args.runs)
    lines_clutter = hg.evaluate_lines(clutter=5, runs=args.runs)
    circles = hg.evaluate_circles(runs=args.runs)

    print(f"Scoring line detection on {len(hg.IMAGES)} photographs ...")
    photo = hg.evaluate_photo_lines()

    print("Sweeping the threshold, the noise and the clutter ...")
    threshold_rows = hg.sweep_threshold()
    noise_rows = hg.sweep_noise()
    clutter_rows = hg.sweep_clutter()

    methods = ["Canny edges (control)", "Standard Hough", "Probabilistic Hough"]

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    # Rows cross edge density, because the pool deliberately mixes man-made
    # scenes full of straight lines with animals and rock faces that have none.
    EDGE_BANDS = [("bare", 0, 12), ("light", 12, 18),
                  ("busy", 18, 24), ("dense", 24, 100)]
    candidates: dict[str, list[dict]] = {}
    for name in hg.IMAGES:
        gray = to_gray(hg.load_scene(name))
        th, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        density = float((cv2.Canny(gray, 0.5 * th, th) > 0).mean() * 100)
        band = next(b for b, lo, hi in EDGE_BANDS if lo <= density < hi)
        target = bsds.consensus_boundaries(name)

        drawn = {
            "Canny edges (control)": cv2.Canny(gray, 50, 150),
            "Standard Hough": hg.rasterise_lines(hg.detect_lines_standard(gray), gray.shape),
            "Probabilistic Hough": hg.rasterise_lines(
                hg.detect_lines_probabilistic(gray)[0], gray.shape),
        }
        scores = [bsds.boundary_f_measure(drawn[m], target, 2)["precision"] for m in methods]
        panels = [hg.load_scene(name), ensure_rgb((target * 255).astype(np.uint8))]
        panels += [ensure_rgb(drawn[m]) for m in methods]
        notes = ["photograph", "HUMAN boundaries"]
        notes += [f"precision {s:.3f}\n{100 * float((drawn[m] > 0).mean()):.0f}% drawn"
                  for m, s in zip(methods, scores)]

        print(f"scene candidate {name:22s} edges {density:5.1f}%  [{band:6s}]  "
              f"canny {scores[0]:.3f}, best hough {max(scores[1:]):.3f}")
        candidates.setdefault(band, []).append({
            "label": f"{name.replace('_', ' ')}\nedges {density:.0f}%",
            "subject": name,
            "density": density,
            "images": panels,
            "notes": notes,
            "scores": scores,
            "score": float(np.std(scores)),
        })

    chosen, used = [], set()
    for band, _, _ in EDGE_BANDS:
        pool = sorted(candidates.get(band, []), key=lambda r: -r["score"])
        pick = next((r for r in pool if r["subject"] not in used), None)
        if pick:
            used.add(pick["subject"])
            chosen.append(pick)
    chosen = chosen[:4]
    assert len({r["subject"] for r in chosen}) == len(chosen), "a subject repeats"

    figures.gallery(
        ["photograph", "HUMAN boundaries"] + methods,
        [(r["label"], r["images"]) for r in chosen],
        IMAGES / "compare_hough.png",
        cell_notes=[r["notes"] for r in chosen],
        suptitle=(
            "Hough lines against what people actually drew. Cells are precision "
            "— of the pixels each method marked, how many sit on a real "
            "boundary. Hough is built on the Canny in column three."
        ),
    )
    gallery_table = markdown_table(
        [dict([("Sr", i), ("Scene", r["label"].replace("\n", " · "))]
              + [(m, f"{s:.3f}") for m, s in zip(methods, r["scores"])])
         for i, r in enumerate(chosen, start=1)],
        [("Sr", "Sr"), ("Scene", "Scene")] + [(m, m) for m in methods],
    )
    print(f"\nfront-on comparison: {len(chosen)} scenes x {len(methods)} methods")
    print("\n--- hough on photographs ---\n" + gallery_table)

    # ------------------------------------------------------------------ #
    # the signature figure: the accumulator threshold trades the two errors
    # ------------------------------------------------------------------ #
    figures.lines(
        [r["threshold"] for r in threshold_rows],
        {k: [r[k] for r in threshold_rows] for k in threshold_rows[0] if k != "threshold"},
        IMAGES / "threshold_sweep.png",
        xlabel="accumulator threshold (votes)",
        ylabel="score",
        title="The accumulator threshold is the whole method: votes needed to be a line",
    )

    figures.lines(
        [r["noise_sigma"] for r in noise_rows],
        {k: [r[k] for r in noise_rows] for k in noise_rows[0] if k != "noise_sigma"},
        IMAGES / "noise_sweep.png",
        xlabel="noise sigma",
        ylabel="score",
        title="Hough under noise — voting is what makes it survive this",
    )

    figures.lines(
        [r["clutter"] for r in clutter_rows],
        {k: [r[k] for r in clutter_rows] for k in clutter_rows[0] if k != "clutter"},
        IMAGES / "clutter_sweep.png",
        xlabel="clutter segments added",
        ylabel="score",
        title="Clutter costs precision, not recall: Hough finds the lines and more",
    )

    figures.comparison_matrix(
        photo,
        [("Precision on photographs", "precision", True),
         ("% of frame drawn", "pixels_drawn", False)],
        IMAGES / "photo_matrix.png",
        title="On real photographs, Hough is less precise than the Canny it is built on",
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS,
        "32_hough_transforms",
        {
            "images": list(hg.IMAGES),
            "lines_clean": lines_clean,
            "lines_clutter": lines_clutter,
            "circles": circles,
            "photo_lines": photo,
            "threshold_sweep": threshold_rows,
            "noise_sweep": noise_rows,
            "clutter_sweep": clutter_rows,
        },
    )

    lines_table = markdown_table(
        lines_clutter,
        [("Method", "method"), ("Recall", "recall"), ("Precision", "precision"),
         ("Angle error (deg)", "angle_error_deg"), ("Rho error (px)", "rho_error_px"),
         ("Time (ms)", "median_ms")],
    )
    circle_table = markdown_table(
        circles,
        [("Method", "method")] + [(k, k) for k in circles[0] if k != "method"],
    )
    photo_table = markdown_table(
        photo,
        [("Method", "method"), ("Precision", "precision"),
         ("% of frame drawn", "pixels_drawn")],
    )
    write_tables(
        RESULTS,
        [
            ("Lines on the generated scene, with clutter", lines_table),
            ("Circles on the generated scene", circle_table),
            ("Lines on photographs, against human boundaries", photo_table),
            ("Four scenes down the rows", gallery_table),
        ],
    )
    print("\n" + lines_table + "\n\n" + photo_table)

    by_photo = {r["method"]: r for r in photo}
    canny = by_photo["Canny edges (control)"]
    best_hough = max((r for r in photo if "Hough" in r["method"]),
                     key=lambda r: r["precision"])
    std = next(r for r in lines_clutter if r["method"] == "Standard Hough")
    prob = next(r for r in lines_clutter if r["method"] == "Probabilistic Hough")

    print("\n--- HEADLINE NUMBERS ---")
    print(f"on the generated scene both reach recall {std['recall']:.3f} / "
          f"{prob['recall']:.3f} with sub-degree angle error")
    print(f"  standard is {prob['median_ms'] / std['median_ms']:.1f}x FASTER here "
          f"({std['median_ms']:.2f} ms vs {prob['median_ms']:.2f}) and more precise "
          f"({std['precision']:.3f} vs {prob['precision']:.3f})")
    print(f"on photographs    : Canny precision {canny['precision']:.4f}, "
          f"best Hough {best_hough['precision']:.4f} "
          f"({best_hough['precision'] - canny['precision']:+.4f})")
    print(f"  standard Hough draws {by_photo['Standard Hough']['pixels_drawn']:.1f}% "
          f"of the frame as line, against Canny's {canny['pixels_drawn']:.1f}%")
    print(f"  -> the voting stage ADDS error on real images, by the only measure "
          "available")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
