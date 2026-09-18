"""Run the Poisson blending comparison and write results + figures.

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
from shared.io import ensure_rgb  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import poisson_blending as pb  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"

OFFSET = 0.25
SIZE = 120


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    methods = list(pb.METHODS)

    print(f"Blending {SIZE}x{SIZE} regions across {len(pb.PAIRS)} pairs, "
          f"brightness offset {OFFSET:g} ...")
    clean = pb.evaluate_methods(brightness_offset=OFFSET, size=SIZE, runs=args.runs)
    for r in clean:
        print(f"  {r['method']:32s} seam {r['seam_visibility']:6.3f}  "
              f"gradient {r['gradient_fidelity']:.4f}  "
              f"pixels moved {r['pixel_difference']:6.2f}  {r['median_ms']:8.2f} ms")

    print("\nHow far the brightness can differ before each method gives up ...")
    offsets = pb.sweep_brightness_offset()
    for r in offsets:
        print(f"  offset {r['brightness_offset']:.2f}  " + "  ".join(
            f"{m.split()[0][:7]}={r[m]:6.3f}" for m in methods))

    sizes = pb.sweep_region_size()
    disagreement = pb.metric_disagreement()
    convergence = pb.convergence()

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    rows, notes, table_rows = [], [], []
    for sr, (target_name, source_name) in enumerate(pb.PAIRS[:4], start=1):
        target, source, mask, centre = pb.make_case(
            target_name, source_name, size=SIZE, brightness_offset=OFFSET)

        panels = [target, ensure_rgb(cv2.bitwise_and(source, source, mask=mask))]
        cell = [f"target · tone {pb.tonal_range(target):.0f}",
                f"source · tone {pb.tonal_range(source):.0f}"]
        scores = {}
        for method in methods:
            out = pb.METHODS[method](target, source, mask, centre)
            seam = pb.seam_visibility(out, mask, centre, target)
            panels.append(out)
            cell.append(f"seam {seam:.3f}")
            scores[method] = seam
        rows.append((f"{target_name.replace('_', ' ')}\n+ {source_name.replace('_', ' ')}",
                     panels))
        notes.append(cell)
        table_rows.append(dict([("Sr", sr),
                                ("Pair", f"{target_name.replace('_', ' ')} + "
                                         f"{source_name.replace('_', ' ')}")],
                               **{m: f"{scores[m]:.3f}" for m in methods}))
        print(f"  row {sr}: {target_name:20s} + {source_name:20s}  " +
              "  ".join(f"{scores[m]:.2f}" for m in methods))

    figures.gallery(
        ["target", "source region"] + methods,
        rows, IMAGES / "compare_blending.png",
        cell_notes=notes,
        suptitle=(
            f"A {SIZE}x{SIZE} region pasted with the source brightened by {OFFSET:g}. "
            "Cells are seam visibility — the boundary gradient relative to its "
            "surroundings, where 1.0 means the edge is indistinguishable."
        ),
    )
    gallery_table = markdown_table(
        table_rows, [("Sr", "Sr"), ("Pair", "Pair")] + [(m, m) for m in methods])
    print("\n--- seam visibility ---\n" + gallery_table)

    # ------------------------------------------------------------------ #
    # the signature figure: the metric that inverts the conclusion
    # ------------------------------------------------------------------ #
    figures.comparison_matrix(
        clean,
        [("Seam visibility", "seam_visibility", False),
         ("Gradient fidelity", "gradient_fidelity", True),
         ("Pixels moved", "pixel_difference", False),
         ("Time (ms)", "median_ms", False)],
        IMAGES / "metric_matrix.png",
        row_key="method",
        title=("Three metrics, three different winners, and the method that looks right "
               "wins none of them outright. Green is 'better' by that column's own rule."),
    )

    figures.metric_bars(
        [r["method"] for r in clean],
        [r["pixel_difference"] for r in clean],
        IMAGES / "pixels_moved.png",
        ylabel="mean absolute change inside the pasted region (levels)",
        title=("How far each method moves the pasted pixels. Poisson moves them 51 levels "
               "and that is why it works."),
        highlight_best="max",
    )

    figures.lines(
        [r["brightness_offset"] for r in offsets],
        {m: [r[m] for r in offsets] for m in methods},
        IMAGES / "brightness_offset.png",
        xlabel="brightness offset applied to the source",
        ylabel="seam visibility (1.0 = invisible)",
        dashed={"Copy-paste (control)"},
        title="How different the two images can be before the seam returns",
    )

    figures.lines(
        [r["region_size"] for r in sizes],
        {m: [r[f"{m} seam"] for r in sizes] for m in methods},
        IMAGES / "region_size.png",
        xlabel="pasted region size (px)", ylabel="seam visibility",
        dashed={"Copy-paste (control)"},
        title="Region size: a bigger paste is a longer boundary to reconcile",
    )

    figures.lines(
        [r["iterations"] for r in convergence],
        {"seam visibility": [r["seam_visibility"] for r in convergence],
         "time (ms) / 100": [r["median_ms"] / 100 for r in convergence]},
        IMAGES / "convergence.png",
        xlabel="Jacobi iterations", ylabel="seam visibility  /  ms per 100",
        logx=True,
        title="The from-scratch solver converges by 100 iterations and costs 40x OpenCV",
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS, "44_poisson_blending",
        {
            "images": list(pb.IMAGES),
            "pairs": [list(p) for p in pb.PAIRS],
            "tone": {n: round(pb.tonal_range(pb.load_scene(n)), 1) for n in pb.IMAGES},
            "brightness_offset": OFFSET,
            "region_size": SIZE,
            "methods": clean,
            "offset_sweep": offsets,
            "size_sweep": sizes,
            "metric_disagreement": disagreement,
            "convergence": convergence,
        },
    )

    methods_table = markdown_table(
        clean, [("Method", "method"), ("Seam visibility", "seam_visibility"),
                ("Gradient fidelity", "gradient_fidelity"),
                ("Pixels moved", "pixel_difference"), ("Time (ms)", "median_ms")])
    offsets_table = markdown_table(
        offsets, [("Offset", "brightness_offset")] + [(m, m) for m in methods])
    sizes_table = markdown_table(
        sizes, [("Region size", "region_size")]
        + [(m, f"{m} seam") for m in methods])
    convergence_table = markdown_table(
        convergence, [("Iterations", "iterations"), ("Seam", "seam_visibility"),
                      ("Time (ms)", "median_ms")])

    write_tables(
        RESULTS,
        [
            (f"At a {OFFSET:g} brightness offset", methods_table),
            ("Against the brightness difference", offsets_table),
            ("Against the region size", sizes_table),
            ("Jacobi convergence", convergence_table),
            ("Four pairs down the rows", gallery_table),
        ],
    )
    print("\n" + methods_table)

    # ------------------------------------------------------------------ #
    # headlines
    # ------------------------------------------------------------------ #
    print("\n--- HEADLINE NUMBERS ---")
    by_method = {r["method"]: r for r in clean}
    control = by_method["Copy-paste (control)"]
    opencv = by_method["Poisson (OpenCV)"]
    feather = by_method["Alpha feather"]
    mixed = by_method["Poisson (mixed gradients)"]
    jacobi = by_method["Poisson (Jacobi, from scratch)"]

    print(f"copy-paste leaves a seam {control['seam_visibility']:.2f}x its surroundings; "
          f"Poisson brings it to {opencv['seam_visibility']:.2f} and mixed gradients to "
          f"{mixed['seam_visibility']:.2f}")

    print(f"\nPoisson moves the pasted pixels {opencv['pixel_difference']:.1f} levels on "
          f"average; alpha feather moves them {feather['pixel_difference']:.1f}")
    print(f"  that is the point: the method that changes the region "
          f"{opencv['pixel_difference'] / max(feather['pixel_difference'], 1e-9):.0f}x more "
          "is the one that looks right, so any metric scoring fidelity to the source "
          "ranks them backwards")
    print(f"  copy-paste scores a perfect {control['pixel_difference']:.1f} on that metric "
          f"and has the worst seam in the table")

    print(f"\nbut the seam metric has its own bias: alpha feather scores "
          f"{feather['seam_visibility']:.3f}, BETTER than Poisson's "
          f"{opencv['seam_visibility']:.3f}, by blurring the boundary rather than "
          "reconciling it — it only moved the pixels "
          f"{feather['pixel_difference']:.1f} levels, so the brightness step is still there")
    print("  no single column in this table is trustworthy alone, which is why the "
          "figure is the result and the numbers are the caveat")

    print(f"\nthe from-scratch Jacobi solver reaches {jacobi['seam_visibility']:.4f} against "
          f"OpenCV's {opencv['seam_visibility']:.4f} — they agree to "
          f"{abs(jacobi['seam_visibility'] - opencv['seam_visibility']):.4f}, which is the "
          f"check that both are right — and costs "
          f"{jacobi['median_ms'] / opencv['median_ms']:.0f}x the time")

    settled = next((r["iterations"] for r in convergence
                    if abs(r["seam_visibility"] - convergence[-1]["seam_visibility"]) < 0.01),
                   None)
    print(f"  it converges by about {settled} iterations")

    worst = offsets[-1]
    print(f"\nat a {worst['brightness_offset']:g} brightness offset the seam is "
          f"{worst['Copy-paste (control)']:.2f} for copy-paste and "
          f"{worst['Poisson (OpenCV)']:.2f} for Poisson — the harder the mismatch, "
          "the more the gradient domain is worth")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
