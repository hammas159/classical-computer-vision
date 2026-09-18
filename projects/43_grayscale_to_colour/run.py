"""Run the colourisation study and write results + figures.

    python run.py

Regenerates `results/results.json`, `results/tables.md` and every figure in
`docs/images/`. Every number in the README comes out of this script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import numpy as np  # noqa: E402

from shared import figures  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import colourise as co  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()
    del args

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    print(f"{len(co.IMAGES)} photographs, spread across greyscale ambiguity")
    ceiling = co.axis_predicts_the_ceiling()
    residuals = co.oracle_residual()
    for r in residuals:
        print(f"  {r['image']:26s} ambiguity {r['ambiguity']:6.2f}  "
              f"oracle still wrong by {r['oracle_chroma_error']:6.2f}")

    print("\nEvery method on all twelve ...")
    overall = co.evaluate()
    for r in overall:
        tag = "  (oracle)" if r["is_oracle"] else ""
        print(f"  {r['method']:32s} chroma {r['chroma_error']:7.3f}  "
              f"PSNR {r['rgb_psnr_db']:7.3f} dB  "
              f"colourfulness {r['colourfulness']:6.2f}{tag}")

    rows_per_image = co.per_image()
    comparison = co.metric_comparison()
    luminance = co.psnr_is_luminance()
    scribbles = co.sweep_scribbles()
    references = co.reference_sensitivity()

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    by_image = {r["image"]: r for r in rows_per_image}
    chosen = [co.IMAGES[i] for i in (0, 3, 7, 11)]

    truths = {n: co.load(n) for n in chosen}
    greys = {n: co.greyscale(truths[n]) for n in chosen}

    rows = [("the original (truth)", [truths[n] for n in chosen]),
            ("the input: no chroma", [greys[n] for n in chosen])]
    notes = [[f"ambiguity {by_image[n]['ambiguity']:.1f}" for n in chosen],
             [f"{by_image[n]['Do nothing (grey, control)']:.1f} chroma error"
              for n in chosen]]

    table_rows = [{"Sr": sr, "Photograph": n.replace("_", " "),
                   "Ambiguity": f"{by_image[n]['ambiguity']:.1f}"}
                  for sr, n in enumerate(chosen, start=1)]

    for method in co.METHODS:
        if method == "Do nothing (grey, control)":
            continue
        panels, cell = [], []
        for pos, n in enumerate(chosen):
            _, prediction, _ = co.colourise(n, method)
            panels.append(prediction)
            cell.append(f"{by_image[n][method]:.1f}")
            table_rows[pos][method] = f"{by_image[n][method]:.1f}"
        rows.append((method, panels))
        notes.append(cell)
        print(f"  {method:32s} {cell}")

    figures.gallery(
        [n.replace("_", " ") for n in chosen], rows,
        IMAGES / "compare_colourisation.png",
        cell_notes=notes,
        suptitle=("Four of the twelve, in order of greyscale ambiguity. Row two is "
                  "what every method is given. Cells are mean chroma error in Lab "
                  "units — lower is better, and the grey input scores 18.7 on "
                  "average."),
    )
    for pos, n in enumerate(chosen):
        table_rows[pos]["Do nothing (grey)"] = \
            f"{by_image[n]['Do nothing (grey, control)']:.1f}"
    gallery_table = markdown_table(
        table_rows,
        [("Sr", "Sr"), ("Photograph", "Photograph"), ("Ambiguity", "Ambiguity"),
         ("Do nothing (grey)", "Do nothing (grey)")]
        + [(m, m) for m in co.METHODS if m != "Do nothing (grey, control)"])
    print("\n--- chroma error per photograph ---\n" + gallery_table)

    # ------------------------------------------------------------------ #
    # the signature figure
    # ------------------------------------------------------------------ #
    figures.scatter_plane(
        {"the twelve photographs": [(r["ambiguity"], r["oracle_chroma_error"])
                                    for r in residuals]},
        target=None,
        out_path=IMAGES / "axis_predicts_ceiling.png",
        xlabel="greyscale ambiguity (measured before anything ran)",
        ylabel="what an oracle handed this image's own mapping still gets wrong",
        title=(f"r = {ceiling['pearson_r']:.4f}. The ceiling on any "
               "luminance-based colouriser was knowable in advance."),
    )

    control = next(r for r in overall
                   if r["method"] == "Do nothing (grey, control)")
    figures.metric_bars(
        [r["method"] for r in overall],
        [r["chroma_error"] for r in overall],
        IMAGES / "chroma_error.png",
        ylabel="mean chroma error (Lab units)",
        title=(f"Two of the four colourisers score worse than returning the grey "
               f"image ({control['chroma_error']:.1f})."),
        highlight_best="min",
    )

    figures.metric_bars(
        [r["method"] for r in overall],
        [r["rgb_psnr_db"] for r in overall],
        IMAGES / "psnr.png",
        ylabel="RGB PSNR (dB)",
        title=("The same six by PSNR. Same order — but a picture with no colour "
               "at all scores 22 dB."),
    )

    figures.metric_bars(
        [r["image"].replace("_", " ") for r in luminance] * 2,
        [r["grey_only_psnr"] for r in luminance]
        + [r["chroma_only_psnr"] for r in luminance],
        IMAGES / "psnr_is_luminance.png",
        ylabel="RGB PSNR (dB)",
        title=("Left twelve: correct luminance, no colour. Right twelve: correct "
               "colour, flattened luminance. PSNR is measuring luminance."),
    )

    figures.lines(
        [r["scribbles"] for r in scribbles],
        {"Levin scribble propagation": [r["chroma_error"] for r in scribbles]},
        IMAGES / "scribbles.png",
        xlabel="scribbles given (each 7x7 px of true colour)",
        ylabel="mean chroma error", logx=True,
        title=("Forty user marks beat knowing the image's entire true "
               "luminance-to-colour mapping."),
    )

    figures.metric_bars(
        [r["image"].replace("_", " ") for r in references],
        [r["spread"] for r in references],
        IMAGES / "reference_spread.png",
        ylabel="worst reference − best reference (chroma error)",
        title=("How much of Welsh transfer's score is the reference photograph. "
               "The method itself is worth 6.6."),
    )

    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS, "43_grayscale_to_colour",
        {
            "images": list(co.IMAGES),
            "overall": overall,
            "per_image": rows_per_image,
            "metric_comparison": comparison,
            "axis_predicts_ceiling": ceiling,
            "oracle_residual": residuals,
            "psnr_is_luminance": luminance,
            "scribble_sweep": scribbles,
            "reference_sensitivity": references,
        },
    )

    overall_table = markdown_table(
        overall, [("Method", "method"), ("Chroma error", "chroma_error"),
                  ("RGB PSNR (dB)", "rgb_psnr_db"),
                  ("Colourfulness", "colourfulness"), ("Oracle?", "is_oracle")])
    residual_table = markdown_table(
        residuals, [("Photograph", "image"), ("Ambiguity", "ambiguity"),
                    ("Oracle chroma error", "oracle_chroma_error")])
    luminance_table = markdown_table(
        luminance, [("Photograph", "image"),
                    ("PSNR, luminance only (dB)", "grey_only_psnr"),
                    ("PSNR, chroma only (dB)", "chroma_only_psnr"),
                    ("Chroma error, luminance only", "grey_only_chroma_error"),
                    ("Chroma error, chroma only", "chroma_only_chroma_error")])
    scribble_table = markdown_table(
        scribbles, [("Scribbles", "scribbles"), ("Chroma error", "chroma_error")])
    reference_table = markdown_table(
        references, [("Photograph", "image"), ("Rule reference", "rule_reference"),
                     ("With rule reference", "with_rule_reference"),
                     ("Best reference", "best_reference"), ("Best", "best"),
                     ("Worst", "worst"), ("Spread", "spread"),
                     ("Grey control", "grey_control")])

    write_tables(
        RESULTS,
        [
            ("Every method on all twelve", overall_table),
            ("The selection axis against the oracle's residual", residual_table),
            ("What PSNR is actually measuring", luminance_table),
            ("How many scribbles Levin's method needs", scribble_table),
            ("Welsh transfer with every other photograph as reference",
             reference_table),
            ("Four photographs down the rows", gallery_table),
        ],
    )

    # ------------------------------------------------------------------ #
    print("\n--- HEADLINE NUMBERS ---")
    print(f"the selection axis predicts the ceiling: r = {ceiling['pearson_r']:.5f} "
          f"between greyscale ambiguity and what an oracle handed this image's own "
          f"luminance-to-colour mapping still gets wrong")
    print(f"  ambiguity {ceiling['ambiguity_range'][0]:.2f} to "
          f"{ceiling['ambiguity_range'][1]:.2f}; oracle error "
          f"{ceiling['oracle_error_range'][0]:.2f} to "
          f"{ceiling['oracle_error_range'][1]:.2f}; slope {ceiling['slope']:.3f}")
    print("  the hardest photograph was identified before a single method ran")

    worse = comparison["methods_worse_than_doing_nothing"]
    print(f"\n{len(worse)} of the methods score worse than returning the grey image "
          f"({control['chroma_error']:.2f}): {', '.join(worse)}")
    for name in worse:
        r = next(x for x in overall if x["method"] == name)
        print(f"  {name:32s} {r['chroma_error']:7.2f}  "
              f"(colourfulness {r['colourfulness']:.1f} against the grey image's "
              f"{control['colourfulness']:.1f})")
    print("  they add a great deal of colour and it is the wrong colour")

    print(f"\nPSNR ranks the six identically to the chroma error "
          f"({'agrees' if comparison['rankings_agree'] else 'DISAGREES'}) — the usual "
          "complaint about it is not what is wrong here")
    grey_psnr = float(np.mean([r["grey_only_psnr"] for r in luminance]))
    chroma_psnr = float(np.mean([r["chroma_only_psnr"] for r in luminance]))
    print(f"  what is wrong is the scale: correct luminance with NO colour scores "
          f"{grey_psnr:.1f} dB, correct colour with flattened luminance "
          f"{chroma_psnr:.1f} dB")
    print(f"  a {grey_psnr - chroma_psnr:.1f} dB gap for the same information, "
          "depending which half of the image it is")

    lookup = next(r for r in overall if r["method"] == "Luminance lookup (oracle)")
    forty = next(r for r in scribbles if r["scribbles"] == 40)
    sixteen = next(r for r in scribbles if r["scribbles"] == 16)
    print(f"\nforty scribbles ({forty['chroma_error']:.2f}) beat knowing the image's "
          f"entire true luminance-to-colour mapping ({lookup['chroma_error']:.2f})")
    print(f"  and sixteen ({sixteen['chroma_error']:.2f}) nearly do. Where the "
          "colour is beats what the colour is.")

    spread = float(np.mean([r["spread"] for r in references]))
    welsh = next(r for r in overall if r["method"] == "Welsh transfer (reference)")
    gap = welsh["chroma_error"] - control["chroma_error"]
    rule_worse = sum(1 for r in references
                     if r["with_rule_reference"] > r["grey_control"])
    best_worse = sum(1 for r in references if r["best"] > r["grey_control"])
    print(f"\nWelsh transfer's score is the reference, not the method: the spread "
          f"across the eleven possible references averages {spread:.1f} chroma units")
    print(f"  the whole difference between running the method and doing nothing is "
          f"{abs(gap):.1f} — so the choice of reference matters "
          f"{spread / abs(gap):.0f}x more than the method")
    print(f"  on {rule_worse} of 12 the rule reference is worse than doing nothing, "
          f"and on {best_worse} of 12 the BEST of eleven references still is")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
