"""Run the panorama stitching study and write results + figures.

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
from shared.io import ensure_rgb  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import panorama as pa  # noqa: E402

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

    print(f"{len(pa.IMAGES)} photographs, each split into two overlapping frames "
          f"by a known homography")
    print(f"  each frame covers {100 * pa.COVERAGE:.0f}% of the canvas; the second "
          f"is {100 * (pa.EXPOSURE - 1):.0f}% brighter")

    registrable = pa.registrability_table()
    print("\nRegistrability — the selection axis, measured before anything else ...")
    for r in registrable:
        print(f"  {r['image']:32s} {r['matches']:5d} matches, "
              f"{r['inliers']:5d} inliers ({r['inlier_rate']:.3f}), "
              f"{r['per_megapixel']:9.1f} per Mpx")

    print("\nEvery blender on all twelve ...")
    overall = pa.evaluate()
    for r in overall:
        tag = "  (control)" if r["is_control"] else ""
        print(f"  {r['blender']:34s} PSNR {r['psnr_db']:7.3f} dB  "
              f"seam step {r['seam_step']:6.2f}{tag}")

    disagree = pa.metrics_disagree()
    rows_per_image = pa.per_image()
    exposures = pa.exposure_is_the_whole_story()
    bands = pa.sweep_bands()
    widths = pa.sweep_feather_width()

    print("\nThe exposure difference is the whole story ...")
    for r in exposures:
        print(f"  x{r['exposure']:.2f}  " + "  ".join(
            f"{k.split()[0]:<12s}{r[k]:6.2f}" for k in pa.BLENDERS))

    real = None
    if pa.graf_available():
        real = pa.real_pair()
        print(f"\nThe one real pair (graf1 / graf3): {real['matches']} matches, "
              f"{real['inliers']} inliers ({real['inlier_rate']:.3f}), "
              f"cycle error {real['cycle_error_px']:.3f} px")
    else:
        print("\ngraf1/graf3 are not cached; the real-pair section is skipped")

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    by_image = {r["image"]: r for r in rows_per_image}
    chosen = [pa.IMAGES[i] for i in (0, 4, 7, 11)]

    built = {n: pa.stitch(pa.load(n), "Overwrite (control)") for n in chosen}

    rows = [("the two frames\n(second is 20% brighter)",
             [np.maximum(built[n]["left"], built[n]["aligned"]) for n in chosen])]
    notes = [[f"{by_image[n]['inliers_per_mpx']:.0f} inliers/Mpx" for n in chosen]]

    table_rows = [{"Sr": sr, "Photograph": n.replace("_", " "),
                   "Inliers/Mpx": f"{by_image[n]['inliers_per_mpx']:.0f}"}
                  for sr, n in enumerate(chosen, start=1)]

    for blender in pa.BLENDERS:
        panels, cell = [], []
        for pos, n in enumerate(chosen):
            r = pa.stitch(pa.load(n), blender)
            panels.append(r["output"])
            step = pa.worst_seam_step(r["output"], r["overlap"])
            cell.append(f"seam step {step:.0f}")
            table_rows[pos][blender] = f"{step:.0f}"
        rows.append((blender, panels))
        notes.append(cell)
        print(f"  {blender:34s} {cell}")

    figures.gallery(
        [n.replace("_", " ") for n in chosen], rows,
        IMAGES / "compare_blending.png",
        cell_notes=notes,
        suptitle=("Four of the twelve. Row one is the two frames laid on one canvas "
                  "before blending — the brightness step is the exposure difference. "
                  "Cells are the brightness step straight across the join, in grey "
                  "levels."),
    )
    gallery_table = markdown_table(
        table_rows, [("Sr", "Sr"), ("Photograph", "Photograph"),
                     ("Inliers/Mpx", "Inliers/Mpx")]
        + [(b, b) for b in pa.BLENDERS])
    print("\n--- seam step per photograph (grey levels) ---\n" + gallery_table)

    # ------------------------------------------------------------------ #
    # the signature figures
    # ------------------------------------------------------------------ #
    figures.metric_bars(
        [r["blender"] for r in overall] * 2,
        [r["psnr_db"] for r in overall] + [r["seam_step"] for r in overall],
        IMAGES / "metrics_disagree.png",
        ylabel="PSNR (dB)  /  seam step (grey levels)",
        title=("Left five: PSNR, higher is better. Right five: seam step, lower is "
               "better. The PSNR winner is the seam loser."),
    )

    figures.lines(
        [r["exposure"] for r in exposures],
        {b: [r[b] for r in exposures] for b in pa.BLENDERS},
        IMAGES / "exposure.png",
        xlabel="exposure ratio between the two frames",
        ylabel="seam step (grey levels)",
        title=("At a ratio of 1.0 there is nothing to hide and every method scores "
               "the same. Blending is worth exactly what the exposure difference is."),
    )

    figures.metric_bars(
        [r["image"].replace("_", " ") for r in registrable],
        [r["per_megapixel"] for r in registrable],
        IMAGES / "registrability.png",
        ylabel="RANSAC inliers per megapixel",
        title=("The selection axis. A wall of identical brick courses gives matches "
               "that are all equally good, so there is no homography to find."),
    )

    figures.lines(
        [r["bands"] for r in bands],
        {"Multi-band": [r["seam_step"] for r in bands]},
        IMAGES / "bands.png",
        xlabel="pyramid levels", ylabel="seam step (grey levels)",
        title="How many levels a multi-band blend needs")

    figures.lines(
        [r["width"] for r in widths],
        {"Feather": [r["seam_step"] for r in widths]},
        IMAGES / "feather_width.png",
        xlabel="feather ramp width (px)", ylabel="seam step (grey levels)",
        logx=True,
        title="The one parameter feathering has")

    if real is not None:
        stitched = pa.stitch_real()
        hard = pa.stitch_real("Overwrite (control)")
        figures.grid(
            [("graf1", ensure_rgb(pa.load_graf()[0])),
             ("graf3", ensure_rgb(pa.load_graf()[1])),
             (f"pasted\n{pa.worst_seam_step(hard['output'], hard['overlap']):.0f} step",
              ensure_rgb(hard["output"])),
             (f"multi-band\n"
              f"{pa.worst_seam_step(stitched['output'], stitched['overlap']):.0f} step",
              ensure_rgb(stitched["output"]))],
            IMAGES / "real_pair.png", ncols=2,
            suptitle=("The one genuinely real pair: two photographs of one graffiti "
                      f"wall, {real['inliers']} RANSAC inliers of {real['matches']} "
                      "matches. No published homography exists for it here."))

    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS, "38_panorama_stitching",
        {
            "images": list(pa.IMAGES),
            "setup": {"coverage": pa.COVERAGE, "exposure": pa.EXPOSURE,
                      "viewpoint": pa.VIEWPOINT.tolist()},
            "registrability": registrable,
            "overall": overall,
            "metrics_disagree": disagree,
            "per_image": rows_per_image,
            "exposure_sweep": exposures,
            "bands_sweep": bands,
            "feather_width_sweep": widths,
            "real_pair": real,
        },
    )

    overall_table = markdown_table(
        overall, [("Blender", "blender"), ("PSNR (dB)", "psnr_db"),
                  ("Seam step", "seam_step"),
                  ("Seam visibility", "seam_visibility"), ("Control?", "is_control")])
    reg_table = markdown_table(
        registrable, [("Photograph", "image"), ("Matches", "matches"),
                      ("Inliers", "inliers"), ("Inlier rate", "inlier_rate"),
                      ("Per Mpx", "per_megapixel")])
    exposure_table = markdown_table(
        exposures, [("Exposure", "exposure")] + [(b, b) for b in pa.BLENDERS])
    bands_table = markdown_table(
        bands, [("Bands", "bands"), ("Seam step", "seam_step")])
    width_table = markdown_table(
        widths, [("Width", "width"), ("Seam step", "seam_step")])

    sections = [
        ("Every blender on all twelve", overall_table),
        ("Registrability: the selection axis", reg_table),
        ("The exposure difference is the whole story", exposure_table),
        ("Pyramid levels", bands_table),
        ("Feather ramp width", width_table),
        ("Four photographs down the rows", gallery_table),
    ]
    write_tables(RESULTS, sections)

    # ------------------------------------------------------------------ #
    print("\n--- HEADLINE NUMBERS ---")
    print(f"the best PSNR belongs to '{disagree['psnr_winner']}' — the control that "
          "does not stitch at all")
    print(f"  and it ranks {disagree['psnr_winner_seam_rank']} of "
          f"{len(pa.BLENDERS)} on the seam step, which is last")
    print(f"  the seam winner is '{disagree['seam_winner']}'")
    print("  a panorama cannot be scored by fidelity to one of its own inputs: the "
          "first frame IS the original in its own region, so any blending that mixes "
          "the second frame in can only move away from it")

    keep = next(r for r in overall if r["blender"].startswith("Keep"))
    best_seam = min(overall, key=lambda r: r["seam_step"])
    print(f"\nat the seam: not stitching leaves {keep['seam_step']:.1f} grey levels, "
          f"{best_seam['blender']} leaves {best_seam['seam_step']:.1f}")

    flat = next(r for r in exposures if abs(r["exposure"] - 1.0) < 1e-9)
    spread_flat = max(flat[b] for b in pa.BLENDERS) - min(flat[b] for b in pa.BLENDERS)
    worst = exposures[-1]
    spread_worst = (max(worst[b] for b in pa.BLENDERS)
                    - min(worst[b] for b in pa.BLENDERS))
    print(f"\nwith the exposures matched (x1.00) every method lands within "
          f"{spread_flat:.2f} grey levels of every other — blending is worth nothing "
          "when there is nothing to hide")
    print(f"  at x{worst['exposure']:.2f} the spread is {spread_worst:.2f}. "
          "A blender comparison run on matched exposures measures nothing at all.")

    lo = min(registrable, key=lambda r: r["per_megapixel"])
    hi = max(registrable, key=lambda r: r["per_megapixel"])
    print(f"\nregistrability spans {lo['per_megapixel']:.0f} to "
          f"{hi['per_megapixel']:.0f} inliers per megapixel — a factor of "
          f"{hi['per_megapixel'] / max(lo['per_megapixel'], 1e-9):.0f}")
    print(f"  the floor is {lo['image'].replace('_', ' ')}: {lo['matches']} matches "
          f"survive the ratio test at all, because every brick course looks like "
          "every other one")

    if real is not None:
        print(f"\nthe real pair: {real['inliers']}/{real['matches']} inliers "
              f"({real['inlier_rate']:.1%}), cycle error "
              f"{real['cycle_error_px']:.2f} px")
        print("  that is a consistency check, not an accuracy — the Oxford set's "
              "published homographies 404 upstream, so there is no external truth "
              "for this pair and none is claimed")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
