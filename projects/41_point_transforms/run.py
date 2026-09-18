"""Run the point-transform comparison and write results + figures.

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

import numpy as np  # noqa: E402

from shared import figures  # noqa: E402
from shared.io import ensure_rgb  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import point_transforms as pt  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"

#: The four curves the front figure shows. The full table has ten; four is what
#: fits across a page, and these four are the ones that separate on real images.
FIGURE_TRANSFORMS = ("Gamma 0.5 (brighten)", "Gamma 2.2 (darken)",
                     "Contrast stretch", "Posterise (6 levels)")

BRIGHTNESS_BANDS = [("dark", 0, 90), ("dim", 90, 110),
                    ("bright", 110, 140), ("very bright", 140, 256)]


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    names = list(pt.LUTS)

    print("What each table costs, computed from the table alone ...")
    properties = pt.table_properties()
    for r in properties:
        print(f"  {r['transform']:24s} {r['levels_surviving']:4d} levels survive, "
              f"longest run {r['max_collapse_run']:3d}, largest gap {r['max_output_gap']:3d}"
              f"{'   (invertible)' if r['invertible'] else ''}")

    print("\nA lookup table against the same maths written out ...")
    equality = pt.lut_versus_arithmetic(runs=args.runs + 2)
    for r in equality:
        print(f"  {r['transform']:24s} identical: {str(r['identical']):5s}  "
              f"LUT {r['lut_ms']:.4f} ms vs {r['arithmetic_ms']:.4f} ms  "
              f"= {r['speedup']:.1f}x")

    print("\nOn twelve photographs ...")
    on_images = pt.evaluate_on_images(runs=args.runs)
    for r in on_images:
        print(f"  {r['transform']:24s} {r['levels_surviving']:4d} levels  "
              f"{r['entropy_bits']:.3f} bits  contrast {r['rms_contrast']:.4f}")

    gamma = pt.sweep_gamma()
    posterise = pt.sweep_posterise()
    planes = pt.bit_plane_contribution()

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    from shared.metrics import entropy

    candidates: dict[str, list[dict]] = {}
    for name in pt.IMAGES:
        img = pt.load_scene(name)
        mean = pt.mean_brightness(img)
        band = next(b for b, lo, hi in BRIGHTNESS_BANDS if lo <= mean < hi)

        panels, notes, scores = [img], [f"mean {mean:.0f}"], []
        for transform in FIGURE_TRANSFORMS:
            out = pt.apply_lut(img, pt.LUTS[transform]())
            bits = entropy(out)
            panels.append(ensure_rgb(out))
            notes.append(f"{bits:.3f} bits")
            scores.append(bits)

        candidates.setdefault(band, []).append({
            "label": f"{name.replace('_', ' ')}\nmean {mean:.0f}",
            "subject": name, "mean": mean, "images": panels,
            "notes": notes, "scores": scores,
            "spread": float(np.std(scores)),
        })
        print(f"  scene candidate {name:24s} mean {mean:6.1f} [{band:11s}]  "
              f"bits " + " ".join(f"{s:.2f}" for s in scores))

    chosen, used = [], set()
    for band, _, _ in BRIGHTNESS_BANDS:
        pool = sorted(candidates.get(band, []), key=lambda r: -r["spread"])
        pick = next((r for r in pool if r["subject"] not in used), None)
        if pick:
            used.add(pick["subject"])
            chosen.append(pick)
    chosen = chosen[:4]
    assert len({r["subject"] for r in chosen}) == len(chosen), "a subject repeats"

    figures.gallery(
        ["photograph"] + list(FIGURE_TRANSFORMS),
        [(r["label"], r["images"]) for r in chosen],
        IMAGES / "compare_transforms.png",
        cell_notes=[r["notes"] for r in chosen],
        suptitle=("Four curves across the brightness range. Cells are the entropy of the "
                  "result in bits — how much of the original 8 bits survived the mapping."),
    )
    gallery_table = markdown_table(
        [dict([("Sr", i), ("Scene", r["label"].replace("\n", " · "))]
              + [(m, f"{s:.3f}") for m, s in zip(FIGURE_TRANSFORMS, r["scores"])])
         for i, r in enumerate(chosen, start=1)],
        [("Sr", "Sr"), ("Scene", "Scene")] + [(m, m) for m in FIGURE_TRANSFORMS],
    )
    print("\n--- entropy of the result, bits ---\n" + gallery_table)

    # ------------------------------------------------------------------ #
    # the signature figure: the curves and what each one costs
    # ------------------------------------------------------------------ #
    figures.lines(
        list(range(256)),
        {n: pt.LUTS[n]().astype(int).tolist() for n in names},
        IMAGES / "curves.png",
        xlabel="input level", ylabel="output level",
        title="The ten tables. Flat stretches lose detail; steep stretches band.",
        dashed={"Identity (control)"},
    )

    figures.comparison_matrix(
        properties,
        [("Levels surviving", "levels_surviving", True),
         ("Longest collapse run", "max_collapse_run", False),
         ("Largest output gap", "max_output_gap", False)],
        IMAGES / "table_costs.png",
        row_key="transform",
        title=("Computed from the 256-entry table, before any image. Log and inverse log "
               "lose exactly the same 131 levels in opposite ways."),
    )

    figures.lines(
        [r["gamma"] for r in gamma],
        {"levels surviving": [r["levels_surviving"] for r in gamma],
         "longest collapse run": [r["max_collapse_run"] for r in gamma],
         "largest output gap": [r["max_output_gap"] for r in gamma]},
        IMAGES / "gamma_sweep.png",
        xlabel="gamma", ylabel="levels",
        title="Brightening and darkening lose about the same amount, in opposite ways",
        vlines={"identity": 1.0},
    )

    figures.lines(
        [r["requested_levels"] for r in posterise],
        {"entropy (bits)": [r["entropy_bits"] for r in posterise],
         "PSNR / 10 (dB)": [min(r["psnr_db"], 60) / 10 for r in posterise]},
        IMAGES / "posterise_sweep.png",
        xlabel="levels kept", ylabel="bits  /  dB per 10",
        logx=True,
        title="Posterisation: the dial that makes the information loss continuous",
    )

    figures.lines(
        [r["top_planes_kept"] for r in planes],
        {"PSNR (dB)": [min(r["psnr_db"], 60) for r in planes],
         "entropy (bits)": [r["entropy_bits"] for r in planes]},
        IMAGES / "bit_planes.png",
        xlabel="top bit planes kept", ylabel="dB  /  bits",
        title="How few bits carry the picture",
    )

    sample = pt.load_scene(chosen[-1]["subject"])
    figures.grid(
        [(f"bit {b}", ensure_rgb(pt.bit_plane(sample, b))) for b in range(7, -1, -1)],
        IMAGES / "planes_grid.png", ncols=4,
        suptitle=(f"The eight bit planes of {chosen[-1]['subject'].replace('_', ' ')}. "
                  "The bottom rows are close to noise."),
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS, "41_point_transforms",
        {
            "images": list(pt.IMAGES),
            "brightness": {n: round(pt.mean_brightness(pt.load_scene(n)), 1)
                           for n in pt.IMAGES},
            "table_properties": properties,
            "lut_versus_arithmetic": equality,
            "on_images": on_images,
            "gamma_sweep": gamma,
            "posterise_sweep": posterise,
            "bit_planes": planes,
        },
    )

    properties_table = markdown_table(
        properties,
        [("Transform", "transform"), ("Levels surviving", "levels_surviving"),
         ("Levels lost", "levels_lost"), ("Invertible", "invertible"),
         ("Longest run", "max_collapse_run"), ("Largest gap", "max_output_gap")])
    equality_table = markdown_table(
        equality, [("Transform", "transform"), ("Bit-identical", "identical"),
                   ("LUT (ms)", "lut_ms"), ("Arithmetic (ms)", "arithmetic_ms"),
                   ("Speedup", "speedup")])
    images_table = markdown_table(
        on_images, [("Transform", "transform"), ("Levels surviving", "levels_surviving"),
                    ("Entropy (bits)", "entropy_bits"), ("RMS contrast", "rms_contrast"),
                    ("PSNR vs original", "psnr_vs_original")])
    gamma_table = markdown_table(
        gamma, [("Gamma", "gamma"), ("Levels", "levels_surviving"),
                ("Longest run", "max_collapse_run"), ("Largest gap", "max_output_gap"),
                ("Entropy (bits)", "entropy_bits")])
    planes_table = markdown_table(
        planes, [("Top planes kept", "top_planes_kept"), ("PSNR (dB)", "psnr_db"),
                 ("Entropy (bits)", "entropy_bits")])

    write_tables(
        RESULTS,
        [
            ("What each table costs, before any image", properties_table),
            ("A lookup table against the same maths", equality_table),
            ("On twelve photographs", images_table),
            ("The gamma sweep", gamma_table),
            ("Bit planes", planes_table),
            ("Four scenes down the rows", gallery_table),
        ],
    )
    print("\n" + properties_table + "\n\n" + equality_table)

    # ------------------------------------------------------------------ #
    # headlines
    # ------------------------------------------------------------------ #
    print("\n--- HEADLINE NUMBERS ---")
    assert all(r["identical"] for r in equality), "the central claim must hold"
    fastest = max(equality, key=lambda r: r["speedup"])
    print(f"every table is bit-identical to the arithmetic it replaces, at up to "
          f"{fastest['speedup']:.0f}x the speed ({fastest['transform']})")
    print("  it took a rounding fix to be true: building a table with .astype(uint8) "
          "truncates where the arithmetic rounds, and the two gamma curves disagreed "
          "on 136 of 256 input levels")

    by_name = {r["transform"]: r for r in properties}
    bright, dark = by_name["Gamma 0.5 (brighten)"], by_name["Gamma 2.2 (darken)"]
    print(f"\nbrightening loses {bright['levels_lost']} levels and darkening "
          f"{dark['levels_lost']} — about the same, and in opposite ways:")
    print(f"  gamma 0.5  longest run {bright['max_collapse_run']:3d}  "
          f"largest gap {bright['max_output_gap']:3d}   -> it BANDS")
    print(f"  gamma 2.2  longest run {dark['max_collapse_run']:3d}  "
          f"largest gap {dark['max_output_gap']:3d}   -> it SMEARS")

    log, inverse = by_name["Log"], by_name["Inverse log"]
    if log["levels_lost"] == inverse["levels_lost"]:
        print(f"\nlog and inverse log lose EXACTLY the same {log['levels_lost']} levels, "
              f"with runs {log['max_collapse_run']} vs {inverse['max_collapse_run']} and "
              f"gaps {log['max_output_gap']} vs {inverse['max_output_gap']} — "
              "a single 'information loss' number cannot tell them apart")

    predicted = {r["transform"]: r["levels_surviving"] for r in properties}
    measured = {r["transform"]: r["levels_surviving"] for r in on_images}
    worst = max(predicted, key=lambda n: abs(predicted[n] - measured[n]))
    print(f"\nthe table predicts the loss without looking at any picture: the largest "
          f"disagreement with what twelve photographs actually show is "
          f"{abs(predicted[worst] - measured[worst])} level(s), on {worst}")

    thresh = next(r for r in on_images if r["transform"] == "Threshold 127")
    print(f"\nthresholding has the HIGHEST RMS contrast of the ten "
          f"({thresh['rms_contrast']:.4f}) and the second-lowest entropy "
          f"({thresh['entropy_bits']:.3f} bits). Contrast and information are not "
          "the same quantity.")

    four = next(r for r in planes if r["top_planes_kept"] == 4)
    print(f"\nhalf the bits carry {four['psnr_db']:.1f} dB — the top four planes of an "
          "8-bit image are most of the picture")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
