"""Run the Canny sensitivity study and write results + figures.

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

from shared import bsds, figures  # noqa: E402
from shared.io import ensure_rgb, to_gray  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import canny_sensitivity as cs  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"

#: Configurations shown in the headline comparison: the textbook default, the
#: best the synthetic scene can find, and the best the photographs can.
GALLERY_CONFIGS = [
    ("textbook (s1.4, 50, 3.0)", 1.4, 50, 3.0),
    ("best on shapes (s1.0, 50, 3.0)", 1.0, 50, 3.0),
    ("best on photos (s2.0, 50, 2.0)", 2.0, 50, 2.0),
    ("no smoothing (s0, 50, 3.0)", 0.0, 50, 3.0),
]


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    print("Sweeping the full parameter grid on the synthetic scene ...")
    grid = cs.full_grid()
    variance = cs.variance_decomposition(grid)
    best_syn = cs.best_configuration(grid)

    print(f"Sweeping a grid on {len(cs.IMAGES)} photographs with human boundaries ...")
    photo = cs.photo_grid()
    ceiling = cs.photo_human_ceiling()
    best_photo = max(photo, key=lambda r: r["f"])

    print("Sweeping noise, tolerance and the precision/recall trade ...")
    noise_rows = cs.sweep_noise()
    tolerance_rows = cs.sweep_tolerance()
    pr_rows = cs.precision_recall_tradeoff()

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    # Rows cross edge density, which is what decides how much there is to detect
    # and how much of it a human bothered to draw.
    EDGE_BANDS = [("bare", 0, 12), ("light", 12, 18),
                  ("busy", 18, 25), ("dense", 25, 100)]

    def edge_density(gray):
        """The selector's own measure: Canny at Otsu-derived thresholds.

        Banding on *this* project's Canny output instead put all twelve
        scenes in one band -- that number is a property of the setting being
        tested, not of the photograph.
        """
        import cv2

        th, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return float((cv2.Canny(gray, 0.5 * th, th) > 0).mean() * 100)
    labels = [c[0] for c in GALLERY_CONFIGS]
    candidates: dict[str, list[dict]] = {}
    for name in cs.IMAGES:
        gray = to_gray(cs.load_scene(name))
        density = edge_density(gray)
        band = next(b for b, lo, hi in EDGE_BANDS if lo <= density < hi)
        target = bsds.consensus_boundaries(name)

        panels = [cs.load_scene(name), ensure_rgb((target * 255).astype(np.uint8))]
        notes, scores = ["photograph", "HUMAN consensus"], []
        for _, sigma, low, ratio in GALLERY_CONFIGS:
            edges = cs.canny(gray, sigma, low, ratio)
            s = bsds.boundary_f_measure(edges, target, cs.TOLERANCE)
            panels.append(ensure_rgb(edges))
            notes.append(f"F {s['f']:.3f}\nP {s['precision']:.2f} R {s['recall']:.2f}")
            scores.append(s["f"])

        print(f"scene candidate {name:22s} edges {density:5.1f}%  [{band:6s}]  "
              f"best config F {max(scores):.3f}")
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
        ["photograph", "HUMAN consensus"] + labels,
        [(r["label"], r["images"]) for r in chosen],
        IMAGES / "compare_canny.png",
        cell_notes=[r["notes"] for r in chosen],
        suptitle=(
            "Canny against what at least two of five human annotators drew. "
            "Two of these settings score a PERFECT 1.000 on the synthetic scene "
            "and differ by 19% here."
        ),
    )
    gallery_table = markdown_table(
        [dict([("Sr", i), ("Scene", r["label"].replace("\n", " · "))]
              + [(lab, f"{s:.3f}") for lab, s in zip(labels, r["scores"])])
         for i, r in enumerate(chosen, start=1)],
        [("Sr", "Sr"), ("Scene", "Scene")] + [(lab, lab) for lab in labels],
    )
    print(f"\nfront-on comparison: {len(chosen)} scenes x {len(labels)} settings")
    print("\n--- canny on photographs ---\n" + gallery_table)

    # ------------------------------------------------------------------ #
    # the signature figure: does tuning on shapes transfer to photographs?
    # ------------------------------------------------------------------ #
    syn_by = {(r["sigma"], r["low"], r["ratio"]): r for r in grid}
    photo_by = {(r["sigma"], r["low"], r["ratio"]): r for r in photo}
    shared_keys = sorted(set(syn_by) & set(photo_by))
    pairs = [(syn_by[k]["f1"], photo_by[k]["f"]) for k in shared_keys]
    order = np.argsort([p[0] for p in pairs])
    figures.lines(
        [pairs[i][0] for i in order],
        {"F on photographs (human boundaries)": [pairs[i][1] for i in order],
         "human ceiling": [ceiling["best"]] * len(order)},
        IMAGES / "transfer.png",
        xlabel="F1 on the synthetic scene",
        ylabel="F against human boundaries",
        title=("Tuning on shapes does not transfer: several settings score a "
               "perfect 1.000 there and differ by 19% on photographs"),
        dashed={"human ceiling"},
    )

    var_params = [k for k in variance if not k.startswith("_")]
    figures.metric_bars(
        var_params, [variance[k]["fraction_of_total"] for k in var_params],
        IMAGES / "variance.png",
        ylabel="fraction of F1 variance explained",
        title="Which Canny parameter actually matters, on the synthetic scene",
    )

    figures.lines(
        [r["noise_sigma"] for r in noise_rows],
        {k: [r[k] for r in noise_rows] for k in noise_rows[0]
         if k not in ("noise_sigma", "best_sigma", "best_low", "best_ratio")},
        IMAGES / "noise.png",
        xlabel="noise sigma",
        ylabel="score",
        title="Canny under noise, and what the best smoothing becomes",
    )

    figures.lines(
        [r["tolerance_px"] for r in tolerance_rows],
        {k: [r[k] for r in tolerance_rows] for k in tolerance_rows[0] if k not in ("tolerance_px", "best_sigma")},
        IMAGES / "tolerance.png",
        xlabel="matching tolerance (px)",
        ylabel="score",
        title="The tolerance is not a detail: it moves the score more than the method does",
    )

    figures.lines(
        [r["low"] for r in pr_rows],
        {"precision": [r["precision"] for r in pr_rows],
         "recall": [r["recall"] for r in pr_rows],
         "F1": [r["f1"] for r in pr_rows]},
        IMAGES / "precision_recall.png",
        xlabel="low threshold",
        ylabel="score",
        title="The threshold is a precision/recall dial, and F1 hides which end you are on",
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS,
        "28_canny_sensitivity",
        {
            "images": list(cs.IMAGES),
            "tolerance_px": cs.TOLERANCE,
            "synthetic_grid": grid,
            "photo_grid": photo,
            "human_ceiling": ceiling,
            "variance_decomposition": variance,
            "best_synthetic": best_syn,
            "best_photo": best_photo,
            "noise_sweep": noise_rows,
            "tolerance_sweep": tolerance_rows,
            "precision_recall": pr_rows,
        },
    )

    photo_table = markdown_table(
        sorted(photo, key=lambda r: -r["f"])[:10],
        [("Sigma", "sigma"), ("Low", "low"), ("Ratio", "ratio"),
         ("F", "f"), ("Precision", "precision"), ("Recall", "recall")],
    )
    variance_table = markdown_table(
        [{"parameter": k, "fraction": variance[k]["fraction_of_total"],
          "best_value": variance[k]["best_value"], "range": variance[k]["range"]}
         for k in var_params],
        [("Parameter", "parameter"), ("Variance explained", "fraction"),
         ("Best value", "best_value"), ("F1 range", "range")],
    )
    write_tables(
        RESULTS,
        [
            ("Best settings against human boundaries", photo_table),
            ("Which parameter matters, on the synthetic scene", variance_table),
            ("Four scenes down the rows", gallery_table),
        ],
    )
    print("\n" + photo_table + "\n\n" + variance_table)

    syn_key = (best_syn["sigma"], best_syn["low"], best_syn["ratio"])
    perfect = [k for k in shared_keys if syn_by[k]["f1"] >= 0.999]
    corr = float(np.corrcoef([p[0] for p in pairs], [p[1] for p in pairs])[0, 1])

    print("\n--- HEADLINE NUMBERS ---")
    print(f"synthetic scene   : best F1 {best_syn['f1']:.4f} at "
          f"sigma {best_syn['sigma']}, low {best_syn['low']}, ratio {best_syn['ratio']}")
    print(f"photographs       : best F {best_photo['f']:.4f} at "
          f"sigma {best_photo['sigma']}, low {best_photo['low']}, ratio {best_photo['ratio']}")
    print(f"human ceiling     : F {ceiling['best']:.4f} (best annotator), "
          f"{ceiling['mean']:.4f} (mean)")
    print(f"  the best setting reaches {best_photo['f'] / ceiling['best']:.0%} of it")
    print(f"settings scoring a PERFECT 1.000 on the synthetic scene: {len(perfect)}")
    if syn_key in photo_by:
        gap = best_photo["f"] - photo_by[syn_key]["f"]
        print(f"  tuning on shapes and applying to photographs costs {gap:+.4f} F "
              f"({gap / max(photo_by[syn_key]['f'], 1e-9):+.0%})")
    print(f"synthetic-to-photo correlation across {len(pairs)} shared settings: {corr:+.3f}")
    print("variance explained: "
          + ", ".join(f"{k} {variance[k]['fraction_of_total']:.3f} "
                      f"(best {variance[k]['best_value']})" for k in var_params))
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
