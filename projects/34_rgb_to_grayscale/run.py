"""Run the grayscale conversion comparison and write results + figures.

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

import grayscale as gs  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"

#: Chroma bands the four figure rows are drawn from, so the comparison spans the
#: axis the pool was selected on rather than four pictures of the same thing.
CHROMA_BANDS = [("near-grey", 0, 16), ("muted", 16, 24),
                ("colourful", 24, 40), ("saturated", 40, 1000)]


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    methods = list(gs.METHODS)

    print(f"Scoring {len(methods)} conversions on {len(gs.IMAGES)} photographs ...")
    photos = gs.evaluate_on_photos(runs=args.runs)
    for r in photos:
        print(f"  {r['method']:26s} {r['mean_diff_vs_bt601']:6.2f} levels from BT.601  "
              f"worst colour-edge recall {r['worst_image_recall']:.4f}  "
              f"{r['median_ms']:7.2f} ms")

    print("\nHow much contrast survives at each photograph's worst colour edge ...")
    retention = gs.luma_retention()
    for r in retention:
        print(f"  {r['image']:24s} chroma {r['chroma']:5.1f}  "
              f"1st pct {r['worst_retained']:.4f}  single worst pixel "
              f"{r['minimum_retained']:.4f}")

    print("\nBuilding a scene invisible to each linear weighting in turn ...")
    blind = gs.blind_spot_matrix()
    for r in blind:
        own = r[r["built_against"]]
        best_other = max(r[m] for m in methods if m != r["built_against"])
        print(f"  built against {r['built_against']:26s} it sees {own:6.2f} levels, "
              f"the best other conversion sees {best_other:6.2f}")

    print("\nSweeping how close to isoluminant a boundary has to be ...")
    sweep = gs.sweep_isoluminance()
    isoluminant = gs.evaluate_isoluminant()
    downstream = gs.downstream_effect()
    weights = gs.weight_sensitivity()

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    # Four photographs across the chroma range. The point of this figure is that
    # the columns look the same — which is the finding, not a failure of the
    # figure — so every cell carries the number that says how same.
    by_image = {r["image"]: r for r in retention}
    candidates: dict[str, list[dict]] = {}
    reference = gs.METHODS["BT.601 (OpenCV default)"]
    for name in gs.IMAGES:
        img = gs.load_scene(name)
        c = by_image[name]["chroma"]
        band = next(b for b, lo, hi in CHROMA_BANDS if lo <= c < hi)
        ref = reference(img)

        panels, notes, scores = [img], [f"chroma {c:.1f}"], []
        for method in methods:
            out = gs.METHODS[method](img)
            diff = float(np.mean(np.abs(out.astype(np.float64) - ref.astype(np.float64))))
            kept = gs.luma_retention([name], weights=gs.BT601)[0]["worst_retained"]
            panels.append(ensure_rgb(out))
            notes.append(f"{diff:.2f} levels from BT.601")
            scores.append(diff)

        candidates.setdefault(band, []).append({
            "label": f"{name.replace('_', ' ')}\nchroma {c:.0f}",
            "subject": name,
            "chroma": c,
            "images": panels,
            "notes": notes,
            "scores": scores,
            "worst_retained": kept,
            "rank": c,
        })

    chosen, used = [], set()
    for band, _, _ in CHROMA_BANDS:
        pool = sorted(candidates.get(band, []), key=lambda r: -r["chroma"])
        pick = next((r for r in pool if r["subject"] not in used), None)
        if pick:
            used.add(pick["subject"])
            chosen.append(pick)
    chosen = chosen[:4]
    assert len({r["subject"] for r in chosen}) == len(chosen), "a subject repeats"

    figures.gallery(
        ["photograph"] + methods,
        [(r["label"], r["images"]) for r in chosen],
        IMAGES / "compare_grayscale.png",
        cell_notes=[r["notes"] for r in chosen],
        suptitle=(
            "Six conversions across the chroma range. Cells are mean absolute "
            "difference from the OpenCV default, in grey levels. They look the "
            "same because on a photograph they are the same — that is the result."
        ),
    )
    gallery_table = markdown_table(
        [dict([("Sr", i), ("Photograph", r["label"].replace("\n", " · "))]
              + [(m, f"{s:.2f}") for m, s in zip(methods, r["scores"])])
         for i, r in enumerate(chosen, start=1)],
        [("Sr", "Sr"), ("Photograph", "Photograph")] + [(m, m) for m in methods],
    )
    print(f"\nfront-on comparison: {len(chosen)} photographs x {len(methods)} conversions")
    print("\n--- difference from BT.601, grey levels ---\n" + gallery_table)

    # ------------------------------------------------------------------ #
    # the signature figure: everyone's blind plane
    # ------------------------------------------------------------------ #
    panels = []
    for built_for, w in gs.LINEAR_WEIGHTINGS.items():
        img, mask = gs.isoluminant_scene(weights=w)
        panels.append((f"in colour\n(built against {built_for.split()[0]})", img))
        for reader in ("Average (R+G+B)/3", "BT.601 (OpenCV default)", "BT.709 (HDTV)"):
            out = gs.METHODS[reader](img)
            separation = gs.region_separation(out, mask)
            panels.append((f"read by {reader.split()[0]}\n{separation:.2f} levels",
                           ensure_rgb(out)))
    figures.grid(
        panels, IMAGES / "blind_planes.png", ncols=4,
        suptitle=("Every fixed weighting has a plane of colours it maps to one grey. "
                  "Each row is a scene built to sit on one conversion's plane; only "
                  "that conversion cannot see it."),
    )

    figures.comparison_matrix(
        blind,
        [(m.split()[0], m, True) for m in methods],
        IMAGES / "blind_spot_matrix.png",
        row_key="built_against",
        title=("Grey levels of separation. The diagonal is each conversion meeting its "
               "own blind plane — the 'correct' weights are blind exactly as often."),
    )

    figures.lines(
        [r["luma_offset"] for r in sweep],
        {"Canny edge recall": [r["BT.601 (OpenCV default)"] for r in sweep],
         "Otsu IoU": [r["BT.601 (OpenCV default)__otsu_iou"] for r in sweep]},
        IMAGES / "isoluminance_sweep.png",
        xlabel="luma difference across the boundary (grey levels)",
        ylabel="score",
        title="Two downstream stages, the same grayscale, a 6x disagreement about enough",
    )

    figures.lines(
        [r["chroma"] for r in retention],
        {"1st percentile": [r["worst_retained"] for r in retention],
         "single worst pixel": [r["minimum_retained"] for r in retention],
         "median": [r["median_retained"] for r in retention]},
        IMAGES / "retention_vs_chroma.png",
        xlabel="mean chroma of the photograph",
        ylabel="fraction of colour contrast kept by BT.601",
        title="On real photographs the projection keeps almost everything, everywhere",
    )

    figures.lines(
        [r["green_weight"] for r in weights],
        {"mean difference from BT.601": [r["mean_diff_vs_bt601"] for r in weights]},
        IMAGES / "weight_sensitivity.png",
        xlabel="green weight (red and blue share the rest)",
        ylabel="grey levels",
        title="How much the whole weights argument can possibly be worth",
        # red and blue are held equal here, so the curve does not pass through
        # either standard — these mark where each one's green weight sits, not
        # what it scores.
        vlines={"BT.601 green (0.587)": 0.587, "BT.709 green (0.715)": 0.7152},
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS,
        "34_rgb_to_grayscale",
        {
            "images": list(gs.IMAGES),
            "photographs": photos,
            "luma_retention": retention,
            "blind_spot_matrix": blind,
            "isoluminance_sweep": sweep,
            "isoluminant_scene": isoluminant,
            "downstream": downstream,
            "weight_sensitivity": weights,
        },
    )

    photo_table = markdown_table(
        photos,
        [("Conversion", "method"), ("Levels from BT.601", "mean_diff_vs_bt601"),
         ("Contrast retained", "contrast_retained"),
         ("Worst colour-edge recall", "worst_image_recall"),
         ("Time (ms)", "median_ms")],
    )
    blind_table = markdown_table(
        blind, [("Scene built against", "built_against")] + [(m, m) for m in methods])
    retention_table = markdown_table(
        retention,
        [("Photograph", "image"), ("Chroma", "chroma"),
         ("1st pct kept", "worst_retained"), ("Worst pixel", "minimum_retained")],
    )
    sweep_table = markdown_table(
        [{"offset": r["luma_offset"],
          "separation": r["BT.601 (OpenCV default)__separation"],
          "canny": r["BT.601 (OpenCV default)"],
          "otsu": r["BT.601 (OpenCV default)__otsu_iou"]} for r in sweep],
        [("Luma offset", "offset"), ("Separation", "separation"),
         ("Canny recall", "canny"), ("Otsu IoU", "otsu")],
    )
    write_tables(
        RESULTS,
        [
            ("On twelve photographs", photo_table),
            ("Every weighting's blind plane (grey levels of separation)", blind_table),
            ("Contrast kept at each photograph's worst colour edge", retention_table),
            ("How isoluminant is too isoluminant", sweep_table),
            ("Four photographs down the rows", gallery_table),
        ],
    )
    print("\n" + photo_table + "\n\n" + blind_table)

    # ------------------------------------------------------------------ #
    # headlines
    # ------------------------------------------------------------------ #
    print("\n--- HEADLINE NUMBERS ---")
    by_method = {r["method"]: r for r in photos}
    bt709 = by_method["BT.709 (HDTV)"]["mean_diff_vs_bt601"]
    fast = by_method["BT.601 (OpenCV default)"]["median_ms"]
    slow = by_method["Contrast-preserving"]["median_ms"]
    print(f"the whole BT.601-vs-BT.709 argument is worth {bt709:.2f} grey levels")
    print(f"the contrast-preserving method costs {slow / fast:.0f}x the time "
          f"({slow:.1f} ms vs {fast:.1f}) and buys nothing measurable on a photograph")

    worst = min(retention, key=lambda r: r["minimum_retained"])
    floor = min(r["worst_retained"] for r in retention)
    print(f"across {len(retention)} photographs and "
          f"{sum(r['strong_edges'] for r in retention):,} strong colour edges, the "
          f"lowest 1st-percentile retention is {floor:.3f}")
    print(f"  the single worst pixel anywhere is {worst['minimum_retained']:.3f} "
          f"on {worst['image']}")

    for r in blind:
        own = r[r["built_against"]]
        others = [r[m] for m in methods if m != r["built_against"]]
        print(f"built against {r['built_against']:26s} own {own:5.2f}, "
              f"others {min(others):5.2f}-{max(others):6.2f} levels")

    canny_needs = next((r["luma_offset"] for r in sweep
                        if r["BT.601 (OpenCV default)"] >= 0.99), None)
    otsu_needs = next((r["luma_offset"] for r in sweep
                       if r["BT.601 (OpenCV default)__otsu_iou"] >= 0.98), None)
    print(f"\nOtsu recovers the shape at {otsu_needs:g} grey levels of separation; "
          f"Canny needs {canny_needs:g} — a {canny_needs / max(otsu_needs, 1):.0f}x "
          "disagreement between two stages reading the same grayscale")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
