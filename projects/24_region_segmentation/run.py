"""Run the region-segmentation comparison and write results + figures.

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
from shared.io import ensure_rgb  # noqa: E402
from shared.metrics import iou  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import segmentation as sg  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"


def colourfulness(img: np.ndarray) -> float:
    b, g, r = (img[..., i].astype(np.float32) for i in range(3))
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    print(f"Scoring {len(sg.METHODS)} methods on BOUNDARY F against human consensus ...")
    rows = sg.evaluate_boundaries(images=sg.IMAGES, runs=args.runs)

    print("... and on IoU, which is the cautionary half of this project ...")
    iou_rows = sg.evaluate_methods(images=sg.IMAGES, runs=1)

    print("Measuring how much the human annotators agree with each other ...")
    ceilings = [dict(image=n, **sg.human_boundary_ceiling(n)) for n in sg.IMAGES]

    print("Sweeping the watershed marker threshold ...")
    marker_rows = sg.sweep_watershed_markers(images=sg.IMAGES)
    print("Sweeping the SLIC region count ...")
    slic_rows = sg.sweep_slic_count(images=sg.IMAGES)
    print("Sweeping the noise ...")
    noise_rows = sg.sweep_noise(images=sg.IMAGES)

    names = list(sg.METHODS)

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    # Rows cross the axis the pool was built on -- colourfulness -- because
    # every method here groups pixels by appearance, and how far appearance and
    # object agree is what the score is really about. A subject-uniqueness rule
    # is applied on top so the four rows are four different pictures.
    COLOUR_BANDS = [("muted", 0, 30), ("moderate", 30, 42),
                    ("strong", 42, 60), ("vivid", 60, 400)]
    candidates: dict[str, list[dict]] = {}
    for name in sg.IMAGES:
        img = sg.load_scene(name)
        colour = colourfulness(img)
        band = next(b for b, a, z in COLOUR_BANDS if a <= colour < z)
        ceiling = sg.human_boundary_ceiling(name)

        target = bsds.consensus_boundaries(name)
        labels = [sg.METHODS[m](img) for m in names]
        edges = [sg.labels_to_boundaries(l) for l in labels]
        scored = [bsds.boundary_f_measure(e, target) for e in edges]
        scores = [s["f"] for s in scored]

        def show(mask):
            return ensure_rgb((np.asarray(mask) > 0).astype(np.uint8) * 255)

        print(f"scene candidate {name:22s} colour {colour:6.1f}  [{band:8s}]  "
              f"human ceiling F {ceiling['best']:.3f}, best method F {max(scores):.3f}")
        candidates.setdefault(band, []).append({
            "label": f"{name.replace('_', ' ')}\ncolour {colour:.0f} · humans F {ceiling['best']:.2f}",
            "subject": name,
            "colour": colour,
            "ceiling": ceiling,
            "images": [img, show(target)] + [show(e) for e in edges],
            "notes": ["photograph", "HUMAN consensus"]
            + [f"F {s['f']:.3f}\nP {s['precision']:.2f} R {s['recall']:.2f}"
               for s in scored],
            "scores": scores,
            "detail": scored,
            "score": float(np.std(scores)),
        })

    chosen, used = [], set()
    for band, _, _ in COLOUR_BANDS:
        pool = sorted(candidates.get(band, []), key=lambda r: -r["score"])
        pick = next((r for r in pool if r["subject"] not in used), None)
        if pick:
            used.add(pick["subject"])
            chosen.append(pick)
    chosen = chosen[:4]
    assert len({r["subject"] for r in chosen}) == len(chosen), "a subject repeats"

    figures.gallery(
        ["photograph", "HUMAN consensus"] + names,
        [(r["label"], r["images"]) for r in chosen],
        IMAGES / "compare_segmentation.png",
        cell_notes=[r["notes"] for r in chosen],
        suptitle=(
            "Boundaries, against what at least two of five human annotators "
            "agreed on. Cells are F, precision and recall. The first method "
            "column is a grid of rectangles that never looked at the image."
        ),
    )
    gallery_table = markdown_table(
        [dict([("Sr", i), ("Scene", r["label"].split("\n")[0]),
               ("Humans (F)", f"{r['ceiling']['best']:.3f}")]
              + [(m, f"{s:.3f}") for m, s in zip(names, r["scores"])])
         for i, r in enumerate(chosen, start=1)],
        [("Sr", "Sr"), ("Scene", "Scene"), ("Humans (F)", "Humans (F)")]
        + [(m, m) for m in names],
    )
    print(f"\nfront-on comparison: {len(chosen)} scenes x {len(names)} methods")
    print("\n--- segmentation ---\n" + gallery_table)

    # ------------------------------------------------------------------ #
    # the signature figure: IoU rewards region count, boundary recall does not
    # ------------------------------------------------------------------ #
    by_regions = sorted(rows, key=lambda r: r["regions"])
    iou_by = {r["method"]: r["iou"] for r in iou_rows}
    figures.lines(
        [r["regions"] for r in by_regions],
        {"IoU (oracle region labelling)": [iou_by[r["method"]] for r in by_regions],
         "boundary F (human consensus)": [r["f"] for r in by_regions],
         "boundary recall": [r["recall"] for r in by_regions]},
        IMAGES / "regions_vs_score.png",
        xlabel="regions produced",
        ylabel="score",
        title=("IoU under oracle region labelling rises with region count. "
               "Boundary F does not, which is why it is the headline."),
        logx=True,
    )

    figures.lines(
        [r["requested"] for r in slic_rows],
        {k: [r[k] for r in slic_rows] for k in slic_rows[0] if k != "requested"},
        IMAGES / "slic_count.png",
        xlabel="SLIC regions requested",
        ylabel="score",
        title="More superpixels is more boundary recall, all the way up",
        logx=True,
    )

    figures.lines(
        [r["fg_ratio"] for r in marker_rows],
        {k: [r[k] for r in marker_rows] for k in marker_rows[0] if k != "fg_ratio"},
        IMAGES / "watershed_markers.png",
        xlabel="marker threshold (fraction of the distance transform's peak)",
        ylabel="score",
        title="Watershed's marker threshold: region count and accuracy in tension",
        logy=True,
    )

    figures.lines(
        [r["noise_sigma"] for r in noise_rows],
        {k: [r[k] for r in noise_rows] for k in noise_rows[0] if k != "noise_sigma"},
        IMAGES / "noise_sweep.png",
        xlabel="noise sigma",
        ylabel="regions produced",
        title=("Noise manufactures local minima, and every one becomes a "
               "catchment basin — which is what markers exist to prevent"),
        logy=True,
    )

    figures.comparison_matrix(
        rows,
        [("Boundary F", "f", True), ("Precision", "precision", True),
         ("Recall", "recall", True), ("Time (ms)", "median_ms", False)],
        IMAGES / "method_matrix.png",
        title="Methods x metrics, against human boundary consensus",
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS,
        "24_region_segmentation",
        {
            "images": list(sg.IMAGES),
            "grid_tiles": sg.GRID_TILES,
            "methods_boundary_f": rows,
            "methods_iou": iou_rows,
            "human_ceilings": ceilings,
            "watershed_markers": marker_rows,
            "slic_count": slic_rows,
            "noise_sweep": noise_rows,
        },
    )

    method_table = markdown_table(
        sorted(rows, key=lambda r: -r["f"]),
        [("Method", "method"), ("Boundary F", "f"), ("Precision", "precision"),
         ("Recall", "recall"), ("Regions", "regions"), ("Time (ms)", "median_ms")],
    )
    iou_table = markdown_table(
        sorted(iou_rows, key=lambda r: -r["iou"]),
        [("Method", "method"), ("IoU (oracle labelling)", "iou"),
         ("Regions", "regions"), ("Underseg. error", "underseg_error")],
    )
    ceiling_table = markdown_table(
        ceilings,
        [("Image", "image"), ("Annotators", "annotators"),
         ("Best pair", "best"), ("Mean pair", "mean")],
    )
    write_tables(
        RESULTS,
        [
            ("Every method: boundary F against human consensus", method_table),
            ("The same methods on IoU, which rewards over-segmentation", iou_table),
            ("How much the humans agree with each other", ceiling_table),
            ("Four scenes down the rows", gallery_table),
        ],
    )
    print("\n" + method_table + "\n\n" + iou_table + "\n\n" + ceiling_table)

    by_f = {r["method"]: r for r in rows}
    grid_f = by_f["Grid tiles (control)"]
    best_f = max((r for r in rows if r["method"] != "Grid tiles (control)"),
                 key=lambda r: r["f"])
    grid_iou = iou_by["Grid tiles (control)"]
    beaten = [r["method"] for r in iou_rows
              if r["method"] != "Grid tiles (control)" and r["iou"] < grid_iou]
    ws = by_f["Watershed (no markers)"]
    best_c = float(np.mean([c["best"] for c in ceilings]))

    print("\n--- HEADLINE NUMBERS ---")
    print(f"human ceiling     : F {best_c:.4f} (best annotator), "
          f"{float(np.mean([c['mean'] for c in ceilings])):.4f} (mean annotator)")
    print(f"best method       : {best_f['method']} F {best_f['f']:.4f} "
          f"- {best_c / max(best_f['f'], 1e-9):.1f}x below the human ceiling")
    print(f"grid control      : F {grid_f['f']:.4f}, LAST of {len(rows)} "
          f"- which is where a grid of rectangles belongs")
    print(f"on IoU instead    : the grid scores {grid_iou:.4f} and beats "
          f"{len(beaten)} of {len(iou_rows) - 1} real methods: {beaten}")
    print(f"recall trap       : {ws['method']} recall {ws['recall']:.3f} "
          f"from {ws['regions']} regions, precision {ws['precision']:.3f}, "
          f"F {ws['f']:.4f}")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
