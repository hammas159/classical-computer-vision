"""Run the plate-localisation comparison and write results + figures.

    python run.py

Regenerates `results/results.json`, `results/tables.md` and every figure in
`docs/images/`. The README's numbers are copied from those files rather than
typed, so this script is the single source of truth for every claim made.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import numpy as np  # noqa: E402

from shared import figures  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import plates as pl  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"

THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)

#: Four of the fourteen, chosen to span the difficulty axis rather than to look
#: alike: the largest plate in the set, the smallest, a dark garage interior, and
#: one of the three non-European plates.
SHOWN = ("eu4", "eu10", "eu1", "AYO9034")
SHOWN_WHY = {
    "eu4": "the largest plate: 18% of the frame",
    "eu10": "the smallest: 0.27%, on a motorway",
    "eu1": "a dark garage, plate in shadow",
    "AYO9034": "a Brazilian plate, aspect 3.1",
}


def main() -> None:
    init_console()
    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    if not pl.available():
        raise SystemExit("Run `python tools/fetch_assets.py --set plates` first.")

    names = pl.image_names()
    shares = [pl.plate_share(n) for n in names]
    print(f"{len(names)} photographs, each with a box somebody drew and the plate's "
          "text typed out.")
    print(f"  the plate occupies {100 * min(shares):.2f}% to {100 * max(shares):.2f}% "
          f"of the frame — a {max(shares) / min(shares):.0f}x range in area")

    # ------------------------------------------------------------------ #
    # 0. what the readability proxy is worth, before it is used
    # ------------------------------------------------------------------ #
    print("\nThe readability proxy, scored against the typed text on the annotated "
          "crops ...")
    proxy = pl.validate_proxy(names)
    for r in proxy["per_image"]:
        flag = "" if r["exact"] else ("  (off by one)" if r["within_one"] else "  <- WRONG")
        print(f"  {r['image']:8s} '{r['text']:9s}' {r['characters']} characters, "
              f"{r['blobs']} blobs{flag}")
    print(f"  exact on {proxy['exact']}/{proxy['frames']}, within one on "
          f"{proxy['within_one']}/{proxy['frames']}.")
    print("  That is the ceiling on anything this proxy says about a detected crop, "
          "and it is\n  stated here rather than left implicit.")

    # ------------------------------------------------------------------ #
    # 1. three metrics
    # ------------------------------------------------------------------ #
    print("\nThree defensible metrics ...")
    results = pl.evaluate_all(names)
    n = len(names)
    print(f"  {'locator':24s} {'IoU>=0.5':>9s} {'coverage':>9s} {'readable':>9s} "
          f"{'med IoU':>8s} {'med cov':>8s} {'boxes':>7s}")
    for r in results:
        print(f"  {r['locator']:24s} {r['hits_iou']:6d}/{n} {r['hits_coverage']:6d}/{n} "
              f"{r['readable']:6d}/{n} {r['median_iou']:8.2f} "
              f"{r['median_coverage']:8.2f} {r['boxes']:7d}")

    by_metric = {}
    for metric in ("hits_iou", "hits_coverage", "readable"):
        real = [r for r in results if r["locator"] in pl.REAL_LOCATORS]
        by_metric[metric] = max(real, key=lambda r: r[metric])["locator"]
    print(f"\n  winner by IoU:       {by_metric['hits_iou']}")
    print(f"  winner by coverage:  {by_metric['hits_coverage']}")
    print(f"  winner by readable:  {by_metric['readable']}")
    if len(set(by_metric.values())) > 1:
        print("  The three metrics do not agree, so quoting one of them is a choice.")

    print("\nWhere two locators swap places between IoU and coverage ...")
    disagree = pl.metrics_disagree(names)
    for r in disagree:
        print(f"  {r['a']} beats {r['b']} on IoU ({r['a_iou']} vs {r['b_iou']}) "
              f"and loses on coverage ({r['a_coverage']} vs {r['b_coverage']})")

    whole = next(r for r in results if r["locator"].startswith("Whole frame"))
    print(f"\n  And coverage is won outright by the whole-frame control: "
          f"{whole['hits_coverage']}/{n}, at a median IoU of {whole['median_iou']:.2f}.")
    print("  That is why coverage is never reported on its own here.")

    # ------------------------------------------------------------------ #
    # 2. what IoU is actually charging for
    # ------------------------------------------------------------------ #
    print("\nShifting the annotated box by a known number of pixels ...")
    shifts = pl.what_a_pixel_of_error_costs(names)
    print(f"  {'shift':>6s} {'IoU sideways':>13s} {'IoU vertically':>15s} "
          f"{'coverage vertically':>20s}")
    for r in shifts:
        print(f"  {r['shift_px']:5d}px {r['iou_shifted_sideways']:13.2f} "
              f"{r['iou_shifted_vertically']:15.2f} "
              f"{r['coverage_shifted_vertically']:20.2f}")
    eight = next(r for r in shifts if r["shift_px"] == 8)
    ratio = ((1 - eight["iou_shifted_vertically"])
             / max(1 - eight["iou_shifted_sideways"], 1e-9))
    aspects = [pl.truth(nm)[0][2] / max(pl.truth(nm)[0][3], 1) for nm in names]
    print(f"\n  The same 8-pixel error costs {ratio:.1f}x more IoU vertically than "
          "sideways,")
    print(f"  because the mean plate here is {np.mean(aspects):.1f} times wider than "
          "it is tall.")

    print("\nAnd the threshold itself is a convention ...")
    sweep = {k: pl.iou_threshold_sweep(k, THRESHOLDS, names) for k in pl.REAL_LOCATORS}
    print(f"  {'locator':24s} " + "  ".join(f"{t:>6.1f}" for t in THRESHOLDS))
    for k, rows in sweep.items():
        print(f"  {k:24s} " + "  ".join(f"{r['hits']:6d}" for r in rows))
    winners = {}
    for i, t in enumerate(THRESHOLDS):
        winners[t] = max(sweep, key=lambda k: sweep[k][i]["hits"])
    print("  winner at each threshold: "
          + ", ".join(f"{t}: {w}" for t, w in winners.items()))

    # ------------------------------------------------------------------ #
    # 3. difficulty
    # ------------------------------------------------------------------ #
    print("\nPer photograph, ordered by how much of the frame the plate fills ...")
    hard = sorted(pl.difficulty(names), key=lambda r: r["plate_share"])
    for r in hard:
        print(f"  {r['image']:8s} {r['plate_px']:>9s} {100 * r['plate_share']:6.2f}% "
              f"aspect {r['aspect']:4.1f}  found by {r['found_iou']}/{r['locators']} "
              f"on IoU, {r['found_coverage']}/{r['locators']} on coverage  "
              f"'{r['text']}'")
    widths = np.array([r["plate_width"] for r in hard], float)
    founds = np.array([r["found_iou"] for r in hard], float)
    r_width = float(np.corrcoef(np.log(widths), founds)[0, 1])
    print(f"\n  log plate width vs how many locators find it: r = {r_width:+.2f}")

    # ------------------------------------------------------------------ #
    # figures
    # ------------------------------------------------------------------ #
    print("\nFigures ...")

    columns = ["Annotation"] + list(pl.REAL_LOCATORS)
    rows, notes = [], []
    for name in SHOWN:
        img = pl.load(name)
        target, text = pl.truth(name)
        cells = [pl.draw_truth(img, target)]
        cell_notes = [f"'{text}'  {target[2]}x{target[3]}"]
        for k in pl.REAL_LOCATORS:
            boxes = pl.LOCATORS[k](img)
            best, v = pl.best_by(boxes, target, pl.iou)
            cov = pl.coverage(best, target) if best is not None else 0.0
            drawn = pl.draw_truth(img, target)
            if best is not None:
                drawn = pl.draw(drawn, [best],
                                colour=(60, 220, 90) if v >= 0.5 else (235, 70, 70))
            cells.append(drawn)
            cell_notes.append(f"IoU {v:.2f}  covers {cov:.2f}"
                              if best is not None else "nothing found")
        rows.append((SHOWN_WHY[name], cells))
        notes.append(cell_notes)
    figures.gallery(columns, rows, IMAGES / "compare.png", cell_notes=notes,
                    suptitle="Amber is the box a person drew. Green where the detection "
                             "passes IoU >= 0.5, red where it does not — "
                             "read the coverage next to it before believing the colour.")

    real = [r for r in results if r["locator"] in pl.REAL_LOCATORS]
    figures.comparison_matrix(
        [{"locator": r["locator"], "iou": r["hits_iou"],
          "coverage": r["hits_coverage"], "readable": r["readable"]} for r in real],
        [("IoU >= 0.5", "iou", True),
         ("coverage >= 0.95", "coverage", True),
         ("characters recoverable", "readable", True)],
        IMAGES / "three_metrics.png",
        row_key="locator",
        title=f"Three defensible metrics over {n} photographs. They do not agree: "
              "coverage alone puts the cascade first, the other two put Sobel first")

    figures.metric_bars(
        [r["locator"].replace(" + ", "+") for r in real],
        [r["hits_iou"] for r in real],
        IMAGES / "by_iou.png", ylabel=f"photographs passing IoU >= 0.5, of {n}",
        title="Ranked by the conventional metric", highlight_best="max")
    figures.metric_bars(
        [r["locator"].replace(" + ", "+") for r in real],
        [r["hits_coverage"] for r in real],
        IMAGES / "by_coverage.png",
        ylabel=f"photographs where the crop contains the whole plate, of {n}",
        title="Ranked by coverage — a different order", highlight_best="max")
    figures.metric_bars(
        [r["locator"].replace(" + ", "+") for r in real],
        [r["readable"] for r in real],
        IMAGES / "by_readable.png",
        ylabel=f"photographs whose crop yields the right character count, of {n}",
        title="Ranked by the only metric that uses the plate's text",
        highlight_best="max")

    figures.lines(
        list(THRESHOLDS),
        {k: [r["hits"] for r in sweep[k]] for k in pl.REAL_LOCATORS},
        IMAGES / "threshold.png",
        xlabel="IoU threshold counted as a hit",
        ylabel=f"photographs passing, of {n}",
        title="0.5 is a convention from PASCAL VOC, chosen for objects as tall as "
              "they are wide")

    figures.lines(
        [r["shift_px"] for r in shifts],
        {"IoU, box shifted sideways": [r["iou_shifted_sideways"] for r in shifts],
         "IoU, box shifted vertically": [r["iou_shifted_vertically"] for r in shifts],
         "coverage, shifted vertically":
             [r["coverage_shifted_vertically"] for r in shifts]},
        IMAGES / "thin_box.png",
        xlabel="the annotated box moved by this many pixels — nothing is detected here",
        ylabel="score against the annotation it was moved from",
        title=f"A plate here is {np.mean(aspects):.1f}x wider than tall, so the same "
              f"pixel error costs {ratio:.1f}x more IoU vertically")

    # ------------------------------------------------------------------ #
    write_results(RESULTS, "48_plate_localisation", {
        "photographs": n,
        "plate_share_min": float(min(shares)),
        "plate_share_max": float(max(shares)),
        "plate_share_range": float(max(shares) / min(shares)),
        "proxy_validation": proxy,
        "results": [{k: v for k, v in r.items() if k != "per_image"} for r in results],
        "per_image": {r["locator"]: r["per_image"] for r in results},
        "metrics_disagree": disagree,
        "winners": by_metric,
        "pixel_shift": shifts,
        "threshold_sweep": sweep,
        "difficulty": hard,
        "log_width_vs_found_r": r_width,
        "shown": list(SHOWN),
    })

    write_tables(RESULTS, [
        ("Three metrics, three winners", markdown_table(
            [{"Locator": r["locator"], "IoU >= 0.5": f"{r['hits_iou']}/{n}",
              "Coverage >= 0.95": f"{r['hits_coverage']}/{n}",
              "Characters recoverable": f"{r['readable']}/{n}",
              "Median IoU": f"{r['median_iou']:.2f}",
              "Median coverage": f"{r['median_coverage']:.2f}"} for r in results],
            [(c, c) for c in ("Locator", "IoU >= 0.5", "Coverage >= 0.95",
                              "Characters recoverable", "Median IoU",
                              "Median coverage")])),
        ("What a pixel of localisation error costs", markdown_table(
            [{"Shift": f"{r['shift_px']} px",
              "IoU sideways": f"{r['iou_shifted_sideways']:.2f}",
              "IoU vertically": f"{r['iou_shifted_vertically']:.2f}",
              "Coverage vertically": f"{r['coverage_shifted_vertically']:.2f}"}
             for r in shifts],
            [(c, c) for c in ("Shift", "IoU sideways", "IoU vertically",
                              "Coverage vertically")])),
        ("The readability proxy against the typed text", markdown_table(
            [{"Photograph": r["image"], "Plate text": r["text"],
              "Characters": r["characters"], "Blobs found": r["blobs"],
              "Exact": "yes" if r["exact"] else "no"}
             for r in proxy["per_image"]],
            [(c, c) for c in ("Photograph", "Plate text", "Characters",
                              "Blobs found", "Exact")])),
        ("Per photograph, by plate size", markdown_table(
            [{"Photograph": r["image"], "Plate": r["plate_px"],
              "Share of frame": f"{100 * r['plate_share']:.2f}%",
              "Aspect": f"{r['aspect']:.1f}",
              "Found (IoU)": f"{r['found_iou']}/{r['locators']}",
              "Found (coverage)": f"{r['found_coverage']}/{r['locators']}"}
             for r in hard],
            [(c, c) for c in ("Photograph", "Plate", "Share of frame", "Aspect",
                              "Found (IoU)", "Found (coverage)")])),
    ])
    print(f"wrote {RESULTS / 'results.json'} and six figures")


if __name__ == "__main__":
    main()
