"""Run the matching + RANSAC comparison and write results + figures.

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
from shared.io import to_gray  # noqa: E402
from shared.metrics import entropy  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import matching as mt  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"

#: Outlier fraction the headline comparison runs at. 0.5 is chosen because it is
#: exactly LMEDS's theoretical breakdown point, and the table should be taken at
#: the place the methods differ rather than a place they all work.
GALLERY_OUTLIERS = 0.5


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    print(f"Comparing descriptors on {len(mt.IMAGES)} photographs ...")
    descriptors = mt.evaluate_descriptors(images=mt.IMAGES, runs=args.runs)
    print("Comparing match filters ...")
    filters = mt.evaluate_filters(images=mt.IMAGES)
    print("Sweeping the ratio-test threshold ...")
    ratio_rows = mt.sweep_ratio(images=mt.IMAGES)
    print("Sweeping the outlier fraction ...")
    outlier_rows = mt.sweep_outliers(images=mt.IMAGES)

    estimators = [k for k in outlier_rows[0]
                  if k not in ("outlier_fraction", "iterations_needed")]

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    # Rows cross the axis the pool was built on -- entropy -- because a
    # descriptor can only describe what is in the frame, and a dancer on a black
    # stage has very little of it. A subject-uniqueness rule is applied on top so
    # the four rows are four different pictures.
    ENTROPY_BANDS = [("sparse", 0, 7.0), ("moderate", 7.0, 7.35),
                     ("busy", 7.35, 7.6), ("dense", 7.6, 10)]
    candidates: dict[str, list[dict]] = {}
    for i, name in enumerate(mt.IMAGES):
        a, b, H = mt.make_pair(name, seed=i)
        bits = entropy(to_gray(a))
        band = next(bd for bd, lo, hi in ENTROPY_BANDS if lo <= bits < hi)

        panels, notes, scores = [a], [f"{bits:.2f} bits"], []
        for est in estimators:
            err, inliers, drawn = mt.estimate_and_draw(
                a, b, H, estimator=est, outlier_fraction=GALLERY_OUTLIERS, seed=i)
            panels.append(drawn)
            notes.append(f"{err:.2f} px\n{inliers} inliers")
            scores.append(err)

        print(f"scene candidate {name:22s} {bits:.2f} bits  [{band:8s}]  "
              f"best {min(scores):7.3f} px, worst {max(scores):9.2f} px")
        candidates.setdefault(band, []).append({
            "label": f"{name.replace('_', ' ')}\n{bits:.2f} bits",
            "subject": name,
            "entropy": bits,
            "images": panels,
            "notes": notes,
            "scores": scores,
            "score": float(np.std(np.log10(np.maximum(scores, 1e-3)))),
        })

    chosen, used = [], set()
    for band, _, _ in ENTROPY_BANDS:
        pool = sorted(candidates.get(band, []), key=lambda r: -r["score"])
        pick = next((r for r in pool if r["subject"] not in used), None)
        if pick:
            used.add(pick["subject"])
            chosen.append(pick)
    chosen = chosen[:4]
    assert len({r["subject"] for r in chosen}) == len(chosen), "a subject repeats"

    figures.gallery(
        ["image A"] + estimators,
        [(r["label"], r["images"]) for r in chosen],
        IMAGES / "compare_estimators.png",
        cell_notes=[r["notes"] for r in chosen],
        suptitle=(
            f"Homography estimated from matches that are {GALLERY_OUTLIERS:.0%} "
            "deliberate outliers. Cells are mean reprojection error against the "
            "TRUE homography, in pixels. Green lines are the inliers each "
            "estimator kept."
        ),
    )
    gallery_table = markdown_table(
        [dict([("Sr", i), ("Scene", r["label"].replace("\n", " · "))]
              + [(e, f"{s:.2f} px") for e, s in zip(estimators, r["scores"])])
         for i, r in enumerate(chosen, start=1)],
        [("Sr", "Sr"), ("Scene", "Scene")] + [(e, e) for e in estimators],
    )
    print(f"\nfront-on comparison: {len(chosen)} scenes x {len(estimators)} estimators")
    print("\n--- estimators ---\n" + gallery_table)

    # ------------------------------------------------------------------ #
    # the signature figure: where each estimator breaks
    # ------------------------------------------------------------------ #
    # Every robust estimator has a breakdown point, and LMEDS's is a theorem:
    # it cannot survive more than half the data being wrong. Plotting the error
    # against the outlier fraction is the one picture that shows a theoretical
    # constant being met exactly.
    figures.lines(
        [r["outlier_fraction"] for r in outlier_rows],
        {e: [r[e] for r in outlier_rows] for e in estimators},
        IMAGES / "outlier_sweep.png",
        xlabel="fraction of matches that are outliers",
        ylabel="reprojection error (px)",
        title=("Breakdown points: least squares fails at 20%, LMEDS at exactly "
               "50% — its theoretical limit — and RANSAC holds to 80%"),
        logy=True,
        vlines={"LMEDS breakdown": 0.5},
    )

    figures.lines(
        [r["outlier_fraction"] for r in outlier_rows],
        {"iterations for 99% confidence":
         [r["iterations_needed"] for r in outlier_rows]},
        IMAGES / "iterations.png",
        xlabel="fraction of matches that are outliers",
        ylabel="iterations",
        title="What robustness costs: 0.2 iterations at 0% outliers, 46,000 at 90%",
        logy=True,
    )

    figures.lines(
        [r["ratio"] for r in ratio_rows],
        {"inlier precision": [r["inlier_precision"] for r in ratio_rows],
         "matches kept / 1000": [r["matches_kept"] / 1000 for r in ratio_rows]},
        IMAGES / "ratio_sweep.png",
        xlabel="Lowe ratio threshold",
        ylabel="score",
        title="The ratio test buys precision with matches, monotonically",
    )

    figures.comparison_matrix(
        descriptors,
        [("Matches", "matches", True), ("Inlier precision", "inlier_precision", True),
         ("Reprojection (px)", "reprojection_px", False),
         ("Time (ms)", "median_ms", False)],
        IMAGES / "descriptor_matrix.png",
        title="Descriptors x metrics — this is the half of SIFT project 19 could not see",
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS,
        "25_matching_ransac",
        {
            "images": list(mt.IMAGES),
            "gallery_outlier_fraction": GALLERY_OUTLIERS,
            "descriptors": descriptors,
            "filters": filters,
            "ratio_sweep": ratio_rows,
            "outlier_sweep": outlier_rows,
        },
    )

    descriptor_table = markdown_table(
        descriptors,
        [("Descriptor", "descriptor"), ("Matches", "matches"),
         ("Inlier precision", "inlier_precision"),
         ("Reprojection (px)", "reprojection_px"), ("Time (ms)", "median_ms")],
    )
    filter_table = markdown_table(
        filters,
        [("Filter", "filter"), ("Matches kept", "matches_kept"),
         ("Inlier precision", "inlier_precision"), ("Inlier recall", "inlier_recall")],
    )
    ratio_table = markdown_table(
        ratio_rows,
        [("Ratio", "ratio"), ("Matches kept", "matches_kept"),
         ("Inlier precision", "inlier_precision")],
    )
    outlier_table = markdown_table(
        outlier_rows,
        [("Outliers", "outlier_fraction"), ("Iterations", "iterations_needed")]
        + [(e, e) for e in estimators],
    )
    write_tables(
        RESULTS,
        [
            ("Descriptors", descriptor_table),
            ("Match filters", filter_table),
            ("The ratio test swept", ratio_table),
            ("Estimators against the outlier fraction", outlier_table),
            ("Four scenes down the rows", gallery_table),
        ],
    )
    print("\n" + descriptor_table + "\n\n" + filter_table + "\n\n" + outlier_table)

    sift = next(r for r in descriptors if r["descriptor"] == "SIFT")
    orb = next(r for r in descriptors if r["descriptor"] == "ORB")
    at = {r["outlier_fraction"]: r for r in outlier_rows}
    lmeds_ok = max(f for f in at if at[f]["LMEDS"] < 1.0)
    lmeds_bad = min(f for f in at if at[f]["LMEDS"] > 10.0)
    ransac_bad = min((f for f in at if at[f]["RANSAC"] > 1.0), default=None)

    print("\n--- HEADLINE NUMBERS ---")
    print(f"SIFT vs ORB       : reprojection {sift['reprojection_px']:.3f} px vs "
          f"{orb['reprojection_px']:.3f} px ({orb['reprojection_px'] / sift['reprojection_px']:.1f}x), "
          f"precision {sift['inlier_precision']:.4f} vs {orb['inlier_precision']:.4f}, "
          f"but ORB is {sift['median_ms'] / orb['median_ms']:.1f}x faster")
    print(f"ratio test        : precision "
          f"{ratio_rows[-1]['inlier_precision']:.4f} at ratio 1.0 -> "
          f"{ratio_rows[0]['inlier_precision']:.4f} at 0.5, "
          f"keeping {ratio_rows[0]['matches_kept']} of {ratio_rows[-1]['matches_kept']} matches")
    print(f"least squares     : {at[0.0]['Least squares (control)']:.2f} px clean -> "
          f"{at[0.2]['Least squares (control)']:.2f} px at 20% outliers")
    print(f"LMEDS breakdown   : fine to {lmeds_ok:.0%} "
          f"({at[lmeds_ok]['LMEDS']:.3f} px), broken by {lmeds_bad:.0%} "
          f"({at[lmeds_bad]['LMEDS']:.1f} px) — theory says exactly 50%")
    print(f"RANSAC            : holds to 80% ({at[0.8]['RANSAC']:.3f} px), "
          f"breaks at 90% ({at[0.9]['RANSAC']:.1f} px)")
    print(f"iterations        : {at[0.0]['iterations_needed']:.1f} at 0% outliers -> "
          f"{at[0.9]['iterations_needed']:.0f} at 90%")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
