"""Run the epipolar geometry study and write results + figures.

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

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from shared import figures  # noqa: E402
from shared.io import ensure_rgb  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import epipolar as ep  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"

FIGURE_ESTIMATORS = ("8-point, raw pixels", "8-point, normalised (Hartley)",
                     "RANSAC", "Assume rectified + offset (1 parameter, control)")

#: How many epipolar lines to draw. More than this and the picture is a grid.
N_LINES = 9


def _lines_drawn(image, F, points, colour=(220, 60, 60)):
    out = ensure_rgb(image).copy()
    if F is None:
        return out
    for (a, b) in ep.epipolar_lines(F, points, image.shape):
        cv2.line(out, a, b, colour, 1)
    return out


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="SIFT", choices=["SIFT", "ORB", "AKAZE"])
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    if not ep.assets_available():
        raise SystemExit("The calibration images are not cached. Run "
                         "`python tools/fetch_assets.py --set calibration` first.")

    truth = ep.calibration_truth()
    pairs = ep.usable_pairs()
    truth_epipole = ep.epipole(truth["F"])
    dy = truth["board_right"][:, 1] - truth["board_left"][:, 1]
    dx = truth["board_right"][:, 0] - truth["board_left"][:, 0]

    print(f"{len(pairs)} stereo pairs, {len(truth['board_left'])} board corners")
    print(f"  the truth F puts every one of them {ep.board_error(truth['F']):.4f} px "
          "from its epipolar line")
    print(f"  it comes from the chessboard alone; every estimator below sees only "
          "scene features")
    print(f"  the epipole is at ({truth_epipole[0]:.0f}, {truth_epipole[1]:.0f}) — "
          f"{abs(truth_epipole[0]) / 640:.0f} image widths outside a 640x480 frame, so "
          "epipole position is not a usable metric on this rig")
    print(f"  board-corner disparity: dx {dx.mean():+.2f} ± {dx.std():.2f} px, "
          f"dy {dy.mean():+.2f} ± {dy.std():.2f} px")
    print(f"  that dy standard deviation of {dy.std():.2f} px is why the "
          "one-parameter control below does so well")

    print(f"\nEvery estimator on {args.detector} matches, board excluded ...")
    overall = ep.evaluate(detector=args.detector, exclude_board=True)
    for r in overall:
        print(f"  {r['estimator']:50s} mean {r['board_px']:9.4f}  "
              f"median {r['board_median_px']:9.4f}")

    normalisation = ep.normalisation_effect(detector=args.detector)
    planar = ep.planar_degeneracy(detector=args.detector)
    inliers = ep.inliers_are_not_quality(detector=args.detector)
    detectors = ep.detector_comparison()
    shares = [ep.board_share(i, args.detector) for i in pairs]

    print("\nHartley's normalisation, pair by pair ...")
    for r in normalisation:
        print(f"  view {r['view']:2d}  {r['matches']:3d} matches  "
              f"raw {r['raw_px']:8.3f}  normalised {r['normalised_px']:8.3f}  "
              f"x{r['ratio']:.1f}")

    print("\nThe planar trap ...")
    for r in planar:
        print(f"  view {r['view']:2d}  board-only {r['board only']:8.3f} px "
              f"(inlier rate {r.get('board_only_inlier_rate', float('nan')):.3f})  "
              f"board excluded {r['board excluded']:8.3f}")

    print("\nRANSAC's threshold ...")
    for r in inliers:
        print(f"  threshold {r['threshold_px']:5.2f} px  "
              f"inlier rate {r['inlier_rate']:.4f}  board error {r['board_px']:8.4f}")

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    by_view = {r["view"]: r for r in normalisation}
    chosen = sorted(by_view, key=lambda v: -by_view[v]["matches"])[:4]
    chosen.sort()

    rows, notes, table_rows = [], [], []
    per_view_F = {}
    for index in chosen:
        p1, p2 = ep.correspondences(index, args.detector, exclude_board=True)
        per_view_F[index] = (p1, p2, {})
        for name in FIGURE_ESTIMATORS:
            try:
                F, _ = ep.ESTIMATORS[name](p1, p2, 1.0)
            except Exception:  # noqa: BLE001
                F = None
            per_view_F[index][2][name] = F

    sample = {i: ep.undistorted_pair(i) for i in chosen}
    drawn_points = {}
    for i in chosen:
        p1, _, _ = per_view_F[i]
        idx = np.linspace(0, len(p1) - 1, min(N_LINES, len(p1))).round().astype(int)
        drawn_points[i] = p1[idx]

    rows.append(("the right image\nwith truth lines",
                 [_lines_drawn(sample[i][1], truth["F"], drawn_points[i], (60, 200, 60))
                  for i in chosen]))
    notes.append([f"{ep.board_error(truth['F']):.3f} px" for _ in chosen])

    table_rows = [{"Sr": sr, "Pair": f"view {i}",
                   "Matches": len(per_view_F[i][0])}
                  for sr, i in enumerate(chosen, start=1)]

    for name in FIGURE_ESTIMATORS:
        panels, cell = [], []
        for pos, i in enumerate(chosen):
            F = per_view_F[i][2][name]
            panels.append(_lines_drawn(sample[i][1], F, drawn_points[i]))
            value = ep.board_error(F)
            cell.append(f"{value:.2f} px" if value == value else "failed")
            table_rows[pos][name] = f"{value:.2f}"
        rows.append((name, panels))
        notes.append(cell)

    figures.gallery(
        [f"view {i}" for i in chosen], rows, IMAGES / "compare_epipolar.png",
        cell_notes=notes,
        suptitle=("Epipolar lines in the right image for nine points of the left. "
                  "Row one is the truth, from the chessboard calibration. Cells are "
                  "the mean distance of all 702 board corners from their lines."),
    )
    gallery_table = markdown_table(
        table_rows, [("Sr", "Sr"), ("Pair", "Pair"), ("Matches", "Matches")]
        + [(n, n) for n in FIGURE_ESTIMATORS])
    print("\n--- board error per pair (px) ---\n" + gallery_table)

    # ------------------------------------------------------------------ #
    # the signature figures
    # ------------------------------------------------------------------ #
    real = [r for r in overall if r["estimator"] != "Calibration (truth)"]
    figures.metric_bars(
        [r["estimator"] for r in real],
        [r["board_median_px"] for r in real],
        IMAGES / "estimators.png",
        ylabel="median distance of 702 board corners from their lines (px)",
        title=("A one-parameter control beats every projective estimate on this rig. "
               "Lower is better."),
        highlight_best="min",
    )

    figures.lines(
        [r["threshold_px"] for r in inliers],
        {"RANSAC inlier rate": [r["inlier_rate"] for r in inliers],
         "board error (px) / 20": [r["board_px"] / 20 for r in inliers]},
        IMAGES / "inliers_not_quality.png",
        xlabel="RANSAC threshold (px)", ylabel="rate  /  px per 20", logx=True,
        dashed={"board error (px) / 20"},
        title=("The inlier rate rises with the threshold and so does the real error. "
               "One is logged; the other is not."),
    )

    figures.metric_bars(
        [f"{r['view']}" for r in normalisation],
        [r["ratio"] for r in normalisation],
        IMAGES / "normalisation.png",
        ylabel="raw error ÷ normalised error",
        title=("Hartley's normalisation, pair by pair. Same code, same "
               "correspondences, one conditioning step."),
    )

    valid = [r for r in planar if r["board only"] == r["board only"]
             and r["board excluded"] == r["board excluded"]]
    figures.metric_bars(
        [f"{r['view']}" for r in valid] * 2,
        [r["board only"] for r in valid] + [r["board excluded"] for r in valid],
        IMAGES / "planar_trap.png",
        ylabel="board error (px)",
        title=("Left: F from chessboard matches only — one plane, which cannot "
               "determine F. Right: the same pairs with the board removed."),
        highlight_best="min",
    )

    figures.metric_bars(
        [r["detector"] for r in detectors] * 2,
        [r["board_px"] for r in detectors] + [r["board_median_px"] for r in detectors],
        IMAGES / "detectors.png",
        ylabel="board error (px)",
        title="Left three: mean. Right three: median. They do not agree.",
        highlight_best="min",
    )

    figures.metric_bars(
        [f"{r['view']}" for r in shares],
        [r["share_on_board"] for r in shares],
        IMAGES / "board_share.png",
        ylabel="fraction of matches on the chessboard",
        title="How much of each correspondence set sits on the one big plane")

    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS, "46_epipolar_geometry",
        {
            "pairs": pairs,
            "detector": args.detector,
            "truth": {
                "F": truth["F"].tolist(),
                "stereo_rms": round(truth["stereo_rms"], 4),
                "board_corners": int(len(truth["board_left"])),
                "board_error_px": round(ep.board_error(truth["F"]), 4),
                "epipole": [round(float(v), 1) for v in truth_epipole],
                "disparity_x_mean": round(float(dx.mean()), 3),
                "disparity_x_sd": round(float(dx.std()), 3),
                "disparity_y_mean": round(float(dy.mean()), 3),
                "disparity_y_sd": round(float(dy.std()), 3),
            },
            "estimators": overall,
            "normalisation": normalisation,
            "planar_degeneracy": planar,
            "inlier_threshold": inliers,
            "detectors": detectors,
            "board_share": shares,
        },
    )

    overall_table = markdown_table(
        overall, [("Estimator", "estimator"), ("Board error mean (px)", "board_px"),
                  ("Board error median (px)", "board_median_px"),
                  ("Error on its own matches (px)", "match_px"),
                  ("Matches used", "inliers")])
    norm_table = markdown_table(
        normalisation, [("View", "view"), ("Matches", "matches"),
                        ("Raw (px)", "raw_px"), ("Normalised (px)", "normalised_px"),
                        ("Ratio", "ratio")])
    planar_table = markdown_table(
        planar, [("View", "view"), ("On board", "on_board"), ("Off board", "off_board"),
                 ("Board only (px)", "board only"),
                 ("Board-only inlier rate", "board_only_inlier_rate"),
                 ("Board excluded (px)", "board excluded"),
                 ("Everything (px)", "everything")])
    inlier_table = markdown_table(
        inliers, [("Threshold (px)", "threshold_px"), ("Inlier rate", "inlier_rate"),
                  ("Board error (px)", "board_px")])
    detector_table = markdown_table(
        detectors, [("Detector", "detector"), ("Matches", "matches"),
                    ("Board error mean (px)", "board_px"),
                    ("Board error median (px)", "board_median_px")])

    write_tables(
        RESULTS,
        [
            ("Every estimator, scored on 702 board corners it never saw", overall_table),
            ("Hartley's normalisation, pair by pair", norm_table),
            ("The planar trap", planar_table),
            ("RANSAC's threshold", inlier_table),
            ("Feature detectors", detector_table),
            ("Four pairs down the rows", gallery_table),
        ],
    )

    # ------------------------------------------------------------------ #
    print("\n--- HEADLINE NUMBERS ---")
    best = min((r for r in overall if r["estimator"] != "Calibration (truth)"),
               key=lambda r: r["board_median_px"])
    ransac = next(r for r in overall if r["estimator"] == "RANSAC")
    offset = next(r for r in overall if r["estimator"].startswith("Assume rectified +"))
    norm8 = next(r for r in overall
                 if r["estimator"] == "8-point, normalised (Hartley)")
    raw8 = next(r for r in overall if r["estimator"] == "8-point, raw pixels")

    print(f"the best estimate of F here has ONE parameter: "
          f"'{offset['estimator']}' at {offset['board_median_px']:.4f} px median")
    print(f"  against RANSAC's {ransac['board_median_px']:.4f} and the normalised "
          f"eight-point algorithm's {norm8['board_median_px']:.4f}")
    print(f"  it is a median of the vertical disparity, and this rig's vertical "
          f"disparity is {dy.mean():+.2f} ± {dy.std():.2f} px — nearly constant, "
          "so one number captures nearly all of it")
    print(f"  the truth still beats it {offset['board_median_px'] / ep.board_error(truth['F']):.0f}x, "
          f"at {ep.board_error(truth['F']):.4f} px")

    ratios = [r["ratio"] for r in normalisation]
    print(f"\nHartley's normalisation improves every one of the {len(ratios)} pairs, "
          f"by {min(ratios):.1f}x to {max(ratios):.1f}x "
          f"(median {np.median(ratios):.1f}x)")
    print(f"  mean over pairs: {raw8['board_px']:.2f} px raw against "
          f"{norm8['board_px']:.2f} px normalised")

    bo = [r["board only"] for r in planar if r["board only"] == r["board only"]]
    be = [r["board excluded"] for r in planar
          if r["board excluded"] == r["board excluded"]]
    rates = [r["board_only_inlier_rate"] for r in planar
             if r.get("board_only_inlier_rate") == r.get("board_only_inlier_rate")]
    print(f"\nthe planar trap: F from chessboard matches alone scores "
          f"{np.median(bo):.2f} px median against {np.median(be):.2f} px with the "
          f"board removed — {np.median(bo) / np.median(be):.1f}x worse")
    print(f"  and RANSAC reports a {np.median(rates):.1%} inlier rate while doing it. "
          "Points on one plane cannot determine F at all; nothing in the output says so.")

    lo, hi = inliers[0], inliers[-1]
    print(f"\ninliers are not quality: from a {lo['threshold_px']:g} px threshold to "
          f"{hi['threshold_px']:g} px the inlier rate rises "
          f"{lo['inlier_rate']:.3f} -> {hi['inlier_rate']:.3f} while the real error "
          f"rises {lo['board_px']:.2f} -> {hi['board_px']:.2f} px")
    print("  the number that goes in the log improves as the answer gets worse")

    sift = next(r for r in detectors if r["detector"] == "SIFT")
    akaze = next(r for r in detectors if r["detector"] == "AKAZE")
    print(f"\ndetectors disagree depending on the statistic: AKAZE has the better mean "
          f"({akaze['board_px']:.2f} vs {sift['board_px']:.2f}) and SIFT the better "
          f"median ({sift['board_median_px']:.2f} vs {akaze['board_median_px']:.2f})")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
