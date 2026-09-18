"""Run the camera calibration study and write results + figures.

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

import calibration as cal  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"

#: The models the comparison figure undistorts with.
FIGURE_MODELS = ("No distortion model (control)", "k1 only",
                 "k1, k2, p1, p2 (OpenCV default)",
                 "Everything (14 coefficients, control)")


def _with_lines(gray, corners):
    """Draw the fitted line through each board row, so bending is visible."""
    out = ensure_rgb(gray).copy()
    rows = corners.reshape(cal.BOARD[1], cal.BOARD[0], 2)
    for row in rows:
        a = row[0]
        b = row[-1]
        cv2.line(out, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (60, 200, 60), 1)
        for p in row:
            cv2.circle(out, (int(p[0]), int(p[1])), 2, (220, 60, 60), -1)
    return out


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=5)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    if not cal.assets_available():
        raise SystemExit("The calibration images are not cached. Run "
                         "`python tools/fetch_assets.py --set calibration` first.")

    left = cal.usable_views("left")
    right = cal.usable_views("right")
    print(f"board {cal.BOARD[0]}x{cal.BOARD[1]} inner corners, "
          f"{cal.image_size('left')[0]}x{cal.image_size('left')[1]} images")
    print(f"  {len(left)} left views usable of {len(cal.VIEW_IDS)}: {left}")
    print(f"  {len(right)} right views usable: {right}")
    print("  (left10/right10 are 404 upstream and were never cached)")

    full = cal.full_calibration()
    print(f"\nall {len(left)} views, OpenCV's default model:")
    print(f"  fx {full['fx']:.2f}  fy {full['fy']:.2f}  "
          f"cx {full['cx']:.2f}  cy {full['cy']:.2f}")
    print(f"  k1 {full['k1']:+.5f}  k2 {full['k2']:+.5f}  "
          f"p1 {full['p1']:+.5f}  p2 {full['p2']:+.5f}")
    print(f"  RMS reprojection {full['rms']:.4f} px")
    print(f"  row straightness {full['straightness_before_px']:.4f} px before, "
          f"{full['straightness_px']:.4f} px after")

    print("\nEvery distortion model, scored three ways ...")
    models = cal.evaluate_models()
    print(f"  fitted on views {models['fit_views']}, "
          f"held out {models['held_out_views']}")
    for r in models["rows"]:
        print(f"  {r['model']:40s} {r['coefficients']:2d} coef  "
              f"fitted {r['rms_fitted']:.4f}  held-out {r['rms_held_out']:.4f}  "
              f"straight {r['straightness_px']:.4f}  cx {r['cx']:.1f}")

    print("\nHow the two errors move as views are added ...")
    views_sweep = cal.sweep_view_count(trials=args.trials)
    for r in views_sweep:
        held = f"{r['rms_held_out']:.4f}" if r["rms_held_out"] == r["rms_held_out"] \
            else "   n/a"
        print(f"  {r['views']:2d} views  fitted {r['rms_fitted']:.4f}  "
              f"held-out {held}  fx {r['fx_mean']:.2f} ± {r['fx_spread']:.2f}")

    stability = cal.intrinsic_stability()
    stereo = cal.stereo_calibration()
    baselines = cal.baseline_stability()

    print(f"\nstereo: baseline {stereo['baseline_squares']:.4f} board squares, "
          f"rotation {stereo['rotation_deg']:.3f}°, RMS {stereo['rms']:.4f}")

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    chosen = [left[i] for i in (0, 3, 7, 11)]
    fit = models["fit_views"]
    calibrations = {name: cal.calibrate(fit, "left", cal.MODELS[name])
                    for name in FIGURE_MODELS}

    rows = [("the photograph", [ensure_rgb(cal.load_view("left", i)) for i in chosen])]
    notes = [[f"view {i}" for i in chosen]]
    table_rows = [{"Sr": sr, "View": f"left{i:02d}"}
                  for sr, i in enumerate(chosen, start=1)]

    for name in FIGURE_MODELS:
        result = calibrations[name]
        K, dist = result["K"], result["dist"]
        panels, cell = [], []
        for pos, i in enumerate(chosen):
            undistorted = cal.undistorted_view(i, K, dist, "left")
            corners = cv2.undistortPoints(cal.find_corners("left", i), K, dist,
                                          P=K).reshape(-1, 1, 2)
            panels.append(_with_lines(undistorted, corners))
            value = cal.line_straightness(K, dist, [i], "left")
            cell.append(f"{value:.3f} px off straight")
            table_rows[pos][name] = f"{value:.3f}"
        rows.append((f"{name}\n({cal.coefficient_count(name)} coefficients)", panels))
        notes.append(cell)

    figures.gallery(
        [f"left{i:02d}" for i in chosen], rows, IMAGES / "compare_undistortion.png",
        cell_notes=notes,
        suptitle=("Four of the thirteen poses, undistorted by four models fitted on "
                  "the same seven views. Green lines join the ends of each board row; "
                  "cells are how far the corners in between sit off that line."),
    )
    gallery_table = markdown_table(
        table_rows, [("Sr", "Sr"), ("View", "View")]
        + [(n, n) for n in FIGURE_MODELS])
    print("\n--- straightness per view (px) ---\n" + gallery_table)

    # ------------------------------------------------------------------ #
    # the signature figure
    # ------------------------------------------------------------------ #
    figures.lines(
        [r["coefficients"] for r in models["rows"]],
        {"RMS on the views it was fitted to": [r["rms_fitted"] for r in models["rows"]],
         "RMS on views it never saw": [r["rms_held_out"] for r in models["rows"]]},
        IMAGES / "fitted_vs_held_out.png",
        xlabel="distortion coefficients the model may move", ylabel="RMS error (px)",
        logy=True, dashed={"RMS on views it never saw"},
        title=("The reported number keeps falling. The honest one stops at four "
               "coefficients and never improves again."),
    )

    figures.lines(
        [r["views"] for r in views_sweep],
        {"RMS on the views it was fitted to": [r["rms_fitted"] for r in views_sweep],
         "RMS on views it never saw":
             [r["rms_held_out"] for r in views_sweep]},
        IMAGES / "view_count.png",
        xlabel="views used to calibrate", ylabel="RMS error (px)",
        dashed={"RMS on views it never saw"},
        title=("Three views give the lowest reported error in this project and the "
               "worst real one. The two curves point opposite ways."),
    )

    figures.metric_bars(
        [r["model"] for r in models["rows"]],
        [r["cx"] for r in models["rows"]],
        IMAGES / "principal_point.png",
        ylabel="principal point cx (px)",
        title=("The image centre is 320. Every sane model puts cx near 343; the "
               "14-coefficient one moves it 67 px to buy 0.005 px of error."),
    )

    figures.lines(
        [r["views"] for r in views_sweep],
        {"spread of fx across random subsets": [r["fx_spread"] for r in views_sweep]},
        IMAGES / "focal_spread.png",
        xlabel="views used to calibrate", ylabel="standard deviation of fx (px)",
        title="How much of the answer came from which views you happened to use",
    )

    figures.scatter_plane(
        {"5-view calibrations": [(r["fx"], r["cx"]) for r in stability]},
        target=(full["fx"], full["cx"]),
        out_path=IMAGES / "intrinsic_spread.png",
        xlabel="fx (px)", ylabel="cx (px)",
        title=("Twelve calibrations of the same lens from five of the same thirteen "
               "photographs each."),
        target_label="all thirteen views",
    )

    figures.metric_bars(
        [f"{i + 1}" for i in range(len(baselines))],
        [r["baseline_squares"] for r in baselines],
        IMAGES / "baseline_stability.png",
        ylabel="stereo baseline (board squares)",
        title=("The same two lenses, six pairs each. A physical distance cannot "
               "change; the spread is the calibration's real uncertainty."),
    )

    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS, "35_camera_calibration",
        {
            "board": list(cal.BOARD),
            "image_size": list(cal.image_size("left")),
            "usable_views": {"left": left, "right": right},
            "full_calibration": full,
            "model_comparison": models,
            "view_count_sweep": views_sweep,
            "intrinsic_stability": stability,
            "stereo": stereo,
            "baseline_stability": baselines,
        },
    )

    model_table = markdown_table(
        models["rows"],
        [("Model", "model"), ("Coefficients", "coefficients"),
         ("RMS fitted", "rms_fitted"), ("RMS held out", "rms_held_out"),
         ("Straightness (px)", "straightness_px"), ("fx", "fx"), ("cx", "cx")])
    views_table = markdown_table(
        views_sweep, [("Views", "views"), ("RMS fitted", "rms_fitted"),
                      ("RMS held out", "rms_held_out"),
                      ("Straightness (px)", "straightness_px"),
                      ("fx mean", "fx_mean"), ("fx spread", "fx_spread")])
    stability_table = markdown_table(
        stability, [("Views", "views"), ("RMS fitted", "rms_fitted"), ("fx", "fx"),
                    ("cx", "cx"), ("cy", "cy"), ("k1", "k1")])
    baseline_table = markdown_table(
        baselines, [("Views", "views"), ("RMS", "rms"),
                    ("Baseline (squares)", "baseline_squares")])

    write_tables(
        RESULTS,
        [
            ("Every distortion model, scored three ways", model_table),
            ("How the two errors move as views are added", views_table),
            ("The same lens from twelve different five-view subsets", stability_table),
            ("The stereo baseline from ten different six-pair subsets", baseline_table),
            ("Straightness per view", gallery_table),
        ],
    )

    # ------------------------------------------------------------------ #
    print("\n--- HEADLINE NUMBERS ---")
    rows = models["rows"]
    best_fitted = min(rows, key=lambda r: r["rms_fitted"])
    best_held = min(rows, key=lambda r: r["rms_held_out"])
    default = next(r for r in rows if r["model"].startswith("k1, k2, p1"))
    everything = next(r for r in rows if r["model"].startswith("Everything"))

    print(f"the model with the LOWEST reported error is "
          f"'{best_fitted['model']}' at {best_fitted['rms_fitted']:.4f} px")
    print(f"  it moves the principal point to cx {everything['cx']:.1f} against "
          f"{default['cx']:.1f} for the default model — "
          f"{abs(everything['cx'] - default['cx']):.0f} px, in a "
          f"{cal.image_size('left')[0]}-pixel-wide image whose centre is "
          f"{cal.image_size('left')[0] // 2}")
    print(f"  and it buys {default['rms_held_out'] - everything['rms_held_out']:+.4f} px "
          f"on views it never saw, for ten extra parameters")

    three = next(r for r in views_sweep if r["views"] == 3)
    ten = next(r for r in views_sweep if r["views"] == 10)
    print(f"\nthree views report {three['rms_fitted']:.4f} px and actually score "
          f"{three['rms_held_out']:.4f} on unseen views")
    print(f"ten views report {ten['rms_fitted']:.4f} px — twice as bad — and actually "
          f"score {ten['rms_held_out']:.4f}")
    print("  the quoted number and the real one move in opposite directions")

    print(f"\nand the reported number is not even stable: from three views fx comes "
          f"out {three['fx_mean']:.1f} ± {three['fx_spread']:.1f} px, from ten "
          f"{ten['fx_mean']:.1f} ± {ten['fx_spread']:.1f}")

    k1s = [r["k1"] for r in stability]
    cxs = [r["cx"] for r in stability]
    print(f"\ntwelve calibrations of the same lens from five of the same thirteen "
          f"photographs each:")
    print(f"  k1 ranges {min(k1s):+.4f} to {max(k1s):+.4f} — "
          f"{100 * (max(k1s) - min(k1s)) / abs(np.mean(k1s)):.0f}% of its own value")
    print(f"  cx ranges {min(cxs):.1f} to {max(cxs):.1f} px")

    bl = [r["baseline_squares"] for r in baselines]
    print(f"\nthe stereo baseline from ten different six-pair subsets: "
          f"{np.mean(bl):.4f} ± {np.std(bl):.4f} board squares "
          f"({100 * np.std(bl) / np.mean(bl):.2f}%)")
    print(f"  a physical distance between two lenses, measured ten times, agreeing "
          f"to {100 * np.std(bl) / np.mean(bl):.2f}% — this is the part that works")

    print(f"\nout-of-objective check: board rows sit {full['straightness_before_px']:.4f} "
          f"px off straight in the raw images and {full['straightness_px']:.4f} px off "
          f"after undistortion — "
          f"{full['straightness_before_px'] / full['straightness_px']:.1f}x better, "
          "on a quantity calibrateCamera never optimised")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
