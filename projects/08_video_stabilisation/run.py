"""Run the video stabilisation study and write results + figures.

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

import stabilise as st  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"

#: Which frames of a segment the comparison figure samples.
FIGURE_FRAMES = (12, 24, 36, 48)


def _overlay_edges(frames, indices):
    """Several frames stacked as edges, so misalignment is visible as ghosting.

    A still cannot show that a video is shaky. Superimposing the edges of four
    frames can: aligned frames give one set of lines, misaligned ones give four.
    """
    stack = []
    for i in indices:
        edges = cv2.Canny(cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY), 80, 160)
        stack.append(edges.astype(np.float32))
    mean = np.mean(stack, axis=0)
    return ensure_rgb(np.clip(mean * 1.6, 0, 255).astype(np.uint8))


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    if not st.video_available():
        raise SystemExit("vtest.avi is not cached. Run "
                         "`python tools/fetch_assets.py --set video` first.")

    _, _, sample_path = st.shaken_segment(st.STARTS[0], seed=0)
    steps = st.path_steps(sample_path)
    print(f"{len(st.STARTS)} segments of {st.SEGMENT} frames, each given its own "
          "known camera path")
    print(f"  shake: step std {steps[:, :2].std():.2f} px, "
          f"{steps[:, 2].std():.3f} deg; path wanders up to "
          f"{np.abs(sample_path[:, :2]).max():.0f} px")

    print("\nHow well each estimator recovers the known motion ...")
    estimators = st.evaluate_estimators(runs=args.runs)
    for r in estimators:
        print(f"  {r['estimator']:24s} {r['translation_error_px']:7.4f} px  "
              f"{r['rotation_error_deg']:7.4f} deg  "
              f"{r['ms_per_frame']:6.2f} ms/frame")

    print("\nThe control nobody runs: the same estimators on the UNJITTERED clip ...")
    zero = st.zero_motion_control()
    for r in zero:
        print(f"  {r['estimator']:24s} invents "
              f"{r['invented_motion_px_per_frame']:7.4f} px/frame, worst drift "
              f"{r['worst_accumulated_drift_px']:7.3f} px")

    pipeline = st.evaluate_pipeline()
    limits = st.what_limits_the_result()
    crop_curve = st.crop_versus_stability()
    shake = st.sweep_shake()

    print("\nWhat actually limits the result ...")
    print(f"  varying the estimator: {limits['varying_the_estimator']}")
    print(f"  varying the smoother:  {limits['varying_the_smoother']}")
    print(f"  spreads: estimator {limits['estimator_spread']:.4f}, "
          f"smoother {limits['smoother_spread']:.4f}")

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    start = st.STARTS[0]
    original, shaken, truth = st.shaken_segment(start, seed=0)

    rows = [("the original clip\n(static camera)",
             [original[i] for i in FIGURE_FRAMES]),
            ("shaken by a known path",
             [shaken[i] for i in FIGURE_FRAMES])]
    notes = [[f"frame {i}" for i in FIGURE_FRAMES],
             [f"({truth[i][0]:+.1f}, {truth[i][1]:+.1f}) px, {truth[i][2]:+.2f} deg"
              for i in FIGURE_FRAMES]]

    table_rows = [{"Sr": sr, "Frame": i,
                   "True offset": f"({truth[i][0]:+.1f}, {truth[i][1]:+.1f})"}
                  for sr, i in enumerate(FIGURE_FRAMES, start=1)]

    # Every combination is aiming at the same thing: a smooth version of the true
    # camera path. Measuring each against *its own* target would make the cells
    # independent of the smoother -- which the first version of this figure did,
    # and it printed two identical columns. One common ideal is the fix.
    ideal = st.smooth_gaussian(truth, sigma=8.0)

    stabilised = {}
    for name in ("Phase correlation", "Features + LK + RANSAC"):
        for smoother in ("Gaussian (sigma=8)", "Kalman (causal)"):
            if name == "Phase correlation" and smoother == "Kalman (causal)":
                continue
            out, est, sm = st.stabilise(shaken, name, smoother)
            stabilised[(name, smoother)] = (out, est, sm)
            effective = st.effective_path(truth, est, sm)
            label = f"{name}\n+ {smoother}"
            rows.append((label, [out[i] for i in FIGURE_FRAMES]))
            notes.append([f"{np.hypot(*(effective[i][:2] - ideal[i][:2])):.2f} px "
                          "from the ideal" for i in FIGURE_FRAMES])
            for pos, i in enumerate(FIGURE_FRAMES):
                table_rows[pos][label.replace("\n", " ")] = \
                    f"{np.hypot(*(effective[i][:2] - ideal[i][:2])):.2f}"

    figures.gallery(
        [f"frame {i}" for i in FIGURE_FRAMES], rows,
        IMAGES / "compare_stabilisation.png",
        cell_notes=notes,
        suptitle=("One segment. Row one is the static original, row two the same "
                  "frames shaken by a known path, and the rest are stabilised. "
                  "Cells are how far the output camera ends up from the ideal "
                  "stabilised path — a smooth curve through the true one."),
    )
    gallery_table = markdown_table(
        table_rows,
        [("Sr", "Sr"), ("Frame", "Frame"), ("True offset", "True offset")]
        + [(k.replace("\n", " "), k.replace("\n", " "))
           for k in [f"{a}\n+ {b}" for (a, b) in stabilised]])
    print("\n--- residual error per frame (px) ---\n" + gallery_table)

    # a still cannot show shake, so stack edges
    best = stabilised[("Features + LK + RANSAC", "Gaussian (sigma=8)")][0]
    figures.grid(
        [("the static original", _overlay_edges(original, FIGURE_FRAMES)),
         ("shaken", _overlay_edges(shaken, FIGURE_FRAMES)),
         ("stabilised", _overlay_edges(best, FIGURE_FRAMES))],
        IMAGES / "edge_stack.png", ncols=3,
        suptitle=("Four frames' edges superimposed. One set of lines means the "
                  "frames are aligned; four means they are not."))

    # ------------------------------------------------------------------ #
    # the signature figures
    # ------------------------------------------------------------------ #
    figures.metric_bars(
        list(limits["varying_the_estimator"]) + list(limits["varying_the_smoother"]),
        list(limits["varying_the_estimator"].values())
        + list(limits["varying_the_smoother"].values()),
        IMAGES / "what_limits.png",
        ylabel="residual jitter (px per frame squared)",
        title=(f"Left four: same smoother, different estimators (spread "
               f"{limits['estimator_spread']:.3f}). Right three: same estimator, "
               f"different smoothers (spread {limits['smoother_spread']:.3f})."),
        highlight_best="min",
    )

    figures.metric_bars(
        [r["estimator"] for r in zero],
        [r["worst_accumulated_drift_px"] for r in zero],
        IMAGES / "zero_motion_control.png",
        ylabel="worst accumulated drift (px) over 60 frames",
        title=("On footage that never moved. The true answer is zero for every "
               "bar."),
        highlight_best="min",
    )

    figures.scatter_plane(
        {"Gaussian smoother, sigma 2 to 32":
            [(r["crop"] * 100, r["residual_jitter"]) for r in crop_curve]},
        target=None,
        out_path=IMAGES / "crop_versus_stability.png",
        xlabel="field of view thrown away (%)",
        ylabel="residual jitter (px per frame squared)",
        title=("There is no best setting, there is a curve. Steadier output is "
               "bought with frame."),
    )

    figures.lines(
        [r["translation_step_px"] for r in shake],
        {name: [r[name] for r in shake] for name in st.ESTIMATORS},
        IMAGES / "shake_sweep.png",
        xlabel="shake: translation step standard deviation (px)",
        ylabel="per-frame translation error (px)", logx=True, logy=True,
        title="How much shake each estimator survives",
    )

    figures.lines(
        list(range(len(truth))),
        {"the true camera path (x)": truth[:, 0].tolist(),
         "estimated": stabilised[("Features + LK + RANSAC",
                                  "Gaussian (sigma=8)")][1][:, 0].tolist(),
         "the smooth path it was warped onto":
             stabilised[("Features + LK + RANSAC",
                         "Gaussian (sigma=8)")][2][:, 0].tolist()},
        IMAGES / "path.png",
        xlabel="frame", ylabel="horizontal camera position (px)",
        dashed={"the smooth path it was warped onto"},
        title=("One segment's camera path. The estimate sits on top of the truth; "
               "the target is the smooth curve through it."),
    )

    figures.metric_bars(
        [f"{r['estimator'][:12]}\n{r['smoother'][:10]}" for r in pipeline],
        [r["crop"] * 100 for r in pipeline],
        IMAGES / "crop_by_pair.png",
        ylabel="field of view thrown away (%)",
        title="What each combination costs in frame",
        highlight_best="min")

    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS, "08_video_stabilisation",
        {
            "video": str(st.VIDEO),
            "segments": list(st.STARTS),
            "segment_length": st.SEGMENT,
            "shake": {"translation_step_px": st.SHAKE_TRANSLATION,
                      "rotation_step_deg": st.SHAKE_ROTATION,
                      "damping": st.SHAKE_DAMPING,
                      "measured_step_std_px": round(float(steps[:, :2].std()), 3),
                      "measured_step_std_deg": round(float(steps[:, 2].std()), 4)},
            "estimators": estimators,
            "zero_motion_control": zero,
            "pipeline": pipeline,
            "what_limits_the_result": limits,
            "crop_versus_stability": crop_curve,
            "shake_sweep": shake,
        },
    )

    estimator_table = markdown_table(
        estimators, [("Estimator", "estimator"),
                     ("Translation error (px)", "translation_error_px"),
                     ("Rotation error (deg)", "rotation_error_deg"),
                     ("ms per frame", "ms_per_frame")])
    zero_table = markdown_table(
        zero, [("Estimator", "estimator"),
               ("Invented motion (px/frame)", "invented_motion_px_per_frame"),
               ("Worst drift (px)", "worst_accumulated_drift_px")])
    pipeline_table = markdown_table(
        pipeline, [("Estimator", "estimator"), ("Smoother", "smoother"),
                   ("Residual jitter", "residual_jitter"),
                   ("Input jitter", "input_jitter"), ("Reduction", "reduction"),
                   ("Crop", "crop")])
    crop_table = markdown_table(
        crop_curve, [("Sigma", "sigma"), ("Residual jitter", "residual_jitter"),
                     ("Crop", "crop")])
    shake_table = markdown_table(
        shake, [("Translation step (px)", "translation_step_px")]
        + [(n, n) for n in st.ESTIMATORS])

    write_tables(
        RESULTS,
        [
            ("How well each estimator recovers the known motion", estimator_table),
            ("The same estimators on footage that never moved", zero_table),
            ("Every estimator against every smoother", pipeline_table),
            ("Stability bought with field of view", crop_table),
            ("How much shake each estimator survives", shake_table),
            ("Four frames down the rows", gallery_table),
        ],
    )

    # ------------------------------------------------------------------ #
    print("\n--- HEADLINE NUMBERS ---")
    real = [r for r in estimators if "control" not in r["estimator"]]
    best_est = min(real, key=lambda r: r["translation_error_px"])
    control = next(r for r in estimators if "control" in r["estimator"])
    print(f"the best estimator recovers the camera path to "
          f"{best_est['translation_error_px']:.4f} px per frame")
    print(f"  against a shake of {steps[:, :2].std():.2f} px per frame — "
          f"{steps[:, :2].std() / best_est['translation_error_px']:.0f}x finer than "
          "the motion it is removing")
    print(f"  and the do-nothing control's error is {control['translation_error_px']:.3f} "
          "px, which is just the shake itself")

    print(f"\nbut the estimator is not the bottleneck:")
    print(f"  swapping the estimator moves the residual jitter by "
          f"{limits['estimator_spread']:.4f}")
    print(f"  swapping the smoother moves it by {limits['smoother_spread']:.4f} — "
          f"{limits['smoother_spread'] / max(limits['estimator_spread'], 1e-9):.1f}x more")
    worst_smoother = max(limits["varying_the_smoother"],
                         key=lambda k: limits["varying_the_smoother"][k])
    best_smoother = min(limits["varying_the_smoother"],
                        key=lambda k: limits["varying_the_smoother"][k])
    print(f"  '{worst_smoother}' is "
          f"{limits['varying_the_smoother'][worst_smoother] / limits['varying_the_smoother'][best_smoother]:.1f}x "
          f"worse than '{best_smoother}', on identical motion estimates — it is "
          "causal, so it lags the path it is following")

    print(f"\non footage that never moved, where the true answer is zero:")
    for r in zero:
        if "control" in r["estimator"]:
            continue
        print(f"  {r['estimator']:24s} invents "
              f"{r['invented_motion_px_per_frame']:.4f} px/frame and drifts "
              f"{r['worst_accumulated_drift_px']:.1f} px over {st.SEGMENT} frames")
    worst_drift = max((r for r in zero if "control" not in r["estimator"]),
                      key=lambda r: r["worst_accumulated_drift_px"])
    print(f"  {worst_drift['estimator']} would introduce "
          f"{worst_drift['worst_accumulated_drift_px']:.0f} px of wander into a "
          "tripod shot. The path is an integral, so a small bias never goes away.")

    lo, hi = crop_curve[0], crop_curve[-1]
    print(f"\nand the bill: from sigma {lo['sigma']:g} to {hi['sigma']:g} the output "
          f"gets {lo['residual_jitter'] / hi['residual_jitter']:.0f}x steadier "
          f"({lo['residual_jitter']:.4f} -> {hi['residual_jitter']:.4f}) and throws "
          f"away {hi['crop'] / lo['crop']:.0f}x more of the frame "
          f"({100 * lo['crop']:.1f}% -> {100 * hi['crop']:.1f}%)")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
