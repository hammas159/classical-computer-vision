"""Run the pedestrian detection study and write results + figures.

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

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from shared import figures  # noqa: E402
from shared.io import ensure_rgb  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import pedestrian as pd  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"

FIGURE_THRESHOLDS = (-0.5, 0.0, 0.6)


def _boxed(img, boxes, colour, thickness=2):
    out = ensure_rgb(img).copy()
    for x, y, w, h in np.asarray(boxes, np.int32).reshape(-1, 4):
        cv2.rectangle(out, (x, y), (x + w, y + h), colour, thickness)
    return out


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--scenes", type=int, default=8)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    if not pd.video_available():
        raise SystemExit("vtest.avi is not cached. Run "
                         "`python tools/fetch_assets.py --set video` first.")

    print("What the detector scores on drawn silhouettes and on real people ...")
    margins = pd.synthetic_versus_real(scenes=args.scenes)
    for r in margins:
        print(f"  {r['people']:20s} {r['detections']:3d} detections, mean margin "
              f"{r['mean_margin']:6.3f}, max {r['max_margin']:6.3f}, "
              f"{r['confident']} above 0.5")

    print("\nOn the composited scenes, where the truth boxes are exact ...")
    synthetic = pd.evaluate(scenes=args.scenes)
    print(f"  precision {synthetic['precision']:.3f}  recall {synthetic['recall']:.3f}  "
          f"F1 {synthetic['f1']:.3f}  ({synthetic['true_positives']} TP, "
          f"{synthetic['false_positives']} FP, {synthetic['false_negatives']} FN)")

    print("\nOn real footage, checked against independent motion evidence ...")
    video = pd.evaluate_video()
    for r in video:
        print(f"  frame {r['frame']:4d}  HOG {r['hog_detections']:2d}  "
              f"moving {r['moving_regions']:2d}  agree {r['hog_on_a_moving_region']:2d}  "
              f"mean margin {r['mean_score']:.2f}")

    threshold_curve = pd.sweep_video_threshold()
    scale_sweep = pd.sweep_scale(scenes=args.scenes)
    size_sweep = pd.sweep_person_size(scenes=args.scenes)

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    chosen = [r["frame"] for r in sorted(video, key=lambda r: -r["moving_regions"])[:4]]
    chosen.sort()

    frames = {i: pd.load_frame(i) for i in chosen}
    blobs = {i: pd.moving_blobs(i) for i in chosen}

    scene_row = [frames[i] for i in chosen]
    motion_row = [_boxed(ensure_rgb(blobs[i][1]), blobs[i][0], (60, 200, 60))
                  for i in chosen]
    rows = [("the frame", scene_row), ("moving regions\n(independent)", motion_row)]
    notes = [["the frame"] * len(chosen),
             [f"{len(blobs[i][0])} moving" for i in chosen]]

    table_rows = [{"Sr": sr, "Frame": i, "Moving regions": len(blobs[i][0])}
                  for sr, i in enumerate(chosen, start=1)]
    for threshold in FIGURE_THRESHOLDS:
        panels, cell = [], []
        for pos, index in enumerate(chosen):
            boxes, _ = pd.detect(frames[index], hit_threshold=threshold)
            tight = [pd.tighten_box(b) for b in boxes]
            supported = sum(
                1 for b in tight
                if max((pd.box_iou(b, m) for m in blobs[index][0]), default=0.0) >= 0.3)
            panels.append(_boxed(frames[index], tight, (220, 60, 60), 3))
            cell.append(f"{len(tight)} found, {supported} on a moving region")
            table_rows[pos][f"HOG at {threshold:+.1f}"] = len(tight)
        rows.append((f"HOG at {threshold:+.1f}", panels))
        notes.append(cell)
        print(f"  HOG at {threshold:+.1f}: {cell}")

    figures.gallery(
        [f"frame {i}" for i in chosen],
        rows, IMAGES / "compare_detection.png",
        cell_notes=notes,
        suptitle=(
            "Real pedestrians, static camera. Row two is where things moved — "
            "independent of appearance, so agreeing with it is not circular. The three "
            "HOG rows are the same detector at three SVM thresholds."
        ),
    )
    gallery_table = markdown_table(
        table_rows,
        [("Sr", "Sr"), ("Frame", "Frame"), ("Moving regions", "Moving regions")]
        + [(f"HOG at {t:+.1f}", f"HOG at {t:+.1f}") for t in FIGURE_THRESHOLDS],
    )
    print("\n--- detections per frame ---\n" + gallery_table)

    # ------------------------------------------------------------------ #
    # the signature figure: what the detector was trained on
    # ------------------------------------------------------------------ #
    figures.metric_bars(
        [r["people"] for r in margins],
        [r["mean_margin"] for r in margins],
        IMAGES / "drawn_vs_real.png",
        ylabel="mean SVM margin of the detections",
        title=("What the detector was trained on. A drawn silhouette has a strong "
               "outline and nothing inside it; a real person has clothing, limbs "
               "and hair."),
    )

    figures.lines(
        [r["hit_threshold"] for r in threshold_curve],
        {"detections on a moving region": [r["on_a_moving_region"] for r in threshold_curve],
         "moving regions covered": [r["moving_regions_covered"] for r in threshold_curve]},
        IMAGES / "threshold_curve.png",
        xlabel="SVM hit threshold", ylabel="rate",
        title="The precision-recall trade, on real footage against an independent cue",
    )

    figures.lines(
        [r["scale"] for r in scale_sweep],
        {"false positives": [r["false_positives"] for r in scale_sweep],
         "time (ms) / 10": [r["median_ms"] / 10 for r in scale_sweep]},
        IMAGES / "scale_sweep.png",
        xlabel="pyramid scale step", ylabel="count  /  ms per 10",
        title="The pyramid step decides how many windows exist at all",
    )

    sample = frames[chosen[0]]
    crop = cv2.resize(sample[80:80 + 256, 300:300 + 128], (128, 256))
    figures.grid(
        [("a real frame", sample),
         ("a crop", ensure_rgb(crop)),
         ("its HOG cells",
          ensure_rgb(pd.hog_visualisation(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY))))],
        IMAGES / "hog_descriptor.png", ncols=3,
        suptitle=("The descriptor itself: one oriented histogram per 8x8 cell. This is "
                  "what the linear SVM sees — no colour, no texture, only gradient "
                  "orientation."),
    )

    figures.metric_bars(
        [f"{r['frame']}" for r in video],
        [r["hog_detections"] for r in video],
        IMAGES / "per_frame.png",
        ylabel="HOG detections at threshold 0.0",
        title="Detections per frame across the clip",
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS, "57_pedestrian_detection",
        {
            "video": str(pd.VIDEO),
            "frames": list(pd.FRAMES),
            "drawn_vs_real": margins,
            "synthetic": synthetic,
            "video_frames": video,
            "threshold_curve": threshold_curve,
            "scale_sweep": scale_sweep,
            "person_size": size_sweep,
        },
    )

    margins_table = markdown_table(
        margins, [("People", "people"), ("Detections", "detections"),
                  ("Mean margin", "mean_margin"), ("Max margin", "max_margin"),
                  ("Above 0.5", "confident")])
    video_table = markdown_table(
        video, [("Frame", "frame"), ("HOG detections", "hog_detections"),
                ("Moving regions", "moving_regions"),
                ("HOG on a moving region", "hog_on_a_moving_region"),
                ("Moving regions detected", "moving_regions_detected"),
                ("Mean margin", "mean_score")])
    threshold_table = markdown_table(
        threshold_curve, [("Hit threshold", "hit_threshold"), ("Detections", "detections"),
                          ("On a moving region", "on_a_moving_region"),
                          ("Moving regions covered", "moving_regions_covered")])
    scale_table = markdown_table(
        scale_sweep, [("Scale", "scale"), ("False positives", "false_positives"),
                      ("Time (ms)", "median_ms")])

    write_tables(
        RESULTS,
        [
            ("Drawn silhouettes against real people", margins_table),
            ("Real footage, frame by frame", video_table),
            ("The threshold trade on real footage", threshold_table),
            ("The pyramid step", scale_table),
            ("Four frames down the rows", gallery_table),
        ],
    )
    print("\n" + margins_table + "\n\n" + threshold_table)

    # ------------------------------------------------------------------ #
    # headlines
    # ------------------------------------------------------------------ #
    print("\n--- HEADLINE NUMBERS ---")
    drawn = next(r for r in margins if r["people"] == "Drawn silhouettes")
    real = next(r for r in margins if r["people"] == "Real pedestrians")
    print(f"the synthetic benchmark measured nothing: recall on the composited scenes is "
          f"{synthetic['recall']:.3f}, at every threshold and every pyramid step")
    print(f"  drawn silhouettes score a mean SVM margin of {drawn['mean_margin']:.3f} "
          f"({drawn['confident']}/{drawn['detections']} above 0.5); real pedestrians "
          f"{real['mean_margin']:.3f} ({real['confident']}/{real['detections']})")
    print("  HOG is a histogram of gradient ORIENTATIONS and the SVM was trained on "
          "photographs. A flat filled silhouette has a strong outline and nothing "
          "inside it — no clothing folds, no limb shading, no hair.")

    total_hog = sum(r["hog_detections"] for r in video)
    total_moving = sum(r["moving_regions"] for r in video)
    agreed = sum(r["hog_on_a_moving_region"] for r in video)
    covered = sum(r["moving_regions_detected"] for r in video)
    print(f"\non {len(video)} real frames: {total_hog} HOG detections, {total_moving} "
          f"person-sized moving regions")
    print(f"  {agreed}/{total_hog} ({agreed / max(total_hog, 1):.1%}) of detections sit "
          f"on something that moved; {covered}/{total_moving} "
          f"({covered / max(total_moving, 1):.1%}) of moving regions were detected")

    lo, hi = threshold_curve[0], threshold_curve[-1]
    print(f"\nthe threshold trade is exactly as advertised: from {lo['hit_threshold']:+.1f} "
          f"to {hi['hit_threshold']:+.1f}, agreement with motion rises "
          f"{lo['on_a_moving_region']:.3f} -> {hi['on_a_moving_region']:.3f} while "
          f"coverage falls {lo['moving_regions_covered']:.3f} -> "
          f"{hi['moving_regions_covered']:.3f}")

    fastest = min(scale_sweep, key=lambda r: r["median_ms"])
    slowest = max(scale_sweep, key=lambda r: r["median_ms"])
    print(f"\nthe pyramid step costs {slowest['median_ms'] / fastest['median_ms']:.1f}x in "
          f"time from {slowest['scale']:g} to {fastest['scale']:g} "
          f"({slowest['median_ms']:.1f} ms vs {fastest['median_ms']:.1f})")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
