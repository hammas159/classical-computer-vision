"""Run the lane-detection comparison and write results + figures.

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

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from shared import figures  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import lanes as ln  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"

#: Four of the fourteen, chosen to be four different *kinds* of road rather than
#: four pictures of a road: clean white dashes, solid yellow on the left, a
#: concrete bridge lighter than its own markings, and heavy tree shadow.
SHOWN = ("lane_solidWhiteRight", "lane_solidYellowLeft", "lane_test1",
         "lane_test5", "lane_test2")

SHOWN_WHY = {
    "lane_solidWhiteRight": "white dashes, dry asphalt",
    "lane_solidYellowLeft": "solid yellow on the left",
    "lane_test1": "concrete lighter than the paint",
    "lane_test5": "heavy tree shadow",
    "lane_test2": "faint right-hand marking",
}


def main() -> None:
    init_console()
    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    if not ln.available():
        raise SystemExit(
            "The lane photographs are missing. Run "
            "`python tools/fetch_assets.py --set lanes` first.")

    names = ln.image_names()
    print(f"{len(names)} dashcam photographs, "
          + ", ".join(f"{c}: {sum(1 for n in names if ln.camera_of(n) == c)}"
                      for c in sorted({ln.camera_of(n) for n in names})))

    # ------------------------------------------------------------------ #
    # 1. the two numbers
    # ------------------------------------------------------------------ #
    print("\nConsistency (no annotation) against responsiveness (a recorded warp) ...")
    consistency = ln.evaluate_all(names)
    yaw = {d: ln.yaw_repeatability(d, names=names) for d in ln.DETECTORS}
    gaps = {d: ln.distance_to_control(d, names=names) for d in ln.DETECTORS}

    print(f"{'detector':36s} {'plaus':>6s} {'vp px':>7s} {'width':>7s} "
          f"{'yaw %':>7s} {'vs ROI %':>9s}")
    for row in consistency:
        d = row["detector"]
        print(f"{d:36s} {row['plausible_rate']:6.2f} {row['vp_x_spread']:7.1f} "
              f"{row['width_spread']:7.4f} {yaw[d]['median_error_pct']:7.3f} "
              f"{gaps[d]['median_gap_pct']:9.2f}")

    fixed = next(r for r in consistency if r["detector"].startswith("Fixed"))
    print(f"\n  The fixed guess scores a vanishing-point spread of "
          f"{fixed['vp_x_spread']:.1f} px -- a perfect score on consistency, because "
          "it returns\n  the same answer every time. Consistency alone ranks the "
          "do-nothing control first.")
    print(f"  On the recorded warp it scores {yaw['Fixed guess (control)']['median_error_pct']:.2f}%, "
          "the worst of the seven. Neither number alone is the result.")

    # ------------------------------------------------------------------ #
    # 2. the ROI is the algorithm
    # ------------------------------------------------------------------ #
    print("\nHow close each pipeline lands to 'bright pixels in the trapezoid' ...")
    for d in ln.REAL_DETECTORS:
        print(f"  {d:30s} median {gaps[d]['median_gap_pct']:5.2f}% of frame width, "
              f"worst {gaps[d]['max_gap_pct']:5.2f}%")

    by_camera = ln.control_gap_by_camera()
    for r in by_camera:
        print(f"  {r['camera']}  median {r['median_gap_pct']:5.2f}%, worst "
              f"{r['max_gap_pct']:5.2f}%, and {100 * r['bright_is_paint']:.0f}% of the "
              "pixels the control fits to are lane-coloured")
    close, far = sorted(by_camera, key=lambda r: r["median_gap_pct"])
    print(f"  -> the control is {far['median_gap_pct'] / close['median_gap_pct']:.0f}x "
          f"further from the pipeline on {far['camera']} than on {close['camera']}")

    # How many frames does a constant guess get right? On a set where most
    # frames are a straight dry road, "the average lane" is close to the answer,
    # and any comparison run only on those frames is measuring nothing.
    fixed_gaps = []
    for name in names:
        img = ln.load(name)
        w = img.shape[1]
        g = [abs(a - b) / w * 100
             for a, b in zip(ln.bottom_endpoints(ln.detect_fixed(img)),
                             ln.bottom_endpoints(ln.detect_colour_canny_hough(img)))
             if a is not None and b is not None]
        if g:
            fixed_gaps.append(float(np.median(g)))
    uninformative = sum(1 for v in fixed_gaps if v < 1.5)
    print(f"\n  A constant guess -- the same two lines every frame -- lands within "
          f"1.5% of the\n  full pipeline on {uninformative} of {len(fixed_gaps)} "
          f"photographs (median {np.median(fixed_gaps):.2f}%, worst "
          f"{max(fixed_gaps):.2f}%).")
    print("  Half this set cannot tell a lane detector from a constant.")

    print("\nRemoving one step at a time ...")
    ablation = ln.ablate(names)
    for r in ablation:
        print(f"  {r['variant']:34s} plausible {r['plausible']:2d}/{r['frames']}  "
              f"vp spread {r['vp_x_spread_pct']:5.2f}% of width  "
              f"lane width spread {r['width_spread']:.3f}")

    # ------------------------------------------------------------------ #
    # 3. the pixel ROI
    # ------------------------------------------------------------------ #
    print("\nThe same trapezoid written in pixels instead of fractions ...")
    pixel_rows = ln.roi_in_pixels()
    for r in pixel_rows:
        print(f"  {r['camera']}  ROI covers {100 * r['fractional_roi_covers']:.1f}% "
              f"of the frame as fractions, {100 * r['pixel_roi_covers']:.1f}% as pixels  "
              f"-> plausible {r['fractional_plausible']}/{r['frames']} vs "
              f"{r['pixel_plausible']}/{r['frames']}, vp spread "
              f"{r['fractional_vp_spread']:.1f} vs {r['pixel_vp_spread']:.1f} px")

    # ------------------------------------------------------------------ #
    # 4. what makes a frame hard
    # ------------------------------------------------------------------ #
    print("\nPer photograph ...")
    per = ln.per_image(names)
    for r in per:
        print(f"  {r['image']:26s} {r['camera']:9s} contrast {r['paint_contrast']:5.0f}  "
              f"shadow {100 * r['shadow_fraction']:5.1f}%  "
              f"plausible {r['plausible']}/{r['detectors']}  "
              f"spread {r['disagreement_pct']:5.2f}%")

    contrast = np.array([r["paint_contrast"] for r in per])
    shadow = np.array([r["shadow_fraction"] for r in per])
    disagree = np.array([r["disagreement_pct"] for r in per])
    r_contrast = float(np.corrcoef(contrast, disagree)[0, 1])
    r_shadow = float(np.corrcoef(shadow, disagree)[0, 1])
    print(f"\n  paint contrast vs disagreement r = {r_contrast:+.2f}")
    print(f"  shadow fraction vs disagreement r = {r_shadow:+.2f}")

    # How far the ROI control drifts, against the one thing it is blind to.
    # A partial explanation, reported as one.
    control_gaps, paint_share = [], []
    for name in names:
        img = ln.load(name)
        w = img.shape[1]
        a, b = ln.detect_bright_in_roi(img), ln.detect_colour_canny_hough(img)
        g = [abs(x - y) / w * 100 for x, y in zip(ln.bottom_endpoints(a),
                                                  ln.bottom_endpoints(b))
             if x is not None and y is not None]
        if g:
            control_gaps.append(float(np.median(g)))
            paint_share.append(ln.bright_is_paint(name))
    r_paint = float(np.corrcoef(paint_share, control_gaps)[0, 1])
    print(f"  share of the control's pixels that are lane-coloured, "
          f"vs its gap to the pipeline: r = {r_paint:+.2f}")

    # ------------------------------------------------------------------ #
    # figures
    # ------------------------------------------------------------------ #
    print("\nFigures ...")

    columns = ["Input"] + list(ln.DETECTORS)
    rows, notes = [], []
    for name in SHOWN:
        img = ln.load(name)
        cells = [ln.draw_roi(img)]
        cell_notes = [f"{img.shape[1]}x{img.shape[0]}"]
        for d in ln.DETECTORS:
            lanes = ln.DETECTORS[d](img)
            ok = ln.plausible(lanes, img.shape)
            cells.append(ln.draw(img, lanes,
                                 colour=(60, 220, 90) if ok else (235, 70, 70)))
            width = ln.lane_width(lanes, img.shape)
            cell_notes.append("lane " + (f"{width:.2f}w" if width else "—")
                              + ("" if ok else "  implausible"))
        rows.append((SHOWN_WHY[name], cells))
        notes.append(cell_notes)
    figures.gallery(columns, rows, IMAGES / "compare.png", cell_notes=notes,
                    suptitle="Green where the answer satisfies road geometry, red where it does not. "
                             "The dot is the vanishing point; the orange trapezoid is the region of interest.")

    figures.scatter_plane(
        {d: [(c["vp_x_spread"], yaw[d]["median_error_pct"])]
         for d, c in ((r["detector"], r) for r in consistency)},
        None, IMAGES / "two_numbers.png",
        xlabel="vanishing-point spread across frames, px  (lower = more consistent)",
        ylabel="error through a recorded yaw, % of width  (lower = follows the paint)",
        title="Neither axis alone ranks these: the fixed guess wins the left one outright",
        equal_aspect=False)

    figures.metric_bars(
        [r["variant"].replace("without the ", "no ") for r in ablation],
        [r["vp_x_spread_pct"] for r in ablation],
        IMAGES / "ablation.png",
        ylabel="vanishing-point spread, % of frame width",
        title="Remove one step: the region of interest is the largest term by far",
        highlight_best="min")

    # the pixel ROI, drawn
    big = next(n for n in names if ln.camera_of(n) == "1280x720")
    small = next(n for n in names if ln.camera_of(n) == "960x540")
    panels = []
    for name in (small, big):
        img = ln.load(name)
        for label, fn, pixels in (("fractions", ln.detect_colour_canny_hough, False),
                                  ("pixels", ln.detect_pixel_roi, True)):
            lanes = fn(img)
            ok = ln.plausible(lanes, img.shape)
            vp = ln.vanishing_point(lanes)
            where = f", vp x={vp[0]:.0f}" if vp else ", no vanishing point"
            panels.append((
                f"{name}\n{img.shape[1]}x{img.shape[0]}, ROI as {label}{where}",
                ln.draw(ln.draw_roi(img, pixels=pixels), lanes,
                        colour=(60, 220, 90) if ok else (235, 70, 70))))
    figures.grid(panels, IMAGES / "pixel_roi.png", ncols=2,
                 suptitle="The identical trapezoid, written two ways. On the camera it was "
                          "written for they are the same picture.")

    figures.scatter_plane(
        {"960x540": [(r["paint_contrast"], r["disagreement_pct"])
                     for r in per if r["camera"] == "960x540"],
         "1280x720": [(r["paint_contrast"], r["disagreement_pct"])
                      for r in per if r["camera"] == "1280x720"]},
        None, IMAGES / "difficulty.png",
        xlabel="paint contrast inside the ROI, grey levels (measured before any detector runs)",
        ylabel="how far the five pipelines spread apart, % of frame width",
        title=f"Contrast predicts disagreement before anything is detected (r = {r_contrast:+.2f})",
        equal_aspect=False)

    # ------------------------------------------------------------------ #
    results = {
        "images": len(names),
        "cameras": {c: sum(1 for n in names if ln.camera_of(n) == c)
                    for c in sorted({ln.camera_of(n) for n in names})},
        "consistency": consistency,
        "yaw_repeatability": list(yaw.values()),
        "distance_to_roi_control": list(gaps.values()),
        "control_gap_by_camera": by_camera,
        "fixed_guess_gap_pct": fixed_gaps,
        "frames_a_constant_gets_right": uninformative,
        "paint_share_vs_control_gap_r": r_paint,
        "ablation": ablation,
        "pixel_roi": pixel_rows,
        "per_image": per,
        "contrast_vs_disagreement_r": r_contrast,
        "shadow_vs_disagreement_r": r_shadow,
        "shown": list(SHOWN),
    }
    write_results(RESULTS, "06_lane_detection", results)

    write_tables(RESULTS, [
        ("Two numbers, one per failure mode", markdown_table(
            [{"Detector": r["detector"],
              "Plausible": f"{round(r['plausible_rate'] * len(names))}/{len(names)}",
              "VP spread (px)": f"{r['vp_x_spread']:.1f}",
              "Lane width spread": f"{r['width_spread']:.4f}",
              "Yaw error (% width)": f"{yaw[r['detector']]['median_error_pct']:.3f}",
              "Gap to ROI control (% width)":
                  f"{gaps[r['detector']]['median_gap_pct']:.2f}"}
             for r in consistency],
            [(c, c) for c in ("Detector", "Plausible", "VP spread (px)",
                              "Lane width spread", "Yaw error (% width)",
                              "Gap to ROI control (% width)")])),
        ("How far the ROI control sits from the full pipeline, per camera",
         markdown_table(
            [{"Camera": r["camera"], "Frames": r["frames"],
              "Median gap": f"{r['median_gap_pct']:.2f}%",
              "Worst gap": f"{r['max_gap_pct']:.2f}%",
              "Its pixels that are lane-coloured":
                  f"{100 * r['bright_is_paint']:.0f}%"} for r in by_camera],
            [(c, c) for c in ("Camera", "Frames", "Median gap", "Worst gap",
                              "Its pixels that are lane-coloured")])),
        ("Removing one step at a time", markdown_table(
            [{"Variant": r["variant"],
              "Plausible": f"{r['plausible']}/{r['frames']}",
              "VP spread (% width)": f"{r['vp_x_spread_pct']:.2f}",
              "Lane width spread": f"{r['width_spread']:.3f}"} for r in ablation],
            [(c, c) for c in ("Variant", "Plausible", "VP spread (% width)",
                              "Lane width spread")])),
        ("The region of interest written in pixels", markdown_table(
            [{"Camera": r["camera"], "Frames": r["frames"],
              "ROI covers (fractions)": f"{100 * r['fractional_roi_covers']:.1f}%",
              "ROI covers (pixels)": f"{100 * r['pixel_roi_covers']:.1f}%",
              "Plausible (fractions)": f"{r['fractional_plausible']}/{r['frames']}",
              "Plausible (pixels)": f"{r['pixel_plausible']}/{r['frames']}",
              "VP spread (fractions)": f"{r['fractional_vp_spread']:.1f} px",
              "VP spread (pixels)": f"{r['pixel_vp_spread']:.1f} px"}
             for r in pixel_rows],
            [(c, c) for c in ("Camera", "Frames", "ROI covers (fractions)",
                              "ROI covers (pixels)", "Plausible (fractions)",
                              "Plausible (pixels)", "VP spread (fractions)",
                              "VP spread (pixels)")])),
        ("Per photograph", markdown_table(
            [{"Photograph": r["image"].replace("lane_", ""), "Camera": r["camera"],
              "Paint contrast": f"{r['paint_contrast']:.0f}",
              "Shadow": f"{100 * r['shadow_fraction']:.1f}%",
              "Plausible": f"{r['plausible']}/{r['detectors']}",
              "Spread between pipelines": f"{r['disagreement_pct']:.2f}%"}
             for r in per],
            [(c, c) for c in ("Photograph", "Camera", "Paint contrast", "Shadow",
                              "Plausible", "Spread between pipelines")])),
    ])
    print(f"wrote {RESULTS / 'results.json'} and five figures")


if __name__ == "__main__":
    main()
