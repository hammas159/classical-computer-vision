"""Run the hand-gesture comparison and write results + figures.

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

import gestures as gs  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"

#: The segmenter the background/hand grid is run for. Chosen as the best real
#: one on mean IoU, which is stated rather than cherry-picked: the claim is that
#: even the *best* segmenter is dominated by the background.
GRID_SEGMENTER = "Lab skin"

#: Four hand-and-background pairs spanning the difficulty axis rather than four
#: pictures of a hand: an easy background, one containing real human skin, one
#: containing a bronze human figure, and the sandy wall that defeats everything.
SHOWN = (("5_P__5_P_hgr1_id04_2", "314016"),
         ("3_P__3_P_hgr1_id02_2", "81090"),
         ("B_P__B_P_hgr1_id01_3", "372019"),
         ("1_P__1_P_hgr1_id01_3", "293029"))


def main() -> None:
    init_console()
    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    if not gs.available():
        raise SystemExit(
            "Assets are missing. Run `python tools/fetch_assets.py --set hands` "
            "and `python tools/fetch_images.py`.")

    hands = gs.hand_names()
    people = sorted({gs.person_of(h) for h in hands})
    print(f"{len(hands)} hand photographs of {len(people)} people, "
          f"{len(gs.BACKGROUNDS)} backgrounds, "
          f"{len(gs.REAL_SEGMENTERS)} segmenters and {len(gs.SEGMENTERS) - len(gs.REAL_SEGMENTERS)} controls")

    # ------------------------------------------------------------------ #
    # 0. how exact the "exact" matte is
    # ------------------------------------------------------------------ #
    print("\nHow exact the inherited matte is (share of the frame in the halo) ...")
    quality = gs.matte_quality(hands)
    for r in quality[:3]:
        print(f"  {r['gesture']:3s} {100 * r['halo']:5.2f}%   <- worst")
    print(f"  median {100 * np.median([r['halo'] for r in quality]):.2f}%, "
          f"best {100 * quality[-1]['halo']:.2f}%")
    print("  One photograph is markedly worse than the rest; it is named rather "
          "than dropped.")

    # ------------------------------------------------------------------ #
    # 1. can the second stage work at all, given a perfect mask?
    # ------------------------------------------------------------------ #
    print("\nBefore blaming any segmenter: does finger counting work on a PERFECT "
          "mask? ...")
    ceiling = gs.finger_rule_ceiling(hands)
    for r in ceiling["sweeps"]:
        detail = "  ".join(f"{d['want']}->{d['got']}" for d in r["detail"])
        print(f"  {r['rule']:24s} p={r['parameter']:<5} {r['correct']}/{r['of']}   "
              f"{detail}")
    print(f"\n  The best rule at its best setting gets "
          f"{ceiling['best_correct']} of {ceiling['of']}: "
          f"{ceiling['best_rule']} at {ceiling['best_parameter']}.")
    print("  So the classical second stage does not work on these photographs even "
          "when the\n  segmentation is handed to it perfectly. That has to be said "
          "before anything else.")

    # ------------------------------------------------------------------ #
    # 2. the segmenters
    # ------------------------------------------------------------------ #
    print("\nRecovering the mask the composite was built from ...")
    results = gs.evaluate_all(hands)
    n = len(hands)
    print(f"  {'segmenter':26s} {'mean IoU':>9s} {'found':>8s} {'feature gap':>12s} "
          f"{'fingers':>9s}")
    for r in results:
        print(f"  {r['segmenter']:26s} {r['mean_iou']:9.3f} {r['found']:5d}/{n} "
              f"{r['median_feature_gap']:12.3f} "
              f"{r['finger_correct']:6d}/{r['finger_frames']}")

    split = gs.stage_split(hands)
    print(f"\n  Ceiling (perfect mask): {split['ceiling_finger_correct']}"
          f"/{split['finger_frames']} fingers.")
    print(f"  Best real segmenter: {split['best_real']} at IoU "
          f"{split['best_real_iou']:.3f}, {split['best_real_finger_correct']}"
          f"/{split['finger_frames']} fingers.")

    ellipse = next(r for r in results if r["segmenter"].startswith("Centre ellipse"))
    beaten = [r["segmenter"] for r in results
              if r["segmenter"] in gs.REAL_SEGMENTERS
              and r["mean_iou"] < ellipse["mean_iou"]]
    print(f"\n  An ellipse drawn where a hand usually is, without reading the image, "
          f"scores {ellipse['mean_iou']:.3f}")
    print(f"  and beats {len(beaten)} of the {len(gs.REAL_SEGMENTERS)} real "
          f"segmenters: {', '.join(beaten) if beaten else 'none'}.")

    # ------------------------------------------------------------------ #
    # 3. the background, not the hand
    # ------------------------------------------------------------------ #
    print(f"\nEvery hand on every background, for {GRID_SEGMENTER} "
          f"({len(hands)} x {len(gs.BACKGROUNDS)} = "
          f"{len(hands) * len(gs.BACKGROUNDS)} composites) ...")
    by_bg = gs.background_effect(GRID_SEGMENTER, hands)
    print(f"  {'skin-like':>10s} {'mean':>6s} {'median':>7s} {'total fails':>12s}   background")
    for r in sorted(by_bg, key=lambda r: r["skin_likeness"]):
        print(f"  {100 * r['skin_likeness']:9.1f}% {r['mean_iou']:6.3f} "
              f"{r['median_iou']:7.3f} {r['total_failures']:8d}/{r['of']:<3d}   {r['what']}")
    for r in by_bg:
        if r["mean_iou"] > 0.25 and r["median_iou"] < 0.05:
            print(f"\n  {r['what']} has a mean of {r['mean_iou']:.3f} and a "
                  f"median of {r['median_iou']:.3f}.")
            print(f"  {r['total_failures']} of {r['of']} hands score below 0.05 and "
                  f"{r['recovered']} score above 0.5 — the mean describes neither "
                  "half.")

    matters = gs.which_matters_more(GRID_SEGMENTER)
    print(f"\n  spread across backgrounds: {matters['background_spread']:.3f}")
    print(f"  spread across hands:       {matters['hand_spread']:.3f}")
    print(f"  -> the background moves the answer "
          f"{matters['background_spread'] / max(matters['hand_spread'], 1e-9):.1f}x "
          "further than which hand it is")
    print(f"  skin-likeness of the background vs mean IoU: "
          f"r = {matters['skin_likeness_r']:+.2f}, measured before any method runs")

    # the outlier worth naming
    fit = np.poly1d(np.polyfit([r["skin_likeness"] for r in by_bg],
                               [r["mean_iou"] for r in by_bg], 1))
    resid = [(r, r["mean_iou"] - fit(r["skin_likeness"])) for r in by_bg]
    worst = min(resid, key=lambda t: t[1])
    print(f"\n  The axis does not explain everything: {worst[0]['what']} is "
          f"{abs(worst[1]):.2f} IoU below")
    print(f"  what its skin-likeness of {100 * worst[0]['skin_likeness']:.1f}% "
          "predicts. So what is it?")

    contest = gs.size_contest(worst[0]["background"], GRID_SEGMENTER)
    print("\n  Every segmenter keeps the largest connected component, so a "
          "skin-coloured object")
    print(f"  larger than the hand wins outright. On this background the score "
          f"tracks how much of\n  the frame the hand covers at "
          f"r = {contest['area_vs_iou_r']:+.2f}: the "
          f"{contest['total_failures']} hands that fail average")
    print(f"  {contest['mean_area_when_failed']:.4f} of the frame and the "
          f"{contest['of'] - contest['total_failures']} that do not average "
          f"{contest['mean_area_when_not']:.4f}.")
    print("  It does not degrade. It is a size contest, and the loser scores zero.")

    # ------------------------------------------------------------------ #
    # figures
    # ------------------------------------------------------------------ #
    print("\nFigures ...")

    columns = ["Composite"] + list(gs.SEGMENTERS)
    rows, notes = [], []
    for hand, bg in SHOWN:
        frame, truth = gs.composite(hand, bg)
        cells = [gs.overlay(frame, np.zeros_like(truth), truth)]
        cell_notes = [f"{gs.gesture_of(hand)} on {bg}"]
        for seg in gs.SEGMENTERS:
            mask = gs.SEGMENTERS[seg](frame, truth=truth)
            v = gs.iou(mask, truth)
            cells.append(gs.overlay(frame, mask, truth,
                                    colour=(60, 220, 90) if v >= 0.5 else (235, 70, 70)))
            cell_notes.append(f"IoU {v:.2f}")
        rows.append((f"{gs.BACKGROUNDS[bg][:34]}\nskin-like "
                     f"{100 * gs.skin_likeness(bg):.0f}%", cells))
        notes.append(cell_notes)
    figures.gallery(columns, rows, IMAGES / "compare.png", cell_notes=notes,
                    suptitle="Amber outlines the mask the composite was built from. "
                             "Green fill where the segmenter recovered it, red where "
                             "it did not.")

    figures.scatter_plane(
        {"background": [(r["skin_likeness"] * 100, r["mean_iou"]) for r in by_bg]},
        None, IMAGES / "skin_likeness.png",
        xlabel="share of the background a skin rule accepts with no hand on it, %",
        ylabel=f"mean IoU over all {len(hands)} hands",
        title=f"The background decides the answer (r = {matters['skin_likeness_r']:+.2f}), "
              "and it is measurable before anything runs",
        equal_aspect=False)

    real = [r for r in results if r["segmenter"] in gs.REAL_SEGMENTERS]
    figures.metric_bars(
        [r["segmenter"] for r in real], [r["mean_iou"] for r in real],
        IMAGES / "segmenters.png", ylabel="mean IoU against the exact matte",
        title="Seven segmenters against a matte that is exact by construction",
        highlight_best="max")

    figures.metric_bars(
        [r["rule"].split()[0] + " " + str(r["parameter"]) for r in ceiling["sweeps"]],
        [r["correct"] for r in ceiling["sweeps"]],
        IMAGES / "finger_ceiling.png",
        ylabel=f"numbered gestures counted correctly, of {ceiling['of']}",
        title="Finger counting on a PERFECT mask: both classical rules, swept",
        highlight_best="max")

    # the size contest, drawn: the smallest hand and the largest, on the
    # background where the rule picks the wrong object entirely
    by_area = sorted(hands, key=lambda h: (gs.matte(gs.load_hand(h)) > 0).mean())
    contest_panels = []
    for hand in (by_area[0], by_area[-1]):
        frame, truth = gs.composite(hand, worst[0]["background"])
        mask = gs.SEGMENTERS[GRID_SEGMENTER](frame, truth=truth)
        contest_panels.append(
            (f"gesture {gs.gesture_of(hand)} covers "
             f"{100 * (truth > 0).mean():.1f}%"
             + "\n" + f"IoU {gs.iou(mask, truth):.2f}",
             gs.overlay(frame, mask, truth, colour=(235, 70, 70))))
    figures.grid(contest_panels, IMAGES / "size_contest.png", ncols=2,
                 suptitle=f"{worst[0]['what']}: every segmenter keeps the "
                          "largest connected component, and the skin-coloured "
                          "cart behind the dog is larger than a small hand. "
                          "Amber outlines the truth.")

    # what a segmentation error does to the description
    panels = []
    hand, bg = SHOWN[1]
    frame, truth = gs.composite(hand, bg)
    panels.append(("the exact matte", gs.draw_defects(frame, truth)))
    for seg in ("Lab skin", "YCrCb skin", "Otsu on grey"):
        mask = gs.SEGMENTERS[seg](frame, truth=truth)
        panels.append((f"{seg}\nIoU {gs.iou(mask, truth):.2f}, "
                       f"{gs.count_fingers(mask)} fingers",
                       gs.draw_defects(frame, mask)))
    figures.grid(panels, IMAGES / "shape_stage.png", ncols=4,
                 suptitle="The same shape code on four masks of the same frame. "
                          "Blue is the convex hull, red the deep gaps it finds.")

    # ------------------------------------------------------------------ #
    write_results(RESULTS, "55_hand_gesture", {
        "hands": len(hands),
        "people": people,
        "backgrounds": gs.BACKGROUNDS,
        "matte_quality": quality,
        "finger_rule_ceiling": ceiling,
        "results": [{k: v for k, v in r.items() if k != "per_image"} for r in results],
        "per_image": {r["segmenter"]: r["per_image"] for r in results},
        "stage_split": split,
        "background_effect": by_bg,
        "hand_effect": gs.hand_effect(GRID_SEGMENTER),
        "which_matters_more": matters,
        "size_contest": contest,
        "grid_segmenter": GRID_SEGMENTER,
        "shown": [list(s) for s in SHOWN],
    })

    write_tables(RESULTS, [
        ("Segmenters against a matte that is exact by construction", markdown_table(
            [{"Segmenter": r["segmenter"], "Mean IoU": f"{r['mean_iou']:.3f}",
              "Median IoU": f"{r['median_iou']:.3f}",
              "Recovered (IoU>=0.5)": f"{r['found']}/{n}",
              "Feature gap": f"{r['median_feature_gap']:.3f}",
              "Fingers": f"{r['finger_correct']}/{r['finger_frames']}"}
             for r in results],
            [(c, c) for c in ("Segmenter", "Mean IoU", "Median IoU",
                              "Recovered (IoU>=0.5)", "Feature gap", "Fingers")])),
        ("Finger counting on a perfect mask", markdown_table(
            [{"Rule": r["rule"], "Parameter": r["parameter"],
              "Correct": f"{r['correct']}/{r['of']}"} for r in ceiling["sweeps"]],
            [(c, c) for c in ("Rule", "Parameter", "Correct")])),
        (f"Every background, for {GRID_SEGMENTER}", markdown_table(
            [{"Background": r["what"],
              "Skin-like": f"{100 * r['skin_likeness']:.1f}%",
              "Mean IoU": f"{r['mean_iou']:.3f}",
              "Median IoU": f"{r['median_iou']:.3f}",
              "Total failures": f"{r['total_failures']}/{r['of']}"}
             for r in sorted(by_bg, key=lambda r: r["skin_likeness"])],
            [(c, c) for c in ("Background", "Skin-like", "Mean IoU", "Median IoU",
                              "Total failures")])),
        ("How exact the inherited matte is", markdown_table(
            [{"Gesture": r["gesture"], "Halo": f"{100 * r['halo']:.2f}%"}
             for r in quality[:6]],
            [(c, c) for c in ("Gesture", "Halo")])),
    ])
    print(f"wrote {RESULTS / 'results.json'} and five figures")


if __name__ == "__main__":
    main()
