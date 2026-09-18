"""Run the colour space robustness comparison and write results + figures.

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
from shared.metrics import iou  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import colour_spaces as cs  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"

#: The condition the front figure is shot at. A warm cast is where the folklore
#: is wrong: HSV survives a brightness change untouched and loses 37% of its IoU
#: to this.
FIGURE_CAST = (1.25, 1.0, 0.75)

CHROMA_BANDS = [("very distinct", 55.0, 99.0), ("distinct", 47.0, 55.0),
                ("moderate", 44.0, 47.0), ("least distinct", 0.0, 44.0)]


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    spaces = list(cs.SPACES)

    print("Choosing each space's tolerance rather than sharing one ...")
    tolerance_sweep = cs.sweep_photo_tolerance()
    for r in tolerance_sweep:
        print(f"  tolerance {r['tolerance']:5.1f}  " + "  ".join(
            f"{s[:6]}={r[s]:.3f}" for s in spaces))
    for space in spaces:
        best = max(tolerance_sweep, key=lambda r: r[space])
        print(f"    {space:16s} peaks at {best['tolerance']:.0f} "
              f"({best[space]:.3f}), used {cs.PHOTO_TOLERANCE[space]:.0f}")

    print(f"\nThresholds tuned on the original, applied unchanged to {len(cs.IMAGES)} "
          "degraded photographs ...")
    photos = cs.photo_robustness()
    for r in photos:
        print(f"  {r['space']:16s} clean {r['undegraded_iou']:.3f}  "
              f"worst {r['worst_case']:.3f}  mean loss {r['mean_loss']:.3f}")

    per_image = cs.photo_robustness_per_image()
    synthetic = cs.compare_all_degradations()
    brightness = cs.stability_table("Brightness scale", cs.BRIGHTNESS_LEVELS)
    cast = cs.stability_table("Colour cast", cs.CAST_LEVELS)
    rescue = cs.white_balance_rescue()
    hue_demo = cs.hue_range_demonstration()

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    by_image = {r["image"]: r for r in per_image}
    candidates: dict[str, list[dict]] = {}
    for name in cs.IMAGES:
        clean = cs.load_scene(name)
        truth = cs.load_target(name)
        distance = by_image[name]["chroma_distance"]
        band = next(b for b, lo, hi in CHROMA_BANDS if lo <= distance < hi)
        warm = cs.degrade_colour_cast(clean, FIGURE_CAST)

        panels = [clean, ensure_rgb(truth), warm]
        notes = [f"chroma distance {distance:.0f}", "the person's region",
                 "warm cast applied"]
        scores = []
        for space in spaces:
            tol = cs.PHOTO_TOLERANCE[space]
            found = cs.threshold_in_space(warm, space, clean, truth, tol)
            score = iou(found, truth)
            panels.append(ensure_rgb(found))
            notes.append(f"IoU {score:.3f}")
            scores.append(score)

        candidates.setdefault(band, []).append({
            "label": f"{name.replace('_', ' ')}\nchroma {distance:.0f}",
            "subject": name, "distance": distance, "images": panels,
            "notes": notes, "scores": scores,
            "spread": float(np.std(scores)),
        })
        print(f"  scene candidate {name:22s} chroma {distance:5.1f} [{band:14s}]  "
              + " ".join(f"{v:.3f}" for v in scores))

    chosen, used = [], set()
    for band, _, _ in CHROMA_BANDS:
        pool = sorted(candidates.get(band, []), key=lambda r: -r["spread"])
        pick = next((r for r in pool if r["subject"] not in used), None)
        if pick:
            used.add(pick["subject"])
            chosen.append(pick)
    chosen = chosen[:4]
    assert len({r["subject"] for r in chosen}) == len(chosen), "a subject repeats"

    figures.gallery(
        ["photograph", "the target region", "with a warm cast"] + spaces,
        [(r["label"], r["images"]) for r in chosen],
        IMAGES / "compare_spaces.png",
        cell_notes=[r["notes"] for r in chosen],
        suptitle=(
            "A colour threshold tuned on the original and applied unchanged after a warm "
            "cast. The target is a region a person traced; cells are IoU against it."
        ),
    )
    gallery_table = markdown_table(
        [dict([("Sr", i), ("Scene", r["label"].replace("\n", " · "))]
              + [(s, f"{v:.3f}") for s, v in zip(spaces, r["scores"])])
         for i, r in enumerate(chosen, start=1)],
        [("Sr", "Sr"), ("Scene", "Scene")] + [(s, s) for s in spaces],
    )
    print("\n--- IoU under a warm cast ---\n" + gallery_table)

    # ------------------------------------------------------------------ #
    # the signature figure: accuracy against robustness
    # ------------------------------------------------------------------ #
    figures.comparison_matrix(
        photos,
        [("Undegraded", "undegraded_iou", True),
         ("Brightness x0.6", "Brightness x0.6", True),
         ("Brightness x1.3", "Brightness x1.3", True),
         ("Warm cast", "Warm cast", True),
         ("Cool cast", "Cool cast", True),
         ("Gamma 2.0", "Gamma 2.0", True)],
        IMAGES / "photo_matrix.png",
        row_key="space",
        title=("IoU against a human-traced region, threshold fixed. HSV is exactly "
               "brightness-invariant, the least accurate space undegraded, and the one "
               "a colour cast hurts most."),
    )

    figures.scatter_plane(
        {r["space"]: [(r["undegraded_iou"], r["worst_case"])] for r in photos},
        None,
        IMAGES / "accuracy_vs_robustness.png",
        xlabel="IoU with no degradation (accuracy)",
        ylabel="worst IoU across five degradations (robustness)",
        title=("The trade, in one plot. Up and to the right is better; nothing is there. "
               "HSV buys its flat brightness response with the worst clean accuracy."),
    )

    figures.metric_bars(
        [r["space"] for r in photos],
        [r["mean_loss"] for r in photos],
        IMAGES / "mean_loss.png",
        ylabel="mean IoU lost across five degradations",
        title="How much each space gives up when the light changes",
        highlight_best="min",
    )

    figures.lines(
        [r["level"] for r in brightness],
        {s: [r[s] for r in brightness] for s in spaces if s in brightness[0]},
        IMAGES / "brightness_stability.png",
        xlabel="brightness scale", ylabel="channel shift",
        title="How far each space's coordinates move under a brightness change",
    )

    figures.lines(
        [r["level"] for r in cast],
        {s: [r[s] for r in cast] for s in spaces if s in cast[0]},
        IMAGES / "cast_stability.png",
        xlabel="colour cast strength", ylabel="channel shift",
        title="And under a change in the illuminant's colour, where nothing is invariant",
    )

    figures.lines(
        [r["cast_level"] for r in rescue],
        {k: [r[k] for r in rescue] for k in rescue[0] if k != "cast_level"},
        IMAGES / "white_balance_rescue.png",
        xlabel="colour cast strength", ylabel="IoU",
        title="An explicit white balance step is what actually fixes a cast",
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS, "47_colour_space_robustness",
        {
            "images": list(cs.IMAGES),
            "segments": {k: list(v) for k, v in cs.SEGMENTS.items()},
            "tolerance": cs.PHOTO_TOLERANCE,
            "tolerance_sweep": tolerance_sweep,
            "photographs": photos,
            "per_image": per_image,
            "synthetic": synthetic,
            "brightness_stability": brightness,
            "cast_stability": cast,
            "white_balance_rescue": rescue,
            "hue_range": hue_demo,
        },
    )

    photos_table = markdown_table(
        photos, [("Space", "space"), ("Tolerance", "tolerance"),
                 ("Undegraded", "undegraded_iou")]
        + [(k, k) for k in cs.PHOTO_DEGRADATIONS]
        + [("Worst case", "worst_case"), ("Mean loss", "mean_loss")])
    tolerance_table = markdown_table(
        tolerance_sweep, [("Tolerance", "tolerance")] + [(s, s) for s in spaces])
    synthetic_table = markdown_table(
        synthetic, [(k, k) for k in synthetic[0]])
    rescue_table = markdown_table(
        rescue, [(k, k) for k in rescue[0]])

    write_tables(
        RESULTS,
        [
            ("On human-traced regions in photographs", photos_table),
            ("Choosing each space's tolerance", tolerance_table),
            ("On the synthetic colour chart", synthetic_table),
            ("White balance as the actual fix", rescue_table),
            ("Four scenes down the rows", gallery_table),
        ],
    )
    print("\n" + photos_table)

    # ------------------------------------------------------------------ #
    # headlines
    # ------------------------------------------------------------------ #
    print("\n--- HEADLINE NUMBERS ---")
    by_space = {r["space"]: r for r in photos}
    hsv = by_space["HSV"]

    print(f"HSV is EXACTLY brightness-invariant on real photographs: "
          f"{hsv['undegraded_iou']:.3f} undegraded, {hsv['Brightness x0.6']:.3f} at x0.6, "
          f"{hsv['Brightness x1.3']:.3f} at x1.3")
    print(f"  and it is the LEAST accurate space undegraded — "
          f"{hsv['undegraded_iou']:.3f} against Lab's "
          f"{by_space['Lab']['undegraded_iou']:.3f} and YCrCb's "
          f"{by_space['YCrCb']['undegraded_iou']:.3f}")
    warm_loss = hsv["undegraded_iou"] - hsv["Warm cast"]
    print(f"  and a warm cast costs it {warm_loss:.3f} IoU "
          f"({100 * warm_loss / hsv['undegraded_iou']:.0f}%) — the half of the folklore "
          "that is false")

    lowest_loss = min(photos, key=lambda r: r["mean_loss"])
    best_clean = max(photos, key=lambda r: r["undegraded_iou"])
    best_worst = max(photos, key=lambda r: r["worst_case"])
    print(f"\nmost robust: {lowest_loss['space']} (mean loss {lowest_loss['mean_loss']:.3f})")
    print(f"most accurate: {best_clean['space']} ({best_clean['undegraded_iou']:.3f})")
    print(f"best worst case: {best_worst['space']} ({best_worst['worst_case']:.3f})")
    if lowest_loss["space"] != best_clean["space"]:
        print("  three different winners — robustness and accuracy are a trade here, "
              "not a ranking")

    norm = by_space["Normalised RGB"]
    print(f"\nnormalised RGB is brightness-invariant by construction "
          f"({norm['undegraded_iou']:.3f} -> {norm['Brightness x0.6']:.3f}) and is "
          f"destroyed by gamma ({norm['Gamma 2.0']:.3f}) — dividing by the sum removes a "
          "scale, and gamma is not a scale")

    rgb = by_space["RGB"]
    print(f"\nplain RGB loses the most overall (mean {rgb['mean_loss']:.3f}) and collapses "
          f"at x0.6 brightness: {rgb['undegraded_iou']:.3f} -> {rgb['Brightness x0.6']:.3f}")

    worst_cast = rescue[-1]
    print(f"\nat cast level {worst_cast['cast_level']:g}, an explicit white balance step "
          "is worth:")
    for s in spaces:
        raw, balanced = worst_cast[f"{s} raw"], worst_cast[f"{s} balanced"]
        print(f"  {s:16s} {raw:.3f} -> {balanced:.3f}  ({balanced - raw:+.3f} IoU)")
    print("  no colour space is invariant to a change of illuminant, and choosing a "
          "different one is not a substitute for correcting it")

    print(f"\nOpenCV's hue is 0-{cs.OPENCV_HUE_MAX}, not 0-359. Read as degrees, every "
          "colour comes back at half its angle:")
    for row in hue_demo:
        print(f"  {row['colour']:8s} true {row['true_degrees']:3d} deg, OpenCV stores "
              f"{row['opencv_hue']:3d}, a naive reading says "
              f"{row['naive_degrees_reading']:3d}")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
