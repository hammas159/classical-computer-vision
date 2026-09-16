"""Run the coin counting and measurement experiments, and write results + figures.

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

from shared import figures, io, synth  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import coins as cn  # noqa: E402

#: Scale of the generated scenes, in millimetres per pixel. At 0.42 the
#: 1.07 mm that separates a 1-rupee coin from a 5-rupee one is about 2.5 px,
#: which is the regime where identification is a measurement problem rather
#: than a lookup.
SCENE_MM_PER_PX = 0.42

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"


def label_overlay(img: np.ndarray, labels: np.ndarray, props: list[dict]) -> np.ndarray:
    """Tint each region a distinct colour and number it at its centroid."""
    rng = np.random.default_rng(0)
    colour = np.zeros_like(img)
    for i in range(1, int(labels.max()) + 1):
        colour[labels == i] = rng.integers(70, 255, 3)
    blend = cv2.addWeighted(img, 0.55, colour, 0.45, 0)
    for j, p in enumerate(sorted(props, key=lambda q: q["centroid"][1]), start=1):
        cx, cy = map(int, p["centroid"])
        cv2.putText(blend, str(j), (cx - 8, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return blend


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    img = io.sample("coins")

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README, and the half that counting
    # cannot answer: which coin is it
    # ------------------------------------------------------------------ #
    # These scenes are GENERATED, and that is the only reason this project can
    # answer "which coin" rather than only "how many". `skimage.data.coins` is a
    # scan of old Greek coins whose values nobody recorded, so a denomination
    # reported against it could never be shown to be wrong. Here the mint's
    # published diameters set every coin, so the count, the millimetres and the
    # denomination are all known by construction.
    # The gate asks whether a scene is SOLVABLE, not whether it is easy. Some
    # method has to be able to count it; a scene no method can read is a broken
    # sample and goes. A scene that only one method can read is the opposite of
    # broken -- it is the entire result, and gating on the median threw exactly
    # those away, leaving three rows of well-separated coins.
    #
    # Selection is then by family, where a family is the CONDITION rather than
    # the tablecloth: four scenes on four backgrounds are four variations of one
    # problem, while spread, touching, side-lit and a light surface are four
    # different problems.
    GALLERY_MIN_COUNT_ACCURACY = 0.75
    scene_pool = [
        ("green felt\nspread out", dict(background="felt"), "spread"),
        ("slate\nspread out", dict(background="slate"), "spread"),
        ("navy cloth\nspread out", dict(background="navy"), "spread"),
        ("wood table\nspread out", dict(background="wood"), "spread"),
        ("green felt\ncoins touching", dict(background="felt", touching=True), "touching"),
        ("slate\ncoins touching", dict(background="slate", touching=True), "touching"),
        ("wood table\ncoins touching", dict(background="wood", touching=True), "touching"),
        ("denim\ncoins touching", dict(background="denim", touching=True), "touching"),
        ("green felt\nside-lit", dict(background="felt", illum_min=0.55), "side-lit"),
        ("wood table\nside-lit, touching", dict(background="wood", touching=True, illum_min=0.55), "side-lit"),
        ("white paper\ncoins darker than the table", dict(background="paper"), "light surface"),
        ("marble\nlight surface, touching", dict(background="marble", touching=True), "light surface"),
    ]
    method_names = list(cn.METHODS)
    survivors: dict[str, dict] = {}
    scored_scenes = []
    for i, (label, kwargs, family) in enumerate(scene_pool):
        scene, truth = synth.coin_scene(seed=i, mm_per_px=SCENE_MM_PER_PX, **kwargs)
        scored_scenes.append((scene, truth, SCENE_MM_PER_PX))
        labels = [cn.METHODS[m](scene) for m in method_names]
        counts = [cn.count_coins(lab) for lab in labels]
        # accuracy of the COUNT, so finding 21 is penalised like finding 19
        accs = [1.0 - abs(c - len(truth)) / len(truth) for c in counts]
        best, typical = max(accs), float(np.median(accs))
        flat = label.replace("\n", " · ")
        if best < GALLERY_MIN_COUNT_ACCURACY:
            print(f"scene candidate {flat:<40} DROP — no method counts it; best is "
                  f"{max(counts)} of {len(truth)}  [{family}]")
            continue
        print(f"scene candidate {flat:<40} keep — counts {counts} of {len(truth)}  [{family}]")
        row = {
            "label": label,
            "scene": scene,
            "truth": truth,
            "images": [scene] + [cn.overlay_regions(scene, lab) for lab in labels],
            "notes": ["scene"] + [f"{c} of {len(truth)}" for c in counts],
            "score": best,
        }
        if family not in survivors or row["score"] > survivors[family]["score"]:
            survivors[family] = row

    chosen = sorted(survivors.values(), key=lambda r: -r["score"])[:4]
    figures.gallery(
        ["scene"] + method_names,
        [(r["label"], r["images"]) for r in chosen],
        IMAGES / "compare_counting.png",
        cell_notes=[r["notes"] for r in chosen],
        suptitle="Four coin scenes, five segmentation methods. Each cell is the count it returned.",
    )
    count_table = markdown_table(
        [
            dict([("Sr", i), ("Scene", r["label"].replace("\n", " · "))]
                 + list(zip(method_names, r["notes"][1:])))
            for i, r in enumerate(chosen, start=1)
        ],
        [("Sr", "Sr"), ("Scene", "Scene")] + [(m, m) for m in method_names],
    )
    print(f"\nfront-on comparison: {len(chosen)} scenes x {len(method_names)} methods")
    print("\n--- counting ---\n" + count_table)

    # the same four scenes, every coin named from its diameter alone
    figures.gallery(
        ["scene", "counted", f"named by diameter ({cn.IDENT_METHOD})"],
        [
            (
                r["label"],
                [
                    r["scene"],
                    cn.overlay_regions(r["scene"], cn.METHODS[cn.IDENT_METHOD](r["scene"])),
                    cn.overlay_denominations(
                        r["scene"],
                        cn.read_scene(cn.METHODS[cn.IDENT_METHOD](r["scene"]), SCENE_MM_PER_PX),
                    ),
                ],
            )
            for r in chosen
        ],
        IMAGES / "compare_denominations.png",
        suptitle=(
            "Every coin named from its diameter alone, with the total it implies. "
            "Rs 10 and Rs 20 are both 27.00 mm, so no size-based method can tell them apart."
        ),
    )

    ident_rows, confusion = cn.evaluate_identification(scored_scenes)
    ident_table = markdown_table(
        ident_rows,
        [
            ("Calibration", "calibration"),
            ("Coins matched", "coins_matched"),
            ("Named right", "identified"),
            ("Accuracy", "accuracy"),
            ("Missed", "missed"),
            ("Spurious", "spurious"),
            ("True value (Rs)", "true_value"),
            ("Read value (Rs)", "read_value"),
            ("Value error %", "value_error_pct"),
        ],
    )
    print("\n--- identification ---\n" + ident_table)

    denoms = sorted({d for pair in confusion for d in pair})
    conf_rows = []
    for true_d in denoms:
        entry = {"True": f"Rs {true_d}"}
        for read_d in denoms:
            entry[f"read {read_d}"] = confusion.get((true_d, read_d), 0)
        conf_rows.append(entry)
    confusion_table = markdown_table(
        conf_rows, [("True", "True")] + [(f"read Rs {d}", f"read {d}") for d in denoms]
    )
    print("\n--- confusion: true down, read across ---\n" + confusion_table)


    print(f"Counting with {len(cn.METHODS)} methods ...")
    method_rows = cn.evaluate_methods(runs=args.runs)

    print("Ablating mask flattening x seeding rule ...")
    ablation_rows = cn.ablate_mask_and_seeding()

    print("Sweeping the tutorial's seed ratio ...")
    seed_rows = cn.sweep_watershed_seed()

    print("Propagating a wrong calibration reference ...")
    calib_rows = cn.calibration_sensitivity()

    print("Measuring every coin ...")
    diam_rows = cn.diameter_distribution()

    # ------------------------------------------------------------------ #
    # figures
    # ------------------------------------------------------------------ #
    panels = [("coins (input)", img)]
    for name, fn in cn.METHODS.items():
        row = next(r for r in method_rows if r["method"] == name)
        labels = fn(img)
        panels.append(
            (f"{name}\n{row['count']} coins ({row['count_error']:+d})", label_overlay(img, labels, cn.region_properties(labels)))
        )
    figures.grid(
        panels,
        IMAGES / "methods.png",
        ncols=3,
        suptitle=f"Five ways to count {cn.TRUE_COIN_COUNT} touching coins",
    )

    # the ablation, as pictures
    figures.grid(
        [
            ("plain Otsu mask", cn._foreground_mask(img, flatten=False)),
            ("top-hat then Otsu", cn._foreground_mask(img, flatten=True)),
            (
                f"global seed, plain mask\n{ablation_rows[0]['plain_otsu']} coins",
                label_overlay(img, cn.segment_watershed_global_seed(img, flatten=False),
                              cn.region_properties(cn.segment_watershed_global_seed(img, flatten=False))),
            ),
            (
                f"local maxima, plain mask\n{ablation_rows[1]['plain_otsu']} coins",
                label_overlay(img, cn.segment_watershed(img, flatten=False),
                              cn.region_properties(cn.segment_watershed(img, flatten=False))),
            ),
        ],
        IMAGES / "ablation.png",
        ncols=2,
        suptitle="The same broken mask: one seeding rule finds 1 coin, the other finds all 24",
    )

    # the distance transform and its seeds — the mechanism
    mask = cn._foreground_mask(img)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    k = 2 * 12 + 1
    peaks = ((dist >= cv2.dilate(dist, np.ones((k, k), np.float32)) - 1e-6) & (dist >= 10.0))
    seed_vis = img.copy()
    ys, xs = np.nonzero(peaks)
    for x, y in zip(xs, ys):
        cv2.circle(seed_vis, (int(x), int(y)), 4, (255, 40, 40), -1)
    figures.grid(
        [
            ("foreground mask", mask),
            ("distance transform", (dist / max(dist.max(), 1e-6) * 255).astype(np.uint8)),
            (f"local maxima = {int(peaks.sum())} seeds", seed_vis),
        ],
        IMAGES / "seeds.png",
        ncols=3,
        suptitle="The distance transform peaks once per coin — even where two coins touch",
    )

    figures.lines(
        [r["fg_ratio"] for r in seed_rows],
        {
            "Watershed (global seed)": [r["global_seed_count"] for r in seed_rows],
            "Watershed (local maxima)": [r["local_maxima_count"] for r in seed_rows],
        },
        IMAGES / "seed_sweep.png",
        xlabel="fg_ratio (fraction of the GLOBAL distance maximum)",
        ylabel="coins counted",
        title=f"The tutorial's knob has no safe setting; the true count is {cn.TRUE_COIN_COUNT}",
    )

    figures.lines(
        [r["reference_error_pct"] for r in calib_rows],
        {"measured diameter error (%)": [r["measured_error_pct"] for r in calib_rows]},
        IMAGES / "calibration.png",
        xlabel="error in the assumed reference diameter (%)",
        ylabel="error in EVERY measured diameter (%)",
        title="Calibration error propagates 1:1 and leaves the output perfectly self-consistent",
    )

    figures.histogram(
        {
            "Watershed (local maxima)": np.array(
                [r["diameter_mm"] for r in cn.diameter_distribution("Watershed (local maxima)")]
            ),
            "Hough circles": np.array(
                [r["diameter_mm"] for r in cn.diameter_distribution("Hough circles")]
            ),
        },
        IMAGES / "diameters.png",
        bins=24,
        value_range=(0, 30),
        xlabel="measured diameter (mm)",
        title="Same count, different measurements: watershed's squeezed basins land below 8 mm",
    )

    figures.comparison_matrix(
        method_rows,
        [
            ("Count error", "abs_count_error", False),
            ("Implausible regions", "implausible", False),
            ("Diameter CV", "diameter_cv", False),
            ("Time (ms)", "median_ms", False),
        ],
        IMAGES / "method_matrix.png",
        title="Counting and measuring are different columns, and they do not agree",
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS,
        "09_coin_counting",
        {
            "true_count": cn.TRUE_COIN_COUNT,
            "reference_diameter_mm": cn.REFERENCE_DIAMETER_MM,
            "methods": method_rows,
            "ablation": ablation_rows,
            "seed_sweep": seed_rows,
            "calibration_sensitivity": calib_rows,
            "diameters": diam_rows,
        },
    )

    method_table = markdown_table(
        method_rows,
        [
            ("Method", "method"),
            ("Count", "count"),
            ("Error", "count_error"),
            ("Smallest (mm)", "min_diameter_mm"),
            ("Largest (mm)", "max_diameter_mm"),
            ("Diameter CV", "diameter_cv"),
            ("Implausible", "implausible"),
            ("Time (ms)", "median_ms"),
        ],
    )
    ablation_table = markdown_table(
        ablation_rows,
        [
            ("Seeding rule", "seeding"),
            ("Plain Otsu", "plain_otsu"),
            ("Top-hat then Otsu", "tophat_otsu"),
        ],
    )
    seed_table = markdown_table(
        seed_rows,
        [
            ("fg_ratio", "fg_ratio"),
            ("Global seed count", "global_seed_count"),
            ("Error", "global_seed_error"),
            ("Local maxima count", "local_maxima_count"),
        ],
    )
    calib_table = markdown_table(
        calib_rows,
        [
            ("Reference error (%)", "reference_error_pct"),
            ("Assumed reference (mm)", "assumed_reference_mm"),
            ("Mean measured (mm)", "mean_diameter_mm"),
            ("Measured error (%)", "measured_error_pct"),
        ],
    )
    diam_table = markdown_table(
        diam_rows,
        [("Rank", "rank"), ("Area (px)", "area_px"), ("Diameter (px)", "diameter_px"),
         ("Diameter (mm)", "diameter_mm")],
    )
    write_tables(
        RESULTS,
        [
            (f"Counting and measuring {cn.TRUE_COIN_COUNT} coins", method_table),
            ("ABLATION: mask flattening x seeding rule (coins counted)", ablation_table),
            ("The tutorial's seed ratio, swept", seed_table),
            ("Calibration error propagates 1:1", calib_table),
            ("Every coin, measured (watershed, local maxima)", diam_table),
        ],
    )
    print("\n" + "\n\n".join([method_table, ablation_table, seed_table, calib_table]))

    exact = [r for r in method_rows if r["count_error"] == 0]
    best_measure = min(method_rows, key=lambda r: (r["implausible"], r["diameter_cv"] or 9))
    hough = next(r for r in method_rows if r["method"] == "Hough circles")
    water = next(r for r in method_rows if r["method"] == "Watershed (local maxima)")
    otsu = next(r for r in method_rows if r["method"] == "Otsu + components")
    tut, loc = ablation_rows[0], ablation_rows[1]

    print("\n--- HEADLINE NUMBERS ---")
    print(f"true count       : {cn.TRUE_COIN_COUNT}")
    print(
        f"exact counts     : {len(exact)} of {len(method_rows)} methods — "
        + ", ".join(r["method"] for r in exact)
    )
    print(
        f"...but only {sum(1 for r in method_rows if r['implausible'] == 0)} of them measured every "
        f"coin plausibly: {best_measure['method']} (0 implausible, CV {best_measure['diameter_cv']})"
    )
    print(
        f"watershed vs hough: both count {water['count']}, but watershed's smallest region implies a "
        f"{water['min_diameter_mm']} mm coin against Hough's {hough['min_diameter_mm']} mm "
        f"(CV {water['diameter_cv']} vs {hough['diameter_cv']})"
    )
    print(
        f"naive baseline   : {otsu['method']} counts {otsu['count']} — touching coins merge into "
        f"one component, and {otsu['implausible']} of its regions are not coin-shaped at all"
    )
    print(
        f"ABLATION         : with a broken mask the tutorial's rule counts {tut['plain_otsu']}, "
        f"local maxima counts {loc['plain_otsu']}. Fixing the mask rescues the tutorial rule to "
        f"{tut['tophat_otsu']} and leaves local maxima at {loc['tophat_otsu']}"
    )
    print(
        f"calibration      : a {calib_rows[0]['reference_error_pct']}% error in the reference "
        f"produces exactly {calib_rows[0]['measured_error_pct']}% error in every diameter, "
        f"with nothing downstream able to detect it"
    )
    print(f"fastest exact    : {min(exact, key=lambda r: r['median_ms'])['method']}")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
