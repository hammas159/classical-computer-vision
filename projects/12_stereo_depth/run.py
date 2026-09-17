"""Run the stereo-matching experiments and write results + figures.

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
from shared.bench import timeit  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import stereo as st  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"
STEREO_DIR = PROJECT_DIR.parents[1] / "assets" / "real" / "stereo"

#: Disparity used by the generated pairs. Kept under `MAX_DISPARITY` so nothing
#: is truncated by the search range — a truncated disparity reads as a wrong
#: match rather than as an out-of-range one, and would be scored as the
#: matcher's fault.
GENERATED_MAX_DISPARITY = 48


def load_aloe():
    """The Middlebury *Aloe* pair with its measured disparity.

    Ground truth here is *measured*, not constructed: structured light, not a
    warp this repo wrote. It is the check that the generated pairs are not
    quietly easier than reality.
    """
    left = io.imread(STEREO_DIR / "aloe_left.png")
    right = io.imread(STEREO_DIR / "aloe_right.png")
    truth = io.imread(STEREO_DIR / "aloe_disparity.png", gray=True).astype(np.float32)
    valid = truth > 0
    truth[~valid] = np.nan
    return left, right, truth, valid


def colourise(disp: np.ndarray, vmax: float) -> np.ndarray:
    """A disparity map as an image, with unanswered pixels in black.

    Unanswered has to be visually distinct from *near* or *far*, or a hole reads
    as a depth. Black is used for it and the colour map starts above black.
    """
    out = np.zeros(disp.shape + (3,), np.uint8)
    known = np.isfinite(disp)
    if known.any():
        # NaN has to be replaced before the cast, not clipped: casting NaN to
        # uint8 is undefined and numpy warns rather than raising, so the holes
        # come out as whatever bit pattern the platform produces
        scaled = np.clip(np.nan_to_num(disp, nan=0.0) / max(vmax, 1e-6), 0, 1)
        tinted = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        out[known] = cv2.cvtColor(tinted, cv2.COLOR_BGR2RGB)[known]
    return out


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    # Scenes are grouped by what makes stereo hard, not by subject: how much
    # texture there is to match, how much of the frame is one flat plane, and
    # how many depth discontinuities there are to get wrong. A pool of six
    # well-textured scenes would measure the easy case six times.
    GALLERY_MAX_BAD = 0.35  # best method must get most of the frame right
    scene_pool = [
        ("windmills\nwhite walls, little texture", "windmills", "low texture"),
        ("rocky coast\nlarge flat sky", "rocky_coast", "low texture"),
        ("stone arch\nstonework, heavy texture", "stone_arch", "high texture"),
        ("tiger on rocks\nstripes and rubble", "tiger_rocks", "high texture"),
        ("harbour and boat\nrigging, thin structures", "harbour_boat", "thin structure"),
        ("temple dragon\nornament against towers", "temple_dragon", "thin structure"),
        ("gallery visitors\npeople at several depths", "gallery_visitors", "many depths"),
        ("elephant herd\nanimals at several depths", "elephant_herd", "many depths"),
        ("penguin on pebbles\nrepeating stones", "penguin_pebbles", "repetitive"),
        ("windows and flowers\nrepeating shutters", "window_flowers", "repetitive"),
        ("coral reef\ndense fine detail", "coral_reef", "fine detail"),
        ("wolf in leaf litter\nscattered fine detail", "wolf_woods", "fine detail"),
    ]
    method_names = list(st.METHODS)
    survivors: dict[str, dict] = {}
    scored: list[tuple] = []
    for i, (label, name, family) in enumerate(scene_pool):
        left, right, truth, valid = synth.stereo_pair(
            io.real_photo(name), max_disparity=GENERATED_MAX_DISPARITY, seed=i
        )
        scored.append((label, left, right, truth, valid))
        preds = [st.METHODS[m](left, right) for m in method_names]
        scores = [st.score_disparity(p, truth, valid) for p in preds]
        fusion = [s for s, m in zip(scores, method_names) if m not in st.CONTROLS]
        best = min(s["bad2_all"] for s in fusion)
        flat = label.replace("\n", " · ")
        if best > GALLERY_MAX_BAD:
            print(f"scene candidate {flat:<44} DROP — best bad-2px {best:.3f}  [{family}]")
            continue
        print(f"scene candidate {flat:<44} keep — best bad-2px {best:.3f}  [{family}]")
        vmax = float(np.nanmax(truth))
        row = {
            "label": label,
            "images": [left, colourise(truth, vmax)]
            + [colourise(p, vmax) for p in preds],
            "notes": ["left view", "true disparity"]
            + [f"{100 * s['bad2_all']:.1f}% bad" for s in scores],
            "score": best,
            "scores": scores,
        }
        if family not in survivors or row["score"] < survivors[family]["score"]:
            survivors[family] = row

    chosen = sorted(survivors.values(), key=lambda r: r["score"])[:4]
    figures.gallery(
        ["left view", "true disparity"] + method_names,
        [(r["label"], r["images"]) for r in chosen],
        IMAGES / "compare_matchers.png",
        cell_notes=[r["notes"] for r in chosen],
        suptitle=(
            "Four stereo pairs, five matchers. Black is a pixel the matcher "
            "declined to answer; the percentage counts those as wrong."
        ),
    )
    gallery_table = markdown_table(
        [
            dict([("Sr", i), ("Scene", r["label"].replace("\n", " · "))]
                 + list(zip(method_names, r["notes"][2:])))
            for i, r in enumerate(chosen, start=1)
        ],
        [("Sr", "Sr"), ("Scene", "Scene")] + [(m, m) for m in method_names],
    )
    print(f"\nfront-on comparison: {len(chosen)} pairs x {len(method_names)} matchers")
    print("\n--- matchers ---\n" + gallery_table)

    # ------------------------------------------------------------------ #
    # aggregate over the generated pairs, and over the real one
    # ------------------------------------------------------------------ #
    def aggregate(pairs):
        rows = []
        for method in method_names:
            acc: dict[str, list] = {}
            ms = []
            for _label, left, right, truth, valid in pairs:
                pred, timing = timeit(
                    lambda l=left, r=right, m=method: st.METHODS[m](l, r),
                    runs=args.runs, warmup=0,
                )
                ms.append(timing.median_ms)
                for k, v in st.score_disparity(pred, truth, valid).items():
                    acc.setdefault(k, []).append(v)
            row = {"method": method, "median_ms": round(float(np.mean(ms)), 3)}
            row.update({k: round(float(np.nanmean(v)), 4) for k, v in acc.items()})
            rows.append(row)
        return rows

    print("\nScoring the generated pairs ...")
    generated_rows = aggregate(scored)

    print("Scoring the real Middlebury pair ...")
    aloe = load_aloe()
    real_rows = aggregate([("aloe",) + aloe])

    # ------------------------------------------------------------------ #
    # where the errors actually are
    # ------------------------------------------------------------------ #
    print("Splitting the error by region ...")
    region_rows = []
    left, right, truth, valid = aloe
    regions = st.error_regions(truth, valid, left)
    for method in method_names:
        pred = st.METHODS[method](left, right)
        row = {"method": method}
        for region_name, mask in regions.items():
            s = st.score_disparity(pred, truth, mask)
            row[region_name] = s.get("bad2_all", float("nan"))
        row["share of frame"] = ""
        region_rows.append(row)
    coverage = {k: round(float(v.sum()) / max(float(valid.sum()), 1), 4)
                for k, v in regions.items()}

    # ------------------------------------------------------------------ #
    # figures
    # ------------------------------------------------------------------ #
    vmax = float(np.nanmax(truth))
    figures.grid(
        [
            ("Left view", left),
            ("Right view", right),
            ("Measured disparity", colourise(truth, vmax)),
            ("SGBM (8-path)", colourise(st.match_sgbm_hh(left, right), vmax)),
        ],
        IMAGES / "aloe.png",
        ncols=4,
        suptitle="The Middlebury Aloe pair — ground truth measured, not constructed",
    )

    figures.grid(
        [(name, mask.astype(np.float32)) for name, mask in regions.items()],
        IMAGES / "error_regions.png",
        ncols=3,
        suptitle=(
            "Where stereo fails, split out. One bad-pixel percentage averages "
            "over all three and describes none of them."
        ),
    )

    # density against accuracy, as a picture. The two axes disagree about the
    # winner, and a scatter is the honest way to show a trade-off that a ranked
    # table has to flatten into one order.
    density_points, labels = [], []
    for method in method_names:
        if method in st.CONTROLS:
            continue
        s = st.score_disparity(st.METHODS[method](left, right), truth, valid)
        density_points.append((s["density"], s["bad2_answered"]))
        labels.append(method)
    figures.lines(
        [p[0] for p in sorted(density_points)],
        {"bad 2px among answered": [p[1] for p in sorted(density_points)]},
        IMAGES / "density_vs_accuracy.png",
        xlabel="density — fraction of the frame answered",
        ylabel="bad pixels among those answered (2 px)",
        title=(
            "Answering more costs accuracy: "
            + ", ".join(f"{l.split(' (')[0]} {100 * p[0]:.0f}%/{100 * p[1]:.1f}%"
                        for l, p in zip(labels, density_points))
        ),
    )

    figures.comparison_matrix(
        generated_rows,
        [
            ("Density", "density", True),
            ("MAE (px)", "mae_px", False),
            ("Bad 2px, answered", "bad2_answered", False),
            ("Bad 2px, all", "bad2_all", False),
            ("Time (ms)", "median_ms", False),
        ],
        IMAGES / "method_matrix.png",
        title="Methods x metrics — density and accuracy point in opposite directions",
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS,
        "12_stereo_depth",
        {
            "max_disparity": st.MAX_DISPARITY,
            "generated_max_disparity": GENERATED_MAX_DISPARITY,
            "generated_pairs": [r[0].replace("\n", " · ") for r in scored],
            "generated": generated_rows,
            "real_aloe": real_rows,
            "error_regions": region_rows,
            "region_coverage": coverage,
        },
    )

    cols = [
        ("Method", "method"),
        ("Density", "density"),
        ("MAE (px)", "mae_px"),
        ("Bad 1px (all)", "bad1_all"),
        ("Bad 2px (answered)", "bad2_answered"),
        ("Bad 2px (all)", "bad2_all"),
        ("Bad 4px (all)", "bad4_all"),
        ("Time (ms)", "median_ms"),
    ]
    gen_table = markdown_table(generated_rows, cols)
    real_table = markdown_table(real_rows, cols)
    region_table = markdown_table(
        region_rows,
        [("Method", "method")] + [(k, k) for k in regions],
    )
    write_tables(
        RESULTS,
        [
            (f"Generated pairs ({len(scored)} scenes)", gen_table),
            ("The real Middlebury Aloe pair", real_table),
            ("Bad-2px rate by region, on Aloe", region_table),
            ("Four pairs down the rows", gallery_table),
        ],
    )
    print("\n" + gen_table + "\n\n" + real_table + "\n\n" + region_table)

    real = [r for r in generated_rows if r["method"] not in st.CONTROLS]
    densest = max(real, key=lambda r: r["density"])
    most_accurate = min(real, key=lambda r: r["bad2_answered"])
    best_overall = min(real, key=lambda r: r["bad2_all"])
    control = next(r for r in generated_rows if r["method"] in st.CONTROLS)

    print("\n--- HEADLINE NUMBERS ---")
    print(f"region coverage  : {coverage}")
    print(f"most accurate    : {most_accurate['method']} @ "
          f"{100 * most_accurate['bad2_answered']:.1f}% bad of "
          f"{100 * most_accurate['density']:.0f}% answered")
    print(f"densest          : {densest['method']} @ "
          f"{100 * densest['density']:.0f}% answered, "
          f"{100 * densest['bad2_answered']:.1f}% of those bad")
    print(f"best all-pixels  : {best_overall['method']} @ "
          f"{100 * best_overall['bad2_all']:.1f}% bad")
    print(f"control          : {100 * control['bad2_all']:.1f}% bad at 100% density")
    for r in region_rows:
        if r["method"] in st.CONTROLS:
            continue
        print(f"  {r['method']:30s} " + "  ".join(
            f"{k} {100 * r[k]:5.1f}%" for k in regions))
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
