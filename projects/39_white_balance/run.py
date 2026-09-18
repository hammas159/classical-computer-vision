"""Run the white balance comparison and write results + figures.

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
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import white_balance as wb  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"

FIGURE_CAST = "tungsten (warm)"

#: Bands of "how far the scene's own mean already sits from grey", so the four
#: figure rows span the axis the pool was selected on rather than four pictures
#: where grey-world happens to work.
GREY_BANDS = [("neutral", 0.0, 4.0), ("mild", 4.0, 8.0),
              ("biased", 8.0, 13.0), ("dominated", 13.0, 99.0)]


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    methods = list(wb.ESTIMATORS)

    print(f"Estimating the illuminant under a {FIGURE_CAST} cast, "
          f"{len(wb.IMAGES)} photographs ...")
    clean = wb.evaluate_estimators(cast=FIGURE_CAST, runs=args.runs)
    for r in clean:
        print(f"  {r['method']:24s} {r['angular_error_deg']:6.3f} deg   "
              f"PSNR {r['psnr_db']:6.2f}   SSIM {r['ssim']:.4f}   {r['median_ms']:7.3f} ms")

    print("\nFour casts, including one that is barely a cast at all ...")
    casts = wb.compare_casts()
    for row in casts:
        print(f"  {row['cast']:20s} " + "  ".join(
            f"{m.split()[0][:6]}={row[m]:5.2f}" for m in methods))

    print("\nTriggering each method's designed failure ...")
    failures = wb.assumption_failure_tests()
    for row in failures:
        print(f"  {row['scene']}")
        for m in methods:
            print(f"      {m:24s} {row[m]:7.3f} deg")

    dominance = wb.sweep_dominance()
    minkowski = wb.sweep_minkowski_p()
    noise = wb.sweep_noise()

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    candidates: dict[str, list[dict]] = {}
    gains = wb.CASTS[FIGURE_CAST]
    for name in wb.IMAGES:
        deviation = wb.grey_deviation(wb.load_scene(name))
        band = next(b for b, lo, hi in GREY_BANDS if lo <= deviation < hi)
        cast, original, truth = wb.make_case(name, gains=gains)

        panels, notes, scores = [original, cast], ["the scene", f"{FIGURE_CAST} cast"], []
        for method in methods:
            estimate = wb.ESTIMATORS[method](cast)
            error = wb.angular_error(estimate, truth)
            panels.append(wb.apply_correction(cast, estimate))
            notes.append(f"{error:.2f} deg")
            scores.append(error)

        candidates.setdefault(band, []).append({
            "label": f"{name.replace('_', ' ')}\n{deviation:.1f} deg from grey",
            "subject": name, "deviation": deviation, "images": panels,
            "notes": notes, "scores": scores, "spread": float(np.std(scores)),
        })
        print(f"  scene candidate {name:24s} {deviation:5.1f} deg [{band:9s}]  "
              f"errors " + " ".join(f"{s:5.2f}" for s in scores))

    chosen, used = [], set()
    for band, _, _ in GREY_BANDS:
        pool = sorted(candidates.get(band, []), key=lambda r: -r["spread"])
        pick = next((r for r in pool if r["subject"] not in used), None)
        if pick:
            used.add(pick["subject"])
            chosen.append(pick)
    chosen = chosen[:4]
    assert len({r["subject"] for r in chosen}) == len(chosen), "a subject repeats"

    figures.gallery(
        ["original", "with the cast"] + methods,
        [(r["label"], r["images"]) for r in chosen],
        IMAGES / "compare_balance.png",
        cell_notes=[r["notes"] for r in chosen],
        suptitle=(
            f"A known {FIGURE_CAST} cast applied, then estimated and undone. Cells are "
            "the angular error of the estimated illuminant in degrees — how wrong the "
            "method was about the light, not how the picture looks."
        ),
    )
    gallery_table = markdown_table(
        [dict([("Sr", i), ("Scene", r["label"].replace("\n", " · "))]
              + [(m, f"{s:.2f}") for m, s in zip(methods, r["scores"])])
         for i, r in enumerate(chosen, start=1)],
        [("Sr", "Sr"), ("Scene", "Scene")] + [(m, m) for m in methods],
    )
    print("\n--- angular error in degrees ---\n" + gallery_table)

    # ------------------------------------------------------------------ #
    # the signature figure: the chromaticity plane
    # ------------------------------------------------------------------ #
    truth = wb.normalise_illuminant(np.asarray(gains, np.float64))
    estimates = {}
    for method in methods:
        points = []
        for name in wb.IMAGES:
            cast, _, _ = wb.make_case(name, gains=gains)
            v = wb.normalise_illuminant(wb.ESTIMATORS[method](cast))
            points.append((v[0] / v.sum(), v[1] / v.sum()))
        estimates[method] = points

    figures.scatter_plane(
        estimates,
        (truth[0] / truth.sum(), truth[1] / truth.sum()),
        IMAGES / "chromaticity.png",
        xlabel="r = R / (R+G+B)", ylabel="g = G / (R+G+B)",
        title=("Estimated illuminant against the true one, on the chromaticity plane. "
               "Distance from the cross is the error; the spread is the method's "
               "dependence on the picture."),
        target_label="the true illuminant",
    )

    figures.lines(
        [r["dominant_fraction"] for r in dominance],
        {m: [r[m] for r in dominance] for m in methods},
        IMAGES / "dominance_sweep.png",
        xlabel="fraction of the frame that is one colour",
        ylabel="angular error (degrees)",
        title="Grey-world's error is linear in how much its assumption is violated",
        dashed={"Do nothing (control)"},
    )

    figures.lines(
        [r["p"] for r in minkowski],
        {"Shades-of-grey": [r["shades_of_grey_deg"] for r in minkowski],
         "Grey-edge": [r["grey_edge_deg"] for r in minkowski]},
        IMAGES / "minkowski_sweep.png",
        xlabel="Minkowski p", ylabel="angular error (degrees)",
        title="The two methods want opposite p, and the usual default suits one of them",
        logx=True, vlines={"the usual p=6": 6.0},
    )

    figures.lines(
        [r["noise_sigma"] for r in noise],
        {m: [r[m] for r in noise] for m in methods},
        IMAGES / "noise_sweep.png",
        xlabel="Gaussian noise sigma", ylabel="angular error (degrees)",
        title="Noise invents bright pixels, which is white-patch's whole input",
        dashed={"Do nothing (control)"},
    )

    figures.comparison_matrix(
        [{"method": m,
          "tungsten": next(r[m] for r in casts if r["cast"] == "tungsten (warm)"),
          "neutral": next(r[m] for r in casts if r["cast"] == "daylight (neutral)"),
          "shade": next(r[m] for r in casts if r["cast"] == "shade (cool)"),
          "green": next(r[m] for r in casts if r["cast"] == "strong green"),
          "dominated": failures[0][m],
          "clipped": failures[1][m]} for m in methods],
        [("Tungsten", "tungsten", False), ("Almost no cast", "neutral", False),
         ("Shade", "shade", False), ("Strong green", "green", False),
         ("Dominated scene", "dominated", False), ("Clipped highlight", "clipped", False)],
        IMAGES / "method_matrix.png",
        row_key="method",
        title="Angular error in degrees, lower is better. Each column is a different "
              "assumption being tested.",
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS, "39_white_balance",
        {
            "images": list(wb.IMAGES),
            "grey_deviation": {n: round(wb.grey_deviation(wb.load_scene(n)), 2)
                               for n in wb.IMAGES},
            "casts": wb.CASTS,
            "tungsten": clean,
            "cast_comparison": casts,
            "assumption_failures": failures,
            "dominance_sweep": dominance,
            "minkowski_sweep": minkowski,
            "noise_sweep": noise,
        },
    )

    clean_table = markdown_table(
        clean, [("Method", "method"), ("Angular error (deg)", "angular_error_deg"),
                ("PSNR (dB)", "psnr_db"), ("SSIM", "ssim"), ("Time (ms)", "median_ms")])
    casts_table = markdown_table(
        casts, [("Cast", "cast")] + [(m, m) for m in methods])
    failures_table = markdown_table(
        failures, [("Scene", "scene")] + [(m, m) for m in methods])
    dominance_table = markdown_table(
        dominance, [("Dominant fraction", "dominant_fraction")] + [(m, m) for m in methods])
    minkowski_table = markdown_table(
        minkowski, [("p", "p"), ("Shades-of-grey", "shades_of_grey_deg"),
                    ("Grey-edge", "grey_edge_deg")])
    noise_table = markdown_table(
        noise, [("Sigma", "noise_sigma")] + [(m, m) for m in methods])

    write_tables(
        RESULTS,
        [
            ("Under a tungsten cast", clean_table),
            ("Four casts", casts_table),
            ("Each method's designed failure", failures_table),
            ("Grey-world against how much of the frame is one colour", dominance_table),
            ("The Minkowski exponent", minkowski_table),
            ("Noise", noise_table),
            ("Four scenes down the rows", gallery_table),
        ],
    )
    print("\n" + clean_table + "\n\n" + failures_table)

    # ------------------------------------------------------------------ #
    # headlines
    # ------------------------------------------------------------------ #
    print("\n--- HEADLINE NUMBERS ---")
    clipped = failures[1]
    control = clipped["Do nothing (control)"]
    max_patch = clipped["White-patch (true max)"]
    print(f"a single clipped highlight puts white-patch (true max) at "
          f"{max_patch:.3f} deg — the do-nothing control is {control:.3f}")
    if abs(max_patch - control) < 0.01:
        print("  exactly the control: it reads the clipped pixel, concludes the light "
              "is white, and applies no correction at all")
    print(f"  the 99th-percentile variant, which throws the top 1% away, gets "
          f"{clipped['White-patch (99th pct)']:.3f} — that is what the percentile is for")

    neutral = next(r for r in casts if r["cast"] == "daylight (neutral)")
    worse = [m for m in methods if m != "Do nothing (control)"
             and neutral[m] > neutral["Do nothing (control)"]]
    print(f"\non a scene with almost no cast ({neutral['Do nothing (control)']:.3f} deg), "
          f"{len(worse)} of {len(methods) - 1} methods make it WORSE — "
          f"grey-world turns {neutral['Do nothing (control)']:.2f} deg into "
          f"{neutral['Grey-world']:.2f}")

    lo, hi = dominance[0], dominance[-1]
    print(f"\ngrey-world goes {lo['Grey-world']:.2f} -> {hi['Grey-world']:.2f} deg as the "
          f"dominant colour covers {lo['dominant_fraction']:.0%} -> "
          f"{hi['dominant_fraction']:.0%} of the frame; grey-edge stays at "
          f"{lo['Grey-edge (p=6)']:.2f} -> {hi['Grey-edge (p=6)']:.2f}")

    best_sg = min(minkowski, key=lambda r: r["shades_of_grey_deg"])
    best_ge = min(minkowski, key=lambda r: r["grey_edge_deg"])
    at_six = next(r for r in minkowski if r["p"] == 6.0)
    print(f"\nshades-of-grey is best at p={best_sg['p']:g} "
          f"({best_sg['shades_of_grey_deg']:.2f} deg); grey-edge at p={best_ge['p']:g} "
          f"({best_ge['grey_edge_deg']:.2f})")
    print(f"  the usual p=6 costs grey-edge "
          f"{at_six['grey_edge_deg'] - best_ge['grey_edge_deg']:.2f} deg "
          f"({100 * (at_six['grey_edge_deg'] / best_ge['grey_edge_deg'] - 1):.0f}% more error) "
          "— one number, and it is set for the wrong method")

    n0, n1 = noise[0], noise[-1]
    print(f"\nat noise sigma {n1['noise_sigma']:g}, white-patch (true max) goes "
          f"{n0['White-patch (true max)']:.2f} -> {n1['White-patch (true max)']:.2f} deg "
          f"while grey-world goes {n0['Grey-world']:.2f} -> {n1['Grey-world']:.2f}")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
