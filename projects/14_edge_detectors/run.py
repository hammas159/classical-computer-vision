"""Run the edge-detector comparison and write results + figures.

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

from shared import figures, io, synth  # noqa: E402
from shared.io import to_gray  # noqa: E402
from shared.metrics import edge_prf  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import edges as ed  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"

#: Every operator, in the order they are compared. The two Canny variants are
#: separate entries because the difference between them is a finding, not a
#: detail: one has fixed thresholds and one derives them from the image.
ALL_OPERATORS = list(ed.GRADIENTS) + ["Canny (fixed 50/150)", "Canny (auto median)"]


def edge_image(edges: np.ndarray) -> np.ndarray:
    """A binary edge map as a white-on-black picture."""
    return np.repeat((edges > 0).astype(np.uint8)[..., None] * 255, 3, axis=2)


def overlay_truth(edges: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Detected edges in white, missed truth in red, false alarms in blue.

    Three colours because precision and recall fail differently and a single
    white edge map cannot show which one happened. A missing edge and an
    invented one look identical in a binary image and are not the same mistake.
    """
    e, t = edges > 0, truth > 0
    out = np.zeros(e.shape + (3,), np.uint8)
    out[e & t] = (255, 255, 255)
    out[t & ~e] = (255, 60, 60)      # missed
    out[e & ~t] = (60, 140, 255)     # invented
    return out


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    print(f"Scoring {len(ALL_OPERATORS)} operators on a clean scene ...")
    clean_rows = ed.evaluate_operators(noise_sigma=0.0, runs=args.runs)

    print("Scoring them again at sigma 30 ...")
    noisy_rows = ed.evaluate_operators(noise_sigma=30.0, runs=args.runs)

    print(f"Sweeping noise over {len(ed.NOISE_LEVELS)} levels ...")
    noise_rows = ed.sweep_noise()

    print("Sweeping the matching tolerance ...")
    tolerance_rows = ed.sweep_tolerance()

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    # Scenes vary by NOISE, because that is the axis every one of these
    # operators is actually distinguished on. On a clean scene six of the seven
    # score 1.000 and the comparison says nothing; the whole result lives in how
    # they come apart as the noise rises.
    GALLERY_MIN_F1 = 0.30
    scene_pool = [
        (f"clean\nsigma 0", 0.0, 0, "clean"),
        (f"clean, other shapes\nsigma 0", 0.0, 1, "clean"),
        (f"light noise\nsigma 5", 5.0, 0, "light"),
        (f"light noise, other shapes\nsigma 5", 5.0, 1, "light"),
        (f"moderate noise\nsigma 15", 15.0, 0, "moderate"),
        (f"moderate noise, other shapes\nsigma 15", 15.0, 1, "moderate"),
        (f"heavy noise\nsigma 30", 30.0, 0, "heavy"),
        (f"heavy noise, other shapes\nsigma 30", 30.0, 1, "heavy"),
        (f"severe noise\nsigma 50", 50.0, 0, "severe"),
        (f"severe noise, other shapes\nsigma 50", 50.0, 1, "severe"),
        (f"severe noise, third scene\nsigma 50", 50.0, 2, "severe"),
        (f"moderate noise, third scene\nsigma 15", 15.0, 2, "moderate"),
    ]
    survivors: dict[str, dict] = {}
    for label, sigma, seed, family in scene_pool:
        img, truth = ed.scene(noise_sigma=sigma, seed=seed)
        gray = to_gray(img)
        maps, scores = [], []
        for name in ALL_OPERATORS:
            if name in ed.GRADIENTS:
                _f1, _t, e = ed.best_threshold(gray, truth, ed.GRADIENTS[name])
            elif name.startswith("Canny (fixed"):
                e = ed.edges_canny(gray)
            else:
                e = ed.edges_canny_auto(gray)
            maps.append(e)
            scores.append(edge_prf(e, truth, ed.TOLERANCE)["f1"])
        best = max(scores)
        flat = label.replace("\n", " · ")
        if best < GALLERY_MIN_F1:
            print(f"scene candidate {flat:<38} DROP — best F1 {best:.3f}  [{family}]")
            continue
        print(f"scene candidate {flat:<38} keep — best F1 {best:.3f}, "
              f"worst {min(scores):.3f}  [{family}]")
        row = {
            "label": label,
            "images": [img, edge_image(truth)] + [overlay_truth(e, truth) for e in maps],
            "notes": ["scene", "true edges"] + [f"F1 {s:.3f}" for s in scores],
            "spread": best - min(scores),
        }
        # keep the scene where the operators DISAGREE most within each family --
        # that is the one carrying information
        if family not in survivors or row["spread"] > survivors[family]["spread"]:
            survivors[family] = row

    BAND_ORDER = ["clean", "moderate", "heavy", "severe"]
    chosen = [survivors[b] for b in BAND_ORDER if b in survivors][:4]
    figures.gallery(
        ["scene", "true edges"] + ALL_OPERATORS,
        [(r["label"], r["images"]) for r in chosen],
        IMAGES / "compare_operators.png",
        cell_notes=[r["notes"] for r in chosen],
        suptitle=(
            "Four noise levels, seven operators, each gradient at its own best "
            "threshold. White = found, red = missed, blue = invented."
        ),
    )
    gallery_table = markdown_table(
        [
            dict([("Sr", i), ("Scene", r["label"].replace("\n", " · "))]
                 + list(zip(ALL_OPERATORS, r["notes"][2:])))
            for i, r in enumerate(chosen, start=1)
        ],
        [("Sr", "Sr"), ("Scene", "Scene")] + [(o, o) for o in ALL_OPERATORS],
    )
    print(f"\nfront-on comparison: {len(chosen)} scenes x {len(ALL_OPERATORS)} operators")
    print("\n--- operators ---\n" + gallery_table)

    # ------------------------------------------------------------------ #
    # the same operators on real photographs, shown and never scored
    # ------------------------------------------------------------------ #
    real_names = ["stone_arch", "albatross_pair", "windmills", "rhino_road"]
    real_rows = []
    for name in real_names:
        img = io.real_photo(name)
        gray = to_gray(img)
        maps = []
        for op in ALL_OPERATORS:
            if op in ed.GRADIENTS:
                mag = ed.GRADIENTS[op](gray)
                maps.append(ed.threshold_magnitude(mag, 0.15))
            elif op.startswith("Canny (fixed"):
                maps.append(ed.edges_canny(gray))
            else:
                maps.append(ed.edges_canny_auto(gray))
        real_rows.append((name.replace("_", " "), [img] + [edge_image(m) for m in maps]))
    figures.gallery(
        ["photograph"] + ALL_OPERATORS,
        real_rows,
        IMAGES / "compare_real.png",
        suptitle=(
            "The same seven operators on real photographs. No score is shown: "
            "nobody recorded where the edges of a real scene are."
        ),
    )

    # ------------------------------------------------------------------ #
    # sweeps
    # ------------------------------------------------------------------ #
    figures.lines(
        [r["noise_sigma"] for r in noise_rows],
        {op: [r[op] for r in noise_rows] for op in ALL_OPERATORS},
        IMAGES / "noise_sweep.png",
        xlabel="Gaussian noise sigma",
        ylabel="F1 at the operator's own best threshold",
        title="Where each operator stops working — and where the auto-threshold recipe falls off",
    )

    figures.lines(
        [r["tolerance_px"] for r in tolerance_rows],
        {op: [r[op] for r in tolerance_rows] for op in tolerance_rows[0] if op != "tolerance_px"},
        IMAGES / "tolerance_sweep.png",
        xlabel="matching tolerance (px)",
        ylabel="F1",
        title="The same detections, scored five ways — the tolerance decides the ranking",
    )

    figures.comparison_matrix(
        noisy_rows,
        [
            ("F1", "f1", True),
            ("Precision", "precision", True),
            ("Recall", "recall", True),
            ("Pratt FOM", "pratt_fom", True),
            ("Time (ms)", "median_ms", False),
        ],
        IMAGES / "operator_matrix.png",
        title="Operators x metrics at sigma 30 — where they actually differ",
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS,
        "14_edge_detectors",
        {
            "tolerance_px": ed.TOLERANCE,
            "noise_levels": list(ed.NOISE_LEVELS),
            "clean": clean_rows,
            "noisy_sigma30": noisy_rows,
            "noise_sweep": noise_rows,
            "tolerance_sweep": tolerance_rows,
        },
    )

    cols = [
        ("Operator", "operator"),
        ("F1", "f1"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("Pratt FOM", "pratt_fom"),
        ("Best threshold", "best_threshold"),
        ("Time (ms)", "median_ms"),
    ]
    clean_table = markdown_table(clean_rows, cols)
    noisy_table = markdown_table(noisy_rows, cols)
    noise_table = markdown_table(
        noise_rows, [("Noise sigma", "noise_sigma")] + [(o, o) for o in ALL_OPERATORS]
    )
    tol_table = markdown_table(
        tolerance_rows,
        [("Tolerance (px)", "tolerance_px")]
        + [(o, o) for o in tolerance_rows[0] if o != "tolerance_px"],
    )
    write_tables(
        RESULTS,
        [
            ("Clean scene — six of seven score 1.000", clean_table),
            ("At sigma 30, where they come apart", noisy_table),
            ("F1 against noise", noise_table),
            ("The same detections, scored at five tolerances", tol_table),
            ("Four scenes down the rows", gallery_table),
        ],
    )
    print("\n" + clean_table + "\n\n" + noisy_table + "\n\n" + noise_table + "\n\n" + tol_table)

    auto = [r[ "Canny (auto median)"] for r in noise_rows]
    fixed = [r["Canny (fixed 50/150)"] for r in noise_rows]
    tol0 = tolerance_rows[0]
    tol3 = next(r for r in tolerance_rows if r["tolerance_px"] == 3)

    print("\n--- HEADLINE NUMBERS ---")
    print(f"clean scene      : {sum(1 for r in clean_rows if r['f1'] >= 0.999)} of "
          f"{len(clean_rows)} operators score 1.000 — the comparison says nothing")
    print(f"auto-Canny       : F1 {auto[0]:.3f} clean -> {auto[-1]:.3f} at sigma "
          f"{ed.NOISE_LEVELS[-1]:.0f}")
    print(f"fixed Canny      : F1 {fixed[0]:.3f} clean -> {fixed[-1]:.3f}")
    print(f"tolerance 0 px   : " + ", ".join(
        f"{k} {v:.3f}" for k, v in tol0.items() if k != "tolerance_px"))
    print(f"tolerance 3 px   : " + ", ".join(
        f"{k} {v:.3f}" for k, v in tol3.items() if k != "tolerance_px"))
    for r in noisy_rows:
        print(f"  sigma30  {r['operator']:22s} F1 {r['f1']:.3f}  P {r['precision']:.3f}  "
              f"R {r['recall']:.3f}  FOM {r['pratt_fom']:.3f}")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
