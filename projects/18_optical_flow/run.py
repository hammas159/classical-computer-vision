"""Run the optical-flow comparison and write results + figures.

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
from shared.metrics import endpoint_error  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import optical_flow as of  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"

#: The displacement the headline comparison runs at. 4 px is past plain LK's
#: limit and well inside the pyramid's, so the rows actually differ.
GALLERY_MAGNITUDE = 4.0

#: The gallery uses a **sinusoidal** field where the sweeps use a pure
#: translation. Two reasons, and both matter:
#:
#: * a translation is the right *controlled* stimulus for "how many pixels
#:   before this breaks", because there is one number to vary and one answer;
#: * but it makes a useless *picture*. A constant flow renders as a flat
#:   rectangle of one colour, so the truth panel carries no information and a
#:   method that gets the direction right anywhere looks right everywhere.
#:
#: A field that varies over the frame shows *where* each method fails, which is
#: the thing a colour-coded flow image is for.
GALLERY_KIND = "sinusoidal"

#: Border cropped before scoring. Every method handles the frame edge
#: differently and the edge is not what is being compared.
MARGIN = 16


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    print(f"Scoring {len(of.METHODS)} methods at {GALLERY_MAGNITUDE} px "
          f"on {len(of.IMAGES)} photographs ...")
    rows = of.evaluate_methods(magnitude=GALLERY_MAGNITUDE, images=of.IMAGES,
                               runs=args.runs)
    print(f"Scoring the same methods on a {GALLERY_KIND} field ...")
    gallery_rows = of.evaluate_methods(magnitude=GALLERY_MAGNITUDE, kind=GALLERY_KIND,
                                       images=of.IMAGES, runs=args.runs)

    print("Sweeping the displacement ...")
    mag_rows = of.sweep_magnitude(images=of.IMAGES)

    print("Sweeping the pyramid depth ...")
    pyr_rows = of.sweep_pyramid_levels(images=of.IMAGES)

    print("Sweeping Horn-Schunck's iteration count ...")
    hs_rows = of.sweep_hs_iterations(images=of.IMAGES)

    method_names = list(of.METHODS)

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    # Rows cross the axis the pool was built on -- texture density -- because
    # every method here recovers displacement from local image structure, and
    # where there is none the 2x2 system is singular. A subject-uniqueness rule
    # is applied on top so the four rows are four different pictures.
    TEXTURE_BANDS = [("smooth", 0, 64), ("mixed", 64, 70),
                     ("textured", 70, 75), ("dense", 75, 200)]
    candidates: dict[str, list[dict]] = {}
    for name in of.IMAGES:
        first, second, truth = of.make_pair(name, GALLERY_MAGNITUDE, kind=GALLERY_KIND)
        tex = of.texture_of(first)
        band = next(b for b, a, z in TEXTURE_BANDS if a <= tex < z)

        preds = [of.METHODS[m](first, second) for m in method_names]
        epes = [endpoint_error(p[MARGIN:-MARGIN, MARGIN:-MARGIN],
                               truth[MARGIN:-MARGIN, MARGIN:-MARGIN]) for p in preds]
        # every flow image shares one magnitude scale, or the colours lie
        scale = float(np.linalg.norm(truth, axis=-1).max())
        print(f"scene candidate {name:22s} texture {tex:5.1f}  [{band:8s}]  "
              f"best {min(epes):6.3f} px, worst {max(epes):7.3f} px")
        candidates.setdefault(band, []).append({
            "label": f"{name.replace('_', ' ')}\ntexture {tex:.0f}",
            "subject": name,
            "texture": tex,
            "images": [first, of.flow_to_colour(truth, scale)]
            + [of.flow_to_colour(p, scale) for p in preds],
            "notes": ["frame 1", "TRUE flow"] + [f"{e:.3f} px" for e in epes],
            "epes": epes,
            "score": float(np.std(epes)),
        })

    chosen, used = [], set()
    for band, _, _ in TEXTURE_BANDS:
        pool = sorted(candidates.get(band, []), key=lambda r: -r["score"])
        pick = next((r for r in pool if r["subject"] not in used), None)
        if pick:
            used.add(pick["subject"])
            chosen.append(pick)
    chosen = chosen[:4]
    assert len({r["subject"] for r in chosen}) == len(chosen), "a subject repeats"

    figures.gallery(
        ["frame 1", "TRUE flow"] + method_names,
        [(r["label"], r["images"]) for r in chosen],
        IMAGES / "compare_flow.png",
        cell_notes=[r["notes"] for r in chosen],
        suptitle=(
            f"A known sinusoidal flow field of up to {GALLERY_MAGNITUDE:g} px, "
            "recovered. Hue is direction, brightness is magnitude, on one shared "
            "scale — so a black panel means no motion was found at all. Cells "
            "are endpoint error in pixels."
        ),
    )
    gallery_table = markdown_table(
        [dict([("Sr", i), ("Scene", r["label"].replace("\n", " · "))]
              + list(zip(method_names, r["notes"][2:])))
         for i, r in enumerate(chosen, start=1)],
        [("Sr", "Sr"), ("Scene", "Scene")] + [(m, m) for m in method_names],
    )
    print(f"\nfront-on comparison: {len(chosen)} scenes x {len(method_names)} methods")
    print("\n--- optical flow ---\n" + gallery_table)

    # ------------------------------------------------------------------ #
    # the signature figure: EPE against the displacement being measured
    # ------------------------------------------------------------------ #
    # The diagonal is what predicting zero scores, so it is the line a method
    # must stay below to be doing anything at all. Where each curve crosses it
    # IS the answer to "how large is large motion".
    series = {m: [r[m] for r in mag_rows] for m in mag_rows[0] if m != "magnitude_px"}
    figures.lines(
        [r["magnitude_px"] for r in mag_rows],
        series,
        IMAGES / "displacement_sweep.png",
        xlabel="true displacement (px)",
        ylabel="endpoint error (px)",
        title=("Where each method dies. The 'predict zero' line is the score for "
               "doing nothing — a curve touching it has failed."),
        logx=True,
        logy=True,
    )

    figures.lines(
        [r["pyramid_levels"] for r in pyr_rows],
        {"largest tractable displacement (px)":
         [r["max_tractable_px"] or 0 for r in pyr_rows]},
        IMAGES / "pyramid_sweep.png",
        xlabel="pyramid levels",
        ylabel="px",
        title="Each pyramid level roughly doubles the motion LK can handle",
        logy=True,
    )

    figures.lines(
        [r["iterations"] for r in hs_rows],
        {"endpoint error (px)": [r["epe_px"] for r in hs_rows]},
        IMAGES / "hs_convergence.png",
        xlabel="Jacobi iterations",
        ylabel="endpoint error (px)",
        title="Horn-Schunck is not wrong at 100 iterations, it is unfinished",
        logx=True,
    )

    figures.comparison_matrix(
        rows,
        [("EPE (px)", "epe_px", False),
         ("EPE / displacement", "epe_relative", False),
         ("Time (ms)", "median_ms", False)],
        IMAGES / "method_matrix.png",
        title=f"Methods x metrics at a {GALLERY_MAGNITUDE:g} px displacement",
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS,
        "18_optical_flow",
        {
            "images": list(of.IMAGES),
            "gallery_magnitude_px": GALLERY_MAGNITUDE,
            "methods": rows,
            "methods_sinusoidal": gallery_rows,
            "magnitude_sweep": mag_rows,
            "pyramid_sweep": pyr_rows,
            "hs_convergence": hs_rows,
        },
    )

    method_table = markdown_table(
        rows,
        [("Method", "method"), ("EPE (px)", "epe_px"),
         ("EPE / displacement", "epe_relative"), ("Time (ms)", "median_ms")],
    )
    mag_table = markdown_table(
        mag_rows,
        [("Displacement (px)", "magnitude_px")] + [(m, m) for m in series],
    )
    hs_table = markdown_table(
        hs_rows,
        [("Iterations", "iterations"), ("EPE (px)", "epe_px"), ("Time (ms)", "median_ms")],
    )
    write_tables(
        RESULTS,
        [
            (f"Every method at {GALLERY_MAGNITUDE:g} px", method_table),
            ("Endpoint error against displacement", mag_table),
            ("Horn-Schunck convergence", hs_table),
            ("Four scenes down the rows", gallery_table),
        ],
    )
    print("\n" + method_table + "\n\n" + mag_table)

    control = next(r for r in rows if r["method"].startswith("Predict zero"))
    best = min((r for r in rows if not r["method"].startswith("Predict zero")),
               key=lambda r: r["epe_px"])
    lk_limit = next(r["max_tractable_px"] for r in pyr_rows if r["pyramid_levels"] == 0)

    print("\n--- HEADLINE NUMBERS ---")
    print(f"control (predict zero) : {control['epe_px']:.3f} px")
    print(f"best method            : {best['method']} @ {best['epe_px']:.3f} px "
          f"({control['epe_px'] / max(best['epe_px'], 1e-9):.0f}x better than the control)")
    print(f"plain LK dies past     : {lk_limit} px")
    print("pyramid levels -> largest tractable displacement: "
          + ", ".join(f"{r['pyramid_levels']}:{r['max_tractable_px']}" for r in pyr_rows))
    print(f"Horn-Schunck 100 iters : {hs_rows[1]['epe_px']:.3f} px, "
          f"3000 iters: {hs_rows[-1]['epe_px']:.3f} px "
          f"({hs_rows[-1]['median_ms'] / max(hs_rows[1]['median_ms'], 1e-9):.0f}x the cost)")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
