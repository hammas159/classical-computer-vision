"""Run the image-quality-metric comparison and write results + figures.

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
from shared.io import to_gray  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import quality_metrics as qm  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"

#: The PSNR every degradation in the headline comparison is tuned to hit. If
#: PSNR were sufficient, all six columns would be interchangeable at this point.
TARGET_PSNR = 28.0


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    print(f"Building equal-PSNR degradations at {TARGET_PSNR:g} dB "
          f"on {len(qm.IMAGES)} photographs ...")
    equal = qm.equal_psnr_comparison(TARGET_PSNR, images=qm.IMAGES)
    print("Ranking them by every metric ...")
    disagreement = qm.ranking_disagreement(TARGET_PSNR, images=qm.IMAGES)

    print("Sweeping degradation strength ...")
    sweeps = {d: qm.sweep_strength(d, images=qm.IMAGES) for d in qm.DEGRADATIONS}

    print("Repeating at other PSNR targets ...")
    by_target = {t: qm.ranking_disagreement(t, images=qm.IMAGES)
                 for t in qm.TARGET_PSNRS}

    metrics = list(qm.METRICS)
    degradations = list(qm.DEGRADATIONS)

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    # Every column is tuned to the SAME PSNR, so the pictures are the argument:
    # if PSNR measured what people mean by quality, these would be equally bad.
    BRIGHT_BANDS = [("dark", 0, 90), ("mid", 90, 110),
                    ("bright", 110, 135), ("very bright", 135, 300)]
    candidates: dict[str, list[dict]] = {}
    for name in qm.IMAGES:
        clean = qm.load_scene(name)
        bright = float(to_gray(clean).mean())
        band = next(b for b, lo, hi in BRIGHT_BANDS if lo <= bright < hi)

        panels, notes, ssims = [clean], ["original"], []
        for deg in degradations:
            strength, got = qm.find_strength_for_psnr(clean, deg, TARGET_PSNR)
            fn, _ = qm.DEGRADATIONS[deg]
            bad = fn(clean, strength)
            ssim_v = qm.METRICS["SSIM"][0](bad, clean)
            ssims.append(ssim_v)
            panels.append(bad)
            notes.append(f"{got:.1f} dB\nSSIM {ssim_v:.3f}")

        print(f"scene candidate {name:22s} brightness {bright:5.1f}  [{band:11s}]  "
              f"SSIM at equal PSNR spans {min(ssims):.3f} to {max(ssims):.3f}")
        candidates.setdefault(band, []).append({
            "label": f"{name.replace('_', ' ')}\nbrightness {bright:.0f}",
            "subject": name,
            "brightness": bright,
            "images": panels,
            "notes": notes,
            "ssims": ssims,
            "score": float(max(ssims) - min(ssims)),
        })

    chosen, used = [], set()
    for band, _, _ in BRIGHT_BANDS:
        pool = sorted(candidates.get(band, []), key=lambda r: -r["score"])
        pick = next((r for r in pool if r["subject"] not in used), None)
        if pick:
            used.add(pick["subject"])
            chosen.append(pick)
    chosen = chosen[:4]
    assert len({r["subject"] for r in chosen}) == len(chosen), "a subject repeats"

    figures.gallery(
        ["original"] + degradations,
        [(r["label"], r["images"]) for r in chosen],
        IMAGES / "compare_degradations.png",
        cell_notes=[r["notes"] for r in chosen],
        suptitle=(
            f"Six different damages, every one tuned to the SAME {TARGET_PSNR:g} dB "
            "PSNR. If PSNR measured quality these would be equally bad. The SSIM "
            "beneath each says they are not."
        ),
    )
    gallery_table = markdown_table(
        [dict([("Sr", i), ("Scene", r["label"].replace("\n", " · "))]
              + [(d, f"{s:.3f}") for d, s in zip(degradations, r["ssims"])])
         for i, r in enumerate(chosen, start=1)],
        [("Sr", "Sr"), ("Scene", "Scene")] + [(d, d) for d in degradations],
    )
    print(f"\nfront-on comparison: {len(chosen)} scenes x {len(degradations)} damages")
    print("\n--- SSIM at equal PSNR ---\n" + gallery_table)

    # ------------------------------------------------------------------ #
    # the signature figure: every metric against strength, on one axis
    # ------------------------------------------------------------------ #
    # A metric is useful if it is monotone in the damage and comparable across
    # damages. Plotting all six degradations for one metric shows whether its
    # scale means the same thing in each -- which is the property PSNR lacks.
    for metric in ("PSNR", "SSIM"):
        figures.lines(
            list(range(len(sweeps[degradations[0]]))),
            {d: [r[metric] for r in sweeps[d]] for d in degradations},
            IMAGES / f"strength_{metric.lower().replace(' ', '_').replace('-', '_')}.png",
            xlabel="degradation strength (step)",
            ylabel=metric,
            title=(f"{metric} against strength, for six different damages — "
                   "a usable metric would have one scale, not six"),
        )

    taus = disagreement["kendall_tau_vs_psnr"]
    figures.metric_bars(
        list(taus), [taus[m] for m in taus],
        IMAGES / "ranking_agreement.png",
        ylabel="Kendall tau against the PSNR ranking",
        title=("How much each metric disagrees with PSNR about which damage is "
               "worse, at equal PSNR"),
    )

    figures.comparison_matrix(
        [{"method": r["degradation"], **{m: r[m] for m in metrics}} for r in equal],
        [(m, m, qm.METRICS[m][1]) for m in metrics],
        IMAGES / "metric_matrix.png",
        title=f"Six damages at {TARGET_PSNR:g} dB x six metrics — only PSNR is flat",
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS,
        "26_quality_metrics",
        {
            "images": list(qm.IMAGES),
            "target_psnr": TARGET_PSNR,
            "equal_psnr": equal,
            "disagreement": disagreement,
            "disagreement_by_target": {str(k): v for k, v in by_target.items()},
            "strength_sweeps": sweeps,
        },
    )

    equal_table = markdown_table(
        equal,
        [("Degradation", "degradation"), ("Strength", "strength"),
         ("PSNR", "achieved_psnr")] + [(m, m) for m in metrics if m != "PSNR"],
    )
    tau_table = markdown_table(
        [{"metric": m, "tau": taus[m]} for m in taus],
        [("Metric", "metric"), ("Kendall tau vs PSNR", "tau")],
    )
    write_tables(
        RESULTS,
        [
            (f"Six damages, all at {TARGET_PSNR:g} dB", equal_table),
            ("How much each metric disagrees with PSNR", tau_table),
            ("Four scenes down the rows", gallery_table),
        ],
    )
    print("\n" + equal_table + "\n\n" + tau_table)

    ssim_col = {r["degradation"]: r["SSIM"] for r in equal}
    shift = next(r for r in equal if r["degradation"] == "Sub-pixel shift")

    print("\n--- HEADLINE NUMBERS ---")
    print(f"at a fixed {TARGET_PSNR:g} dB, SSIM spans "
          f"{min(ssim_col.values()):.4f} ({min(ssim_col, key=ssim_col.get)}) to "
          f"{max(ssim_col.values()):.4f} ({max(ssim_col, key=ssim_col.get)}) "
          f"— a spread of {max(ssim_col.values()) - min(ssim_col.values()):.4f}")
    print(f"worst damage by PSNR : {disagreement['worst_by_psnr']}")
    print(f"worst damage by SSIM : {disagreement['worst_by_ssim']}")
    print(f"PSNR and SSIM agree  : {disagreement['psnr_and_ssim_agree']}")
    print("kendall tau vs PSNR  : "
          + ", ".join(f"{m} {t:.3f}" for m, t in taus.items() if m != "PSNR"))
    print(f"sub-pixel shift      : could not be made mild enough to reach "
          f"{TARGET_PSNR:g} dB — it tops out at {shift['achieved_psnr']:.2f} dB "
          f"while being nearly invisible")
    for t, d in by_target.items():
        print(f"  at {t:g} dB the PSNR/SSIM rankings agree: {d['psnr_and_ssim_agree']}")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
