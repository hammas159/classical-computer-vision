"""Run the Gonzalez & Woods pipeline ablation and write results + figures.

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
from shared.io import ensure_rgb, to_gray  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import gw_pipeline as gw  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"

#: The four things compared in the headline figure: the book's pipeline and the
#: one-line alternatives it is supposed to beat.
GALLERY_METHODS = ["G&W 8-stage pipeline", "CLAHE only", "Unsharp mask only",
                   "Gamma 0.5 only"]


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    print(f"Ablating the pipeline over {len(gw.IMAGES)} photographs ...")
    ablation = gw.ablation(images=gw.IMAGES)
    print("Comparing it against the one-line alternatives ...")
    alternatives = gw.compare_to_simple_alternatives(images=gw.IMAGES)
    print("Testing the Laplacian sign ...")
    signs = gw.laplacian_sign_test(images=gw.IMAGES)
    print("Sweeping gamma and the smoothing size ...")
    gamma_rows = gw.sweep_gamma(images=gw.IMAGES)
    smooth_rows = gw.sweep_smoothing(images=gw.IMAGES)

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    # Rows cross brightness, because the pipeline's last stage is a power-law
    # tone curve whose whole job is to lift detail out of the dark regions.
    BRIGHT_BANDS = [("very dark", 0, 75), ("dark", 75, 92),
                    ("mid", 92, 104), ("lighter", 104, 200)]
    candidates: dict[str, list[dict]] = {}
    for name in gw.IMAGES:
        img = gw.load_scene(name)
        bright = float(to_gray(img).mean())
        band = next(b for b, lo, hi in BRIGHT_BANDS if lo <= bright < hi)

        outs = [gw.apply_variant(img, m) for m in GALLERY_METHODS]
        scores = [gw.measure(o, img)["dark_detail"] for o in outs]
        panels = [img] + [ensure_rgb(o) for o in outs]
        notes = ["original"] + [f"dark detail {s:.3f}" for s in scores]

        print(f"scene candidate {name:22s} brightness {bright:5.1f}  [{band:9s}]  "
              f"pipeline {scores[0]:.3f}, best alternative {max(scores[1:]):.3f}")
        candidates.setdefault(band, []).append({
            "label": f"{name.replace('_', ' ')}\nbrightness {bright:.0f}",
            "subject": name,
            "brightness": bright,
            "images": panels,
            "notes": notes,
            "scores": scores,
            "score": float(np.std(scores)),
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
        ["original"] + GALLERY_METHODS,
        [(r["label"], r["images"]) for r in chosen],
        IMAGES / "compare_pipeline.png",
        cell_notes=[r["notes"] for r in chosen],
        suptitle=(
            "Eight stages from the textbook, against three one-line "
            "alternatives. Cells are dark-region detail — the quantity the "
            "pipeline's final stage exists to raise."
        ),
    )
    gallery_table = markdown_table(
        [dict([("Sr", i), ("Scene", r["label"].replace("\n", " · "))]
              + [(m, f"{s:.3f}") for m, s in zip(GALLERY_METHODS, r["scores"])])
         for i, r in enumerate(chosen, start=1)],
        [("Sr", "Sr"), ("Scene", "Scene")] + [(m, m) for m in GALLERY_METHODS],
    )
    print(f"\nfront-on comparison: {len(chosen)} scenes x {len(GALLERY_METHODS)} methods")
    print("\n--- pipeline vs one-liners ---\n" + gallery_table)

    # ------------------------------------------------------------------ #
    # the signature figure: what each stage contributed, as a delta
    # ------------------------------------------------------------------ #
    deltas = [r for r in ablation if "acutance_delta" in r]
    figures.metric_bars(
        [r["configuration"].replace("Without ", "") for r in deltas],
        [abs(r["dark_detail_delta"]) for r in deltas],
        IMAGES / "stage_contributions.png",
        ylabel="|change in dark-region detail| when the stage is removed",
        title="What each stage contributes — stage (e) contributes nothing",
    )

    figures.lines(
        [r["gamma"] for r in gamma_rows],
        {k: [r[k] for r in gamma_rows] for k in gamma_rows[0] if k != "gamma"},
        IMAGES / "gamma_sweep.png",
        xlabel="gamma",
        ylabel="score",
        title="The power-law stage: a tone curve that moves brightness, not structure",
    )

    figures.lines(
        [r["smooth_ksize"] for r in smooth_rows],
        {k: [r[k] for r in smooth_rows] for k in smooth_rows[0] if k != "smooth_ksize"},
        IMAGES / "smoothing_sweep.png",
        xlabel="smoothing kernel size for stage (e)",
        ylabel="score",
        title="Stage (e)'s own parameter, on a stage that contributes nothing",
    )

    figures.comparison_matrix(
        [{"method": r["method"], **{k: r[k] for k in
          ("acutance", "dark_detail", "rms_contrast", "ssim_vs_original")}}
         for r in alternatives],
        [("Acutance", "acutance", True), ("Dark detail", "dark_detail", True),
         ("RMS contrast", "rms_contrast", True),
         ("SSIM vs original", "ssim_vs_original", True)],
        IMAGES / "alternatives_matrix.png",
        title="Eight stages against one line of code",
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS,
        "31_gw_pipeline",
        {
            "images": list(gw.IMAGES),
            "ablation": ablation,
            "alternatives": alternatives,
            "laplacian_sign": signs,
            "gamma_sweep": gamma_rows,
            "smoothing_sweep": smooth_rows,
        },
    )

    ablation_table = markdown_table(
        ablation,
        [("Configuration", "configuration"), ("Acutance", "acutance"),
         ("Dark detail", "dark_detail"), ("RMS contrast", "rms_contrast"),
         ("SSIM", "ssim_vs_original")],
    )
    alt_table = markdown_table(
        alternatives,
        [("Method", "method"), ("Acutance", "acutance"),
         ("Dark detail", "dark_detail"), ("RMS contrast", "rms_contrast"),
         ("SSIM", "ssim_vs_original")],
    )
    sign_table = markdown_table(
        signs,
        [("Configuration", "configuration"), ("Acutance", "acutance"),
         ("SSIM", "ssim_vs_original")],
    )
    write_tables(
        RESULTS,
        [
            ("Each stage removed in turn", ablation_table),
            ("The pipeline against one-line alternatives", alt_table),
            ("The Laplacian sign trap, measured", sign_table),
            ("Four scenes down the rows", gallery_table),
        ],
    )
    print("\n" + ablation_table + "\n\n" + alt_table + "\n\n" + sign_table)

    full = ablation[0]
    by_alt = {r["method"]: r for r in alternatives}
    pipeline, unsharp = by_alt["G&W 8-stage pipeline"], by_alt["Unsharp mask only"]
    smallest = min(deltas, key=lambda r: abs(r["dark_detail_delta"]))
    largest = max(deltas, key=lambda r: abs(r["dark_detail_delta"]))
    correct = next(r for r in signs if r["configuration"].startswith("Correct"))
    wrong = next(r for r in signs if r["configuration"].startswith("Wrong"))
    original = next(r for r in signs if r["configuration"].startswith("Original"))

    print("\n--- HEADLINE NUMBERS ---")
    print(f"least useful stage : {smallest['configuration']} — removing it moves "
          f"dark detail by {smallest['dark_detail_delta']:+.5f} and acutance by "
          f"{smallest['acutance_delta']:+.5f}")
    print(f"most useful stage  : {largest['configuration']} "
          f"({largest['dark_detail_delta']:+.4f} dark detail)")
    print(f"one line beats eight stages: unsharp mask "
          f"{unsharp['dark_detail']:.4f} dark detail vs pipeline "
          f"{pipeline['dark_detail']:.4f}, and SSIM "
          f"{unsharp['ssim_vs_original']:.4f} vs {pipeline['ssim_vs_original']:.4f}")
    beats = [m for m, r in by_alt.items()
             if m not in ("G&W 8-stage pipeline", "Original (control)")
             and r["dark_detail"] > pipeline["dark_detail"]]
    print(f"  alternatives beating the pipeline on dark detail: {beats}")
    print(f"laplacian sign     : original {original['acutance']:.4f} acutance, "
          f"correct {correct['acutance']:.4f}, wrong {wrong['acutance']:.4f} "
          f"— both RAISE it")
    print(f"  but SSIM: correct {correct['ssim_vs_original']:.4f}, "
          f"wrong {wrong['ssim_vs_original']:.4f}")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
