"""Run the histogram-equalisation comparison and write results + figures.

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
from shared.metrics import entropy, psnr, rms_contrast  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import histogram_eq as he  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"


def transfer_curve(src: np.ndarray, out: np.ndarray) -> np.ndarray:
    """The empirical input-level -> output-level curve a method actually applied.

    Measured rather than read out of the method, because CLAHE and AHE have no
    single curve to read: their map depends on where the pixel is. Taking the
    **median** output for each input level gives every method one comparable
    curve, and the spread around it is precisely what makes a local method local.
    """
    s, o = to_gray(src).ravel(), to_gray(out).ravel()
    curve = np.full(256, np.nan, np.float32)
    order = np.argsort(s, kind="stable")
    s_sorted, o_sorted = s[order], o[order]
    edges = np.searchsorted(s_sorted, np.arange(257))
    for level in range(256):
        lo, hi = edges[level], edges[level + 1]
        if hi > lo:
            curve[level] = np.median(o_sorted[lo:hi])
    return curve


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    print(f"Equalising {len(he.SCORED_IMAGES)} photographs with "
          f"{len(he.METHODS)} methods + 1 oracle ...")
    rows, degraded = he.evaluate_methods(runs=args.runs)

    print("Sweeping the CLAHE clip limit ...")
    clip_rows = he.sweep_clip_limit()

    print("Sweeping the tile grid ...")
    grid_rows = he.sweep_grid()

    print("Ranking the methods by each metric ...")
    disagreement = he.metric_disagreement()

    method_names = list(he.METHODS) + [he.ORACLE_NAME]

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    # Rows are chosen across the axis the pool was built on -- how much of the
    # tone range the ORIGINAL already used -- because that is what decides
    # whether equalisation has anything to give. A subject-uniqueness rule is
    # applied on top: four rows that differ only by tone are a comparison of
    # exposures, not of pictures.
    TONE_BANDS = [("very low", 0, 60), ("low", 60, 78), ("mid", 78, 90), ("high", 90, 101)]
    candidates: dict[str, list[dict]] = {}
    for name in he.SCORED_IMAGES:
        clean = he.load_scene(name)
        g = to_gray(clean)
        lo, hi = np.percentile(g, (1, 99))
        tone = float(hi - lo) / 255.0 * 100
        band = next(b for b, a, z in TONE_BANDS if a <= tone < z)

        bad = he.degrade(clean, kind="low_contrast", noise_sigma=3.0, seed=0)
        outs = [he.METHODS[m](bad) for m in he.METHODS] + [he.eq_match_oracle(bad, clean)]
        scores = [psnr(o, clean) for o in outs]
        best = max(scores[:-1])  # best REAL method, the oracle is not competing
        print(f"scene candidate {name:22s} tone {tone:5.1f}%  [{band:8s}]  "
              f"degraded {psnr(bad, clean):5.2f} dB -> best method {best:5.2f} dB, "
              f"oracle {scores[-1]:5.2f} dB")
        candidates.setdefault(band, []).append({
            "label": f"{name.replace('_', ' ')}\ntone {tone:.0f}%",
            "subject": name,
            "tone": tone,
            "images": [clean, bad] + outs,
            "notes": ["original", f"{psnr(bad, clean):.1f} dB"]
            + [f"{s:.1f} dB" for s in scores],
            "scores": scores,
            # the scene where the methods disagree most carries the most
            # information; one they all handle identically carries none
            "score": float(np.std(scores[:-1])),
        })

    chosen, used = [], set()
    for band, _, _ in TONE_BANDS:
        pool = sorted(candidates.get(band, []), key=lambda r: -r["score"])
        pick = next((r for r in pool if r["subject"] not in used), None)
        if pick:
            used.add(pick["subject"])
            chosen.append(pick)
    chosen = chosen[:4]
    assert len({r["subject"] for r in chosen}) == len(chosen), "a subject repeats"

    figures.gallery(
        ["original", "degraded input"] + method_names,
        [(r["label"], r["images"]) for r in chosen],
        IMAGES / "compare_equalisers.png",
        cell_notes=[r["notes"] for r in chosen],
        suptitle=(
            "Four photographs flattened to a third of their tone range, then "
            "restored. Cells are PSNR against the original. The last column is "
            "handed the original's histogram — it is a ceiling, not a method."
        ),
    )
    gallery_table = markdown_table(
        [dict([("Sr", i), ("Scene", r["label"].replace("\n", " · "))]
              + [("Degraded", r["notes"][1])]
              + list(zip(method_names, r["notes"][2:])))
         for i, r in enumerate(chosen, start=1)],
        [("Sr", "Sr"), ("Scene", "Scene"), ("Degraded", "Degraded")]
        + [(m, m) for m in method_names],
    )
    print(f"\nfront-on comparison: {len(chosen)} scenes x {len(method_names)} columns")
    print("\n--- equalisation ---\n" + gallery_table)

    # ------------------------------------------------------------------ #
    # the signature figure: the transfer curve IS the method
    # ------------------------------------------------------------------ #
    # Every row in the table is a map from input grey to output grey. Drawing
    # that map says more than any metric: the oracle's curve is the correct
    # answer, so the distance from it is the error, visible directly.
    # The widest-toned of the four chosen scenes. A curve can only be drawn
    # where the degraded image actually has pixels, and the 25%-tone night scene
    # leaves three quarters of the axis empty.
    curve_scene = max(chosen, key=lambda r: r["tone"])["subject"]
    clean = he.load_scene(curve_scene)
    bad = he.degrade(clean, kind="low_contrast", noise_sigma=3.0, seed=0)
    curves = {m: transfer_curve(bad, he.METHODS[m](bad)) for m in he.METHODS}
    curves[he.ORACLE_NAME] = transfer_curve(bad, he.eq_match_oracle(bad, clean))
    levels = np.arange(256)
    # NaN where the degraded image has no pixels at that level. Left as NaN on
    # purpose: matplotlib draws a gap, where substituting 0 drew a spike to the
    # floor and made every curve look discontinuous.
    occupied = np.flatnonzero(~np.isnan(curves[he.ORACLE_NAME]))
    lo, hi = int(occupied.min()), int(occupied.max()) + 1
    figures.lines(
        levels[lo:hi].tolist(),
        {m: c[lo:hi].tolist() for m, c in curves.items()},
        IMAGES / "transfer_curves.png",
        xlabel="input level (the degraded image)",
        ylabel="output level (median)",
        title=(f"The curve IS the method — {curve_scene.replace('_', ' ')}. "
               "The oracle's curve is the correct answer; distance from it is error."),
    )

    figures.lines(
        [r["clip_limit"] for r in clip_rows],
        {
            "PSNR / 25 (normalised)": [r["psnr_db"] / 25.0 for r in clip_rows],
            "SSIM": [r["ssim"] for r in clip_rows],
            "entropy / 8 (normalised)": [r["entropy_bits"] / 8.0 for r in clip_rows],
            "noise sigma / 40": [r["noise_sigma"] / 40.0 for r in clip_rows],
        },
        IMAGES / "clip_sweep.png",
        xlabel="CLAHE clip limit",
        ylabel="normalised score",
        title="The clip limit has an optimum — and entropy does not know where it is",
    )

    figures.lines(
        [r["grid"] for r in grid_rows],
        {"PSNR (dB)": [r["psnr_db"] for r in grid_rows],
         "entropy (bits)": [r["entropy_bits"] for r in grid_rows]},
        IMAGES / "grid_sweep.png",
        xlabel="tiles per side",
        ylabel="score",
        title="Tile count: too few is global HE, too many is per-pixel noise",
    )

    figures.comparison_matrix(
        rows,
        [("PSNR (dB)", "psnr_db", True), ("SSIM", "ssim", True),
         ("Entropy (bits)", "entropy_bits", True),
         ("RMS contrast", "rms_contrast", True),
         ("Noise sigma", "noise_sigma", False)],
        IMAGES / "method_matrix.png",
        title="Methods x metrics — the no-reference columns disagree with the rest",
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS,
        "17_histogram_equalization",
        {
            "images": list(he.IMAGES),
            "scored_images": list(he.SCORED_IMAGES),
            "match_reference": he.MATCH_REFERENCE,
            "methods": rows,
            "degraded_input": degraded,
            "clip_sweep": clip_rows,
            "grid_sweep": grid_rows,
            "disagreement": disagreement,
        },
    )

    method_table = markdown_table(
        rows,
        [("Method", "method"), ("PSNR (dB)", "psnr_db"), ("SSIM", "ssim"),
         ("Entropy (bits)", "entropy_bits"), ("RMS contrast", "rms_contrast"),
         ("Noise sigma", "noise_sigma"), ("Time (ms)", "median_ms")],
    )
    clip_table = markdown_table(
        clip_rows,
        [("Clip limit", "clip_limit"), ("PSNR (dB)", "psnr_db"), ("SSIM", "ssim"),
         ("Entropy (bits)", "entropy_bits"), ("Noise sigma", "noise_sigma")],
    )
    write_tables(
        RESULTS,
        [
            ("Every method, full-reference and no-reference", method_table),
            ("CLAHE clip limit swept", clip_table),
            ("Four scenes down the rows", gallery_table),
        ],
    )
    print("\n" + method_table + "\n\n" + clip_table)

    control = next(r for r in rows if r["method"].startswith("Do nothing"))
    oracle = next(r for r in rows if r["method"] == he.ORACLE_NAME)
    real = [r for r in rows if r["method"] != he.ORACLE_NAME
            and not r["method"].startswith("Do nothing")]
    best = max(real, key=lambda r: r["psnr_db"])
    most_entropy = max(real, key=lambda r: r["entropy_bits"])
    best_clip = max(clip_rows, key=lambda r: r["psnr_db"])

    print("\n--- HEADLINE NUMBERS ---")
    print(f"degraded input   : {degraded['psnr']:.2f} dB, entropy {degraded['entropy']:.3f}")
    print(f"control          : {control['psnr_db']:.2f} dB")
    print(f"best real method : {best['method']} @ {best['psnr_db']:.2f} dB "
          f"(+{best['psnr_db'] - control['psnr_db']:.2f} over doing nothing)")
    print(f"oracle           : {oracle['psnr_db']:.2f} dB "
          f"(+{oracle['psnr_db'] - best['psnr_db']:.2f} over the best method), "
          f"entropy {oracle['entropy_bits']:.3f}")
    print(f"highest entropy  : {most_entropy['method']} @ "
          f"{most_entropy['entropy_bits']:.3f} bits but only "
          f"{most_entropy['psnr_db']:.2f} dB")
    print(f"best clip limit  : {best_clip['clip_limit']} @ {best_clip['psnr_db']:.2f} dB")
    print(f"rank by PSNR     : {disagreement['rank_by_psnr']}")
    print(f"rank by entropy  : {disagreement['rank_by_entropy']}")
    print(f"winners agree    : {disagreement['winners_agree']}")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
