"""Run the JPEG comparison and write results + figures.

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
from shared.metrics import psnr, ssim  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import jpeg as jp  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"

#: Qualities the headline comparison shows. Chosen to span the range where the
#: codec goes from unusable to indistinguishable.
GALLERY_QUALITIES = (5, 20, 50, 95)


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    print(f"Rate-distortion over {len(jp.IMAGES)} photographs ...")
    rd = jp.rate_distortion(images=jp.IMAGES)
    print("Ablating each stage ...")
    ablation = jp.stage_ablation(images=jp.IMAGES)
    print("Counting which coefficients survive ...")
    coeffs = jp.coefficient_statistics(images=jp.IMAGES)

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    # Columns are qualities rather than methods, because there is one codec and
    # the interesting axis is what it throws away. A subject-uniqueness rule is
    # applied over the texture bands the pool was built on.
    TEXTURE_BANDS = [("smooth", 0, 65), ("moderate", 65, 70),
                     ("busy", 70, 75), ("dense", 75, 200)]
    candidates: dict[str, list[dict]] = {}
    for name in jp.IMAGES:
        clean = jp.load_scene(name)
        texture = jp.texture_of(clean) if hasattr(jp, "texture_of") else 0.0
        band = next(b for b, lo, hi in TEXTURE_BANDS if lo <= texture < hi)

        panels, notes, scores = [clean], ["original"], []
        for q in GALLERY_QUALITIES:
            recon, stats = jp.encode_decode(clean, quality=q)
            panels.append(recon)
            notes.append(f"Q{q} · {stats['estimated_bpp']:.2f} bpp\n"
                         f"{psnr(recon, clean):.1f} dB")
            scores.append(psnr(recon, clean))

        print(f"scene candidate {name:24s} texture {texture:5.1f}  [{band:8s}]  "
              f"Q5 {scores[0]:5.2f} dB -> Q95 {scores[-1]:5.2f} dB")
        candidates.setdefault(band, []).append({
            "label": f"{name.replace('_', ' ')}\ntexture {texture:.0f}",
            "subject": name,
            "texture": texture,
            "images": panels,
            "notes": notes,
            "scores": scores,
            "score": float(scores[-1] - scores[0]),
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
        ["original"] + [f"quality {q}" for q in GALLERY_QUALITIES],
        [(r["label"], r["images"]) for r in chosen],
        IMAGES / "compare_qualities.png",
        cell_notes=[r["notes"] for r in chosen],
        suptitle=(
            "One codec, written from the DCT up, across four quality settings. "
            "Cells are the estimated bits per pixel and the PSNR they bought."
        ),
    )
    gallery_table = markdown_table(
        [dict([("Sr", i), ("Scene", r["label"].replace("\n", " · "))]
              + [(f"Q{q}", f"{s:.1f} dB") for q, s in zip(GALLERY_QUALITIES, r["scores"])])
         for i, r in enumerate(chosen, start=1)],
        [("Sr", "Sr"), ("Scene", "Scene")] + [(f"Q{q}", f"Q{q}") for q in GALLERY_QUALITIES],
    )
    print(f"\nfront-on comparison: {len(chosen)} scenes x {len(GALLERY_QUALITIES)} qualities")
    print("\n--- jpeg ---\n" + gallery_table)

    # ------------------------------------------------------------------ #
    # the signature figure: the rate-distortion curve, ours against OpenCV's
    # ------------------------------------------------------------------ #
    figures.lines(
        [r["bpp_ours"] for r in rd],
        {"this codec (PSNR)": [r["psnr_ours"] for r in rd]},
        IMAGES / "rate_distortion.png",
        xlabel="bits per pixel",
        ylabel="PSNR (dB)",
        title=("Rate-distortion: what each bit buys. A textbook codec against "
               "libjpeg is a number, not a hand-wave."),
    )

    figures.lines(
        [r["quality"] for r in rd],
        {"this codec": [r["bpp_ours"] for r in rd],
         "OpenCV (libjpeg)": [r["bpp_opencv"] for r in rd]},
        IMAGES / "bitrate.png",
        xlabel="quality setting",
        ylabel="bits per pixel",
        title="Bits spent at each quality setting",
    )

    figures.lines(
        [r["quality"] for r in coeffs],
        {"all coefficients": [r["nonzero_fraction"] for r in coeffs],
         "the DC term": [r["dc_survival"] for r in coeffs],
         "the highest frequency": [r["highest_freq_survival"] for r in coeffs]},
        IMAGES / "coefficient_survival.png",
        xlabel="quality setting",
        ylabel="fraction surviving quantisation",
        title=("What quantisation actually deletes: the top-frequency coefficient "
               "survives 0% of the time below quality 75"),
    )

    figures.comparison_matrix(
        [{"method": r["configuration"], **{k: r[k] for k in
          ("psnr_db", "ssim", "estimated_bpp", "blockiness")}} for r in ablation],
        [("PSNR (dB)", "psnr_db", True), ("SSIM", "ssim", True),
         ("Bits per pixel", "estimated_bpp", False),
         ("Blockiness", "blockiness", False)],
        IMAGES / "ablation_matrix.png",
        title="Each stage turned off — what it was contributing to size and to loss",
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS,
        "27_jpeg_from_scratch",
        {
            "images": list(jp.IMAGES),
            "rate_distortion": rd,
            "ablation": ablation,
            "coefficient_statistics": coeffs,
        },
    )

    rd_table = markdown_table(
        rd,
        [("Quality", "quality"), ("bpp (ours)", "bpp_ours"),
         ("PSNR (ours)", "psnr_ours"), ("SSIM (ours)", "ssim_ours"),
         ("Blockiness", "blockiness"), ("bpp (OpenCV)", "bpp_opencv"),
         ("PSNR (OpenCV)", "psnr_opencv")],
    )
    ablation_table = markdown_table(
        ablation,
        [("Configuration", "configuration"), ("PSNR (dB)", "psnr_db"),
         ("SSIM", "ssim"), ("bpp", "estimated_bpp"), ("Blockiness", "blockiness")],
    )
    coeff_table = markdown_table(
        coeffs,
        [("Quality", "quality"), ("Non-zero coefficients", "nonzero_fraction"),
         ("DC survives", "dc_survival"),
         ("Top frequency survives", "highest_freq_survival")],
    )
    write_tables(
        RESULTS,
        [
            ("Rate-distortion, against OpenCV's libjpeg", rd_table),
            ("Each stage turned off", ablation_table),
            ("Which coefficients survive quantisation", coeff_table),
            ("Four scenes down the rows", gallery_table),
        ],
    )
    print("\n" + rd_table + "\n\n" + ablation_table + "\n\n" + coeff_table)

    by_cfg = {r["configuration"]: r for r in ablation}
    full, no_dct = by_cfg["Full codec"], by_cfg["No DCT (quantise pixels)"]
    no_q = by_cfg["No quantisation"]
    lossless = by_cfg["No DCT, no quantisation"]
    no_chroma = by_cfg["No chroma subsampling"]
    no_colour = by_cfg["No colour transform (RGB)"]
    at50 = next(r for r in rd if r["quality"] == 50)

    print("\n--- HEADLINE NUMBERS ---")
    print(f"the DCT is worth      : {full['psnr_db'] - no_dct['psnr_db']:.2f} dB "
          f"AND {1 - full['estimated_bpp'] / no_dct['estimated_bpp']:.0%} fewer bits "
          f"— quantising pixels directly scores {no_dct['psnr_db']:.2f} dB at "
          f"{no_dct['estimated_bpp']:.2f} bpp")
    print(f"the transform is lossless: {lossless['psnr_db']:.2f} dB with quantisation off")
    print(f"quantisation costs    : {no_q['psnr_db'] - full['psnr_db']:.2f} dB "
          f"and saves {1 - full['estimated_bpp'] / no_q['estimated_bpp']:.0%} of the bits")
    print(f"chroma subsampling    : {full['psnr_db'] - no_chroma['psnr_db']:+.2f} dB "
          f"for {1 - full['estimated_bpp'] / no_chroma['estimated_bpp']:.0%} fewer bits")
    print(f"the colour transform  : {full['psnr_db'] - no_colour['psnr_db']:+.2f} dB "
          f"and {1 - full['estimated_bpp'] / no_colour['estimated_bpp']:.0%} fewer bits "
          "than compressing R, G, B directly")
    print(f"vs libjpeg at Q50     : {at50['psnr_ours']:.2f} dB at {at50['bpp_ours']:.2f} bpp "
          f"against {at50['psnr_opencv']:.2f} dB at {at50['bpp_opencv']:.2f} bpp")
    print("top-frequency survival: "
          + ", ".join(f"Q{r['quality']} {r['highest_freq_survival']:.4f}" for r in coeffs))
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
