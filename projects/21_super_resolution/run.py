"""Run the super-resolution comparison and write results + figures.

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

from shared import figures, synth  # noqa: E402
from shared.metrics import psnr  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import super_resolution as sr  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"

#: The scale factor the headline comparison runs at.
GALLERY_SCALE = 4


def edge_profile(img: np.ndarray, row: int | None = None) -> list[float]:
    """One horizontal scanline of luminance, for the edge-profile figure.

    The signature picture for this project. A PSNR table says the interpolators
    differ by half a decibel; a scanline says *why* that is the whole story.
    Nearest visibly steps and the rest ramp, but next to the original they are
    all the same curve -- the truth swings between 36 and 243 across four pixels
    and every method draws a smooth line through the middle of it.
    """
    from shared.io import to_gray

    g = to_gray(img)
    return g[row if row is not None else g.shape[0] // 2].astype(float).tolist()


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    print(f"Upscaling {len(sr.IMAGES)} photographs x{GALLERY_SCALE} "
          f"with {len(sr.METHODS)} methods + 1 oracle ...")
    rows, extra = sr.evaluate_methods(scale=GALLERY_SCALE, images=sr.IMAGES,
                                      runs=args.runs)

    print("Sweeping the scale factor ...")
    scale_rows = sr.sweep_scale(images=sr.IMAGES)
    print("Comparing the two downsampling operators ...")
    degradation = sr.compare_degradations(images=sr.IMAGES, scale=GALLERY_SCALE)
    print("Measuring recovered high-frequency energy ...")
    freq_rows = sr.evaluate_frequency(scale=GALLERY_SCALE, images=sr.IMAGES)

    names = list(sr.METHODS)

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    # Rows cross the axis the pool was built on -- texture density -- because
    # that is exactly the content the downsample destroys. A subject-uniqueness
    # rule is applied on top so the four rows are four different pictures.
    # Banded on this project's own `frequency_content` -- the percentage of
    # spectral energy above half Nyquist -- rather than on the selector's
    # texture axis. The two measure the same idea on different scales, and
    # mixing them put all twelve scenes into a single band.
    TEXTURE_BANDS = [("smooth", 0, 47), ("mixed", 47, 52),
                     ("textured", 52, 58), ("dense", 58, 200)]
    candidates: dict[str, list[dict]] = {}
    for name in sr.IMAGES:
        hr = sr.load_scene(name)
        h = (hr.shape[0] // GALLERY_SCALE) * GALLERY_SCALE
        w = (hr.shape[1] // GALLERY_SCALE) * GALLERY_SCALE
        hr = hr[:h, :w]
        texture = sr.frequency_content(hr) * 100
        band = next(b for b, a, z in TEXTURE_BANDS if a <= texture < z)

        lr = synth.downsample_for_sr(hr, scale=GALLERY_SCALE)
        outs = [sr.METHODS[m](lr, GALLERY_SCALE)[:h, :w] for m in names]
        oracle = sr.oracle_band_limited(hr, GALLERY_SCALE)
        scores = [psnr(o, hr) for o in outs]
        ceiling = psnr(oracle, hr)

        # the low-resolution input is shown at its true size relative to the
        # others, upscaled with NEAREST so the missing detail is visible as
        # blocks rather than hidden by a smooth resample
        lr_shown = cv2.resize(lr, (w, h), interpolation=cv2.INTER_NEAREST)
        print(f"scene candidate {name:22s} hf {texture:5.2f}%  [{band:8s}]  "
              f"best {max(scores):5.2f} dB, ceiling {ceiling:5.2f} dB, "
              f"headroom {ceiling - max(scores):+.2f}")
        candidates.setdefault(band, []).append({
            "label": f"{name.replace('_', ' ')}\nhigh-freq {texture:.1f}%",
            "subject": name,
            "texture": texture,
            "images": [hr, lr_shown] + outs + [oracle],
            "notes": ["original", f"{w // GALLERY_SCALE}x{h // GALLERY_SCALE}"]
            + [f"{s:.1f} dB" for s in scores] + [f"{ceiling:.1f} dB"],
            "scores": scores,
            "ceiling": ceiling,
            "score": float(np.std(scores)),
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
        ["original", f"input (1/{GALLERY_SCALE})"] + names + [sr.ORACLE_NAME],
        [(r["label"], r["images"]) for r in chosen],
        IMAGES / "compare_upscalers.png",
        cell_notes=[r["notes"] for r in chosen],
        suptitle=(
            f"A {GALLERY_SCALE}x downsample, then put back. Cells are PSNR "
            "against the original. The last column is the original low-passed by "
            "the same blur the downsample applied — what a perfect "
            "reconstruction is aiming at."
        ),
    )
    gallery_table = markdown_table(
        [dict([("Sr", i), ("Scene", r["label"].replace("\n", " · "))]
              + [(m, f"{s:.1f} dB") for m, s in zip(names, r["scores"])]
              + [("Oracle", f"{r['ceiling']:.1f} dB")])
         for i, r in enumerate(chosen, start=1)],
        [("Sr", "Sr"), ("Scene", "Scene")] + [(m, m) for m in names]
        + [("Oracle", "Oracle")],
    )
    print(f"\nfront-on comparison: {len(chosen)} scenes x {len(names)} methods")
    print("\n--- super-resolution ---\n" + gallery_table)

    # ------------------------------------------------------------------ #
    # the signature figure: how each interpolator crosses an edge
    # ------------------------------------------------------------------ #
    edge_scene = chosen[-1]["subject"]
    hr = sr.load_scene(edge_scene)
    h = (hr.shape[0] // GALLERY_SCALE) * GALLERY_SCALE
    w = (hr.shape[1] // GALLERY_SCALE) * GALLERY_SCALE
    hr = hr[:h, :w]
    lr = synth.downsample_for_sr(hr, scale=GALLERY_SCALE)
    # pick the scanline with the strongest single step in the original, so the
    # figure is about an edge rather than about whatever was in the middle row
    from shared.io import to_gray

    g = to_gray(hr).astype(float)
    row = int(np.argmax(np.abs(np.diff(g, axis=1)).max(axis=1)))
    col = int(np.argmax(np.abs(np.diff(g[row]))))
    lo, hi = max(0, col - 12), min(w, col + 13)

    series = {"Original (truth)": edge_profile(hr, row)[lo:hi]}
    for m in names:
        series[m] = edge_profile(sr.METHODS[m](lr, GALLERY_SCALE)[:h, :w], row)[lo:hi]
    figures.lines(
        list(range(lo, hi)),
        series,
        IMAGES / "edge_profile.png",
        xlabel="column (pixels)",
        ylabel="luminance",
        title=(f"One scanline, {edge_scene.replace('_', ' ')}: the truth (dashed) "
               "oscillates, and every method returns the same smooth ramp"),
        dashed={"Original (truth)"},
    )

    figures.lines(
        [r["scale"] for r in scale_rows],
        {m: [r[m] for r in scale_rows] for m in names}
        | {"spread between methods": [r["spread_db"] for r in scale_rows],
           "headroom to the oracle": [r["oracle_headroom_db"] for r in scale_rows]},
        IMAGES / "scale_sweep.png",
        xlabel="upscale factor",
        ylabel="PSNR (dB)",
        title="Every method falls together; the choice between them stays small",
        dashed={"spread between methods", "headroom to the oracle"},
    )

    figures.metric_bars(
        [r["method"] for r in freq_rows],
        [r["high_freq_energy"] for r in freq_rows],
        IMAGES / "frequency_content.png",
        ylabel="fraction of spectral energy above half Nyquist",
        title="Nothing is added: no method approaches the original's high-frequency energy",
    )

    figures.comparison_matrix(
        rows,
        [("PSNR (dB)", "psnr_db", True), ("SSIM", "ssim", True),
         ("Time (ms)", "median_ms", False)],
        IMAGES / "method_matrix.png",
        title=f"Methods x metrics at {GALLERY_SCALE}x",
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS,
        "21_super_resolution",
        {
            "images": list(sr.IMAGES),
            "scale": GALLERY_SCALE,
            "methods": rows,
            "summary": extra,
            "scale_sweep": scale_rows,
            "degradation_comparison": degradation,
            "frequency_content": freq_rows,
        },
    )

    method_table = markdown_table(
        rows,
        [("Method", "method"), ("PSNR (dB)", "psnr_db"), ("SSIM", "ssim"),
         ("Time (ms)", "median_ms")],
    )
    scale_table = markdown_table(
        scale_rows,
        [("Scale", "scale")] + [(m, m) for m in names]
        + [("Spread", "spread_db"), ("Headroom", "oracle_headroom_db")],
    )
    freq_table = markdown_table(
        freq_rows, [("Method", "method"), ("High-freq energy", "high_freq_energy")],
    )
    write_tables(
        RESULTS,
        [
            (f"Every method at {GALLERY_SCALE}x", method_table),
            ("Scale factor swept", scale_table),
            ("High-frequency energy recovered", freq_table),
            ("Four scenes down the rows", gallery_table),
        ],
    )
    print("\n" + method_table + "\n\n" + scale_table + "\n\n" + freq_table)

    real = [r for r in rows if r["method"] != sr.ORACLE_NAME]
    best = max(real, key=lambda r: r["psnr_db"])
    oracle = next(r for r in rows if r["method"] == sr.ORACLE_NAME)
    classics = [r for r in real if r["method"] in
                ("Nearest", "Bilinear", "Bicubic", "Lanczos-4", "Edge-directed")]
    classic_best = max(classics, key=lambda r: r["psnr_db"])
    classic_worst = min(classics, key=lambda r: r["psnr_db"])

    print("\n--- HEADLINE NUMBERS ---")
    print(f"band-limited ref  : {oracle['psnr_db']:.2f} dB")
    print(f"best method       : {best['method']} @ {best['psnr_db']:.2f} dB "
          f"({best['psnr_db'] - oracle['psnr_db']:+.2f} dB against the reference)")
    print(f"classic spread    : {classic_worst['method']} {classic_worst['psnr_db']:.2f} -> "
          f"{classic_best['method']} {classic_best['psnr_db']:.2f} = "
          f"{classic_best['psnr_db'] - classic_worst['psnr_db']:.2f} dB")
    print(f"back-projection   : {best['psnr_db'] - classic_best['psnr_db']:+.2f} dB "
          f"over the best classic interpolator")
    print(f"degradation       : {degradation}")
    print("headroom by scale : "
          + ", ".join(f"{r['scale']}x:{r['oracle_headroom_db']:.2f}" for r in scale_rows))
    orig_hf = next(r["high_freq_energy"] for r in freq_rows if r["method"].startswith("Original"))
    best_hf = max(r["high_freq_energy"] for r in freq_rows if not r["method"].startswith("Original"))
    print(f"high-freq energy  : original {orig_hf:.5f}, best method {best_hf:.5f} "
          f"({best_hf / orig_hf * 100:.0f}% of it)")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
