"""Run the multi-frame super-resolution comparison and write results + figures.

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

from shared import figures  # noqa: E402
from shared.io import ensure_rgb  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import multiframe_sr as sr  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"

SCALE = 3
FRAMES = 8

DETAIL_BANDS = [("smooth", 0, 700), ("moderate", 700, 2000),
                ("busy", 2000, 3000), ("dense", 3000, 99999)]


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    methods = list(sr.METHODS)

    print(f"Reconstructing at x{SCALE} from {FRAMES} frames, "
          f"{len(sr.IMAGES)} photographs ...")
    clean = sr.evaluate_methods(n_frames=FRAMES, scale=SCALE)
    for r in clean:
        print(f"  {r['method']:28s} {r['psnr_db']:6.2f} dB  SSIM {r['ssim']:.4f}  "
              f"{r['gain_over_single_db']:+6.2f} dB over one frame  {r['median_ms']:8.1f} ms")

    print("\nHow many frames it takes — including the n=1 control ...")
    counts = sr.sweep_frame_count(scale=SCALE)
    for r in counts:
        print(f"  {r['frames']:3d} frames  " + "  ".join(
            f"{m.split()[0][:7]}={r[m]:6.2f}" for m in methods))

    single_frame_ibp = counts[0]["Iterative back-projection"]
    bicubic = counts[0]["Single frame (bicubic)"]
    print(f"\n  the n=1 control: IBP scores {single_frame_ibp:.2f} dB from ONE frame, "
          f"{single_frame_ibp - bicubic:+.2f} dB over bicubic — that is deblurring, "
          "not fusion")

    print("\nSub-pixel against whole-pixel offsets ...")
    offsets = sr.subpixel_vs_integer(scale=SCALE, n_frames=FRAMES)
    for r in offsets:
        print(f"  {r['offsets']:24s} " + "  ".join(
            f"{m.split()[0][:7]}={r[m]:6.2f}" for m in methods))

    print("\nRegistration accuracy ...")
    registration = sr.evaluate_registration(scale=SCALE, n_frames=FRAMES)
    for r in registration:
        print(f"  {r['method']:20s} mean {r['mean_offset_error_px']:.4f} px, "
              f"median {r['median_offset_error_px']:.4f}")

    reg_errors = sr.sweep_registration_error(scale=SCALE, n_frames=FRAMES)
    scales = sr.sweep_scale()

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    from shared.metrics import psnr

    candidates: dict[str, list[dict]] = {}
    for name in sr.IMAGES:
        hr = sr._prepare(name, SCALE)
        d = sr.detail(hr)
        band = next(b for b, lo, hi in DETAIL_BANDS if lo <= d < hi)
        frames, offs = sr.make_stack(hr, n_frames=FRAMES, scale=SCALE, seed=0)

        lr_shown = cv2.resize(frames[0], (hr.shape[1], hr.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        panels, notes, scores = [hr, lr_shown], ["the truth", f"one frame, x{SCALE} smaller"], []
        for method in methods:
            out = sr.METHODS[method](frames, offs, scale=SCALE)
            value = psnr(out, hr)
            panels.append(ensure_rgb(out))
            notes.append(f"{value:.2f} dB")
            scores.append(value)

        candidates.setdefault(band, []).append({
            "label": f"{name.replace('_', ' ')}\ndetail {d:.0f}",
            "subject": name, "detail": d, "images": panels,
            "notes": notes, "scores": scores,
            "gain": scores[-1] - scores[0],
        })
        print(f"  scene candidate {name:24s} detail {d:7.1f} [{band:8s}]  "
              f"best {max(scores):.2f} dB, IBP gain {scores[-1] - scores[0]:+.2f}")

    chosen, used = [], set()
    for band, _, _ in DETAIL_BANDS:
        pool = sorted(candidates.get(band, []), key=lambda r: -r["gain"])
        pick = next((r for r in pool if r["subject"] not in used), None)
        if pick:
            used.add(pick["subject"])
            chosen.append(pick)
    chosen = chosen[:4]
    assert len({r["subject"] for r in chosen}) == len(chosen), "a subject repeats"

    figures.gallery(
        ["ground truth", "one low-res frame"] + methods,
        [(r["label"], r["images"]) for r in chosen],
        IMAGES / "compare_sr.png",
        cell_notes=[r["notes"] for r in chosen],
        suptitle=(
            f"{FRAMES} low-resolution frames with known sub-pixel offsets, reconstructed "
            f"at x{SCALE}. Cells are PSNR against the truth in column one."
        ),
    )
    gallery_table = markdown_table(
        [dict([("Sr", i), ("Scene", r["label"].replace("\n", " · "))]
              + [(m, f"{s:.2f}") for m, s in zip(methods, r["scores"])])
         for i, r in enumerate(chosen, start=1)],
        [("Sr", "Sr"), ("Scene", "Scene")] + [(m, m) for m in methods],
    )
    print("\n--- PSNR in dB ---\n" + gallery_table)

    # ------------------------------------------------------------------ #
    # the signature figure: where the gain actually comes from
    # ------------------------------------------------------------------ #
    figures.lines(
        [r["frames"] for r in counts],
        {m: [r[m] for r in counts] for m in methods},
        IMAGES / "frame_count.png",
        xlabel="frames available", ylabel="PSNR (dB)",
        logx=True,
        dashed={"Single frame (bicubic)"},
        vlines={f"one frame: IBP already {single_frame_ibp - bicubic:+.2f} dB": 1.0},
        title=("Where the gain comes from. The n=1 point is deblurring; everything right "
               "of it is fusion."),
    )

    figures.metric_bars(
        ["bicubic\n(one frame)", "IBP\n(one frame)", f"IBP\n({FRAMES} frames)",
         "IBP\n(32 frames)"],
        [bicubic, single_frame_ibp,
         next(r["Iterative back-projection"] for r in counts if r["frames"] == FRAMES),
         counts[-1]["Iterative back-projection"]],
        IMAGES / "gain_decomposition.png",
        ylabel="PSNR (dB)",
        title="Deblurring one frame, then adding frames — the two halves of the gain",
    )

    figures.lines(
        [r["registration_error_px"] for r in reg_errors],
        {m: [r[m] for r in reg_errors] for m in methods},
        IMAGES / "registration_error.png",
        xlabel="registration error injected (high-res pixels)", ylabel="PSNR (dB)",
        dashed={"Single frame (bicubic)"},
        title="How well the frames must be aligned before fusion stops paying",
    )

    figures.lines(
        [r["scale"] for r in scales],
        {"Single frame (bicubic)": [r["single_frame_db"] for r in scales],
         "Best multi-frame method": [r["best_db"] for r in scales],
         "the gain": [r["gain_db"] for r in scales]},
        IMAGES / "scale_sweep.png",
        xlabel="upscaling factor", ylabel="PSNR (dB)",
        dashed={"Single frame (bicubic)"},
        title=("The harder the upscale, the LESS a fixed number of frames buys — "
               "x2 needs a quarter of the samples x4 does"),
    )

    figures.metric_bars(
        [r["method"] for r in registration],
        [r["mean_offset_error_px"] for r in registration],
        IMAGES / "registration_methods.png",
        ylabel="mean offset error (high-res px)",
        title="Registration accuracy — ECC against phase correlation",
        highlight_best="min",
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS, "40_multiframe_super_resolution",
        {
            "images": list(sr.IMAGES),
            "detail": {n: round(sr.detail(sr._prepare(n, SCALE)), 1) for n in sr.IMAGES},
            "scale": SCALE, "frames": FRAMES,
            "methods": clean,
            "frame_count": counts,
            "subpixel_vs_integer": offsets,
            "registration": registration,
            "registration_error": reg_errors,
            "scale_sweep": scales,
            "single_frame_ibp_db": round(single_frame_ibp, 3),
            "bicubic_db": round(bicubic, 3),
        },
    )

    methods_table = markdown_table(
        clean, [("Method", "method"), ("PSNR (dB)", "psnr_db"), ("SSIM", "ssim"),
                ("Gain over one frame (dB)", "gain_over_single_db"),
                ("Time (ms)", "median_ms")])
    counts_table = markdown_table(
        counts, [("Frames", "frames")] + [(m, m) for m in methods])
    offsets_table = markdown_table(
        offsets, [("Offsets", "offsets")] + [(m, m) for m in methods])
    registration_table = markdown_table(
        registration, [("Method", "method"), ("Mean error (px)", "mean_offset_error_px"),
                       ("Median (px)", "median_offset_error_px")])
    reg_error_table = markdown_table(
        reg_errors, [("Injected error (px)", "registration_error_px")]
        + [(m, m) for m in methods])

    write_tables(
        RESULTS,
        [
            (f"At x{SCALE} from {FRAMES} frames", methods_table),
            ("Against the number of frames", counts_table),
            ("Sub-pixel against whole-pixel offsets", offsets_table),
            ("Registration accuracy", registration_table),
            ("Tolerance to registration error", reg_error_table),
            ("Four scenes down the rows", gallery_table),
        ],
    )
    print("\n" + methods_table + "\n\n" + counts_table)

    # ------------------------------------------------------------------ #
    # headlines
    # ------------------------------------------------------------------ #
    print("\n--- HEADLINE NUMBERS ---")
    by_method = {r["method"]: r for r in clean}
    best = max(clean, key=lambda r: r["psnr_db"])
    print(f"best is {best['method']} at {best['psnr_db']:.2f} dB, "
          f"{best['gain_over_single_db']:+.2f} dB over the best single-frame method")
    print(f"  project 21 found the entire nearest-to-Lanczos argument worth 0.49 dB; "
          f"this is {best['gain_over_single_db'] / 0.49:.1f}x that")

    most = counts[-1]["Iterative back-projection"]
    print(f"\nbut {single_frame_ibp - bicubic:+.2f} dB of it is available from ONE frame "
          f"({single_frame_ibp:.2f} against bicubic's {bicubic:.2f}) — IBP inverts the "
          "blur whether or not there is anything to fuse")
    print(f"  the fusion is the rest: {most - single_frame_ibp:+.2f} dB from 1 to "
          f"{counts[-1]['frames']} frames")

    sub = next(r for r in offsets if "Sub-pixel" in r["offsets"])
    whole = next(r for r in offsets if "Integer" in r["offsets"])
    print(f"\nsub-pixel offsets are worth {sub['Iterative back-projection'] - whole['Iterative back-projection']:+.2f} dB "
          f"over whole-pixel ones ({sub['Iterative back-projection']:.2f} vs "
          f"{whole['Iterative back-projection']:.2f}) — that part is genuinely resolution")

    naive = by_method["Naive average"]
    print(f"\nnaive averaging scores {naive['gain_over_single_db']:+.2f} dB — WORSE than "
          "one frame. Averaging frames that are not aligned is a blur.")

    ecc = next(r for r in registration if r["method"] == "ECC")
    phase = next(r for r in registration if r["method"] == "Phase correlation")
    print(f"\nECC registers to {ecc['mean_offset_error_px']:.4f} px against phase "
          f"correlation's {phase['mean_offset_error_px']:.4f} — "
          f"{phase['mean_offset_error_px'] / ecc['mean_offset_error_px']:.1f}x better, "
          "after a sign inversion that had it reported as 11x worse")

    print(f"
sixteen frames are worth {scales[0]['gain_db']:.2f} dB at x{scales[0]['scale']} "
          f"and {scales[-1]['gain_db']:.2f} dB at x{scales[-1]['scale']} — the same frames "
          "spread over four times as many output pixels")

    breakeven = next((r["registration_error_px"] for r in reg_errors
                      if r["Iterative back-projection"] <= r["Single frame (bicubic)"]), None)
    print(f"\nfusion stops paying at about {breakeven:g} px of registration error"
          if breakeven else "\nfusion still pays at every registration error tested")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
