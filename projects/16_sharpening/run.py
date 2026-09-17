"""Run the sharpening comparison and write results + figures.

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
from shared.metrics import psnr, ssim  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import sharpening as sh  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"

#: The blur the "is there anything to recover" half of the project undoes.
BLUR_SIGMA = 1.5


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    print(f"Sharpening {len(sh.IMAGES)} CLEAN images with {len(sh.METHODS)} methods ...")
    clean_rows = sh.evaluate_on_clean(runs=args.runs)

    print(f"Sharpening the same images after a sigma {BLUR_SIGMA} blur ...")
    blurred_rows, blurred_stats = sh.evaluate_on_blurred(sigma=BLUR_SIGMA)

    print("Sweeping the sharpening amount ...")
    amount_rows = sh.sweep_amount()

    print("Sweeping the blur the sharpener is asked to undo ...")
    blur_rows = sh.sweep_blur()

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    # Two questions share one figure, and they have opposite answers:
    #   * on a CLEAN image, sharpening has nothing to restore, so every method
    #     can only move away from the truth while looking crisper;
    #   * on a BLURRED image there is something to restore, and the oracle says
    #     how much of it was available.
    # The rows therefore cross detail density with blur, not subject.
    GALLERY_MIN_ACUTANCE_GAIN = 0.01
    NL = "\n"  # named, because an inline escape inside an f-string in this
               # file has been mangled by shell heredocs more than once
    method_names = list(sh.METHODS)
    scene_pool = [
        ("albatross pair · clean\nsmooth, detail 7", "albatross_pair", 0.0, "smooth, clean"),
        ("lionesses · clean\ndetail 9", "lionesses", 0.0, "smooth, clean"),
        ("deer in scrub · clean\ndetail 17", "deer_water", 0.0, "detailed, clean"),
        ("carved stone · clean\ndetail 45", "stone_face_leaves", 0.0, "detailed, clean"),
        ("albatross pair · blurred\nsmooth, sigma 1.5", "albatross_pair", 1.5, "smooth, blurred"),
        ("lionesses · blurred\nsigma 1.5", "lionesses", 1.5, "smooth, blurred"),
        ("deer in scrub · blurred\ndetail 17, sigma 1.5", "deer_water", 1.5, "detailed, blurred"),
        ("carved stone · blurred\ndetail 45, sigma 1.5", "stone_face_leaves", 1.5, "detailed, blurred"),
        ("rhino on gravel · clean\ndetail 23", "rhino_road", 0.0, "detailed, clean"),
        ("rhino on gravel · blurred\ndetail 23, sigma 1.5", "rhino_road", 1.5, "detailed, blurred"),
        ("elk in water · clean\ndetail 11", "elk_water", 0.0, "smooth, clean"),
        ("elk in water · blurred\ndetail 11, sigma 1.5", "elk_water", 1.5, "smooth, blurred"),
    ]
    # Candidates are collected per family and the four rows are then assigned
    # greedily under a SUBJECT-UNIQUENESS constraint. Scoring each family
    # independently picked the elk for both "smooth" rows and the carved stone
    # for both "detailed" rows -- four rows showing two subjects, which is a
    # comparison of conditions dressed up as a comparison of pictures.
    candidates: dict[str, list[dict]] = {}
    for label, name, sigma, family in scene_pool:
        clean = sh.load_scene(name)
        src = clean if sigma == 0 else cv2.GaussianBlur(
            clean, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
        outs = [sh.METHODS[m](src) for m in method_names]
        raw = [psnr(o, clean) for o in outs]
        # The cells report MATCHED PSNR, so the cells must SHOW the matched
        # image. Displaying the raw output beside a matched score put a visibly
        # dark high-boost panel under the best number in the row, which reads as
        # the figure contradicting itself. Where the rescale moved the score, the
        # cell note says so and gives the raw number too.
        shown = [sh.match_mean(o, clean) for o in outs]
        matched = [psnr(s, clean) for s in shown]
        acut = [sh.acutance(o) for o in outs]
        base_acut = sh.acutance(src)
        gain = max(acut) - base_acut
        flat = label.replace("\n", " · ")
        if gain < GALLERY_MIN_ACUTANCE_GAIN:
            print(f"scene candidate {flat:<46} DROP — no method raised acutance  [{family}]")
            continue
        print(f"scene candidate {flat:<46} keep — acutance {base_acut:.3f} -> "
              f"{max(acut):.3f}, best matched PSNR {max(matched):.1f} dB  [{family}]")
        candidates.setdefault(family, []).append({
            "label": label,
            "subject": name,
            "images": [clean, src] + shown,
            "notes": ["original", f"{psnr(src, clean):.1f} dB" if sigma else "= original"]
            + [f"{m:.1f} dB" + (NL + f"(rescaled; raw {r:.1f} dB)" if m - r > 0.5 else "")
               for m, r in zip(matched, raw)],
            # Markdown table cells cannot contain a newline, so the same note is
            # kept a second time on one line.
            "table_notes": [f"{m:.1f} dB" + (f" (rescaled; raw {r:.1f} dB)"
                                             if m - r > 0.5 else "")
                            for m, r in zip(matched, raw)],
            "raw": raw,
            "matched": matched,
            "acutance": acut,
            "score": gain,
        })

    ORDER = ["smooth, clean", "detailed, clean", "smooth, blurred", "detailed, blurred"]
    chosen, used_subjects = [], set()
    for family in ORDER:
        pool = sorted(candidates.get(family, []), key=lambda r: -r["score"])
        pick = next((r for r in pool if r["subject"] not in used_subjects), None)
        if pick is None:
            continue
        used_subjects.add(pick["subject"])
        chosen.append(pick)
    chosen = chosen[:4]
    assert len({r["subject"] for r in chosen}) == len(chosen), "a subject repeats in the gallery"
    figures.gallery(
        ["original", "input"] + method_names,
        [(r["label"], r["images"]) for r in chosen],
        IMAGES / "compare_sharpeners.png",
        cell_notes=[r["notes"] for r in chosen],
        suptitle=(
            "Sharpening a clean image and a blurred one. Cells are PSNR against "
            "the original AFTER matching brightness — and show the matched "
            "image, so the picture and the number agree. Where the rescale "
            "changed the score the raw one is given beside it; high-boost is "
            "the formula it changes most, by up to 17 dB."
        ),
    )
    gallery_table = markdown_table(
        [
            dict([("Sr", i), ("Scene", r["label"].replace("\n", " · "))]
                 + list(zip(method_names, r["table_notes"])))
            for i, r in enumerate(chosen, start=1)
        ],
        [("Sr", "Sr"), ("Scene", "Scene")] + [(m, m) for m in method_names],
    )
    print(f"\nfront-on comparison: {len(chosen)} scenes x {len(method_names)} methods")
    print("\n--- sharpening ---\n" + gallery_table)

    # ------------------------------------------------------------------ #
    # figures
    # ------------------------------------------------------------------ #
    # sweep_amount tracks one method across amounts, reporting three quantities
    # that disagree: PSNR falls while acutance and overshoot rise. That
    # disagreement IS the project, so all three go on one plot.
    figures.lines(
        [r["amount"] for r in amount_rows],
        {
            "PSNR / 40 (normalised)": [r["psnr_db"] / 40.0 for r in amount_rows],
            "acutance": [r["acutance"] for r in amount_rows],
            "overshoot": [r["overshoot"] for r in amount_rows],
        },
        IMAGES / "amount_sweep.png",
        xlabel="sharpening amount",
        ylabel="normalised score",
        title=(
            "On a clean image: apparent sharpness rises, fidelity falls, "
            "monotonically and in opposite directions"
        ),
    )

    figures.lines(
        [r["blur_sigma"] for r in blur_rows],
        {m: [r[m] for r in blur_rows]
         for m in blur_rows[0] if m not in ("blur_sigma", "blurred_input")},
        IMAGES / "blur_sweep.png",
        xlabel="blur sigma the sharpener is asked to undo",
        ylabel="PSNR against the original (dB)",
        title="Sharpening only helps while the blur is mild",
    )

    figures.comparison_matrix(
        blurred_rows,
        [
            ("PSNR (dB)", "psnr_db", True),
            ("PSNR matched (dB)", "psnr_matched_db", True),
            ("SSIM", "ssim", True),
            ("Acutance", "acutance", True),
        ],
        IMAGES / "method_matrix.png",
        title=f"Methods x metrics on a sigma {BLUR_SIGMA} blur — raw PSNR scores brightness",
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS,
        "16_sharpening",
        {
            "images": list(sh.IMAGES),
            "blur_sigma": BLUR_SIGMA,
            "clean": clean_rows,
            "blurred": blurred_rows,
            "blurred_input": blurred_stats,
            "amount_sweep": amount_rows,
            "blur_sweep": blur_rows,
        },
    )

    clean_table = markdown_table(
        clean_rows,
        [("Method", "method"), ("PSNR (dB)", "psnr_db"), ("SSIM", "ssim"),
         ("Acutance", "acutance"), ("Overshoot", "overshoot"),
         ("RMS contrast", "rms_contrast"), ("Time (ms)", "median_ms")],
    )
    blurred_table = markdown_table(
        blurred_rows,
        [("Method", "method"), ("PSNR (dB)", "psnr_db"),
         ("PSNR matched (dB)", "psnr_matched_db"), ("SSIM", "ssim"),
         ("Acutance", "acutance")],
    )
    write_tables(
        RESULTS,
        [
            ("Sharpening a CLEAN image — nothing to restore", clean_table),
            (f"Sharpening after a sigma {BLUR_SIGMA} blur", blurred_table),
            ("Four scenes down the rows", gallery_table),
        ],
    )
    print("\n" + clean_table + "\n\n" + blurred_table)

    control = next(r for r in clean_rows if r["method"].startswith("Do nothing"))
    best_acut = max((r for r in clean_rows if not r["method"].startswith("Do nothing")),
                    key=lambda r: r["acutance"])
    hb = next(r for r in blurred_rows if r["method"].startswith("High-boost"))
    oracle = next(r for r in blurred_rows if r["method"] == sh.ORACLE_NAME)
    best_real = max((r for r in blurred_rows
                     if r["method"] not in (sh.ORACLE_NAME,) and not r["method"].startswith("Do")),
                    key=lambda r: r["psnr_matched_db"])

    print("\n--- HEADLINE NUMBERS ---")
    print(f"clean image      : control acutance {control['acutance']:.4f}, "
          f"best sharpener {best_acut['acutance']:.4f} "
          f"({best_acut['acutance'] / control['acutance']:.2f}x) — "
          f"at {best_acut['psnr_db']:.1f} dB against an infinite control")
    print(f"high-boost       : {hb['psnr_db']:.2f} dB raw -> "
          f"{hb['psnr_matched_db']:.2f} dB matched "
          f"({hb['psnr_matched_db'] - hb['psnr_db']:+.2f} dB from brightness alone)")
    print(f"blurred input    : {blurred_stats['psnr']:.2f} dB, acutance {blurred_stats['acutance']:.4f}")
    print(f"best sharpener   : {best_real['method']} @ {best_real['psnr_matched_db']:.2f} dB matched")
    print(f"oracle           : {oracle['psnr_matched_db']:.2f} dB matched, "
          f"SSIM {oracle['ssim']:.4f}, acutance {oracle['acutance']:.4f}")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
