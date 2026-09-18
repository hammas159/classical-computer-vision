"""Run the red-eye removal study and write results + figures.

    python run.py

Regenerates `results/results.json`, `results/tables.md` and every figure in
`docs/images/`. Every number in the README comes out of this script.
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
from shared.io import ensure_rgb, real_photo  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import red_eye as re58  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"

#: The four portraits the comparison figure is built from: the two with four
#: pupils each, the one whose own lipstick is redder than the planted red-eye,
#: and the one whose frame is full of tulips.
FIGURE_PHOTOS = ("two_women_in_headdress", "girl_with_tulips",
                 "woman_in_red_scarf", "woman_with_curly_hair")


def _overlay(img, mask, colour=(255, 40, 40)):
    """The detected mask painted over the photograph, so misses are visible."""
    out = ensure_rgb(img).copy()
    sel = mask > 0
    out[sel] = (0.35 * out[sel] + 0.65 * np.array(colour, np.float32)).astype(np.uint8)
    return out


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=6)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    print("Detectors on six real portraits, against the true pupils ...")
    real = re58.evaluate_on_portraits()
    for r in real:
        print(f"  {r['detector']:24s} IoU {r['iou']:.4f}  recall {r['pupil_recall']:.4f}  "
              f"FP area x{r['false_positive_area_ratio']:.3f}  {r['median_ms']:.2f} ms")

    print("\nThe same detectors on the generated scene ...")
    generated = re58.evaluate_detectors(scenes=args.scenes, runs=args.runs)
    for r in generated:
        print(f"  {r['detector']:24s} IoU {r['iou']:.4f}  recall {r['pupil_recall']:.4f}  "
              f"FP area x{r['false_positive_area_ratio']:.3f}")

    both = re58.generated_versus_real_background(scenes=args.scenes)
    rows_per_photo = re58.per_portrait()
    clean = re58.false_positives_on_clean_photographs()
    corrections_real = re58.corrections_on_portraits()
    corrections_gen = re58.evaluate_corrections(scenes=args.scenes, runs=args.runs)
    threshold_real = re58.sweep_threshold_on_portraits()
    pipeline = re58.end_to_end_on_portraits()
    distractors = re58.sweep_distractors(scenes=args.scenes)
    pupil_sizes = re58.sweep_pupil_size(scenes=args.scenes)

    print("\nOn six photographs with no red-eye in them at all ...")
    for r in clean:
        print(f"  {r['photograph']:24s} blobs {r['pupil_like_blobs']:3d}  "
              + "  ".join(f"{d}={r[d]}" for d in re58.DETECTORS))

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    scenes = {n: re58.portrait_scene(n) for n in FIGURE_PHOTOS}

    rows = [("with red-eye", [scenes[n][0] for n in FIGURE_PHOTOS])]
    notes = [[f"{len(re58.PORTRAITS[n]['pupils'])} pupils" for n in FIGURE_PHOTOS]]

    by_photo = {r["photograph"]: r for r in rows_per_photo}
    for det in re58.DETECTORS:
        rows.append((det, [_overlay(scenes[n][0], re58.DETECTORS[det](scenes[n][0]))
                           for n in FIGURE_PHOTOS]))
        notes.append([f"IoU {by_photo[n][det]:.3f}" for n in FIGURE_PHOTOS])

    fixed = []
    for n in FIGURE_PHOTOS:
        flash, _, truth = scenes[n]
        fixed.append(re58.correct_desaturate(flash, re58.detect_face_constrained(flash)))
    rows.append(("corrected\n(face-constrained\n+ desaturate)", fixed))
    notes.append(["" for _ in FIGURE_PHOTOS])

    figures.gallery(
        [n.replace("_", " ") for n in FIGURE_PHOTOS],
        rows, IMAGES / "compare_detection.png",
        cell_notes=notes,
        suptitle=("Red-eye planted at recorded pupil positions in four photographs. "
                  "Everything else in each frame — the skin, the tulips, the scarf, the "
                  "lipstick — is real, and is what the detectors have to reject. Cells "
                  "are IoU against the true pupil mask."),
    )

    table_rows = []
    for sr, n in enumerate(FIGURE_PHOTOS, start=1):
        row = {"Sr": sr, "Photograph": n.replace("_", " "),
               "Pupils": len(re58.PORTRAITS[n]["pupils"]),
               "Blobs": by_photo[n]["pupil_like_blobs"]}
        row |= {d: f"{by_photo[n][d]:.3f}" for d in re58.DETECTORS}
        table_rows.append(row)
    gallery_table = markdown_table(
        table_rows,
        [("Sr", "Sr"), ("Photograph", "Photograph"), ("Pupils", "Pupils"),
         ("Pupil-like blobs", "Blobs")] + [(d, d) for d in re58.DETECTORS])
    print("\n--- the four rows ---\n" + gallery_table)

    # ------------------------------------------------------------------ #
    # the signature figure: what each detector does to a clean photograph
    # ------------------------------------------------------------------ #
    figures.metric_bars(
        list(re58.DETECTORS),
        [sum(r[d] for r in clean) for d in re58.DETECTORS],
        IMAGES / "clean_photographs.png",
        ylabel="wrongly detected pixels over six photographs",
        title=("Six photographs with no red-eye in them. Truth is empty, so every "
               "pixel here is a mistake — and this is the case a red-eye remover "
               "meets most often."),
        highlight_best="min",
    )

    clean_panels = []
    for r in clean[:4]:
        img = real_photo(r["photograph"])
        clean_panels.append((f"{r['photograph'].replace('_', ' ')}\n"
                             f"colour only: {r['Colour only (control)']} px",
                             _overlay(img, re58.detect_colour_only(img))))
    figures.grid(clean_panels, IMAGES / "clean_false_positives.png", ncols=2,
                 suptitle=("What the naive detector 'finds' in photographs that contain "
                           "no eyes at all."))

    figures.metric_bars(
        [r["detector"] for r in both] * 2,
        [r["generated_iou"] for r in both] + [r["real_iou"] for r in both],
        IMAGES / "generated_vs_real.png",
        ylabel="IoU against the true pupil mask",
        title=("Left four: the generated scene. Right four: the same detectors on real "
               "photographs. The generated scene reports the problem solved."),
    )

    figures.metric_bars(
        [r["photograph"].replace("_", " ") for r in corrections_real]
        + ["mean, did nothing"],
        [r["Zero red channel"] for r in corrections_real]
        + [float(np.mean([r["did nothing"] for r in corrections_real]))],
        IMAGES / "zero_red_is_worse.png",
        ylabel="pupil PSNR (dB) after zeroing the red channel",
        title=("Zeroing the red channel, per photograph, against the mean of leaving "
               "the red-eye alone. Higher is better."),
    )

    figures.lines(
        [r["threshold"] for r in threshold_real],
        {d: [r[f"{d} IoU"] for r in threshold_real] for d in re58.DETECTORS},
        IMAGES / "threshold_real.png",
        xlabel="redness threshold", ylabel="IoU on real portraits",
        title="The redness threshold on real photographs",
        vlines={"default 0.25": 0.25},
    )

    figures.lines(
        [r["distractors"] for r in distractors],
        {d: [r[d] for r in distractors] for d in re58.DETECTORS},
        IMAGES / "distractor_sweep.png",
        xlabel="drawn distractors in the generated scene",
        ylabel="false-positive area / pupil area",
        title="The generated scene's own distractor sweep, for comparison",
    )

    figures.lines(
        [r["pupil_radius"] for r in pupil_sizes],
        {d: [r[d] for r in pupil_sizes] for d in re58.DETECTORS},
        IMAGES / "pupil_size_sweep.png",
        xlabel="pupil radius (px)", ylabel="IoU",
        title=("The shape filter's working range, with both ends visible: below "
               "min_area at radius 2 and above max_area at 45, it reports nothing."),
    )

    figures.metric_bars(
        [r["detector"] for r in pipeline],
        [r["pupil_psnr_db"] for r in pipeline],
        IMAGES / "pipeline_real.png",
        ylabel="pupil PSNR (dB), detected mask",
        title=("The whole pipeline on real photographs. The control at the right is "
               "what leaving the red-eye alone scores."),
    )

    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS, "58_red_eye_removal",
        {
            "photographs": list(re58.PHOTOGRAPHS),
            "portraits": {k: {"pupils": v["pupils"], "source": v["source"],
                              "note": v["note"]} for k, v in re58.PORTRAITS.items()},
            "detectors_on_portraits": real,
            "detectors_on_generated": generated,
            "generated_versus_real": both,
            "per_portrait": rows_per_photo,
            "clean_photographs": clean,
            "corrections_on_portraits": corrections_real,
            "corrections_on_generated": corrections_gen,
            "threshold_on_portraits": threshold_real,
            "pipeline_on_portraits": pipeline,
            "distractor_sweep": distractors,
            "pupil_size_sweep": pupil_sizes,
        },
    )

    real_table = markdown_table(
        real, [("Detector", "detector"), ("IoU", "iou"),
               ("Pupil recall", "pupil_recall"),
               ("FP area / pupil area", "false_positive_area_ratio"),
               ("ms", "median_ms")])
    both_table = markdown_table(
        both, [("Detector", "detector"), ("Generated IoU", "generated_iou"),
               ("Real IoU", "real_iou"), ("Generated FP", "generated_fp_area"),
               ("Real FP", "real_fp_area")])
    clean_table = markdown_table(
        clean, [("Photograph", "photograph"), ("Pupil-like blobs", "pupil_like_blobs")]
        + [(d, d) for d in re58.DETECTORS])
    corr_table = markdown_table(
        corrections_real, [("Photograph", "photograph")]
        + [(c, c) for c in re58.CORRECTIONS] + [("Did nothing", "did nothing")])
    pipe_table = markdown_table(
        pipeline, [("Detector", "detector"), ("Correction", "correction"),
                   ("Pupil PSNR (dB)", "pupil_psnr_db")])

    write_tables(
        RESULTS,
        [
            ("Detectors on six real portraits", real_table),
            ("Generated against real", both_table),
            ("Six photographs with no red-eye at all", clean_table),
            ("Corrections on real skin, true mask", corr_table),
            ("The whole pipeline on real photographs", pipe_table),
            ("Four photographs down the rows", gallery_table),
        ],
    )

    # ------------------------------------------------------------------ #
    print("\n--- HEADLINE NUMbERS ---".replace("NUMbERS", "NUMBERS"))
    face_gen = next(r for r in both if r["detector"] == "Face-constrained")
    print(f"the generated scene says the problem is solved and the photographs do not: "
          f"face-constrained IoU {face_gen['generated_iou']:.3f} generated vs "
          f"{face_gen['real_iou']:.3f} real")

    worse = [r for r in corrections_real if r["Zero red channel"] < r["did nothing"]]
    print(f"\nzeroing the red channel is worse than leaving the red-eye alone on "
          f"{len(worse)} of {len(corrections_real)} real portraits")
    for r in corrections_real:
        flag = "WORSE" if r["Zero red channel"] < r["did nothing"] else "better"
        print(f"  {r['photograph']:24s} zero-red {r['Zero red channel']:6.2f} dB vs "
              f"do-nothing {r['did nothing']:6.2f} dB  {flag}")

    total_clean = {d: sum(r[d] for r in clean) for d in re58.DETECTORS}
    print(f"\non six photographs containing no eyes, the naive detector marks "
          f"{total_clean['Colour only (control)']} pixels, the shape filter "
          f"{total_clean['Colour + shape']}, and the face constraint "
          f"{total_clean['Face-constrained']}")
    print(f"  the eye cascade finds eyes in a pile of sweets: "
          f"{next(r for r in clean if r['photograph'] == 'scattered_sweets')['Eye-constrained']}"
          " pixels there")

    best_pipe = max((r for r in pipeline if r["detector"] != "Did nothing (control)"),
                    key=lambda r: r["pupil_psnr_db"])
    nothing = next(r for r in pipeline if r["detector"] == "Did nothing (control)")
    naive = next(r for r in pipeline if r["detector"] == "Colour only (control)")
    face = next(r for r in pipeline if r["detector"] == "Face-constrained")
    print(f"\nend to end on real photographs: {best_pipe['detector']} reaches "
          f"{best_pipe['pupil_psnr_db']:.2f} dB against {nothing['pupil_psnr_db']:.2f} dB "
          "for doing nothing")
    fp_naive = next(r for r in real if r["detector"] == "Colour only (control)")
    fp_face = next(r for r in real if r["detector"] == "Face-constrained")
    print(f"  but the naive detector scores {naive['pupil_psnr_db']:.3f} dB against the "
          f"face-constrained {face['pupil_psnr_db']:.3f} — "
          f"{abs(face['pupil_psnr_db'] - naive['pupil_psnr_db']):.3f} dB apart, while "
          f"their false-positive areas differ by "
          f"{fp_naive['false_positive_area_ratio'] / max(fp_face['false_positive_area_ratio'], 1e-9):.0f}x")
    print("  pupil PSNR is computed on pupil pixels, so it cannot see what a detector "
          "does to the rest of the frame. That is what the clean-photograph control "
          "is for.")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
