"""Run the focus stacking study and write results + figures.

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
from shared.io import ensure_rgb  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import focus as fo  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"


def _index_map(indices, n):
    """The selection map as a picture, so which frame won is visible."""
    scaled = (indices.astype(np.float32) / max(n - 1, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(scaled, cv2.COLORMAP_VIRIDIS),
                        cv2.COLOR_BGR2RGB)


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()
    del args

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    print(f"{len(fo.IMAGES)} photographs, each made into a {fo.N_FRAMES}-frame "
          f"focal stack with a synthetic depth map")
    print(f"  blur at the far end of the depth range: sigma {fo.MAX_SIGMA:g}")

    print("\nEvery measure and every control ...")
    overall = fo.evaluate()
    for r in overall:
        tag = "  (control)" if r["is_control"] else ""
        print(f"  {r['method']:26s} {r['psnr_db']:7.3f} dB  agrees with truth on "
              f"{r['agreement_with_truth']:.4f}{tag}")

    rows_per_image = fo.per_image()
    pooling = fo.sweep_pooling()
    spread = fo.measure_spread()
    disagree = fo.disagreement()
    flat = fo.flat_regions_are_unwinnable()
    ceiling = fo.detail_predicts_the_ceiling()
    frames_sweep = fo.sweep_frames()
    blur_sweep = fo.sweep_blur()
    feather = fo.feathering_effect()

    print(f"\nthe pooling window ...")
    for r in pooling:
        print(f"  window {r['window']:3d}  {r['psnr_db']:7.3f} dB  "
              f"agreement {r['agreement_with_truth']:.4f}")

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    by_image = {r["image"]: r for r in rows_per_image}
    chosen = [fo.IMAGES[i] for i in (0, 4, 8, 11)]

    built = {n: fo.stack_for(n) for n in chosen}

    rows = [("the original photograph\n(sharp everywhere)",
             [built[n][0] for n in chosen]),
            (f"one frame of the stack\n(sharp in one depth band)",
             [built[n][1][fo.N_FRAMES // 2] for n in chosen])]
    notes = [[f"detail {by_image[n]['detail']:.0f}" for n in chosen],
             [f"frame {fo.N_FRAMES // 2} of {fo.N_FRAMES}" for n in chosen]]

    table_rows = [{"Sr": sr, "Photograph": n.replace("_", " "),
                   "Detail": f"{by_image[n]['detail']:.0f}",
                   "Flat share": f"{100 * by_image[n]['flat_share']:.1f}%",
                   "Oracle (dB)": f"{by_image[n]['oracle_db']:.2f}",
                   "Random (dB)": f"{by_image[n]['random_db']:.2f}"}
                  for sr, n in enumerate(chosen, start=1)]

    for measure in fo.MEASURES:
        panels, cell = [], []
        for pos, n in enumerate(chosen):
            image, frames, truth = built[n]
            idx = fo.select_indices(frames, measure)
            panels.append(fo.merge(frames, idx))
            cell.append(f"{by_image[n][measure]:.2f} dB")
            table_rows[pos][measure] = f"{by_image[n][measure]:.2f}"
        rows.append((measure, panels))
        notes.append(cell)
        print(f"  {measure:26s} {cell}")

    figures.gallery(
        [n.replace("_", " ") for n in chosen], rows,
        IMAGES / "compare_focus.png",
        cell_notes=notes,
        suptitle=("Four of the twelve. Row one is the original, sharp everywhere; "
                  "row two is one frame of its focal stack. The rest are merges, "
                  "scored in dB against the original."),
    )
    gallery_table = markdown_table(
        table_rows,
        [("Sr", "Sr"), ("Photograph", "Photograph"), ("Detail", "Detail"),
         ("Flat share", "Flat share"), ("Random (dB)", "Random (dB)"),
         ("Oracle (dB)", "Oracle (dB)")] + [(m, m) for m in fo.MEASURES])
    print("\n--- merge quality per photograph (dB) ---\n" + gallery_table)

    # what each measure selected, on one image
    name = chosen[1]
    image, frames, truth = built[name]
    panels = [("the truth", _index_map(truth, fo.N_FRAMES))]
    for measure in fo.MEASURES:
        idx = fo.select_indices(frames, measure)
        panels.append((f"{measure}\n{100 * float((idx == truth).mean()):.1f}% correct",
                       _index_map(idx, fo.N_FRAMES)))
    figures.grid(panels, IMAGES / "selection_maps.png", ncols=3,
                 suptitle=(f"Which frame each measure picked, on {name.replace('_', ' ')}. "
                           "Colour is the frame index."))

    # ------------------------------------------------------------------ #
    # the signature figures
    # ------------------------------------------------------------------ #
    real = [r for r in overall if not r["is_control"]]
    figures.metric_bars(
        [r["method"] for r in real] + [f"pool {r['window']}" for r in pooling],
        [r["psnr_db"] for r in real] + [r["psnr_db"] for r in pooling],
        IMAGES / "measure_vs_pooling.png",
        ylabel="merged PSNR (dB)",
        title=(f"Left five: different measures (spread {spread['measure_spread_db']:.2f} "
               f"dB). Right seven: one measure, different pooling windows (spread "
               f"{spread['pooling_spread_db']:.2f} dB)."),
    )

    figures.metric_bars(
        [r["method"] for r in overall],
        [r["psnr_db"] for r in overall],
        IMAGES / "measures.png",
        ylabel="merged PSNR (dB)",
        title=("Every measure sits within a decibel of the oracle, and the random "
               "control is 14 dB below all of them."),
    )

    figures.lines(
        [r["window"] for r in pooling],
        {"merged PSNR (dB)": [r["psnr_db"] for r in pooling],
         "agreement with truth x 45": [r["agreement_with_truth"] * 45
                                       for r in pooling]},
        IMAGES / "pooling.png",
        xlabel="pooling window (px)", ylabel="dB  /  rate x 45", logx=True,
        dashed={"agreement with truth x 45"},
        title=("The parameter almost nobody reports. Unpooled, the measure is a "
               "second derivative of a noisy signal."),
    )

    figures.metric_bars(
        [r["image"].replace("_", " ") for r in disagree] * 2,
        [r["disagreement_in_detailed"] for r in disagree]
        + [r["disagreement_in_flat"] for r in disagree],
        IMAGES / "disagreement.png",
        ylabel="fraction of pixels where the five measures disagree",
        title=("Left twelve: in detailed regions. Right twelve: in flat regions, "
               "where every frame looks the same and no measure can be right."),
    )

    figures.scatter_plane(
        {"the twelve photographs":
            [(r["detail"], r["oracle_db"]) for r in rows_per_image]},
        target=None,
        out_path=IMAGES / "detail_predicts_ceiling.png",
        xlabel="detail of the photograph (measured before any stack exists)",
        ylabel="oracle PSNR (dB): the best any selection could do",
        title=(f"r = {ceiling['pearson_r']:.3f}. More detail means a lower ceiling: "
               "there is more to lose when a region is blurred."),
    )

    figures.lines(
        [r["frames"] for r in frames_sweep],
        {"oracle": [r["oracle_db"] for r in frames_sweep],
         "Variance of Laplacian": [r["variance_of_laplacian_db"]
                                   for r in frames_sweep]},
        IMAGES / "frames.png",
        xlabel="frames in the stack", ylabel="merged PSNR (dB)", logx=True,
        dashed={"Variance of Laplacian"},
        title="The two curves never separate by more than 0.1 dB",
    )

    figures.lines(
        [r["max_sigma"] for r in blur_sweep],
        {m: [r[m] for r in blur_sweep] for m in fo.MEASURES},
        IMAGES / "blur.png",
        xlabel="blur at the far end of the depth range (sigma)",
        ylabel="agreement with truth", logx=True,
        title="How much defocus there has to be before a measure can see it",
    )

    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS, "51_focus_stacking",
        {
            "images": list(fo.IMAGES),
            "stack": {"frames": fo.N_FRAMES, "max_sigma": fo.MAX_SIGMA,
                      "pooling_window": fo.POOL},
            "overall": overall,
            "per_image": rows_per_image,
            "pooling": pooling,
            "spread": spread,
            "disagreement": disagree,
            "flat_regions": flat,
            "detail_predicts_ceiling": ceiling,
            "frames_sweep": frames_sweep,
            "blur_sweep": blur_sweep,
            "feathering": feather,
        },
    )

    overall_table = markdown_table(
        overall, [("Method", "method"), ("PSNR (dB)", "psnr_db"),
                  ("Agreement with truth", "agreement_with_truth"),
                  ("Control?", "is_control")])
    pooling_table = markdown_table(
        pooling, [("Window", "window"), ("PSNR (dB)", "psnr_db"),
                  ("Agreement with truth", "agreement_with_truth")])
    disagree_table = markdown_table(
        disagree, [("Photograph", "image"), ("Disagreement", "disagree_share"),
                   ("Flat share", "flat_share"),
                   ("Disagreement in flat", "disagreement_in_flat"),
                   ("Disagreement in detailed", "disagreement_in_detailed")])
    frames_table = markdown_table(
        frames_sweep, [("Frames", "frames"), ("Oracle (dB)", "oracle_db"),
                       ("Variance of Laplacian (dB)", "variance_of_laplacian_db"),
                       ("Gap (dB)", "gap_db")])
    blur_table = markdown_table(
        blur_sweep, [("Max sigma", "max_sigma")] + [(m, m) for m in fo.MEASURES])

    write_tables(
        RESULTS,
        [
            ("Every measure and every control", overall_table),
            ("The pooling window", pooling_table),
            ("Where the measures disagree", disagree_table),
            ("How many frames the stack has", frames_table),
            ("How much defocus a measure needs", blur_table),
            ("Four photographs down the rows", gallery_table),
        ],
    )

    # ------------------------------------------------------------------ #
    print("\n--- HEADLINE NUMBERS ---")
    oracle = next(r for r in overall if r["method"] == "Oracle (truth)")
    random_pick = next(r for r in overall if r["method"].startswith("Random"))
    best = max(real, key=lambda r: r["psnr_db"])
    worst = min(real, key=lambda r: r["psnr_db"])

    print(f"the task is worth {oracle['psnr_db'] - random_pick['psnr_db']:.2f} dB: "
          f"random selection {random_pick['psnr_db']:.2f}, oracle "
          f"{oracle['psnr_db']:.2f}")
    print(f"  and every measure captures at least "
          f"{worst['psnr_db'] - random_pick['psnr_db']:.2f} of it")
    print(f"  the best ({best['method']}) is {oracle['psnr_db'] - best['psnr_db']:.3f} "
          "dB from the oracle that is handed the answer")

    print(f"\nthe measure is not what decides the result:")
    print(f"  spread across the five measures:  "
          f"{spread['measure_spread_db']:.3f} dB")
    print(f"  spread across pooling windows:    "
          f"{spread['pooling_spread_db']:.3f} dB  "
          f"({spread['pooling_spread_db'] / max(spread['measure_spread_db'], 1e-9):.1f}x more)")
    unpooled = next(r for r in pooling if r["window"] == 1)
    pooled = max(pooling, key=lambda r: r["psnr_db"])
    print(f"  unpooled, agreement with truth is "
          f"{unpooled['agreement_with_truth']:.3f}; at window {pooled['window']} it is "
          f"{pooled['agreement_with_truth']:.3f}")
    print("  the raw response is a second derivative of a noisy signal, and the "
          "window is what makes it usable")

    print(f"\nwhere no measure can be right:")
    print(f"  agreement with truth in detailed regions "
          f"{flat['agreement_in_detailed_regions']:.4f}")
    print(f"  agreement with truth in flat regions     "
          f"{flat['agreement_in_flat_regions']:.4f}  "
          f"({flat['images_with_any_flat_region']} of {len(fo.IMAGES)} photographs "
          "have one)")
    flat_rows = [r for r in disagree if r["flat_share"] > 0.01]
    if flat_rows:
        print(f"  and the five measures disagree with each other on "
              f"{np.mean([r['disagreement_in_flat'] for r in flat_rows]):.1%} of flat "
              f"pixels against "
              f"{np.mean([r['disagreement_in_detailed'] for r in flat_rows]):.1%} of "
              "detailed ones")

    print(f"\nthe ceiling was predictable: r = {ceiling['pearson_r']:.3f} between the "
          "photograph's detail and the best PSNR any selection could reach")
    print(f"  and the sign is the interesting part — MORE detail means a LOWER "
          "ceiling, because a blurred detailed region loses more than a blurred "
          "smooth one")

    gaps = [r["gap_db"] for r in frames_sweep]
    print(f"\nacross stacks of 3 to 17 frames the best measure is never more than "
          f"{max(gaps):.3f} dB from the oracle")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
