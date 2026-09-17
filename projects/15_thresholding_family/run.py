"""Run the thresholding comparison and write results + figures.

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

from shared import figures, io  # noqa: E402
from shared.io import to_gray  # noqa: E402
from shared.metrics import iou  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import thresholding as th  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"


def best_polarity(mask: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, float]:
    """Score a binarisation both ways up and keep the better.

    The scene is dark-on-light, but a thresholder is free to call either class
    "foreground" — `cv2.THRESH_BINARY` and `THRESH_BINARY_INV` are the same
    method. Fixing a polarity would score half the table on a convention rather
    than on whether it separated the two populations.
    """
    a, b = iou(mask > 0, truth > 0), iou(mask == 0, truth > 0)
    return (mask if a >= b else (mask == 0).astype(np.uint8) * 255), max(a, b)


def as_image(mask: np.ndarray) -> np.ndarray:
    return np.repeat((mask > 0).astype(np.uint8)[..., None] * 255, 3, axis=2)


def overlay(mask: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """White = correct foreground, red = missed, blue = invented."""
    m, t = mask > 0, truth > 0
    out = np.zeros(m.shape + (3,), np.uint8)
    out[m & t] = (255, 255, 255)
    out[t & ~m] = (255, 60, 60)
    out[m & ~t] = (60, 140, 255)
    return out


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    method_names = list(th.METHODS)

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    # Two axes, deliberately crossed: how EVEN the light is, and what SHAPE the
    # foreground has. Sweeping illumination alone is the usual experiment and it
    # reaches the wrong conclusion, because the adaptive family's failure on
    # filled shapes is about geometry and gets blamed on lighting.
    GALLERY_MIN_IOU = 0.50
    scene_pool = [
        ("solid shapes\neven light", dict(kind="solid", illum_min=1.0), "solid, even"),
        ("solid shapes\nslight gradient", dict(kind="solid", illum_min=0.8), "solid, even"),
        ("solid shapes\nstrong gradient", dict(kind="solid", illum_min=0.3), "solid, uneven"),
        ("solid shapes\nsevere gradient", dict(kind="solid", illum_min=0.15), "solid, uneven"),
        ("thin strokes\neven light", dict(kind="thin", illum_min=1.0), "thin, even"),
        ("thin strokes\nslight gradient", dict(kind="thin", illum_min=0.8), "thin, even"),
        ("thin strokes\nstrong gradient", dict(kind="thin", illum_min=0.3), "thin, uneven"),
        ("thin strokes\nsevere gradient", dict(kind="thin", illum_min=0.15), "thin, uneven"),
        ("solid shapes\neven light, noisy", dict(kind="solid", illum_min=1.0, noise_sigma=25.0), "noisy"),
        ("thin strokes\neven light, noisy", dict(kind="thin", illum_min=1.0, noise_sigma=25.0), "noisy"),
        ("solid shapes\nfew and small", dict(kind="solid", fg_fraction=0.03), "imbalanced"),
        ("thin strokes\nsparse", dict(kind="thin", fg_fraction=0.03), "imbalanced"),
    ]
    survivors: dict[str, dict] = {}
    for label, kwargs, family in scene_pool:
        kwargs = {"fg_fraction": 0.12, **kwargs}
        img, truth = th.synthetic_scene(seed=0, **kwargs)
        gray = to_gray(img)
        masks, scores = [], []
        for name in method_names:
            m, s = best_polarity(th.METHODS[name](gray), truth)
            masks.append(m)
            scores.append(s)
        oracle_mask, oracle_t = th.thresh_best_global(gray, truth)
        oracle_mask, oracle_score = best_polarity(oracle_mask, truth)
        best = max(scores)
        flat = label.replace("\n", " · ")
        if best < GALLERY_MIN_IOU:
            print(f"scene candidate {flat:<36} DROP — best IoU {best:.3f}  [{family}]")
            continue
        print(f"scene candidate {flat:<36} keep — best IoU {best:.3f}, "
              f"worst {min(scores):.3f}, oracle {oracle_score:.3f}  [{family}]")
        row = {
            "label": label,
            "images": [img, as_image(truth)] + [overlay(m, truth) for m in masks]
            + [overlay(oracle_mask, truth)],
            "notes": ["scene", "truth"] + [f"{s:.3f}" for s in scores] + [f"{oracle_score:.3f}"],
            "spread": best - min(scores),
            "scores": dict(zip(method_names, scores)),
        }
        if family not in survivors or row["spread"] > survivors[family]["spread"]:
            survivors[family] = row

    ORDER = ["solid, even", "solid, uneven", "thin, even", "thin, uneven"]
    chosen = [survivors[k] for k in ORDER if k in survivors][:4]
    figures.gallery(
        ["scene", "truth"] + method_names + [th.ORACLE_NAME],
        [(r["label"], r["images"]) for r in chosen],
        IMAGES / "compare_thresholds.png",
        cell_notes=[r["notes"] for r in chosen],
        suptitle=(
            "Two foreground shapes crossed with two lighting conditions. "
            "White = correct, red = missed, blue = invented. Cells are IoU."
        ),
    )
    gallery_table = markdown_table(
        [
            dict([("Sr", i), ("Scene", r["label"].replace("\n", " · "))]
                 + list(zip(method_names + [th.ORACLE_NAME], r["notes"][2:])))
            for i, r in enumerate(chosen, start=1)
        ],
        [("Sr", "Sr"), ("Scene", "Scene")]
        + [(m, m) for m in method_names + [th.ORACLE_NAME]],
    )
    print(f"\nfront-on comparison: {len(chosen)} scenes x {len(method_names)} methods")
    print("\n--- thresholding ---\n" + gallery_table)

    # ------------------------------------------------------------------ #
    # the crossing point, measured on both scene kinds
    # ------------------------------------------------------------------ #
    print("\nSweeping illumination on both foreground shapes ...")
    cross_rows = []
    for kind in ("solid", "thin"):
        for illum in th.ILLUM_LEVELS:
            img, truth = th.synthetic_scene(kind=kind, fg_fraction=0.12,
                                            illum_min=illum, seed=0)
            gray = to_gray(img)
            row = {"kind": kind, "illum_min": illum}
            for name in method_names:
                row[name] = round(best_polarity(th.METHODS[name](gray), truth)[1], 4)
            cross_rows.append(row)

    for kind in ("solid", "thin"):
        rows = [r for r in cross_rows if r["kind"] == kind]
        figures.lines(
            [r["illum_min"] for r in rows],
            {m: [r[m] for r in rows] for m in method_names},
            IMAGES / f"illumination_{kind}.png",
            xlabel="illumination at the darkest corner (1.0 = even)",
            ylabel="IoU",
            title=(
                f"{kind.capitalize()} foreground: "
                + ("local thresholding never wins, at any light level"
                   if kind == "solid" else
                   "local thresholding holds 1.000 where global collapses")
            ),
        )

    print("Scoring every method on the default scene ...")
    method_rows, scene_stats = th.evaluate_methods(runs=args.runs)

    print("Sweeping the foreground fraction ...")
    fraction_rows = th.sweep_foreground_fraction()

    print("Sweeping noise ...")
    noise_rows = th.sweep_noise()

    figures.comparison_matrix(
        method_rows,
        [("IoU", "iou", True), ("Dice", "dice", True), ("Time (ms)", "median_ms", False)],
        IMAGES / "method_matrix.png",
        title="Methods x metrics on the default scene",
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS,
        "15_thresholding_family",
        {
            "contrast_ratio": th.CONTRAST_RATIO,
            "thin_stroke_px": th.THIN_STROKE_PX,
            "methods": method_rows,
            "scene": scene_stats,
            "shape_x_illumination": cross_rows,
            "foreground_fraction": fraction_rows,
            "noise": noise_rows,
        },
    )

    cross_table = markdown_table(
        cross_rows,
        [("Foreground", "kind"), ("Illumination", "illum_min")]
        + [(m, m) for m in method_names],
    )
    method_table = markdown_table(
        method_rows,
        [("Method", "method"), ("IoU", "iou"), ("Dice", "dice"), ("Time (ms)", "median_ms")],
    )
    write_tables(
        RESULTS,
        [
            ("Foreground shape crossed with illumination", cross_table),
            ("Methods on the default scene", method_table),
            ("Four scenes down the rows", gallery_table),
        ],
    )
    print("\n" + cross_table)

    def pick(kind, illum, method):
        return next(r[method] for r in cross_rows
                    if r["kind"] == kind and r["illum_min"] == illum)

    print("\n--- HEADLINE NUMBERS ---")
    for kind in ("solid", "thin"):
        for illum in (1.0, 0.3):
            rows = [r for r in cross_rows if r["kind"] == kind and r["illum_min"] == illum]
            row = rows[0]
            best = max(method_names, key=lambda m: row[m])
            print(f"{kind:6s} illum {illum:.1f}  best {best:22s} {row[best]:.3f}   "
                  f"Otsu {row['Otsu']:.3f}  Sauvola {row['Sauvola']:.3f}")
    print(f"\nglobal vs local under bad light:")
    print(f"  solid foreground: Otsu {pick('solid', 0.3, 'Otsu'):.3f} vs "
          f"Sauvola {pick('solid', 0.3, 'Sauvola'):.3f}  "
          f"(local {pick('solid', 0.3, 'Sauvola') - pick('solid', 0.3, 'Otsu'):+.3f})")
    print(f"  thin  foreground: Otsu {pick('thin', 0.3, 'Otsu'):.3f} vs "
          f"Sauvola {pick('thin', 0.3, 'Sauvola'):.3f}  "
          f"(local {pick('thin', 0.3, 'Sauvola') - pick('thin', 0.3, 'Otsu'):+.3f})")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
