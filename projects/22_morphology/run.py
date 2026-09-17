"""Run the morphology comparison and write results + figures.

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
from shared.io import ensure_rgb, to_gray  # noqa: E402
from shared.metrics import iou  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import morphology as mo  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"

#: Structuring-element size the headline comparison runs at. 3 is the smallest
#: useful kernel and, as the results show, the only one where opening and
#: closing beat leaving the image alone.
GALLERY_SIZE = 3


def edge_density(img: np.ndarray) -> float:
    g = to_gray(img)
    t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float((cv2.Canny(g, 0.5 * t, t) > 0).mean() * 100)


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    print("Checking the algebraic identities ...")
    identities = mo.verify_algebraic_identities()
    print("Comparing structuring elements on each structure kind ...")
    elements = mo.compare_elements()
    print("Sweeping the element size ...")
    size_rows = mo.sweep_size()
    print("Denoising the synthetic scene ...")
    denoise_rows = mo.evaluate_denoising()
    print(f"Cleaning {len(mo.IMAGES)} binarised photographs ...")
    photo_rows = mo.evaluate_photo_cleaning()
    print("Skeletonising ...")
    skeletons = mo.evaluate_skeletons(runs=args.runs)
    print("Hit-or-miss corner detection ...")
    hitmiss = mo.evaluate_hit_or_miss()

    ops = list(mo.OPERATIONS)
    columns = ["Do nothing (control)", "Median filter"] + ops

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    # Rows cross the axis the pool was built on -- edge density -- because
    # morphology acts on the size of structures, and a silhouette and a hedge
    # are opposite problems for it. A subject-uniqueness rule is applied on top
    # so the four rows are four different pictures.
    EDGE_BANDS = [("silhouette", 0, 12), ("open", 12, 18),
                  ("busy", 18, 26), ("thicket", 26, 100)]
    se = mo.element("ellipse", GALLERY_SIZE)
    candidates: dict[str, list[dict]] = {}
    for name in mo.IMAGES:
        density = edge_density(mo.load_scene(name))
        band = next(b for b, a, z in EDGE_BANDS if a <= density < z)
        truth, noisy = mo.photo_binary(name)

        outs = [noisy, cv2.medianBlur(noisy, GALLERY_SIZE)] + [
            mo.OPERATIONS[o](noisy, se) for o in ops]
        scores = [iou(o > 0, truth > 0) for o in outs]
        print(f"scene candidate {name:24s} edges {density:5.1f}%  [{band:10s}]  "
              f"noisy {scores[0]:.3f} -> best {max(scores):.3f}")
        candidates.setdefault(band, []).append({
            "label": f"{name.replace('_', ' ')}\nedges {density:.0f}%",
            "subject": name,
            "density": density,
            "images": [ensure_rgb(truth)] + [ensure_rgb(o) for o in outs],
            "notes": ["TRUTH (Otsu)"] + [f"{s:.3f}" for s in scores],
            "scores": scores,
            "score": float(np.std(scores)),
        })

    chosen, used = [], set()
    for band, _, _ in EDGE_BANDS:
        pool = sorted(candidates.get(band, []), key=lambda r: -r["score"])
        pick = next((r for r in pool if r["subject"] not in used), None)
        if pick:
            used.add(pick["subject"])
            chosen.append(pick)
    chosen = chosen[:4]
    assert len({r["subject"] for r in chosen}) == len(chosen), "a subject repeats"

    figures.gallery(
        ["TRUTH (Otsu)"] + columns,
        [(r["label"], r["images"]) for r in chosen],
        IMAGES / "compare_morphology.png",
        cell_notes=[r["notes"] for r in chosen],
        suptitle=(
            f"Otsu's binarisation IS the truth; {mo.PHOTO_NOISE:.0%} salt-and-pepper "
            f"is added and each operation tries to clean it off with a "
            f"{GALLERY_SIZE}x{GALLERY_SIZE} ellipse. Cells are IoU against that truth."
        ),
    )
    gallery_table = markdown_table(
        [dict([("Sr", i), ("Scene", r["label"].replace("\n", " · "))]
              + [(c, f"{s:.3f}") for c, s in zip(columns, r["scores"])])
         for i, r in enumerate(chosen, start=1)],
        [("Sr", "Sr"), ("Scene", "Scene")] + [(c, c) for c in columns],
    )
    print(f"\nfront-on comparison: {len(chosen)} scenes x {len(columns)} operations")
    print("\n--- morphology ---\n" + gallery_table)

    # ------------------------------------------------------------------ #
    # the signature figure: the structuring element decides what survives
    # ------------------------------------------------------------------ #
    # Morphology's whole behaviour is set by the shape of the element, and the
    # effect is different for each KIND of structure. One grid says that; a
    # single IoU per element cannot.
    figures.comparison_matrix(
        [{"method": r["element"], **{k: r[k] for k in
          ("axis_aligned", "diagonal", "curved", "thin")}} for r in elements],
        [("Axis-aligned", "axis_aligned", True), ("Diagonal", "diagonal", True),
         ("Curved", "curved", True), ("Thin", "thin", True)],
        IMAGES / "element_matrix.png",
        title=("Fraction of each structure kind surviving a 9 px erosion — "
               "the element's shape IS the method"),
    )

    figures.lines(
        sorted({r["size"] for r in photo_rows}),
        {op: [next(r["iou"] for r in photo_rows
                   if r["size"] == s and r["operation"] == op)
              for s in sorted({r["size"] for r in photo_rows})]
         for op in ("Do nothing (control)", "Median filter", "Open", "Close",
                    "Erode", "Dilate")},
        IMAGES / "photo_cleaning.png",
        xlabel="structuring element size (px)",
        ylabel="IoU against the true binarisation",
        title="Cleaning real binarised photographs — a median filter wins",
        dashed={"Do nothing (control)"},
    )

    figures.lines(
        [r["size"] for r in size_rows],
        {"fraction of foreground kept": [r["fraction_kept"] for r in size_rows]},
        IMAGES / "size_sweep.png",
        xlabel="structuring element size (px)",
        ylabel="fraction kept",
        title="Opening removes everything smaller than the element, by construction",
    )

    figures.lines(
        sorted({r["size"] for r in denoise_rows}),
        {el: [next(r["iou"] for r in denoise_rows
                   if r["size"] == s and r["element"] == el)
              for s in sorted({r["size"] for r in denoise_rows})]
         for el in sorted({r["element"] for r in denoise_rows})},
        IMAGES / "denoise_sweep.png",
        xlabel="structuring element size (px)",
        ylabel="IoU after opening then closing",
        title="On the synthetic scene: bigger elements remove more noise and more signal",
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS,
        "22_morphology",
        {
            "images": list(mo.IMAGES),
            "gallery_element_size": GALLERY_SIZE,
            "photo_noise_density": mo.PHOTO_NOISE,
            "identities": identities,
            "elements": elements,
            "size_sweep": size_rows,
            "denoise_sweep": denoise_rows,
            "photo_cleaning": photo_rows,
            "skeletons": skeletons,
            "hit_or_miss": hitmiss,
        },
    )

    identity_table = markdown_table(
        identities,
        [("Element", "element"), ("Opening idempotent", "opening_idempotent"),
         ("Opening anti-extensive", "opening_anti_extensive"),
         ("Closing extensive", "closing_extensive"),
         ("Erode/dilate dual", "erode_dilate_dual")],
    )
    element_table = markdown_table(
        elements,
        [("Element", "element"), ("Axis-aligned", "axis_aligned"),
         ("Diagonal", "diagonal"), ("Curved", "curved"), ("Thin", "thin")],
    )
    photo_table = markdown_table(
        sorted(photo_rows, key=lambda r: (r["size"], -r["iou"])),
        [("Size", "size"), ("Operation", "operation"), ("IoU", "iou")],
    )
    skeleton_table = markdown_table(
        skeletons,
        [("Method", "method"), ("Components", "components"),
         ("Thinness", "thinness"), ("Time (ms)", "median_ms")],
    )
    write_tables(
        RESULTS,
        [
            ("The algebraic identities morphology must obey", identity_table),
            ("What survives a 9 px erosion, by element and structure kind", element_table),
            ("Cleaning binarised photographs", photo_table),
            ("Skeletonisation", skeleton_table),
            ("Four scenes down the rows", gallery_table),
        ],
    )
    print("\n" + identity_table + "\n\n" + element_table + "\n\n" + skeleton_table)

    at3 = {r["operation"]: r["iou"] for r in photo_rows if r["size"] == GALLERY_SIZE}
    control = at3["Do nothing (control)"]
    best_morph = max((v, k) for k, v in at3.items()
                     if k not in ("Do nothing (control)", "Median filter"))
    morph_skel = next(r for r in skeletons if r["method"].startswith("Morph"))
    thin_skel = next(r for r in skeletons if r["method"].startswith("Zhang"))

    print("\n--- HEADLINE NUMBERS ---")
    print(f"all algebraic identities hold: "
          f"{all(all(v for k, v in r.items() if k != 'element') for r in identities)}")
    print(f"photo cleaning, size {GALLERY_SIZE}: control {control:.4f}, "
          f"median {at3['Median filter']:.4f}, best morphology "
          f"{best_morph[1]} {best_morph[0]:.4f}")
    print(f"  -> median beats the best morphological operation by "
          f"{at3['Median filter'] - best_morph[0]:+.4f} IoU")
    beaten = [s for s in sorted({r['size'] for r in photo_rows})
              if max(r["iou"] for r in photo_rows
                     if r["size"] == s and r["operation"] not in
                     ("Do nothing (control)", "Median filter")) < control]
    print(f"  -> element sizes where NO morphology beats doing nothing: {beaten}")
    diag = {r["element"]: r["diagonal"] for r in elements}
    print(f"diagonal structure surviving a 9 px erosion: "
          + ", ".join(f"{k} {v:.4f}" for k, v in diag.items()))
    print(f"skeletons: morphological {morph_skel['components']:.0f} components in "
          f"{morph_skel['median_ms']:.0f} ms, Zhang-Suen {thin_skel['components']:.0f} "
          f"in {thin_skel['median_ms']:.0f} ms "
          f"({thin_skel['median_ms'] / morph_skel['median_ms']:.0f}x slower)")
    print(f"hit-or-miss: {sum(r['detections'] == r['expected'] for r in hitmiss)}"
          f"/{len(hitmiss)} corners found exactly")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
