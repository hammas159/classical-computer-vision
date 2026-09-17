"""Run the keypoint-detector comparison and write results + figures.

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
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import keypoints as kp  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"

#: The rotation the headline comparison runs at. 30 degrees is off-axis, so it
#: cannot be done by transposing the pixel grid — which is exactly what makes
#: 90 and 180 degrees uninformative about rotation invariance.
GALLERY_DEGREES = 30.0


def edge_density(img: np.ndarray) -> float:
    """Percentage of pixels Canny calls an edge — the axis this pool was built on."""
    import cv2

    g = to_gray(img)
    t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float((cv2.Canny(g, 0.5 * t, t) > 0).mean() * 100)


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    print(f"Scoring {len(kp.DETECTORS)} detectors at {GALLERY_DEGREES} degrees "
          f"on {len(kp.IMAGES)} photographs ...")
    rows = kp.evaluate_detectors("rotation", GALLERY_DEGREES, images=kp.IMAGES,
                                 runs=args.runs)

    print("Sweeping rotation ...")
    rot_rows = kp.sweep_rotation(images=kp.IMAGES)
    print("Sweeping scale ...")
    scale_rows = kp.sweep_scale(images=kp.IMAGES)
    print("Sweeping noise ...")
    noise_rows = kp.sweep_noise(images=kp.IMAGES)
    print("Controlling for keypoint density ...")
    budget_rows = kp.sweep_keypoint_budget(images=kp.IMAGES)

    names = list(kp.DETECTORS)

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    # Rows cross the axis the pool was built on -- edge density -- because every
    # detector here needs distinctive local structure and the failure mode worth
    # showing is what happens when there is none. A subject-uniqueness rule is
    # applied on top so the four rows are four different pictures.
    EDGE_BANDS = [("bare", 0, 12), ("light", 12, 18), ("busy", 18, 26), ("dense", 26, 100)]
    candidates: dict[str, list[dict]] = {}
    for name in kp.IMAGES:
        img = kp.load_scene(name)
        gray = to_gray(img)
        density = edge_density(img)
        band = next(b for b, a, z in EDGE_BANDS if a <= density < z)

        H = kp.homography_rotation(gray.shape, GALLERY_DEGREES)
        warped = kp.apply_homography(img, H)
        wgray = to_gray(warped)

        panels, notes, scores = [img], ["original"], []
        from shared.metrics import repeatability

        for det in names:
            a, b = kp.DETECTORS[det](gray), kp.DETECTORS[det](wgray)
            rep = repeatability(a, b, H, kp.MATCH_THRESHOLD, shape=wgray.shape)
            scores.append(rep)
            panels.append(kp.draw_keypoints(img, a))
            notes.append(f"{rep:.3f}\n{len(a)} kp")

        print(f"scene candidate {name:22s} edges {density:5.1f}%  [{band:6s}]  "
              f"best {max(scores):.3f}, worst {min(scores):.3f}, "
              f"spread {max(scores) - min(scores):.3f}")
        candidates.setdefault(band, []).append({
            "label": f"{name.replace('_', ' ')}\nedges {density:.0f}%",
            "subject": name,
            "density": density,
            "images": panels,
            "notes": notes,
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
        ["original"] + names,
        [(r["label"], r["images"]) for r in chosen],
        IMAGES / "compare_detectors.png",
        cell_notes=[r["notes"] for r in chosen],
        suptitle=(
            f"Keypoints found, and how many survive a {GALLERY_DEGREES:g}-degree "
            "rotation. Cells are repeatability against the known homography, "
            "counting only keypoints the rotation kept in frame."
        ),
    )
    gallery_table = markdown_table(
        [dict([("Sr", i), ("Scene", r["label"].replace("\n", " · "))]
              + [(d, f"{s:.3f}") for d, s in zip(names, r["scores"])])
         for i, r in enumerate(chosen, start=1)],
        [("Sr", "Sr"), ("Scene", "Scene")] + [(d, d) for d in names],
    )
    print(f"\nfront-on comparison: {len(chosen)} scenes x {len(names)} detectors")
    print("\n--- keypoints ---\n" + gallery_table)

    # ------------------------------------------------------------------ #
    # the signature figure: three invariances, three different rankings
    # ------------------------------------------------------------------ #
    for rows_, key, fname, xlabel, title, logx in (
        (rot_rows, "rotation_deg", "rotation_sweep.png", "rotation (degrees)",
         "Rotation: 90 and 180 are free — only off-axis angles test anything", False),
        (scale_rows, "scale", "scale_sweep.png", "scale factor",
         "Scale: the detectors with explicit scale selection do worst", False),
        (noise_rows, "noise_sigma", "noise_sweep.png", "noise sigma",
         "Noise: SIFT's scale-space extrema are the least stable", False),
    ):
        ordered = sorted(rows_, key=lambda r: r[key])
        figures.lines(
            [r[key] for r in ordered],
            {d: [r[d] for r in ordered] for d in names},
            IMAGES / fname,
            xlabel=xlabel,
            ylabel="repeatability",
            title=title,
            logx=logx,
        )

    figures.lines(
        [r["budget"] for r in budget_rows],
        {"Harris repeatability at 30 degrees": [r["repeatability"] for r in budget_rows]},
        IMAGES / "density_control.png",
        xlabel="Harris keypoints kept (randomly subsampled)",
        ylabel="repeatability",
        title="Harris does not win by finding more points — the line is flat",
        logx=True,
    )

    figures.comparison_matrix(
        rows,
        [("Repeatability", "repeatability", True),
         ("Coverage %", "coverage_pct", True),
         ("Keypoints", "keypoints", True),
         ("Time (ms)", "median_ms", False)],
        IMAGES / "method_matrix.png",
        title=f"Detectors x metrics at {GALLERY_DEGREES:g} degrees of rotation",
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS,
        "19_keypoint_detectors",
        {
            "images": list(kp.IMAGES),
            "gallery_degrees": GALLERY_DEGREES,
            "match_threshold_px": kp.MATCH_THRESHOLD,
            "detectors": rows,
            "rotation_sweep": rot_rows,
            "scale_sweep": scale_rows,
            "noise_sweep": noise_rows,
            "density_control": budget_rows,
        },
    )

    detector_table = markdown_table(
        rows,
        [("Detector", "detector"), ("Repeatability", "repeatability"),
         ("Coverage %", "coverage_pct"), ("Keypoints", "keypoints"),
         ("Time (ms)", "median_ms"), ("Slowdown", "slowdown_vs_fastest")],
    )
    rot_table = markdown_table(
        sorted(rot_rows, key=lambda r: r["rotation_deg"]),
        [("Rotation", "rotation_deg")] + [(d, d) for d in names],
    )
    scale_table = markdown_table(
        sorted(scale_rows, key=lambda r: r["scale"]),
        [("Scale", "scale")] + [(d, d) for d in names],
    )
    noise_table = markdown_table(
        sorted(noise_rows, key=lambda r: r["noise_sigma"]),
        [("Noise sigma", "noise_sigma")] + [(d, d) for d in names],
    )
    write_tables(
        RESULTS,
        [
            (f"Every detector at {GALLERY_DEGREES:g} degrees", detector_table),
            ("Rotation swept", rot_table),
            ("Scale swept", scale_table),
            ("Noise swept", noise_table),
            ("Four scenes down the rows", gallery_table),
        ],
    )
    print("\n" + detector_table + "\n\n" + scale_table + "\n\n" + noise_table)

    fastest = min(rows, key=lambda r: r["median_ms"])
    slowest = max(rows, key=lambda r: r["median_ms"])
    best = max(rows, key=lambda r: r["repeatability"])
    sift = next(r for r in rows if r["detector"] == "SIFT")
    orb = next(r for r in rows if r["detector"] == "ORB")

    print("\n--- HEADLINE NUMBERS ---")
    print(f"most repeatable  : {best['detector']} @ {best['repeatability']:.4f}")
    print(f"fastest          : {fastest['detector']} @ {fastest['median_ms']:.2f} ms")
    print(f"slowest          : {slowest['detector']} @ {slowest['median_ms']:.2f} ms "
          f"({slowest['median_ms'] / fastest['median_ms']:.0f}x the fastest)")
    print(f"ORB vs SIFT      : {sift['median_ms'] / orb['median_ms']:.1f}x faster, "
          f"and {orb['repeatability'] - sift['repeatability']:+.4f} repeatability")
    print("coverage         : "
          + ", ".join(f"{r['detector']} {r['coverage_pct']:.0f}%" for r in rows))
    print("density control  : "
          + ", ".join(f"{r['budget']}:{r['repeatability']:.3f}" for r in budget_rows))
    worst_scale = min(scale_rows, key=lambda r: r["scale"])
    print(f"at scale {worst_scale['scale']}     : "
          + ", ".join(f"{d} {worst_scale[d]:.3f}" for d in names))
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
