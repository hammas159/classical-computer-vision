"""Run the shape descriptor comparison and write results + figures.

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

import shapes_desc as sd  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"

CHANCE = 1.0 / len(sd.REAL_IMAGES)

#: The figure shows retrieval at this rotation, because upright is where the
#: rankings are least interesting: the descriptor that wins upright is the one
#: that is not rotation invariant, and turning the shape is what shows it.
FIGURE_ROTATION = 30.0


def _outline(mask: np.ndarray) -> np.ndarray:
    """A silhouette drawn as a filled shape with its contour picked out."""
    rgb = ensure_rgb((mask > 0).astype(np.uint8) * 235)
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8),
                                   cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(rgb, contours, -1, (30, 90, 200), 2)
    return rgb


def _fit(mask: np.ndarray, box: int = 260) -> np.ndarray:
    ys, xs = np.nonzero(mask > 0)
    if not len(ys):
        return np.zeros((box, box), np.uint8)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    scale = (box - 20) / max(crop.shape)
    resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    out = np.zeros((box, box), np.uint8)
    y = (box - resized.shape[0]) // 2
    x = (box - resized.shape[1]) // 2
    out[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return out


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    descriptors = list(sd.DESCRIPTORS)

    print("Measuring each claimed invariance on generated shapes ...")
    generated = {
        "rotation": sd.invariance_table("rotation", sd.ROTATIONS),
        "scale": sd.invariance_table("scale", sd.SCALES),
        "translation": sd.invariance_table("translation", sd.TRANSLATIONS),
    }
    for transform, rows in generated.items():
        print(f"  {transform:12s} " + "  ".join(
            f"{r['descriptor'].split()[0][:7]}={r['mean_relative_change']:.4f}" for r in rows))

    print("\nSame transforms, on twelve silhouettes people traced ...")
    real = {
        "rotation": sd.real_invariance_table("rotation", sd.ROTATIONS),
        "scale": sd.real_invariance_table("scale", sd.SCALES),
        "translation": sd.real_invariance_table("translation", sd.TRANSLATIONS),
    }
    for transform, rows in real.items():
        print(f"  {transform:12s} " + "  ".join(
            f"{r['descriptor'].split()[0][:7]}={r['mean_relative_change']:.4f}" for r in rows))

    print("\nHow far apart two people are when they trace the same object ...")
    human = sd.annotator_variation()
    for r in human:
        print(f"  {r['descriptor']:28s} {r['mean_relative_change']:.4f} "
              f"(max {r['max_relative_change']:.3f}, {r['pairs']} pairs)")

    print("\nRecognising an object from another person's tracing ...")
    upright = sd.cross_annotator_classification()
    turned = sd.cross_annotator_classification(rotation=FIGURE_ROTATION)
    for a, b in zip(upright, turned):
        print(f"  {a['descriptor']:28s} upright {a['accuracy']:.4f}  "
              f"turned {FIGURE_ROTATION:g}deg {b['accuracy']:.4f}")

    discretisation = sd.discretisation_error()
    reflection = sd.reflection_test()
    real_reflection = sd.real_reflection_test()
    noise = sd.noise_robustness()
    shape_accuracy = sd.classification_accuracy(runs=args.runs)

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    # Rows are objects; the last columns are the silhouette each descriptor
    # *retrieved* when handed a second person's tracing of that object, turned 30
    # degrees. Green if it is the same object, red if not.
    gallery, gallery_labels, probes, probe_labels = [], [], [], []
    for label, image in enumerate(sd.REAL_IMAGES):
        traced = sd.silhouettes_by_annotator(image)
        gallery.append(sd.transform_mask(traced[0]))
        gallery_labels.append(label)
        for other in traced[1:]:
            probes.append(sd.transform_mask(other, rotation=FIGURE_ROTATION))
            probe_labels.append(label)
    gallery_labels = np.asarray(gallery_labels)
    probe_labels = np.asarray(probe_labels)

    retrieved = {}
    for name, fn in sd.DESCRIPTORS.items():
        g = np.stack([fn(m) for m in gallery])
        p = np.stack([fn(m) for m in probes])
        mean = g.mean(axis=0, keepdims=True)
        std = np.maximum(g.std(axis=0, keepdims=True), sd.EPS)
        d = np.linalg.norm(((p - mean) / std)[:, None, :]
                           - ((g - mean) / std)[None, :, :], axis=2)
        retrieved[name] = np.argmin(d, axis=1)

    # pick four probes the descriptors disagree about most, one object each
    disagreement = sorted(
        range(len(probes)),
        key=lambda i: -len({gallery_labels[retrieved[d][i]] for d in descriptors}),
    )
    chosen, seen = [], set()
    for i in disagreement:
        if probe_labels[i] in seen:
            continue
        seen.add(probe_labels[i])
        chosen.append(i)
        if len(chosen) == 4:
            break
    chosen.sort(key=lambda i: probe_labels[i])
    assert len({probe_labels[i] for i in chosen}) == 4, "an object repeats down the rows"

    condition_accuracy = {
        d: float((gallery_labels[retrieved[d]] == probe_labels).mean()) for d in descriptors
    }
    columns = ["the photograph", "person A traced", f"person B, turned {FIGURE_ROTATION:g}deg"]
    columns += [f"{d}\n{condition_accuracy[d]:.3f} over all {len(probes)}" for d in descriptors]

    rows, notes, table_rows = [], [], []
    for sr, i in enumerate(chosen, start=1):
        true_name = sd.REAL_IMAGES[probe_labels[i]]
        panels = [sd.load_scene(true_name),
                  _outline(_fit(gallery[probe_labels[i]])),
                  _outline(_fit(probes[i]))]
        cell = ["photograph", "the gallery shape", "the probe"]
        scores = {}
        for d in descriptors:
            j = retrieved[d][i]
            correct = gallery_labels[j] == probe_labels[i]
            got = sd.REAL_IMAGES[gallery_labels[j]]
            panels.append(cv2.copyMakeBorder(
                _outline(_fit(gallery[j], 240)), 10, 10, 10, 10, cv2.BORDER_CONSTANT,
                value=(60, 170, 60) if correct else (200, 60, 60)))
            cell.append(("OK  " if correct else "WRONG  ") + got.replace("_", " "))
            scores[d] = "correct" if correct else got.replace("_", " ")
        rows.append((true_name.replace("_", " "), panels))
        notes.append(cell)
        table_rows.append(dict([("Sr", sr), ("Object", true_name.replace("_", " "))], **scores))
        print(f"  row {sr}: {true_name:22s} " +
              "  ".join(f"{d.split()[0]}={scores[d]}" for d in descriptors))

    figures.gallery(
        columns, rows, IMAGES / "compare_shapes.png",
        cell_notes=notes,
        suptitle=(
            f"Twelve silhouettes traced by people. The probe is another person's "
            f"tracing of the same object, turned {FIGURE_ROTATION:g} degrees; each cell is "
            f"the shape that descriptor retrieved. Chance is {CHANCE:.3f}."
        ),
    )
    gallery_table = markdown_table(
        table_rows, [("Sr", "Sr"), ("Object", "Object")] + [(d, d) for d in descriptors])

    # ------------------------------------------------------------------ #
    # the signature figure: the invariance that should be a flat line
    # ------------------------------------------------------------------ #
    angles = [0, 15, 30, 45, 60, 90, 135, 180]
    masks = {n: sd.load_silhouette(n) for n in sd.REAL_IMAGES}
    curves: dict[str, list[float]] = {}
    for name, fn in sd.DESCRIPTORS.items():
        series = []
        for angle in angles:
            changes = []
            for image in sd.REAL_IMAGES:
                ref = fn(sd.transform_mask(masks[image]))
                moved = fn(sd.transform_mask(masks[image], rotation=float(angle)))
                changes.append(sd.relative_change(ref, moved))
            series.append(float(np.mean(changes)))
        curves[name] = series

    human_floor = {r["descriptor"]: r["mean_relative_change"] for r in human}
    figures.lines(
        angles, curves, IMAGES / "rotation_invariance.png",
        xlabel="rotation applied (degrees)",
        ylabel="relative change in the descriptor",
        title=(f"An invariance should be a flat line at zero. Two people tracing the "
               f"same object already differ by {min(human_floor.values()):.2f}-"
               f"{max(human_floor.values()):.2f} on these descriptors."),
        logy=True,
        dashed={"Hu moments (textbook log)"},
    )

    figures.lines(
        angles,
        {f"{d} (generated shapes)": [
            float(np.mean([sd.relative_change(
                sd.DESCRIPTORS[d](sd.make_shape(s)),
                sd.DESCRIPTORS[d](sd.make_shape(s, rotation=float(a))))
                for s in sd.SHAPE_NAMES])) for a in angles]
         for d in ("Hu moments (log)", "Hu moments (textbook log)")},
        IMAGES / "hu_log_bug.png",
        xlabel="rotation applied (degrees)",
        ylabel="relative change in the descriptor",
        title="sign(h)*log(|h|+eps) on symmetric shapes: the zero that is not a zero",
        logy=True,
        dashed={"Hu moments (textbook log)"},
    )

    figures.comparison_matrix(
        [{"descriptor": d,
          "generated_rotation": next(r["mean_relative_change"]
                                     for r in generated["rotation"] if r["descriptor"] == d),
          "real_rotation": next(r["mean_relative_change"]
                                for r in real["rotation"] if r["descriptor"] == d),
          "real_scale": next(r["mean_relative_change"]
                             for r in real["scale"] if r["descriptor"] == d),
          "human_spread": human_floor[d],
          "upright": next(r["accuracy"] for r in upright if r["descriptor"] == d),
          "turned": next(r["accuracy"] for r in turned if r["descriptor"] == d)}
         for d in descriptors],
        [("Rotation (drawn)", "generated_rotation", False),
         ("Rotation (traced)", "real_rotation", False),
         ("Scale (traced)", "real_scale", False),
         ("Two people differ by", "human_spread", False),
         ("Recognition upright", "upright", True),
         ("Recognition turned", "turned", True)],
        IMAGES / "descriptor_matrix.png",
        row_key="descriptor",
        title=("The descriptor that wins upright is the one with no rotation invariance — "
               "and it is the one that collapses when the shape is turned."),
    )

    figures.lines(
        [r["raster_size"] for r in discretisation],
        {d: [r[d] for r in discretisation] for d in descriptors},
        IMAGES / "discretisation.png",
        xlabel="raster size (px)", ylabel="relative change under a 30 deg rotation",
        title="Hu moments are invariant in continuous maths; images are discrete",
        logy=True, dashed={"Hu moments (textbook log)"},
    )

    figures.lines(
        [r["boundary_noise"] for r in noise],
        {d: [r[d] for r in noise] for d in descriptors},
        IMAGES / "boundary_noise.png",
        xlabel="fraction of boundary pixels flipped", ylabel="relative change",
        title="Roughening the outline: low-frequency descriptors should not care",
        dashed={"Hu moments (textbook log)"},
    )

    figures.grid(
        [(n.replace("_", " "), _outline(_fit(sd.load_silhouette(n))))
         for n in sd.REAL_IMAGES],
        IMAGES / "silhouettes.png", ncols=4,
        suptitle="The twelve outlines, exactly as one annotator drew them.",
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS, "36_shape_descriptors",
        {
            "images": list(sd.REAL_IMAGES),
            "silhouettes": {k: list(v) for k, v in sd.SILHOUETTES.items()},
            "chance": round(CHANCE, 4),
            "generated_invariance": generated,
            "real_invariance": real,
            "annotator_variation": human,
            "cross_annotator_upright": upright,
            "cross_annotator_turned": turned,
            "shape_classification": shape_accuracy,
            "discretisation": discretisation,
            "reflection_generated": reflection,
            "reflection_real": real_reflection,
            "boundary_noise": noise,
        },
    )

    def invariance_table(rows_by_transform):
        merged = []
        for d in descriptors:
            row = {"descriptor": d}
            for transform, rows in rows_by_transform.items():
                row[transform] = next(r["mean_relative_change"]
                                      for r in rows if r["descriptor"] == d)
            merged.append(row)
        return markdown_table(
            merged, [("Descriptor", "descriptor"), ("Rotation", "rotation"),
                     ("Scale", "scale"), ("Translation", "translation")])

    human_table = markdown_table(
        human, [("Descriptor", "descriptor"), ("Mean change", "mean_relative_change"),
                ("Max", "max_relative_change"), ("Pairs", "pairs")])
    recognition_table = markdown_table(
        [{"descriptor": a["descriptor"], "upright": a["accuracy"], "turned": b["accuracy"]}
         for a, b in zip(upright, turned)],
        [("Descriptor", "descriptor"), ("Upright", "upright"),
         (f"Turned {FIGURE_ROTATION:g} deg", "turned")])
    reflection_table = markdown_table(
        real_reflection,
        [("Object", "image"), ("Mirror symmetry", "mirror_symmetry_iou"),
         ("h7 flips", "hu7_sign_flipped"), ("Fourier change", "fourier_change"),
         ("Geometry change", "geometry_change")])

    write_tables(
        RESULTS,
        [
            ("Invariance on generated shapes", invariance_table(generated)),
            ("Invariance on human-traced silhouettes", invariance_table(real)),
            ("How far apart two people are on the same object", human_table),
            ("Recognising an object from another person's tracing", recognition_table),
            ("Reflection: only the 7th Hu moment notices", reflection_table),
            ("Four objects down the rows", gallery_table),
        ],
    )
    print("\n" + invariance_table(real) + "\n\n" + recognition_table)

    # ------------------------------------------------------------------ #
    # headlines
    # ------------------------------------------------------------------ #
    print("\n--- HEADLINE NUMBERS ---")
    gen_rot = {r["descriptor"]: r["mean_relative_change"] for r in generated["rotation"]}
    real_rot = {r["descriptor"]: r["mean_relative_change"] for r in real["rotation"]}
    print(f"the textbook log-Hu recipe moves {gen_rot['Hu moments (textbook log)']:.3f} under "
          f"rotation on drawn shapes; flooring the zero properly takes it to "
          f"{gen_rot['Hu moments (log)']:.3f} "
          f"({gen_rot['Hu moments (textbook log)'] / max(gen_rot['Hu moments (log)'], 1e-9):.0f}x)")
    print("  the whole difference is that np.sign(0) is 0, so an exactly-zero Hu moment "
          "logs to 0 instead of to the floor")

    for d in descriptors:
        ratio = real_rot[d] / max(human_floor[d], 1e-9)
        verdict = "BELOW the human spread" if ratio < 1 else "ABOVE it"
        print(f"  {d:28s} rotation {real_rot[d]:.4f} vs people {human_floor[d]:.4f} "
              f"-> {ratio:6.3f}x, {verdict}")

    up = {r["descriptor"]: r["accuracy"] for r in upright}
    tu = {r["descriptor"]: r["accuracy"] for r in turned}
    best_up = max(up, key=up.get)
    best_turned = max(tu, key=tu.get)
    print(f"\nbest upright: {best_up} at {up[best_up]:.4f} -> "
          f"{tu[best_up]:.4f} once turned {FIGURE_ROTATION:g} degrees")
    print(f"best turned : {best_turned} at {tu[best_turned]:.4f} "
          f"(it was {up[best_turned]:.4f} upright)")
    print(f"chance is {CHANCE:.4f}")

    flipped = sum(1 for r in real_reflection if r["hu7_sign_flipped"])
    print(f"\nthe 7th Hu moment flips sign on {flipped}/{len(real_reflection)} traced "
          "silhouettes; every other descriptor here is exactly blind to a mirror "
          f"(geometry change {max(r['geometry_change'] for r in real_reflection):.4f})")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
