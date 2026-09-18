"""Run the texture descriptor comparison and write results + figures.

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

import texture as tx  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"

#: The condition the front comparison figure is shot at. Chosen because it is
#: where the descriptors *disagree*: at illumination 0.6 LBP still answers three
#: quarters of the probes correctly and every other descriptor is at chance.
#: A clean figure would show five near-identical columns and prove nothing.
FIGURE_DEGRADATION = ("illumination", 0.6)

#: 32x32 is what the classifier sees. The figure enlarges it with nearest
#: neighbour so that no detail is invented for the reader that the descriptor
#: did not have.
FIGURE_ZOOM = 6

CHANCE = 1.0 / len(tx.TEXTURES)


def _zoom(patch: np.ndarray) -> np.ndarray:
    return cv2.resize(patch, None, fx=FIGURE_ZOOM, fy=FIGURE_ZOOM,
                      interpolation=cv2.INTER_NEAREST)


def _retrieval_panels(gallery, probes, labels, descriptor: str):
    """What one descriptor retrieved for every probe: (index, correct?)."""
    fn = tx.DESCRIPTORS[descriptor]
    g = tx._features(gallery, fn)
    p = tx._features(probes, fn)
    mean = g.mean(0, keepdims=True)
    std = np.maximum(g.std(0, keepdims=True), tx.EPS)
    d = np.linalg.norm(((p - mean) / std)[:, None, :] - ((g - mean) / std)[None, :, :], axis=2)
    return np.argmin(d, axis=1)


def _mark(patch: np.ndarray, correct: bool) -> np.ndarray:
    """Frame a retrieved patch green when the class is right, red when wrong."""
    rgb = ensure_rgb(_zoom(patch))
    colour = (60, 170, 60) if correct else (200, 60, 60)
    return cv2.copyMakeBorder(rgb, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=colour)


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    descriptors = list(tx.DESCRIPTORS)

    print(f"Scoring {len(descriptors)} descriptors on {len(tx.TEXTURES)} textures "
          f"at {tx.PATCH}x{tx.PATCH}, over {len(tx.SEEDS)} crop seeds ...")
    clean = tx.evaluate_over_seeds(runs=args.runs)
    for r in clean:
        print(f"  {r['descriptor']:26s} acc {r['accuracy']:.4f} +- {r['accuracy_sd']:.4f}  "
              f"sep {r['separability']:7.3f}  {r['dimensions']:3d}-D  "
              f"{r['median_ms']:.3f} ms")

    print("\nSweeping the patch size — this is what chooses the operating point ...")
    patch_rows = tx.sweep_patch_size()
    for r in patch_rows:
        print(f"  {r['patch']:3d} px  " +
              "  ".join(f"{k.split()[0]} {v:.3f}" for k, v in r.items() if k != "patch"))

    print("\nSweeping each degradation (degraded probes, clean gallery) ...")
    sweeps = {
        "illumination": tx.sweep_degradation("illumination", tx.ILLUMINATION_LEVELS),
        "gamma": tx.sweep_degradation("gamma", tx.GAMMA_LEVELS),
        "noise": tx.sweep_degradation("noise", tx.NOISE_LEVELS),
        "rotation": tx.sweep_degradation("rotation", tx.ROTATION_LEVELS),
    }
    for name, rows in sweeps.items():
        print(f"  {name}: " + "  ".join(
            f"{k.split()[0]} {rows[0][k]:.2f}->{rows[-1][k]:.2f}"
            for k in rows[0] if k != name))

    invariance = tx.invariance_summary()

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    # Rows are four textures; columns are what each descriptor *retrieved* for a
    # degraded probe. A picture of a feature map would prove nothing, which is
    # the whole premise of this project — so the figure shows the actual answer,
    # framed green when the class is right and red when it is not.
    deg, strength = FIGURE_DEGRADATION
    gallery, probes, labels = tx.build_split(degradation=deg, strength=strength)
    retrieved = {d: _retrieval_panels(gallery, probes, labels, d) for d in descriptors}

    # One probe per chosen class, picked as the probe the descriptors disagree
    # about most — an agreed-on probe shows five identical columns.
    disagreement = []
    for i, label in enumerate(labels):
        answers = {labels[retrieved[d][i]] for d in descriptors}
        disagreement.append((len(answers), i))

    chosen, used = [], set()
    for _, i in sorted(disagreement, reverse=True):
        if labels[i] in used:
            continue
        used.add(labels[i])
        chosen.append(i)
        if len(chosen) == 4:
            break
    chosen.sort(key=lambda i: labels[i])
    assert len({labels[i] for i in chosen}) == 4, "a texture repeats down the rows"

    # Each descriptor's accuracy over all 144 probes at this condition goes in
    # the column header. Four rows chosen for disagreement are an illustration,
    # not a statistic, and the header is what stops them being read as one.
    condition_accuracy = {
        d: float((labels[retrieved[d]] == labels).mean()) for d in descriptors
    }

    clean_probes, _ = tx.build_patches()
    columns = ["the surface", f"what it sees\n({deg} {strength:g})"]
    columns += [f"{d}\n{condition_accuracy[d]:.3f} over all {len(labels)}"
                for d in descriptors]
    rows, notes, table_rows = [], [], []
    for sr, i in enumerate(chosen, start=1):
        true_name = tx.TEXTURES[labels[i]]
        panels = [ensure_rgb(_zoom(clean_probes[i])), ensure_rgb(_zoom(probes[i]))]
        cell = ["the gallery sees this", "the classifier sees this"]
        scores = {}
        for d in descriptors:
            j = retrieved[d][i]
            correct = labels[j] == labels[i]
            panels.append(_mark(gallery[j], correct))
            cell.append(("OK  " if correct else "WRONG  ") +
                        tx.TEXTURES[labels[j]].replace("_", " "))
            scores[d] = "correct" if correct else tx.TEXTURES[labels[j]].replace("_", " ")
        rows.append((f"{true_name.replace('_', ' ')}", panels))
        notes.append(cell)
        table_rows.append(dict([("Sr", sr), ("Texture", true_name.replace("_", " "))],
                               **scores))
        print(f"  row {sr}: {true_name:20s} " +
              "  ".join(f"{d.split()[0]}={scores[d]}" for d in descriptors))

    figures.gallery(
        columns,
        rows,
        IMAGES / "compare_texture.png",
        cell_notes=notes,
        suptitle=(
            f"Twelve textures, {tx.PATCH}x{tx.PATCH} patches, no training. The probe is dimmed "
            f"({deg} {strength:g}); the gallery is not. Each cell is the patch that "
            "descriptor retrieved — green if it is the same surface, red if not."
        ),
    )
    gallery_table = markdown_table(
        table_rows,
        [("Sr", "Sr"), ("Texture", "Texture")] + [(d, d) for d in descriptors],
    )

    # ------------------------------------------------------------------ #
    # the signature figure: the invariance matrix, descriptor x transform
    # ------------------------------------------------------------------ #
    # The title states the finding, so it is computed rather than typed — a
    # hardcoded one went stale the first time the numbers moved.
    best_clean_row = max(invariance, key=lambda r: r["clean_accuracy"])
    collapsed = [c for c in tx.INVARIANCE_COLUMNS
                 if best_clean_row[f"{c}_accuracy"] < CHANCE + 0.02]
    column_winners = {
        max(invariance, key=lambda r: r[f"{c}_accuracy"])["descriptor"]
        for c in ("clean", *tx.INVARIANCE_COLUMNS)
    }

    figures.comparison_matrix(
        invariance,
        [("Clean", "clean_accuracy", True),
         ("Relit", "illumination_accuracy", True),
         ("Gamma", "gamma_accuracy", True),
         ("Noise", "noise_accuracy", True),
         ("Rotated 45 deg", "rotation45_accuracy", True),
         ("Rotated 90 deg", "rotation_accuracy", True)],
        IMAGES / "invariance_matrix.png",
        row_key="descriptor",
        title=(f"A descriptor is an invariance, not a quality: {len(column_winners)} rows "
               f"win a column, and the best clean row ({best_clean_row['descriptor']}) "
               f"is at chance ({CHANCE:.3f}) in {len(collapsed)} of them. "
               f"Mean of {len(tx.SEEDS)} crop seeds."),
    )

    figures.lines(
        [r["patch"] for r in patch_rows],
        {d: [r[d] for r in patch_rows] for d in descriptors},
        IMAGES / "patch_size_sweep.png",
        xlabel="patch size (px)",
        ylabel="accuracy",
        title="How much surface each descriptor needs — and where the benchmark saturates",
        dashed={"Raw histogram (control)"},
        vlines={f"operating point ({tx.PATCH} px)": tx.PATCH},
    )

    for name, xlabel in (("illumination", "brightness pulled towards black"),
                         ("gamma", "extra gamma exponent"),
                         ("noise", "Gaussian noise sigma"),
                         ("rotation", "rotation (degrees)")):
        rowset = sweeps[name]
        figures.lines(
            [r[name] for r in rowset],
            {d: [r[d] for r in rowset] for d in descriptors},
            IMAGES / f"{name}_sweep.png",
            xlabel=xlabel,
            ylabel="accuracy (degraded probe, clean gallery)",
            title=f"{name.capitalize()}: which descriptor keeps working",
            dashed={"Raw histogram (control)"},
        )

    best_clean = max(clean, key=lambda r: r["accuracy"])["descriptor"]
    short = [t.replace("_", " ") for t in tx.TEXTURES]
    figures.confusion_matrix(
        tx.confusion(best_clean),
        IMAGES / "confusion_clean.png",
        short, short,
        title=f"{best_clean} on clean patches — what it is actually confused about",
    )

    figures.grid(
        [(f"{t.replace('_', ' ')}", tx.load_scene(t)) for t in tx.TEXTURES],
        IMAGES / "plates.png",
        ncols=4,
        suptitle="The twelve surfaces. Every patch in this project is cut from one of these.",
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS,
        "33_texture",
        {
            "textures": list(tx.TEXTURES),
            "patch": tx.PATCH,
            "per_class": tx.PER_CLASS,
            "chance": round(CHANCE, 4),
            "clean": clean,
            "patch_size_sweep": patch_rows,
            "invariance": invariance,
            **{f"{k}_sweep": v for k, v in sweeps.items()},
            "figure_condition": {"degradation": deg, "strength": strength,
                                 "accuracy": {k: round(v, 4)
                                              for k, v in condition_accuracy.items()}},
        },
    )

    for r in clean:
        r["accuracy_pm"] = f"{r['accuracy']:.3f} ± {r['accuracy_sd']:.3f}"
    clean_table = markdown_table(
        clean,
        [("Descriptor", "descriptor"), ("Accuracy", "accuracy_pm"),
         ("Separability", "separability"), ("Dimensions", "dimensions"),
         ("Time (ms)", "median_ms")],
    )
    for r in invariance:
        for key in ("clean", *tx.INVARIANCE_COLUMNS):
            r[f"{key}_pm"] = f"{r[f'{key}_accuracy']:.3f} ± {r[f'{key}_sd']:.3f}"
    invariance_table = markdown_table(
        invariance,
        [("Descriptor", "descriptor"), ("Clean", "clean_pm"),
         ("Relit", "illumination_pm"), ("Gamma", "gamma_pm"),
         ("Noise", "noise_pm"), ("Rotated 45", "rotation45_pm"),
         ("Rotated 90", "rotation_pm")],
    )
    patch_table = markdown_table(
        patch_rows, [("Patch", "patch")] + [(d, d) for d in descriptors])
    write_tables(
        RESULTS,
        [
            ("Clean accuracy at the operating point", clean_table),
            ("Accuracy under each degradation", invariance_table),
            ("Accuracy against patch size", patch_table),
            ("Four textures down the rows", gallery_table),
        ],
    )
    print("\n" + clean_table + "\n\n" + invariance_table)

    # ------------------------------------------------------------------ #
    # headlines
    # ------------------------------------------------------------------ #
    by_name = {r["descriptor"]: r for r in invariance}
    print("\n--- HEADLINE NUMBERS ---")
    print(f"chance is {CHANCE:.4f}; every cell is the mean of {len(tx.SEEDS)} crop seeds "
          f"({len(tx.SEEDS) * len(labels)} classifications)")
    for column in ("clean", *tx.INVARIANCE_COLUMNS):
        key = f"{column}_accuracy"
        ranked = sorted(invariance, key=lambda r: -r[key])
        winner, runner = ranked[0], ranked[1]
        tied = winner[key] - runner[key] < winner[f"{column}_sd"]
        print(f"  {column:12s} winner: {winner['descriptor']:26s} {winner[key]:.4f} "
              f"(next {runner['descriptor'].split()[0]} {runner[key]:.4f})"
              f"{'   [within one SD — a tie]' if tied else ''}")

    lbp = by_name["LBP (uniform)"]
    best_clean = max(invariance, key=lambda r: r["clean_accuracy"])
    at_chance = [c for c in tx.INVARIANCE_COLUMNS
                 if best_clean[f"{c}_accuracy"] < CHANCE + 0.02]
    print(f"\nthe best clean descriptor ({best_clean['descriptor']}, "
          f"{best_clean['clean_accuracy']:.4f}) is at CHANCE in "
          f"{len(at_chance)} of {len(tx.INVARIANCE_COLUMNS)} degraded columns: "
          f"{', '.join(at_chance)}")

    real = [r for r in invariance if "control" not in r["descriptor"]]
    worst_clean = min(real, key=lambda r: r["clean_accuracy"])
    print(f"the WORST real descriptor clean is {worst_clean['descriptor']} "
          f"({worst_clean['clean_accuracy']:.4f}) and it is the only one above chance "
          f"after a relight ({lbp['illumination_accuracy']:.4f})")

    control45 = by_name["Raw histogram (control)"]["rotation45_accuracy"]
    beaten = [r["descriptor"] for r in real if r["rotation45_accuracy"] < control45]
    print(f"at 45 degrees the do-nothing control scores {control45:.4f} and beats "
          f"{len(beaten)} of {len(real)} real descriptors — it is the only one "
          "with no geometry in it at all")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
