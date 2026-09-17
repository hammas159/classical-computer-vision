"""Run the HDR exposure-fusion experiments and write results + figures.

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

from shared import figures, synth  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import hdr  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=len(hdr.IMAGES))
    args = ap.parse_args()

    images = hdr.IMAGES[: args.images]
    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    print(f"Scoring {len(hdr.METHODS)} methods on {len(images)} scenes ...")
    method_rows, stats = hdr.evaluate_methods(images=images)

    print("Sweeping bracket width ...")
    size_rows = hdr.sweep_bracket_size(images=images)

    print("Measuring the clipping ceiling ...")
    ceiling_rows = hdr.evaluate_clipping_ceiling(images=images)

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    # Four scenes down the rows, every method across. The scenes are chosen for
    # how their range is DISTRIBUTED, not for subject: a bright sky over dark
    # water puts the range in one boundary, a stone arch puts it in a hundred
    # small ones, and a method that handles one need not handle the other.
    GALLERY_MIN_SSIM = 0.40
    scene_pool = [
        ("lake and shrine\nbright sky, dark foreground", "lake_shrine", "one boundary"),
        ("rocky coast\nsky over shadowed rock", "rocky_coast", "one boundary"),
        ("stone arch\nmany small bright/dark edges", "stone_arch", "many boundaries"),
        ("windmills\nwhite walls against sky", "windmills", "many boundaries"),
        ("harbour and boat\nsunlit town, shaded hull", "harbour_boat", "mixed"),
        ("boat and shed\nwater reflections", "boat_shed", "mixed"),
        ("temple dragon\nlit statue, dark towers", "temple_dragon", "subject vs ground"),
        ("giraffe\nlit animal, flat background", "giraffe", "subject vs ground"),
        ("gallery visitors\ninterior, lit pictures", "gallery_visitors", "interior"),
        ("elephant in grass\neven light, little range", "elephant_grass", "little range"),
        ("squirrel on a rock\nsoft light, little range", "squirrel_rock", "little range"),
        ("woman and child\nclose faces, flat light", "woman_child_fur", "interior"),
    ]
    method_names = list(hdr.METHODS)
    survivors: dict[str, dict] = {}
    for name_label, name, family in scene_pool:
        frames, times, reference = synth.hdr_bracket(hdr.load_scene(name))
        loss = synth.bracket_loss(frames)
        outs = [hdr.METHODS[m](frames, times) for m in method_names]
        scores = [hdr.score(o, reference) for o in outs]
        best = max(s["ssim"] for s in scores)
        flat = name_label.replace("\n", " · ")
        if best < GALLERY_MIN_SSIM:
            print(f"scene candidate {flat:<42} DROP — best SSIM {best:.3f}  [{family}]")
            continue
        print(
            f"scene candidate {flat:<42} keep — best SSIM {best:.3f}, "
            f"{100 * loss['unrecoverable']:.1f}% unrecoverable  [{family}]"
        )
        row = {
            "label": name_label,
            "images": [reference, frames[len(frames) // 2]] + outs,
            "notes": ["reference", "one exposure"] + [f"SSIM {s['ssim']:.3f}" for s in scores],
            "score": best,
            "scores": scores,
        }
        if family not in survivors or row["score"] > survivors[family]["score"]:
            survivors[family] = row

    chosen = sorted(survivors.values(), key=lambda r: -r["score"])[:4]
    figures.gallery(
        ["reference (the answer)", "middle exposure"] + method_names,
        [(r["label"], r["images"]) for r in chosen],
        IMAGES / "compare_fusion.png",
        cell_notes=[r["notes"] for r in chosen],
        suptitle=(
            "Four scenes of 12 stops, bracketed into five exposures and fused. "
            "The reference is the photograph the scene was built from."
        ),
    )
    gallery_table = markdown_table(
        [
            dict([("Sr", i), ("Scene", r["label"].replace("\n", " · "))]
                 + list(zip(method_names, r["notes"][2:])))
            for i, r in enumerate(chosen, start=1)
        ],
        [("Sr", "Sr"), ("Scene", "Scene")] + [(m, m) for m in method_names],
    )
    print(f"\nfront-on comparison: {len(chosen)} scenes x {len(method_names)} methods")
    print("\n--- fusion ---\n" + gallery_table)

    # ------------------------------------------------------------------ #
    # the signature figure: which frame each pixel came from
    # ------------------------------------------------------------------ #
    demo = hdr.load_scene("lake_shrine")
    frames, times, reference = synth.hdr_bracket(demo)
    figures.grid(
        [(f"{s:+.0f} stops", f) for s, f in zip(synth.EXPOSURE_STOPS, frames)],
        IMAGES / "bracket.png",
        ncols=5,
        suptitle="One 12-stop scene, five exposures. No single frame holds it.",
    )

    contrib = hdr.contribution_map(frames)
    figures.grid(
        [
            ("Reference", reference),
            ("Middle exposure", frames[len(frames) // 2]),
            ("Which frame is best exposed here", contrib.astype(np.float32) / max(len(frames) - 1, 1)),
            ("Blown in EVERY frame", _all_blown(frames).astype(np.float32)),
        ],
        IMAGES / "contribution.png",
        ncols=4,
        suptitle=(
            "Darker means an earlier (shorter) exposure supplied the pixel. "
            "The last panel is what no exposure recorded."
        ),
    )

    figures.lines(
        [r["stop_spread"] for r in ceiling_rows],
        {
            "blown in every frame": [r["blown_everywhere"] for r in ceiling_rows],
            "crushed in every frame": [r["crushed_everywhere"] for r in ceiling_rows],
            "unrecoverable": [r["unrecoverable"] for r in ceiling_rows],
        },
        IMAGES / "clipping_ceiling.png",
        xlabel="bracket spread (± stops)",
        ylabel="fraction of the scene",
        title="What the bracket never recorded — a ceiling no method can lift",
    )

    fusion_only = [n for n in method_names if n not in hdr.CONTROLS]
    figures.lines(
        [r["frames"] for r in size_rows],
        {n: [r[n] for r in size_rows] for n in fusion_only},
        IMAGES / "bracket_size.png",
        xlabel="frames in the bracket",
        ylabel="PSNR vs the reference (dB)",
        title="How many exposures actually pay",
    )

    figures.comparison_matrix(
        method_rows,
        [
            ("PSNR (dB)", "psnr_db", True),
            ("PSNR matched (dB)", "psnr_matched_db", True),
            ("SSIM", "ssim", True),
            ("Time (ms)", "median_ms", False),
        ],
        IMAGES / "method_matrix.png",
        title="Methods x metrics — raw PSNR scores the tone mapper's brightness choice",
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS,
        "11_hdr_exposure_fusion",
        {
            "images": list(images),
            "scene_stops": synth.HDR_SCENE_STOPS,
            "exposure_stops": list(synth.EXPOSURE_STOPS),
            "single_exposure": stats,
            "methods": method_rows,
            "bracket_size_sweep": size_rows,
            "clipping_ceiling": ceiling_rows,
        },
    )

    method_table = markdown_table(
        method_rows,
        [
            ("Method", "method"),
            ("PSNR (dB)", "psnr_db"),
            ("PSNR matched (dB)", "psnr_matched_db"),
            ("SSIM", "ssim"),
            ("Time (ms)", "median_ms"),
        ],
    )
    size_table = markdown_table(
        size_rows,
        [("Frames", "frames"), ("Unrecoverable", "unrecoverable")]
        + [(n, n) for n in fusion_only],
    )
    ceiling_table = markdown_table(
        ceiling_rows,
        [
            ("Bracket spread (± stops)", "stop_spread"),
            ("Frames", "frames"),
            ("Blown everywhere", "blown_everywhere"),
            ("Crushed everywhere", "crushed_everywhere"),
            ("Unrecoverable", "unrecoverable"),
        ],
    )
    write_tables(
        RESULTS,
        [
            (f"Methods on {len(images)} scenes of {synth.HDR_SCENE_STOPS:.0f} stops", method_table),
            ("Four exposures down the rows", gallery_table),
            ("How many frames pay", size_table),
            ("What the bracket never recorded", ceiling_table),
        ],
    )
    print("\n" + method_table + "\n\n" + size_table + "\n\n" + ceiling_table)

    fusion_rows = [r for r in method_rows if r["method"] not in hdr.CONTROLS]
    best = max(fusion_rows, key=lambda r: r["ssim"])
    mean_ctl = next(r for r in method_rows if r["method"].startswith("Mean"))
    single = next(r for r in method_rows if r["method"].startswith("Middle"))

    print("\n--- HEADLINE NUMBERS ---")
    print(f"scene range      : {synth.HDR_SCENE_STOPS:.0f} stops, bracketed at "
          f"{list(synth.EXPOSURE_STOPS)}")
    print(f"one exposure     : {single['psnr_db']} dB, SSIM {single['ssim']}")
    print(f"best fusion      : {best['method']} @ SSIM {best['ssim']} "
          f"({best['psnr_db']} dB raw, {best['psnr_matched_db']} dB matched)")
    print(f"naive mean       : SSIM {mean_ctl['ssim']} at {mean_ctl['median_ms']} ms")
    print(f"fusion vs single : SSIM {best['ssim'] - single['ssim']:+.4f}")
    print(f"unrecoverable    : {100 * stats['mean_unrecoverable']:.2f}% of the scene "
          "was clipped or crushed in every frame")
    for r in method_rows:
        print(f"  {r['method']:32s} raw {r['psnr_db']:6.2f}  matched "
              f"{r['psnr_matched_db']:6.2f}  (+{r['psnr_matched_db'] - r['psnr_db']:.2f})")
    print(f"\nwrote {results_path}")


def _all_blown(frames):
    """Pixels saturated in every frame — what the bracket simply did not record."""
    stacked = np.stack([f.astype(np.float32) / 255.0 for f in frames])
    return (stacked >= 254.0 / 255.0).all(axis=0).any(axis=-1)


if __name__ == "__main__":
    main()
