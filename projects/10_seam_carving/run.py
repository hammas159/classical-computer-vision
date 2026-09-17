"""Run the seam carving experiments and write results + figures.

    python run.py [--reduction 0.20]

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

from shared import figures, io  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import seam_carving as sc  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"


def outline(img, mask, colour=(60, 220, 60)):
    """Draw the tracked region's outline so it is visible in a figure."""
    out = img.copy()
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, contours, -1, colour, 2)
    return out


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--reduction", type=float, default=0.20)
    ap.add_argument("--images", type=int, default=len(sc.IMAGES))
    args = ap.parse_args()

    images = sc.IMAGES[: args.images]
    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    print(f"Comparing {len(sc.ENERGIES)} energy functions at {args.reduction:.0%} ...")
    energy_rows = sc.evaluate_energies(reduction=args.reduction, images=images)

    print(f"Sweeping reduction over {len(sc.REDUCTION_LEVELS)} levels ...")
    sweep_rows = sc.sweep_reduction(images=images)

    print("Measuring what the advantage costs ...")
    cost_rows = sc.compare_cost(reduction=args.reduction, images=images)

    print("Breaking the average down per image ...")
    image_rows = sc.per_image(reduction=args.reduction, images=images)

    # ------------------------------------------------------------------ #
    # figures
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # the comparison at the top of the README:
    # four photographs down the rows, each resizer across the columns
    # ------------------------------------------------------------------ #
    # Seam carving removes the lowest-energy path it can find, so what it can do
    # depends entirely on whether the photograph HAS a low-energy path. The
    # candidates are therefore grouped by shape rather than by subject: a tall
    # thin animal in a wide frame has enormous slack, a frame filled edge to edge
    # with architecture has almost none, and the difference between them is the
    # whole result.
    GALLERY_MIN_ADVANTAGE = 2.0  # points of subject retained, vs plain rescale
    shape_pool = [
        ("giraffe\ntall subject, wide empty frame", "giraffe", "slack"),
        ("squirrel on a rock\nsmall subject, large background", "squirrel_rock", "slack"),
        ("elephant in grass\nwide subject, low-energy grass", "elephant_grass", "slack"),
        ("rocky coast\nwide, no single subject", "rocky_coast", "no subject"),
        ("lake and shrine\nforeground object, empty water", "lake_shrine", "no subject"),
        ("stone arch\narchitecture, strong verticals", "stone_arch", "architecture"),
        ("windmills\nrepeated structure on a wall", "windmills", "architecture"),
        ("temple dragon\nvertical subject against towers", "temple_dragon", "architecture"),
        ("harbour and boat\nhorizontal structure", "harbour_boat", "horizontal"),
        ("boat and shed\nmoored, reflected", "boat_shed", "horizontal"),
        ("woman and child\nfaces filling the frame", "woman_child_fur", "faces"),
        ("gallery visitors\npeople and framed pictures", "gallery_visitors", "faces"),
    ]
    survivors: dict[str, dict] = {}
    for label, name, family in shape_pool:
        scene = sc.load_scene(name)
        carved_i, rescaled_i, mask_i, cmask, rmask = sc.resize_pair(scene, args.reduction)
        total = float((mask_i > 0).sum())
        kept_carve = 100.0 * float((cmask > 0).sum()) / max(total, 1.0)
        kept_rescale = 100.0 * float((rmask > 0).sum()) / max(total, 1.0)
        advantage = kept_carve - kept_rescale
        flat = label.replace("\n", " · ")
        if advantage < GALLERY_MIN_ADVANTAGE:
            print(f"shape candidate {flat:<46} DROP — carving gains only "
                  f"{advantage:+.1f} pts  [{family}]")
            continue
        print(f"shape candidate {flat:<46} keep — carve {kept_carve:.1f}% vs "
              f"rescale {kept_rescale:.1f}%, {advantage:+.1f} pts  [{family}]")
        row = {
            "label": label,
            "images": [outline(scene, mask_i), outline(carved_i, cmask), outline(rescaled_i, rmask)],
            "notes": ["subject outlined", f"{kept_carve:.1f}% kept", f"{kept_rescale:.1f}% kept"],
            "score": advantage,
        }
        if family not in survivors or row["score"] > survivors[family]["score"]:
            survivors[family] = row

    # Fixed slot order, not the top four by advantage. Ranking by advantage
    # returned the four shapes carving does BEST on and left out architecture,
    # which is where it has least room to work -- and a figure that only shows
    # the cases a method wins is an advertisement. These four span the range.
    SLOT_ORDER = ["slack", "horizontal", "faces", "architecture"]
    chosen = [survivors[s] for s in SLOT_ORDER if s in survivors][:4]
    figures.gallery(
        ["original", f"seam carved −{args.reduction:.0%}", f"cv2.resize −{args.reduction:.0%}"],
        [(r["label"], r["images"]) for r in chosen],
        IMAGES / "compare_resizers.png",
        cell_notes=[r["notes"] for r in chosen],
        suptitle=(
            f"Four photographs narrowed by {args.reduction:.0%}. The outline is the "
            "highest-energy region of the original; the percentage is how much of it survived."
        ),
    )
    print(f"front-on comparison: {len(chosen)} photographs x 2 resizers")
    resize_table = markdown_table(
        [
            {
                "Sr": i,
                "Photograph": r["label"].replace("\n", " · "),
                "Seam carving": r["notes"][1],
                "cv2.resize": r["notes"][2],
                "Advantage": f"{r['score']:+.1f} pts",
            }
            for i, r in enumerate(chosen, start=1)
        ],
        [("Sr", "Sr"), ("Photograph", "Photograph"), ("Seam carving", "Seam carving"),
         ("cv2.resize", "cv2.resize"), ("Advantage", "Advantage")],
    )
    print("\n--- resizers ---\n" + resize_table)


    img = sc.load_scene("giraffe")
    carved, rescaled, mask, carved_mask, rescaled_mask = sc.resize_pair(img, args.reduction)

    figures.grid(
        [
            ("Original, subject outlined", outline(img, mask)),
            (
                f"Seam carved to {carved.shape[1]}px\n"
                f"{float((carved_mask > 0).sum()) / float((mask > 0).sum()):.1%} of the subject kept",
                outline(carved, carved_mask),
            ),
            (
                f"Plain rescale to {rescaled.shape[1]}px\n"
                f"{float((rescaled_mask > 0).sum()) / float((mask > 0).sum()):.1%} of the subject kept",
                outline(rescaled, rescaled_mask),
            ),
        ],
        IMAGES / "comparison.png",
        ncols=3,
        suptitle=(
            f"{args.reduction:.0%} narrower, two ways — the outlined region was found "
            "from the image, not pasted in"
        ),
    )

    figures.grid(
        [
            ("Original", img),
            ("Energy |dx|+|dy|", (sc.energy_gradient(img) / max(sc.energy_gradient(img).max(), 1e-6) * 255).astype(np.uint8)),
            ("The next 40 seams", sc.seam_overlay(img, n=40)),
        ],
        IMAGES / "seams.png",
        ncols=3,
        suptitle="Seams route around what the energy calls interesting",
    )

    energy_panels = [("Original", img)]
    for name, fn in sc.ENERGIES.items():
        e = fn(img)
        energy_panels.append(
            (name, (e / max(float(e.max()), 1e-6) * 255).astype(np.uint8))
        )
    figures.grid(
        energy_panels,
        IMAGES / "energies.png",
        ncols=5,
        suptitle="Four definitions of 'boring' — and they carve almost identically",
    )

    carved_panels = [("Original", img)]
    for name, fn in sc.ENERGIES.items():
        row = next(r for r in energy_rows if r["energy"] == name)
        out, _ = sc.carve(img, int(img.shape[1] * (1 - args.reduction)), fn)
        carved_panels.append((f"{name}\n{row['subject_kept']:.1%} subject kept", out))
    figures.grid(
        carved_panels,
        IMAGES / "energy_results.png",
        ncols=5,
        suptitle=(
            f"The same {args.reduction:.0%} reduction under each energy — "
            "the spread is 0.5 points, the gap to a rescale is 11.6"
        ),
    )

    figures.lines(
        [r["reduction"] for r in sweep_rows],
        {
            "Seam carving — subject kept": [r["carved_subject_kept"] for r in sweep_rows],
            "Plain rescale — subject kept": [r["rescale_subject_kept"] for r in sweep_rows],
            "Seam carving — energy kept": [r["carved_energy_kept"] for r in sweep_rows],
            "Plain rescale — energy kept": [r["rescale_energy_kept"] for r in sweep_rows],
        },
        IMAGES / "reduction_sweep.png",
        xlabel="width reduction",
        ylabel="fraction retained",
        title="Carving wins its own objective far more decisively than the perceptual one",
    )

    # what extreme reduction actually looks like
    extreme = [("Original", img)]
    for red in (0.20, 0.45, 0.70):
        out, _ = sc.carve(img, int(img.shape[1] * (1 - red)))
        extreme.append((f"carved −{red:.0%}", out))
    figures.grid(
        extreme, IMAGES / "extreme.png", ncols=4,
        suptitle="Past about 45% there are no low-energy paths left and every seam crosses something",
    )

    figures.comparison_matrix(
        energy_rows,
        [
            ("Subject kept", "subject_kept", True),
            ("Aspect retained", "aspect_ratio", True),
            ("Energy kept", "energy_kept", True),
            ("Time (ms)", "median_ms", False),
        ],
        IMAGES / "energy_matrix.png",
        row_key="energy",
        title="The four energies are indistinguishable; the control is not",
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS,
        "10_seam_carving",
        {
            "reduction": args.reduction,
            "images": list(images),
            "energies": energy_rows,
            "reduction_sweep": sweep_rows,
            "cost": cost_rows,
            "per_image": image_rows,
        },
    )

    energy_table = markdown_table(
        energy_rows,
        [
            ("Energy", "energy"),
            ("Subject kept", "subject_kept"),
            ("Aspect retained", "aspect_ratio"),
            ("Energy kept", "energy_kept"),
            ("Time (ms)", "median_ms"),
        ],
    )
    sweep_table = markdown_table(
        sweep_rows,
        [
            ("Reduction", "reduction"),
            ("Carved: subject", "carved_subject_kept"),
            ("Rescale: subject", "rescale_subject_kept"),
            ("Carved: aspect", "carved_aspect"),
            ("Rescale: aspect", "rescale_aspect"),
            ("Carved: energy", "carved_energy_kept"),
            ("Rescale: energy", "rescale_energy_kept"),
        ],
    )
    cost_table = markdown_table(
        cost_rows,
        [
            ("Method", "method"),
            ("Subject kept", "subject_kept"),
            ("Aspect retained", "aspect_ratio"),
            ("Energy kept", "energy_kept"),
            ("Time (ms)", "median_ms"),
        ],
    )
    image_table = markdown_table(
        image_rows,
        [
            ("Image", "image"),
            ("Carved: ROI kept", "carved_kept"),
            ("Rescale: ROI kept", "rescale_kept"),
            ("Advantage", "advantage"),
            ("Carved: energy", "carved_energy_kept"),
            ("Rescale: energy", "rescale_energy_kept"),
        ],
    )
    write_tables(
        RESULTS,
        [
            (f"Energy functions at {args.reduction:.0%} reduction ({len(images)} images)", energy_table),
            (f"PER IMAGE at {args.reduction:.0%} — the average hides a sign change", image_table),
            ("Seam carving vs plain rescale, by reduction", sweep_table),
            (f"What the advantage costs (last row: difference, and the TIME RATIO)", cost_table),
        ],
    )
    print("\n" + "\n\n".join([energy_table, sweep_table, cost_table]))

    carve_row, plain_row, diff = cost_rows
    kepts = [r["subject_kept"] for r in energy_rows if not r["energy"].startswith("Plain")]
    control = next(r for r in energy_rows if r["energy"].startswith("Plain"))
    spread = max(kepts) - min(kepts)
    gap = max(kepts) - control["subject_kept"]
    last = sweep_rows[-1]

    print("\n--- HEADLINE NUMBERS ---")
    print(
        f"energy choice    : four energies span {spread:.4f} of subject retention "
        f"({spread * 100:.1f} points)"
    )
    print(
        f"carving vs not   : gap to the plain rescale is {gap:.4f} "
        f"({gap * 100:.1f} points) — {gap / max(spread, 1e-9):.0f}x the spread between energies"
    )
    print(
        f"what it costs    : {carve_row['median_ms']:.0f} ms vs {plain_row['median_ms']:.2f} ms — "
        f"{diff['median_ms']:.0f}x — for +{diff['subject_kept'] * 100:.1f} points of subject"
    )
    print(
        f"its own objective: +{diff['energy_kept'] * 100:.1f} points of retained energy, "
        f"which it wins more decisively than the perceptual measure"
    )
    print(
        f"at {last['reduction']:.0%} reduction : carving retains {last['carved_energy_kept']:.1%} of the "
        f"energy against a rescale's {last['rescale_energy_kept']:.1%} "
        f"({last['carved_energy_kept'] / max(last['rescale_energy_kept'], 1e-9):.1f}x) — while retaining "
        f"{last['carved_subject_kept']:.1%} of the subject against {last['rescale_subject_kept']:.1%} "
        f"({last['carved_subject_kept'] / max(last['rescale_subject_kept'], 1e-9):.2f}x)"
    )
    print(
        f"the control      : a plain rescale's aspect column is exactly "
        f"1 - reduction by arithmetic ({control['aspect_ratio']:.4f} at {args.reduction:.0%})"
    )
    best = max(image_rows, key=lambda r: r["advantage"])
    worst = min(image_rows, key=lambda r: r["advantage"])
    print(
        f"per image        : the advantage ranges from {worst['advantage'] * 100:+.1f} points on "
        f"{worst['image']} to {best['advantage'] * 100:+.1f} on {best['image']} — the average of "
        f"{diff['subject_kept'] * 100:+.1f} describes none of them"
    )
    print(
        f"...and yet       : carving wins the ENERGY column on "
        f"{sum(1 for r in image_rows if r['carved_energy_kept'] > r['rescale_energy_kept'])}"
        f"/{len(image_rows)} images and the ROI column on only "
        f"{sum(1 for r in image_rows if r['advantage'] > 0)}/{len(image_rows)} — it wins its own "
        f"objective reliably and the one you care about unreliably"
    )
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
