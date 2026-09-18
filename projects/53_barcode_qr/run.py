"""Run the barcode/QR comparison and write results + figures.

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

import barcode as bc  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"

LOCATORS = ("Gradient + morphology", "Local variance", "QRCodeDetector")

CLUTTER_BANDS = [("plain", 0.0, 20.0), ("some", 20.0, 30.0),
                 ("striped", 30.0, 45.0), ("busy", 45.0, 101.0)]


def _drawn(img: np.ndarray, corners, colour) -> np.ndarray:
    out = ensure_rgb(img).copy()
    if corners is not None:
        cv2.polylines(out, [np.int32(corners).reshape(-1, 1, 2)], True, colour, 3)
    return out


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    if not bc.barcode_decoder_available():
        print("WARNING: the 1-D barcode decoder is unavailable; that table will be empty")

    print("Generated clutter against real photographs ...")
    background = bc.photo_versus_generated_background()
    for r in background:
        print(f"  {r['background']:20s} " + "  ".join(
            f"{k.split()[0][:8]}={r[k]:.3f}" for k in LOCATORS)
            + f"  decoded={r['decoded']:.3f}")

    print(f"\nPer photograph, ordered by how barcode-like the background is ...")
    per_photo = bc.evaluate_on_photographs()
    for r in per_photo:
        print(f"  {r['image']:22s} {r['barcode_like']:5.1f}%  " + "  ".join(
            f"{k.split()[0][:8]}={r[k]:.3f}" for k in LOCATORS)
            + f"  decoded={r['decoded']:.3f}")

    blur = bc.sweep_blur()
    rotation = bc.sweep_rotation()
    scale = bc.sweep_scale()
    perspective = bc.sweep_perspective()
    noise = bc.sweep_noise()
    barcode_1d = bc.evaluate_barcode_1d()

    print("\nWhere localisation and decoding part company ...")
    for label, rows, key in (("blur", blur, "blur_sigma"),
                             ("scale", scale, "scale"),
                             ("perspective", perspective, "perspective"),
                             ("noise", noise, "noise_sigma")):
        first_gap = next((r for r in rows
                          if r["best_found_rate"] - r["decode_rate"] >= 0.5), None)
        if first_gap:
            print(f"  {label:12s} at {key}={first_gap[key]:g}: found "
                  f"{first_gap['best_found_rate']:.3f}, decoded "
                  f"{first_gap['decode_rate']:.3f}")

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    candidates: dict[str, list[dict]] = {}
    by_image = {r["image"]: r for r in per_photo}
    for name in bc.IMAGES:
        share = by_image[name]["barcode_like"]
        band = next(b for b, lo, hi in CLUTTER_BANDS if lo <= share < hi)
        code = bc.make_qr("CLASSICAL")
        img, truth = bc.place_on_background(code, seed=0, background=name)

        panels = [_drawn(img, truth, (60, 200, 60))]
        notes = [f"barcode-like {share:.0f}% · truth in green"]
        scores = []
        for locator in LOCATORS:
            fn = {"Gradient + morphology": bc.locate_gradient_morphology,
                  "Local variance": bc.locate_variance,
                  "QRCodeDetector": bc.locate_qr_detector}[locator]
            corners = fn(img)
            value = bc.localisation_iou(corners, truth, img.shape)
            panels.append(_drawn(img, corners,
                                 (60, 200, 60) if value >= bc.FOUND_IOU else (220, 60, 60)))
            notes.append(f"IoU {value:.3f}" + ("" if value >= bc.FOUND_IOU else "  MISS"))
            scores.append(value)

        decoded = bc.decode_qr(img)
        panels.append(_drawn(img, truth, (60, 60, 220)))
        notes.append(f"decoded: {decoded or 'nothing'}")
        scores.append(1.0 if decoded == "CLASSICAL" else 0.0)

        candidates.setdefault(band, []).append({
            "label": f"{name.replace('_', ' ')}\nbarcode-like {share:.0f}%",
            "subject": name, "share": share, "images": panels,
            "notes": notes, "scores": scores,
            "spread": float(np.std(scores[:3])),
        })
        print(f"  scene candidate {name:22s} {share:5.1f}% [{band:8s}]  "
              + " ".join(f"{v:.2f}" for v in scores))

    chosen, used = [], set()
    for band, _, _ in CLUTTER_BANDS:
        pool = sorted(candidates.get(band, []), key=lambda r: -r["share"])
        pick = next((r for r in pool if r["subject"] not in used), None)
        if pick:
            used.add(pick["subject"])
            chosen.append(pick)
    chosen = chosen[:4]
    assert len({r["subject"] for r in chosen}) == len(chosen), "a subject repeats"

    figures.gallery(
        ["the scene"] + list(LOCATORS) + ["decoded?"],
        [(r["label"], r["images"]) for r in chosen],
        IMAGES / "compare_localisation.png",
        cell_notes=[r["notes"] for r in chosen],
        suptitle=(
            "One QR code on four real backgrounds. Green is a hit, red a miss; cells are "
            "localisation IoU. The last column is the only objective answer in the "
            "project — the payload either comes back or it does not."
        ),
    )
    gallery_table = markdown_table(
        [dict([("Sr", i), ("Background", r["label"].replace("\n", " · "))]
              + [(m, f"{v:.3f}") for m, v in zip(LOCATORS, r["scores"])]
              + [("Decoded", "yes" if r["scores"][3] else "no")])
         for i, r in enumerate(chosen, start=1)],
        [("Sr", "Sr"), ("Background", "Background")] + [(m, m) for m in LOCATORS]
        + [("Decoded", "Decoded")],
    )
    print("\n--- localisation IoU on real backgrounds ---\n" + gallery_table)

    # ------------------------------------------------------------------ #
    # the signature figure: found is not decoded
    # ------------------------------------------------------------------ #
    figures.lines(
        [r["blur_sigma"] for r in blur],
        {"best localiser found it": [r["best_found_rate"] for r in blur],
         "actually decoded": [r["decode_rate"] for r in blur],
         "the gap": [r["gap"] for r in blur]},
        IMAGES / "found_vs_decoded_blur.png",
        xlabel="blur sigma", ylabel="rate",
        title="Blur: still located at 100%, decodable at 0%",
    )

    figures.lines(
        [r["approx_px_per_module"] for r in scale],
        {"best localiser found it": [r["best_found_rate"] for r in scale],
         "actually decoded": [r["decode_rate"] for r in scale]},
        IMAGES / "found_vs_decoded_scale.png",
        xlabel="pixels per QR module", ylabel="rate", invert_x=True,
        title="Resolution: the sharpest version of the same gap",
    )

    figures.lines(
        [r["perspective"] for r in perspective],
        {"best localiser found it": [r["best_found_rate"] for r in perspective],
         "actually decoded": [r["decode_rate"] for r in perspective]},
        IMAGES / "found_vs_decoded_perspective.png",
        xlabel="perspective jitter", ylabel="rate",
        title="Perspective: the one degradation where decoding degrades gradually",
    )

    figures.lines(
        [r["noise_sigma"] for r in noise],
        {"best localiser found it": [r["best_found_rate"] for r in noise],
         "actually decoded": [r["decode_rate"] for r in noise]},
        IMAGES / "found_vs_decoded_noise.png",
        xlabel="noise sigma", ylabel="rate",
        title="Noise: located throughout, unreadable past sigma 50",
    )

    figures.metric_bars(
        [f"{r['background']}\n{name.split()[0]}"
         for r in background for name in LOCATORS],
        [r[name] for r in background for name in LOCATORS],
        IMAGES / "background_matters.png",
        ylabel="localisation rate",
        title=("Generated clutter against real photographs. The gradient localiser "
               "scores 1.000 on one and 0.021 on the other."),
    )

    figures.lines(
        [r["barcode_like"] for r in per_photo],
        {name: [r[name] for r in per_photo] for name in LOCATORS},
        IMAGES / "clutter_sweep.png",
        xlabel="how barcode-like the background is (% of frame responding)",
        ylabel="localisation rate",
        title="Localisation against background clutter, on twelve photographs",
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS, "53_barcode_qr",
        {
            "images": list(bc.IMAGES),
            "barcode_like_share": {r["image"]: r["barcode_like"] for r in per_photo},
            "payloads": list(bc.PAYLOADS),
            "background_comparison": background,
            "per_photograph": per_photo,
            "blur": blur, "rotation": rotation, "scale": scale,
            "perspective": perspective, "noise": noise,
            "barcode_1d": barcode_1d,
            "found_iou_threshold": bc.FOUND_IOU,
        },
    )

    background_table = markdown_table(
        background, [("Background", "background")] + [(m, m) for m in LOCATORS]
        + [("Decoded", "decoded")])
    per_photo_table = markdown_table(
        per_photo, [("Photograph", "image"), ("Barcode-like %", "barcode_like")]
        + [(m, m) for m in LOCATORS] + [("Decoded", "decoded")])
    blur_table = markdown_table(
        blur, [("Blur sigma", "blur_sigma"), ("Found", "best_found_rate"),
               ("Decoded", "decode_rate"), ("Gap", "gap")])
    scale_table = markdown_table(
        scale, [("Scale", "scale"), ("Px per module", "approx_px_per_module"),
                ("Found", "best_found_rate"), ("Decoded", "decode_rate")])

    write_tables(
        RESULTS,
        [
            ("Generated clutter against real photographs", background_table),
            ("Per photograph", per_photo_table),
            ("Blur", blur_table),
            ("Resolution", scale_table),
            ("Four backgrounds down the rows", gallery_table),
        ],
    )
    print("\n" + background_table)

    # ------------------------------------------------------------------ #
    # headlines
    # ------------------------------------------------------------------ #
    print("\n--- HEADLINE NUMBERS ---")
    generated = next(r for r in background if r["background"] == "Generated clutter")
    photographs = next(r for r in background if r["background"] == "Photographs")

    print("the generated background was flattering one of the three localisers:")
    for name in LOCATORS:
        ratio = (generated[name] / photographs[name]) if photographs[name] else float("inf")
        print(f"  {name:24s} generated {generated[name]:.3f}, photographs "
              f"{photographs[name]:.3f}"
              + (f"  ({ratio:.0f}x)" if np.isfinite(ratio) and ratio > 1.5 else ""))
    print("  random rectangles have no repeating vertical structure, and a barcode "
          "localiser looks for exactly that — so a photograph of zebras, saguaro ribs "
          "or coiled rope produces false positives the synthetic scene never will")

    print(f"\nand decoding does not care about the background at all: "
          f"{photographs['decoded']:.3f} on photographs, "
          f"{generated['decoded']:.3f} on generated clutter")

    print("\nthe gap between FOUND and DECODED, which is the project's point:")
    for label, rows, key, unit in (("blur", blur, "blur_sigma", "sigma"),
                                   ("resolution", scale, "approx_px_per_module", "px/module"),
                                   ("perspective", perspective, "perspective", ""),
                                   ("noise", noise, "noise_sigma", "sigma")):
        collapsed = next((r for r in rows if r["decode_rate"] <= 0.25), None)
        if collapsed:
            print(f"  {label:12s} at {key}={collapsed[key]:g} {unit}: located "
                  f"{collapsed['best_found_rate']:.3f}, decoded "
                  f"{collapsed['decode_rate']:.3f}")
    print("  a detection rate overstates a pipeline by exactly this much")

    readable = [r for r in scale if r["decode_rate"] >= 0.99]
    if readable:
        limit = min(r["approx_px_per_module"] for r in readable)
        print(f"\nthe resolution limit is between {limit:.1f} and "
              f"{max(r['approx_px_per_module'] for r in scale if r['decode_rate'] < 0.5):.1f} "
              "pixels per module — below it the code is still found every time and read none")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
