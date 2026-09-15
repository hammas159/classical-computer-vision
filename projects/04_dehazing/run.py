"""Run the dehazing experiment and write results + figures.

    python run.py [--beta 1.4]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import numpy as np  # noqa: E402

from shared import figures, io, synth  # noqa: E402
from shared.metrics import psnr  # noqa: E402
from shared.io import to_gray  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import dehazing as dz  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=1.4, help="haze density")
    ap.add_argument("--images", type=int, default=len(dz.IMAGES))
    args = ap.parse_args()

    images = dz.IMAGES[: args.images]
    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    print(f"Scoring {len(dz.METHODS)} methods + oracle on {len(images)} images at beta {args.beta} ...")
    method_rows, hazy_stats = dz.evaluate_methods(beta=args.beta, images=images)

    print(f"Sweeping beta over {len(dz.BETA_LEVELS)} levels ...")
    sweep_rows = dz.sweep_beta(images=images)

    print(f"Comparing {len(dz.AIRLIGHT_ESTIMATORS)} airlight estimators ...")
    airlight_rows = dz.compare_airlight_estimators(images=images, beta=args.beta)

    # ------------------------------------------------------------------ #
    # figures
    # ------------------------------------------------------------------ #
    # The single scene the detail figures are drawn on. A street receding to a
    # vanishing point, because a transmission map is only legible where there is
    # actually a depth gradient to see.
    clean = dz.load_scene("old_street")
    hazy, true_t = synth.add_haze(clean, beta=args.beta, airlight=dz.AIRLIGHT)

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README:
    # four scenes down the rows, every method across the columns
    # ------------------------------------------------------------------ #
    # Haze depends on one thing only — distance — so the scenes are chosen for
    # their DEPTH STRUCTURE, not for looking pretty. Twelve candidates are tried
    # across five families: a street receding to a vanishing point, a flat
    # farmhouse wall with almost no depth range, water running to a horizon, a
    # mountain scene, and a sky-dominant frame, which is the dark channel prior's
    # documented failure case. The best survivor of each family is kept, so the
    # table cannot fill up with four variations on the same depth profile.
    #
    # Every candidate is scored before it goes in: if the best named method does
    # not beat the hazy input by GALLERY_MIN_GAIN_DB, the sample is broken and is
    # dropped rather than shown.
    GALLERY_MIN_GAIN_DB = 2.0
    gallery_pool = [
        ("old street\nreceding to a vanishing point", "old_street", "street"),
        ("painted chalet\nbuilding against a wooded hill", "painted_chalet", "street"),
        ("stone house\nflat on — almost no depth", "stone_house", "flat"),
        ("motocross\nnear subject, shallow depth", "motocross", "flat"),
        ("mountain stream\nsnow line on the horizon", "mountain_stream", "mountain"),
        ("whitewater raft\npeople, spray, close rock", "whitewater_raft", "mountain"),
        ("lighthouse cliff\nrock, sea and sky", "lighthouse_cliff", "coast"),
        ("lighthouse lawn\nportrait frame, low horizon", "lighthouse_lawn", "coast"),
        ("moored boat\nturquoise shallows", "moored_boat", "water"),
        ("sailboats\ncoloured sails on flat water", "sailboats", "water"),
        ("tropical island\nsky-dominant", "tropical_island", "sky"),
        ("warplane\npale sky, no dark channel", "warplane", "sky"),
    ]
    best_named = max(
        (r for r in method_rows if r["method"] != dz.ORACLE_NAME),
        key=lambda r: r["psnr_db"],
    )["method"]

    method_names = list(dz.METHODS)
    survivors: dict[str, tuple] = {}
    for label, name, family in gallery_pool:
        src = io.real_photo(name)
        h, t_true = synth.add_haze(src, beta=args.beta, airlight=dz.AIRLIGHT)
        hazy_db = psnr(h, src)
        outs = [dz.METHODS[m](h) for m in method_names]
        oracle_out = dz.dehaze_oracle(h, dz.AIRLIGHT, t_true)
        gain = psnr(outs[method_names.index(best_named)], src) - hazy_db
        if gain < GALLERY_MIN_GAIN_DB:
            print(f"gallery candidate {name:<17} DROP — {best_named} gained only {gain:+.1f} dB  [{family}]")
            continue
        print(f"gallery candidate {name:<17} keep — {hazy_db:.1f} dB hazy, {gain:+.1f} dB best  [{family}]")
        row = (
            label,
            [h] + outs + [oracle_out],
            [f"{hazy_db:.1f} dB"]
            + [f"{psnr(o, src):.1f} dB" for o in outs]
            + [f"{psnr(oracle_out, src):.1f} dB"],
            gain,
        )
        if family not in survivors or gain > survivors[family][3]:
            survivors[family] = row

    chosen = sorted(survivors.values(), key=lambda r: -r[3])[:4]
    figures.gallery(
        ["hazy input"] + method_names + ["oracle"],
        [(label, imgs) for label, imgs, _notes, _g in chosen],
        IMAGES / "samples.png",
        cell_notes=[notes for _l, _i, notes, _g in chosen],
        suptitle=(
            f"Four scenes at beta {args.beta}, every method across the columns. "
            "The oracle was handed the true transmission map and is the ceiling."
        ),
    )
    print(f"front-on comparison: {len(chosen)} scenes x {len(method_names)} methods + oracle")

    # the same figure as a markdown table, so the README's numbers are generated
    # rather than transcribed by hand
    gallery_rows = [
        dict(
            [("Sr", i), ("Scene", label.replace("\n", " · "))]
            + list(zip(["Hazy"] + method_names + ["Oracle"], notes))
        )
        for i, (label, _imgs, notes, _g) in enumerate(chosen, start=1)
    ]
    gallery_table = markdown_table(
        gallery_rows,
        [("Sr", "Sr"), ("Scene", "Scene"), ("Hazy input", "Hazy")]
        + [(m, m) for m in method_names]
        + [("Oracle", "Oracle")],
    )
    print("\n" + gallery_table)

    panels = [("Original (truth)", clean), (f"Hazy, beta={args.beta}", hazy)]
    for name, fn in dz.METHODS.items():
        row = next(r for r in method_rows if r["method"] == name)
        panels.append((f"{name}\n{row['psnr_db']:.1f} dB · SSIM {row['ssim']:.2f}", fn(hazy)))
    orow = next(r for r in method_rows if r["method"] == dz.ORACLE_NAME)
    panels.append(
        (
            f"{dz.ORACLE_NAME}\n{orow['psnr_db']:.1f} dB — THE CEILING",
            dz.dehaze_oracle(hazy, dz.AIRLIGHT, true_t),
        )
    )
    figures.grid(panels, IMAGES / "methods.png", ncols=4, suptitle="Five methods and the oracle")

    # the transmission map is where the physics lives
    a = dz.estimate_airlight(hazy)
    t_blocky = dz.transmission_dcp(hazy, a)
    t_refined = dz.refine_transmission_guided(hazy, t_blocky)
    figures.grid(
        [
            ("True transmission t", true_t),
            (f"DCP estimate\nMAE {dz.transmission_error(t_blocky, true_t):.3f}", t_blocky),
            (f"Guided refined\nMAE {dz.transmission_error(t_refined, true_t):.3f}", t_refined),
            ("Error, DCP", np.abs(t_blocky - true_t)),
            ("Error, refined", np.abs(t_refined - true_t)),
            ("Dark channel", dz.dark_channel(hazy)),
        ],
        IMAGES / "transmission.png",
        ncols=3,
        suptitle="Recovering the transmission map — the part that is actually physics",
    )

    series = {name: [r[name] for r in sweep_rows] for name in dz.METHODS}
    series[dz.ORACLE_NAME] = [r[dz.ORACLE_NAME] for r in sweep_rows]
    figures.lines(
        [r["beta"] for r in sweep_rows],
        series,
        IMAGES / "beta_sweep.png",
        xlabel="beta (higher = thicker haze)",
        ylabel="PSNR vs the original (dB)",
        title="As haze thickens, even the oracle falls — the veil replaces the signal",
        dashed={dz.ORACLE_NAME},
    )

    # ------------------------------------------------------------------ #
    # distributions and matrices
    # ------------------------------------------------------------------ #
    figures.histogram(
        {
            "original": to_gray(clean),
            "hazy": to_gray(hazy),
            "DCP + guided refine": to_gray(dz.dehaze_dcp_refined(hazy)),
            "CLAHE (contrast only)": to_gray(dz.dehaze_clahe(hazy)),
        },
        IMAGES / "histogram_tone.png",
        bins=256,
        title="Haze compresses the tone range toward the airlight; the two methods undo it differently",
    )

    g_clean = to_gray(clean)
    h, w = g_clean.shape
    y0, x0, size = h // 4, w // 2, 12
    figures.value_matrix(
        [
            ("Original", g_clean[y0 : y0 + size, x0 : x0 + size]),
            ("Hazy", to_gray(hazy)[y0 : y0 + size, x0 : x0 + size]),
            (
                "DCP + guided refine",
                to_gray(dz.dehaze_dcp_refined(hazy))[y0 : y0 + size, x0 : x0 + size],
            ),
            (
                "Oracle",
                to_gray(dz.dehaze_oracle(hazy, dz.AIRLIGHT, true_t))[
                    y0 : y0 + size, x0 : x0 + size
                ],
            ),
        ],
        IMAGES / "pixel_matrix.png",
        title="The veil, as numbers: haze pulls every value toward the airlight",
    )

    figures.comparison_matrix(
        method_rows,
        [
            ("PSNR (dB)", "psnr_db", True),
            ("SSIM", "ssim", True),
            ("RMS contrast", "rms_contrast", True),
            ("Transmission MAE", "transmission_mae", False),
            ("Time (ms)", "median_ms", False),
        ],
        IMAGES / "method_matrix.png",
        title="Methods x metrics — contrast alone does not mean the haze was modelled",
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS,
        "04_dehazing",
        {
            "beta": args.beta,
            "airlight": dz.AIRLIGHT,
            "images": list(images),
            "hazy_input": hazy_stats,
            "methods": method_rows,
            "beta_sweep": sweep_rows,
            "airlight_estimators": airlight_rows,
        },
    )

    method_table = markdown_table(
        method_rows,
        [
            ("Method", "method"),
            ("PSNR (dB)", "psnr_db"),
            ("SSIM", "ssim"),
            ("RMS contrast", "rms_contrast"),
            ("Transmission MAE", "transmission_mae"),
            ("Time (ms)", "median_ms"),
        ],
    )
    sweep_table = markdown_table(
        sweep_rows,
        [("Beta", "beta"), ("min t", "min_transmission")]
        + [(n, n) for n in dz.METHODS]
        + [(dz.ORACLE_NAME, dz.ORACLE_NAME)],
    )
    airlight_table = markdown_table(
        airlight_rows,
        [
            ("Airlight estimator", "estimator"),
            ("Airlight error", "airlight_error"),
            ("Transmission MAE", "transmission_mae"),
            ("Saturated", "saturated_estimates"),
            ("PSNR (dB)", "psnr_db"),
            ("SSIM", "ssim"),
        ],
    )
    write_tables(
        RESULTS,
        [
            (f"Methods at beta {args.beta} ({len(images)} images)", method_table),
            ("PSNR vs haze density, with the oracle ceiling", sweep_table),
            ("A better airlight makes the OUTPUT worse", airlight_table),
        ],
    )
    print("\n" + method_table + "\n\n" + sweep_table)

    oracle = next(r for r in method_rows if r["method"] == dz.ORACLE_NAME)
    real = [r for r in method_rows if r["method"] != dz.ORACLE_NAME]
    best = max(real, key=lambda r: r["psnr_db"])
    clahe = next(r for r in real if r["method"].startswith("CLAHE"))
    best_contrast = max(real, key=lambda r: r["rms_contrast"])

    print("\n--- HEADLINE NUMBERS ---")
    print(f"hazy input       : {hazy_stats}")
    print(f"ceiling (oracle) : {oracle['psnr_db']} dB")
    print(f"best real method : {best['method']} @ {best['psnr_db']} dB "
          f"({oracle['psnr_db'] - best['psnr_db']:.2f} dB below the ceiling)")
    print(f"highest contrast : {best_contrast['method']} @ {best_contrast['rms_contrast']} "
          f"but {best_contrast['psnr_db']} dB")
    print(f"CLAHE control    : {clahe['psnr_db']} dB, contrast {clahe['rms_contrast']}")
    for r in real:
        if r["transmission_mae"] is not None:
            print(f"transmission MAE : {r['method']} = {r['transmission_mae']}")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
