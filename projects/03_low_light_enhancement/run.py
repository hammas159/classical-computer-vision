"""Run the low-light enhancement experiment and write results + figures.

    python run.py [--gamma 3.0] [--images 6]

Every number published for this project comes out of this script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import numpy as np  # noqa: E402

from shared import figures, io, synth  # noqa: E402
from shared.io import to_gray  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import low_light as ll  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=3.0, help="darkness of the headline condition")
    ap.add_argument("--images", type=int, default=len(ll.IMAGES))
    ap.add_argument("--noise", type=float, default=4.0, help="read noise sigma, 0-255")
    args = ap.parse_args()

    images = ll.IMAGES[: args.images]
    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    print(f"Scoring {len(ll.METHODS)} methods + oracle on {len(images)} images at gamma {args.gamma} ...")
    method_rows, dark_stats = ll.evaluate_methods(
        gamma=args.gamma, noise_sigma=args.noise, images=images
    )

    print(f"Sweeping gamma over {len(ll.GAMMA_LEVELS)} levels ...")
    sweep_rows = ll.sweep_gamma(images=images, noise_sigma=args.noise)

    print("Measuring noise amplification ...")
    noise_rows = ll.evaluate_noise_amplification(
        gamma=args.gamma, noise_sigma=args.noise, images=images
    )

    # ------------------------------------------------------------------ #
    # figures
    # ------------------------------------------------------------------ #
    clean = io.sample("coffee")
    dark = synth.low_light(clean, gamma=args.gamma, noise_sigma=args.noise, seed=0)

    panels = [("Original (truth)", clean), (f"Darkened, gamma={args.gamma}", dark)]
    for name, fn in ll.METHODS.items():
        row = next(r for r in method_rows if r["method"] == name)
        panels.append((f"{name}\n{row['psnr_db']:.1f} dB · SSIM {row['ssim']:.2f}", fn(dark)))
    orow = next(r for r in method_rows if r["method"] == ll.ORACLE_NAME)
    panels.append(
        (
            f"{ll.ORACLE_NAME}\n{orow['psnr_db']:.1f} dB — THE CEILING",
            ll.enhance_oracle(dark, args.gamma),
        )
    )
    figures.grid(
        panels,
        IMAGES / "methods.png",
        ncols=5,
        suptitle=f"Seven classical methods and the oracle, on a scene darkened by gamma {args.gamma}",
    )

    # the ceiling: PSNR vs darkness, with the oracle as a dashed reference
    series = {name: [r[name] for r in sweep_rows] for name in ll.METHODS}
    series[ll.ORACLE_NAME] = [r[ll.ORACLE_NAME] for r in sweep_rows]
    figures.lines(
        [r["gamma"] for r in sweep_rows],
        series,
        IMAGES / "gamma_sweep.png",
        xlabel="gamma (higher = darker)",
        ylabel="PSNR vs the original (dB)",
        title="No method can cross the dashed line — that is what quantisation destroyed",
        dashed={ll.ORACLE_NAME},
    )

    # how many tone levels survive the darkening, computed with no images at all
    figures.lines(
        [r["gamma"] for r in sweep_rows],
        {"distinct tone levels left": [r["levels_left"] for r in sweep_rows]},
        IMAGES / "quantisation_ceiling.png",
        xlabel="gamma (higher = darker)",
        ylabel="distinct 8-bit levels surviving (of 256)",
        title="Darkening is not invertible: levels collapse before any method runs",
    )

    # ------------------------------------------------------------------ #
    # distributions and matrices
    # ------------------------------------------------------------------ #

    # 1. the histogram that shows the tone collapse and the comb it leaves
    figures.histogram(
        {
            "original": to_gray(clean),
            "darkened": to_gray(dark),
            "oracle (inverse gamma)": to_gray(ll.enhance_oracle(dark, args.gamma)),
        },
        IMAGES / "histogram_tone.png",
        bins=256,
        title=(
            f"Darkening crushes the tone range; inverting it stretches the survivors "
            f"back apart, leaving gaps (gamma {args.gamma})"
        ),
    )

    # 2. raw pixel values: the same patch before, after darkening, and restored
    h, w = to_gray(clean).shape
    y0, x0, size = h // 2, w // 2, 12
    g_clean = to_gray(clean)[y0 : y0 + size, x0 : x0 + size]
    g_dark = to_gray(dark)[y0 : y0 + size, x0 : x0 + size]
    g_oracle = to_gray(ll.enhance_oracle(dark, args.gamma))[y0 : y0 + size, x0 : x0 + size]
    lost = len(np.unique(g_clean)) - len(np.unique(g_dark))
    figures.value_matrix(
        [
            (f"Original\n{len(np.unique(g_clean))} distinct values", g_clean),
            (f"Darkened\n{len(np.unique(g_dark))} distinct values", g_dark),
            (f"Oracle restored\n{len(np.unique(g_oracle))} distinct values", g_oracle),
        ],
        IMAGES / "pixel_matrix.png",
        title=(
            f"One 12x12 patch. Darkening merged {lost} distinct grey levels into fewer; "
            "inverting cannot invent them back"
        ),
    )

    # 3. noise amplification, the hidden cost of brightening
    figures.metric_bars(
        [r["method"] for r in noise_rows],
        [r["amplification"] for r in noise_rows],
        IMAGES / "noise_amplification.png",
        ylabel="noise sigma after / before",
        title=f"Brightening multiplies the noise hiding in the shadows (gamma {args.gamma})",
        highlight_best="min",
    )

    # 4. the comparative matrix
    figures.comparison_matrix(
        method_rows,
        [
            ("PSNR (dB)", "psnr_db", True),
            ("SSIM", "ssim", True),
            ("PSNR matched", "psnr_matched_db", True),
            ("SSIM matched", "ssim_matched", True),
            ("Entropy", "entropy_bits", True),
            ("Noise sigma", "noise_sigma", False),
            ("Time (ms)", "median_ms", False),
        ],
        IMAGES / "method_matrix.png",
        title="Methods x metrics — the oracle row is the ceiling, not a competitor",
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS,
        "03_low_light_enhancement",
        {
            "gamma": args.gamma,
            "noise_sigma": args.noise,
            "images": list(images),
            "darkened_input": dark_stats,
            "levels_surviving_at_gamma": {
                str(g): ll.quantisation_ceiling(g) for g in ll.GAMMA_LEVELS
            },
            "methods": method_rows,
            "gamma_sweep": sweep_rows,
            "noise_amplification": noise_rows,
        },
    )

    method_table = markdown_table(
        method_rows,
        [
            ("Method", "method"),
            ("PSNR (dB)", "psnr_db"),
            ("SSIM", "ssim"),
            ("PSNR matched", "psnr_matched_db"),
            ("SSIM matched", "ssim_matched"),
            ("Entropy (bits)", "entropy_bits"),
            ("Noise sigma", "noise_sigma"),
            ("Time (ms)", "median_ms"),
        ],
    )
    sweep_table = markdown_table(
        sweep_rows,
        [("Gamma", "gamma"), ("Levels left", "levels_left")]
        + [(n, n) for n in ll.METHODS]
        + [(ll.ORACLE_NAME, ll.ORACLE_NAME)],
    )
    noise_table = markdown_table(
        noise_rows,
        [("Method", "method"), ("Noise after", "noise_after"), ("Amplification", "amplification")],
    )

    write_tables(
        RESULTS,
        [
            (f"Methods at gamma {args.gamma} ({len(images)} images)", method_table),
            ("PSNR vs darkness, with the oracle ceiling", sweep_table),
            (f"Noise amplification at gamma {args.gamma}", noise_table),
        ],
    )
    print("\n" + method_table + "\n\n" + sweep_table + "\n\n" + noise_table)

    oracle = next(r for r in method_rows if r["method"] == ll.ORACLE_NAME)
    real = [r for r in method_rows if r["method"] != ll.ORACLE_NAME]
    best = max(real, key=lambda r: r["psnr_db"])
    best_matched = max(real, key=lambda r: r["psnr_matched_db"])
    noisiest = max(noise_rows, key=lambda r: r["amplification"])

    print("\n--- HEADLINE NUMBERS ---")
    print(f"ceiling (oracle)   : {oracle['psnr_db']} dB, SSIM {oracle['ssim']} "
          f"— not infinite, because quantisation already destroyed levels")
    print(f"levels surviving   : {ll.quantisation_ceiling(args.gamma)} of 256 at gamma {args.gamma}")
    print(f"best real method   : {best['method']} @ {best['psnr_db']} dB "
          f"({oracle['psnr_db'] - best['psnr_db']:.2f} dB below the ceiling), "
          f"{best['median_ms']} ms")
    print(f"best after exposure match: {best_matched['method']} @ "
          f"{best_matched['psnr_matched_db']} dB")
    print(f"noisiest           : {noisiest['method']} amplifies noise {noisiest['amplification']}x")
    print(f"darkened input     : {dark_stats}")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
