"""Run the denoising shootout and write results + figures.

    python run.py [--retune]

`--retune` re-runs the parameter grid search and prints a TUNED table to paste
back into `src/denoising.py`. Without it the stored tuned parameters are used,
which keeps a normal run to a couple of minutes.
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
from shared.metrics import psnr  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import denoising as dn  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--retune", action="store_true", help="re-run the parameter grid search")
    ap.add_argument("--images", type=int, default=len(dn.IMAGES))
    args = ap.parse_args()

    images = dn.IMAGES[: args.images]
    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    # Four photographs down the rows, every filter across. The candidates are
    # ordered by DETAIL DENSITY -- mean |Laplacian| -- because that is exactly
    # the quantity a denoiser destroys, and a pool of equally detailed images
    # cannot show the trade any of them makes.
    GALLERY_MIN_GAIN_DB = 2.0
    detail_pool = [
        ("albatross pair\nsmooth plumage · detail 7", "albatross_pair", "smooth"),
        ("bear in grass\nsoft, low contrast · detail 7", "bear_grass", "smooth"),
        ("bear on a riverbank\nsoft · detail 8", "bear_riverbank", "smooth"),
        ("lionesses\ndry grass · detail 9", "lionesses", "light texture"),
        ("elk in water\nripples · detail 11", "elk_water", "light texture"),
        ("lions on a plain\nopen grass · detail 15", "lions_plain", "light texture"),
        ("deer in scrub\nwinter branches · detail 17", "deer_water", "medium texture"),
        ("iguana in surf\nspray and weed · detail 21", "iguana_surf", "medium texture"),
        ("rhino on gravel\nroad and hide · detail 23", "rhino_road", "medium texture"),
        ("bear against bark\ntree bark · detail 35", "bear_tree_bark", "heavy texture"),
        ("carved stone in leaves\nstone and foliage · detail 45", "stone_face_leaves", "heavy texture"),
    ]
    method_names = [m for m in dn.METHODS if m != "Do nothing (control)"]
    all_columns = list(dn.METHODS)
    survivors: dict[str, dict] = {}
    for i, (label, name, family) in enumerate(detail_pool):
        clean = dn.load_scene(name)
        noisy = dn.make_noisy(clean, "gaussian", 25.0, seed=i)
        base = psnr(noisy, clean)
        outs = [dn.METHODS[m](noisy) for m in all_columns]
        scores = [psnr(o, clean) for o in outs]
        gain = max(scores) - base
        flat = label.replace("\n", " · ")
        if gain < GALLERY_MIN_GAIN_DB:
            print(f"detail candidate {flat:<48} DROP — best gained {gain:+.1f} dB  [{family}]")
            continue
        print(f"detail candidate {flat:<48} keep — noisy {base:.1f} dB, "
              f"best {max(scores):.1f} dB ({gain:+.1f})  [{family}]")
        row = {
            "label": label,
            "images": [clean, noisy] + outs,
            "notes": ["clean", f"{base:.1f} dB"] + [f"{s:.1f} dB" for s in scores],
            "score": gain,
        }
        if family not in survivors or row["score"] > survivors[family]["score"]:
            survivors[family] = row

    # one row per detail band, in order, so the figure spans the axis rather
    # than showing the four images the filters happen to do best on
    BAND_ORDER = ["smooth", "light texture", "medium texture", "heavy texture"]
    chosen = [survivors[b] for b in BAND_ORDER if b in survivors][:4]
    figures.gallery(
        ["clean", "noisy (sigma 25)"] + all_columns,
        [(r["label"], r["images"]) for r in chosen],
        IMAGES / "compare_filters.png",
        cell_notes=[r["notes"] for r in chosen],
        suptitle=(
            "Four photographs of increasing detail, Gaussian noise at sigma 25, "
            "every filter across the columns. Cells are PSNR against the clean image."
        ),
    )
    gallery_table = markdown_table(
        [
            dict([("Sr", i), ("Photograph", r["label"].replace("\n", " · ")),
                  ("Noisy", r["notes"][1])]
                 + list(zip(all_columns, r["notes"][2:])))
            for i, r in enumerate(chosen, start=1)
        ],
        [("Sr", "Sr"), ("Photograph", "Photograph"), ("Noisy", "Noisy")]
        + [(m, m) for m in all_columns],
    )
    print(f"\nfront-on comparison: {len(chosen)} photographs x {len(all_columns)} filters")
    print("\n--- filters ---\n" + gallery_table)


    print(f"Scoring {len(dn.METHODS)} filters at default parameters ...")
    default_rows, noisy_stats = dn.evaluate_methods(images=images, runs=3)

    print("Scoring every filter at its tuned parameter, on every noise type ...")
    tuned_rows = dn.compare_noise_types_tuned(images=images)

    print("Measuring what a default parameter costs ...")
    cost_rows = dn.default_vs_tuned(images=images)

    print("Held-out transfer check ...")
    transfer_rows = dn.transfer_check()

    print("Sweeping the noise level ...")
    sweep_rows = dn.sweep_level(images=images)

    tune_rows = []
    if args.retune:
        print("Re-tuning (slow) ...")
        for kind, level, label in dn.NOISE_TYPES:
            for r in dn.tune_parameter(images=images, kind=kind, level=level):
                tune_rows.append({**r, "noise": label, "kind": kind})

    # ------------------------------------------------------------------ #
    # figures
    # ------------------------------------------------------------------ #
    clean = io.sample("astronaut")

    for kind, level, label in dn.NOISE_TYPES:
        noisy = dn.make_noisy(clean, kind, level, seed=0)
        row = next(r for r in tuned_rows if r["noise"] == label)
        panels = [("Clean (truth)", clean), (f"{label}\n{row['Do nothing (control)']:.1f} dB", noisy)]
        for method in dn.TUNED[kind]:
            panels.append((f"{method}\n{row[method]:.2f} dB", dn.tuned_call(method, noisy, kind)))
        figures.grid(
            panels,
            IMAGES / f"methods_{kind}.png",
            ncols=4,
            suptitle=f"{label} — every filter at its own measured-best parameter",
        )

    # the role reversal, side by side
    reversal = []
    for kind, level, label in (dn.NOISE_TYPES[0], dn.NOISE_TYPES[1]):
        noisy = dn.make_noisy(clean, kind, level, seed=0)
        row = next(r for r in tuned_rows if r["noise"] == label)
        reversal.append((f"{label}\ninput {row['Do nothing (control)']:.1f} dB", noisy))
        reversal.append((f"Median  {row['Median']:.2f} dB", dn.tuned_call("Median", noisy, kind)))
        reversal.append((f"Bilateral  {row['Bilateral']:.2f} dB", dn.tuned_call("Bilateral", noisy, kind)))
    figures.grid(
        reversal,
        IMAGES / "role_reversal.png",
        ncols=3,
        suptitle="The same two filters, two noise types — the winner and the loser swap places",
    )

    method_names = [m for m in dn.METHODS if not m.startswith("Do nothing")]
    figures.lines(
        [r["level"] for r in sweep_rows],
        {"Noisy input (do nothing)": [r["noisy_input"] for r in sweep_rows]}
        | {m: [r[m] for r in sweep_rows] for m in method_names},
        IMAGES / "level_sweep.png",
        xlabel="Gaussian noise sigma",
        ylabel="PSNR (dB)",
        title="Below about sigma 10 the filters' blurring costs more than the noise does",
    )

    figures.metric_bars(
        [r["method"] for r in transfer_rows],
        [r["transfer_gain_db"] for r in transfer_rows],
        IMAGES / "transfer.png",
        ylabel="dB gained on HELD-OUT images by tuning",
        title="Tuning transfers for one filter of six",
    )

    figures.histogram(
        {
            "clean": to_gray(clean).ravel(),
            "Gaussian sigma=25": to_gray(dn.make_noisy(clean, "gaussian", 25.0)).ravel(),
            "Salt & pepper 6%": to_gray(dn.make_noisy(clean, "salt_pepper", 0.06)).ravel(),
            "Poisson lambda=30": to_gray(dn.make_noisy(clean, "poisson", 30.0)).ravel(),
        },
        IMAGES / "noise_histograms.png",
        bins=128,
        title="Three noise models, three different shapes — which is why one filter cannot win all three",
    )

    figures.comparison_matrix(
        default_rows,
        [
            ("PSNR (dB)", "psnr_db", True),
            ("SSIM", "ssim", True),
            ("Time (ms)", "median_ms", False),
        ],
        IMAGES / "method_matrix.png",
        title="Gaussian sigma=25 at DEFAULT parameters — note where the control lands",
    )

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    payload = {
        "images": list(images),
        "noise_types": [{"kind": k, "level": lv, "label": lb} for k, lv, lb in dn.NOISE_TYPES],
        "noisy_input": noisy_stats,
        "defaults": default_rows,
        "tuned_by_noise": tuned_rows,
        "cost_of_default": cost_rows,
        "transfer_check": transfer_rows,
        "level_sweep": sweep_rows,
        "tuned_parameters": {k: {m: list(v) for m, v in d.items()} for k, d in dn.TUNED.items()},
    }
    if tune_rows:
        payload["retuned"] = tune_rows
    results_path = write_results(RESULTS, "13_denoising_shootout", payload)

    default_table = markdown_table(
        default_rows,
        [("Filter", "method"), ("PSNR (dB)", "psnr_db"), ("SSIM", "ssim"), ("Time (ms)", "median_ms")],
    )
    tuned_table = markdown_table(
        tuned_rows,
        [("Noise", "noise")] + [(m, m) for m in dn.TUNED["gaussian"]] + [("Do nothing", "Do nothing (control)")],
    )
    cost_table = markdown_table(
        cost_rows,
        [
            ("Filter", "method"),
            ("Default (dB)", "default_psnr_db"),
            ("Tuned (dB)", "tuned_psnr_db"),
            ("Gain (dB)", "cost_of_default_db"),
        ],
    )
    transfer_table = markdown_table(
        transfer_rows,
        [
            ("Filter", "method"),
            ("Fitted on 3 images", "fitted_param"),
            ("Train (dB)", "train_psnr_db"),
            ("Held-out (dB)", "test_psnr_db"),
            ("Held-out at default", "test_default_db"),
            ("Transfer gain (dB)", "transfer_gain_db"),
        ],
    )
    sweep_table = markdown_table(
        sweep_rows,
        [("Sigma", "level"), ("Noisy input", "noisy_input")] + [(m, m) for m in method_names],
    )
    tables = [
        ("Gaussian sigma=25 at DEFAULT parameters", default_table),
        ("EVERY FILTER x EVERY NOISE, each at its own tuned parameter (PSNR dB)", tuned_table),
        ("What a default parameter costs, on Gaussian sigma=25", cost_table),
        ("HELD-OUT transfer: tuned on 3 images, scored on 3 others", transfer_table),
        ("PSNR vs Gaussian noise level", sweep_table),
    ]
    if tune_rows:
        tables.append(
            (
                "Re-tuned parameters (paste into TUNED)",
                markdown_table(
                    tune_rows,
                    [("Noise", "noise"), ("Filter", "method"), ("Parameter", "parameter"),
                     ("Best value", "best_value"), ("Best PSNR (dB)", "best_psnr_db")],
                ),
            )
        )
    write_tables(RESULTS, tables)
    print("\n" + "\n\n".join(t for _, t in tables))

    print("\n--- HEADLINE NUMBERS ---")
    for row in tuned_rows:
        filters = {k: v for k, v in row.items() if k not in ("noise", "Do nothing (control)")}
        best = max(filters, key=filters.get)
        worst = min(filters, key=filters.get)
        runner = sorted(filters.values(), reverse=True)[1]
        print(
            f"{row['noise']:<22} best {best} {filters[best]:.3f} dB "
            f"(+{filters[best] - runner:.3f} over the runner-up), "
            f"worst {worst} {filters[worst]:.3f}, input {row['Do nothing (control)']:.3f}"
        )
    winners = set()
    for row in tuned_rows:
        filters = {k: v for k, v in row.items() if k not in ("noise", "Do nothing (control)")}
        winners.add(max(filters, key=filters.get))
    print(f"\ndistinct winners : {len(winners)} of {len(tuned_rows)} noise types — {', '.join(sorted(winners))}")

    transfers = [r for r in transfer_rows if r["transfer_gain_db"] > 0.5]
    print(
        f"tuning transfers : for {len(transfers)} filter(s) of {len(transfer_rows)} — "
        + ", ".join(f"{r['method']} {r['transfer_gain_db']:+.2f} dB" for r in transfers)
    )
    print(
        "worst transfer   : "
        + ", ".join(
            f"{r['method']} {r['transfer_gain_db']:+.2f} dB"
            for r in sorted(transfer_rows, key=lambda r: r["transfer_gain_db"])[:2]
        )
        + "  (a fitted parameter that LOSES on held-out images)"
    )

    fastest = min((r for r in default_rows if not r["method"].startswith("Do nothing")),
                  key=lambda r: r["median_ms"])
    slowest = max(default_rows, key=lambda r: r["median_ms"])
    print(
        f"cost spread      : {slowest['method']} {slowest['median_ms']:.1f} ms vs "
        f"{fastest['method']} {fastest['median_ms']:.2f} ms — "
        f"{slowest['median_ms'] / max(fastest['median_ms'], 1e-9):.0f}x"
    )
    crossovers = []
    for m in method_names:
        for r in sweep_rows:
            if r[m] > r["noisy_input"]:
                crossovers.append((m, r["level"]))
                break
    print(
        "worth denoising  : "
        + ", ".join(f"{m} beats doing nothing from sigma {lv:g}" for m, lv in crossovers)
    )
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
