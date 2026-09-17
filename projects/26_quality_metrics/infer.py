"""Score your own image pair with every full-reference metric at once.

    python infer.py original.jpg damaged.jpg
    python infer.py original.jpg damaged.jpg --verbose
    python infer.py photo.jpg --equal-psnr 28     # build the six equal-PSNR damages

This is one of the few projects here where nothing has to be withheld: a
full-reference metric needs only the two images, so every number below is real.

What this adds to running SSIM yourself is the **disagreement**. Each metric is
reported as a percentile against this project's twelve photographs under six
known damages, so a score can be read as "as bad as a 20% contrast loss" rather
than as a bare number — and where the metrics rank your damage differently, that
is said out loud, because it means the single number you were about to quote is
not enough.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import numpy as np  # noqa: E402

from shared import io  # noqa: E402
from shared.report import init_console  # noqa: E402

import quality_metrics as qm  # noqa: E402

#: Spread in the metrics' normalised rankings above which they are calling your
#: damage genuinely different things.
DISAGREEMENT_WARN = 0.25


def nearest_known_damage(value: float, metric: str, reference: list[dict]) -> str:
    """Which of the six known damages this score most resembles."""
    higher_better = qm.METRICS[metric][1]
    best, gap = None, float("inf")
    for row in reference:
        d = abs(row[metric] - value)
        if d < gap:
            best, gap = row["degradation"], d
    direction = "better than" if (
        (reference[0][metric] < value) == higher_better) else "worse than"
    return f"{direction} the reference {best.lower()}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("original")
    ap.add_argument("damaged", nargs="?", default=None)
    ap.add_argument("--equal-psnr", type=float, default=None,
                    help="ignore the second image; build all six damages at this PSNR")
    ap.add_argument("--out", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    init_console()
    clean = io.imread(args.original)
    print(f"{args.original}  {clean.shape[1]}x{clean.shape[0]}")

    if args.equal_psnr is not None:
        target = args.equal_psnr
        print(f"\nbuilding six damages, each bisected to {target:g} dB:\n")
        print(f"{'damage':18s} {'strength':>10s} {'PSNR':>8s} "
              + " ".join(f"{m[:9]:>9s}" for m in qm.METRICS if m != "PSNR"))
        panels, scores = [clean], {}
        for deg in qm.DEGRADATIONS:
            strength, got = qm.find_strength_for_psnr(clean, deg, target)
            fn, _ = qm.DEGRADATIONS[deg]
            bad = fn(clean, strength)
            panels.append(bad)
            row = {m: qm.METRICS[m][0](bad, clean) for m in qm.METRICS}
            scores[deg] = row
            print(f"{deg:18s} {strength:10.4f} {got:8.2f} "
                  + " ".join(f"{row[m]:9.4f}" for m in qm.METRICS if m != "PSNR"))

        ssims = {d: s["SSIM"] for d, s in scores.items()}
        print(f"\nAt a fixed {target:g} dB, SSIM spans {min(ssims.values()):.4f} "
              f"({min(ssims, key=ssims.get)}) to {max(ssims.values()):.4f} "
              f"({max(ssims, key=ssims.get)}).")
        print("If PSNR measured what you mean by quality, those would be equal.")
        out = np.hstack(panels)
    else:
        if args.damaged is None:
            ap.error("give a damaged image, or pass --equal-psnr")
        bad = io.imread(args.damaged)
        if bad.shape != clean.shape:
            ap.error(f"images differ in size: {clean.shape} vs {bad.shape}")
        print(f"{args.damaged}  {bad.shape[1]}x{bad.shape[0]}")

        reference = qm.equal_psnr_comparison(28.0, images=qm.IMAGES[:4])
        print(f"\n{'metric':14s} {'value':>10s}   compared with six known damages at 28 dB")
        values = {}
        for name, (metric, higher_better) in qm.METRICS.items():
            v = metric(bad, clean)
            values[name] = v
            note = "" if name in ("MSE", "PSNR") else nearest_known_damage(
                v, name, reference)
            print(f"{name:14s} {v:10.4f}   {note}")

        # where do the metrics place this damage relative to the known six?
        places = {}
        for name, (_, higher_better) in qm.METRICS.items():
            col = sorted(r[name] for r in reference)
            rank = sum(1 for c in col if c < values[name]) / len(col)
            places[name] = rank if higher_better else 1.0 - rank
        spread = max(places.values()) - min(places.values())
        print(f"\nThe metrics place this damage between the "
              f"{min(places.values()):.0%} and {max(places.values()):.0%} points "
              f"of the known six — a spread of {spread:.0%}.")
        if spread > DISAGREEMENT_WARN:
            worst = min(places, key=places.get)
            best = max(places, key=places.get)
            print(f"  They disagree: {worst} says this is among the worst damage "
                  f"here and {best} says it is among the mildest. Quoting either "
                  "on its own would be choosing an answer.")
        else:
            print("  They broadly agree, which is the case where one number is "
                  "defensible.")
        if args.verbose:
            print("\nreference table (six damages at 28 dB):")
            for r in reference:
                print("  " + ", ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                                       for k, v in r.items() if not k.endswith("__exact")))
        out = np.hstack([clean, bad])

    out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "compared.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    io.imwrite(out_path, out)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
