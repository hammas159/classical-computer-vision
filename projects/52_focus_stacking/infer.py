"""Build a focal stack from one photograph and merge it — with both controls.

    python infer.py --image mountain_lake_and_scree
    python infer.py --image whitewashed_bell_tower --pool 1
    python infer.py --image picnic_in_the_snow --sigma 1
    python infer.py photo.jpg --frames 9

Every run prints the **oracle** and the **random** control alongside the five
measures, because that is the only way to read them. On this project's twelve
photographs the task is worth 17 dB between those two, and all five measures land
in the last 1.4 dB of it.

`--pool 1` turns off the pooling window, which is the parameter this project
finds matters more than the measure: unpooled, the best measure gets 0.66 of
pixels right instead of 0.97.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import numpy as np  # noqa: E402

from shared import figures, io  # noqa: E402
from shared.report import init_console  # noqa: E402

import focus as fo  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default=None,
                    help="your own photograph; a stack is built from it")
    ap.add_argument("--image", default=None, choices=list(fo.IMAGES))
    ap.add_argument("--frames", type=int, default=fo.N_FRAMES)
    ap.add_argument("--sigma", type=float, default=fo.MAX_SIGMA,
                    help="blur at the far end of the depth range")
    ap.add_argument("--pool", type=int, default=fo.POOL,
                    help="the pooling window; 1 turns it off")
    ap.add_argument("--feather", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    init_console()

    if args.image:
        image = fo.load(args.image)
        name = args.image
    elif args.path:
        image = io.imread(args.path)
        name = Path(args.path).name
    else:
        ap.error("give an image path or --image with one of the project's twelve")

    frames, truth, _ = fo.build_stack(image, n=args.frames, seed=args.seed,
                                      max_sigma=args.sigma)

    detail = fo.detail(image)
    flat = fo.flat_share(image)
    print(f"{name}  {image.shape[1]}x{image.shape[0]}")
    print(f"  detail {detail:.0f}, flat share {100 * flat:.1f}%")
    print(f"  stack: {args.frames} frames, blur up to sigma {args.sigma:g}, "
          f"pooling window {args.pool}")
    if flat > 0.02:
        print(f"  {100 * flat:.0f}% of this photograph is flat enough that no focus "
              "measure can be right there — every frame looks the same")

    oracle = fo.psnr(fo.merge(frames, truth, args.feather), image)
    random_idx = fo.select_random(frames, seed=args.seed)
    random_db = fo.psnr(fo.merge(frames, random_idx, args.feather), image)
    first_db = fo.psnr(fo.merge(frames, fo.select_first(frames), args.feather), image)

    print(f"\n  the task is worth {oracle - random_db:.2f} dB here: random "
          f"{random_db:.2f}, oracle {oracle:.2f}")

    print(f"\n{'measure':26s} {'PSNR (dB)':>10s} {'agreement':>10s} "
          f"{'of the 17 dB':>13s}")

    panels = [("the original (truth)", image),
              (f"frame {args.frames // 2} of the stack", frames[args.frames // 2])]
    results = {}
    for measure in fo.MEASURES:
        idx = fo.select_indices(frames, measure, args.pool)
        merged = fo.merge(frames, idx, args.feather)
        db = fo.psnr(merged, image)
        agree = float((idx == truth).mean())
        results[measure] = (merged, db, agree)
        share = (db - random_db) / max(oracle - random_db, 1e-9)
        print(f"{measure:26s} {db:10.3f} {agree:10.4f} {share:12.1%}")
        panels.append((f"{measure}\n{db:.2f} dB", merged))

    print(f"{'Oracle (truth)':26s} {oracle:10.3f} {1.0:10.4f} {1.0:12.1%}")
    print(f"{'Random pick (control)':26s} {random_db:10.3f} "
          f"{float((random_idx == truth).mean()):10.4f} {0.0:12.1%}")
    print(f"{'First frame (control)':26s} {first_db:10.3f} "
          f"{float((np.zeros_like(truth) == truth).mean()):10.4f}")

    # ------------------------------------------------------------------ #
    best = max(results, key=lambda m: results[m][1])
    worst = min(results, key=lambda m: results[m][1])
    print(f"\nbest measure here: {best} ({results[best][1]:.3f} dB), "
          f"{oracle - results[best][1]:.3f} dB from the oracle")
    print(f"spread across the five: {results[best][1] - results[worst][1]:.3f} dB")

    if args.pool != 1:
        unpooled = {}
        for measure in fo.MEASURES:
            idx = fo.select_indices(frames, measure, 1)
            unpooled[measure] = fo.psnr(fo.merge(frames, idx, args.feather), image)
        gain = results[best][1] - unpooled[best]
        print(f"\nwith the pooling window turned off, {best} scores "
              f"{unpooled[best]:.3f} dB")
        print(f"  the window is worth {gain:.3f} dB — against "
              f"{results[best][1] - results[worst][1]:.3f} dB between the best and "
              "worst measure")
        if gain > results[best][1] - results[worst][1]:
            print("  — the parameter nobody reports is worth more than the choice "
                  "everybody argues about")

    picks = np.stack([fo.select_indices(frames, m, args.pool) for m in fo.MEASURES])
    disagree = (picks != picks[0]).any(axis=0)
    print(f"\nthe five measures disagree on {100 * float(disagree.mean()):.1f}% of "
          "pixels")

    out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "merged.png"
    figures.grid(panels, out_path, ncols=4,
                 suptitle=(f"{name}: a {args.frames}-frame stack merged five ways "
                           f"(oracle {oracle:.2f} dB, random {random_db:.2f} dB)"))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
