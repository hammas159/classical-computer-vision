"""Run the eight-stage pipeline on your own image, against the one-liners.

    python infer.py photo.jpg
    python infer.py photo.jpg --stages          # every intermediate stage
    python infer.py photo.jpg --ablate          # each stage removed, on your image
    python infer.py photo.jpg --method "Unsharp mask only" --out enhanced.png

Enhancement has no ground truth — there is no correctly enhanced version of your
photograph — so **no fidelity score is reported as a quality judgement**.

Three numbers are printed instead, and all three are needed. Dark-region detail
and acutance say what changed; SSIM against the original says how far it moved.
The Laplacian sign test in this project's README is exactly the case where the
first two are satisfied by a result that is structurally inverted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import numpy as np  # noqa: E402

from shared import io  # noqa: E402
from shared.io import to_gray  # noqa: E402
from shared.report import init_console  # noqa: E402

import gw_pipeline as gw  # noqa: E402

#: SSIM against the original below which the result has moved so far that the
#: enhancement numbers stop meaning what they appear to.
STRUCTURE_WARN = 0.3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--method", default=None, choices=list(gw.VARIANTS))
    ap.add_argument("--stages", action="store_true")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    init_console()
    img = io.imread(args.image)
    bright = float(to_gray(img).mean())
    print(f"{args.image}  {img.shape[1]}x{img.shape[0]}  mean luma {bright:.1f}")
    print("  (this pipeline is for dark images with buried detail; its pool "
          "spans luma 36 to 110)")

    if args.stages:
        final, stages = gw.run_pipeline(img, gamma=args.gamma)
        print(f"\n{'stage':22s} {'acutance':>9s} {'dark detail':>12s} {'SSIM':>7s}")
        panels = []
        for name, stage in stages.items():
            m = gw.measure(stage, img)
            panels.append(io.ensure_rgb(stage))
            print(f"{name:22s} {m['acutance']:9.4f} {m['dark_detail']:12.4f} "
                  f"{m['ssim_vs_original']:7.4f}")
        out = np.hstack(panels)

    elif args.ablate:
        rows = [("Full pipeline", None)] + [(f"Without {s}", s) for s in gw.ABLATABLE]
        full = gw.measure(gw.run_pipeline(img, gamma=args.gamma)[0], img)
        print(f"\n{'configuration':26s} {'dark detail':>12s} {'delta':>9s} {'SSIM':>7s}")
        panels, results = [], {}
        for label, skip in rows:
            out_img, _ = gw.run_pipeline(img, skip=skip, gamma=args.gamma)
            m = gw.measure(out_img, img)
            results[label] = m
            panels.append(io.ensure_rgb(out_img))
            delta = m["dark_detail"] - full["dark_detail"]
            print(f"{label:26s} {m['dark_detail']:12.4f} "
                  f"{'' if skip is None else f'{delta:+9.4f}'} "
                  f"{m['ssim_vs_original']:7.4f}")

        deltas = {k: abs(v["dark_detail"] - full["dark_detail"])
                  for k, v in results.items() if k != "Full pipeline"}
        least = min(deltas, key=deltas.get)
        print(f"\nLeast useful stage on this image: {least} "
              f"(changes dark detail by {deltas[least]:.4f}).")
        print("On this project's twelve photographs that is always stage (e), "
              "the smoothed Sobel — it is in the book and does nothing.")
        out = np.hstack(panels)

    else:
        names = [args.method] if args.method else list(gw.VARIANTS)
        print(f"\n{'method':26s} {'acutance':>9s} {'dark detail':>12s} {'SSIM':>7s}")
        panels, scores = [], {}
        for name in names:
            result = gw.apply_variant(img, name)
            m = gw.measure(result, img)
            scores[name] = m
            panels.append(io.ensure_rgb(result))
            print(f"{name:26s} {m['acutance']:9.4f} {m['dark_detail']:12.4f} "
                  f"{m['ssim_vs_original']:7.4f}")

        if len(scores) > 1:
            pipeline = scores.get("G&W 8-stage pipeline")
            beats = [n for n, m in scores.items()
                     if n not in ("G&W 8-stage pipeline", "Original (control)")
                     and pipeline and m["dark_detail"] > pipeline["dark_detail"]]
            if beats:
                print(f"\nBeating the eight-stage pipeline on dark detail here: "
                      f"{', '.join(beats)}.")
                print("  On this project's twelve photographs a single unsharp "
                      "mask beats it on every column, including SSIM.")
            else:
                print("\nThe pipeline wins on this image, which is not the case "
                      "on any of this project's twelve.")
        risky = [n for n, m in scores.items() if m["ssim_vs_original"] < STRUCTURE_WARN]
        if risky:
            print(f"\nSSIM below {STRUCTURE_WARN:g} for: {', '.join(risky)}. "
                  "The enhancement numbers for those are not describing the same "
                  "picture any more — look before trusting them.")
        out = np.hstack(panels)

    out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "enhanced.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    io.imwrite(out_path, out)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
