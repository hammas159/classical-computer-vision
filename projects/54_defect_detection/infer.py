"""Plant a defect in a surface and see which detectors find it — and what they
flag when there is nothing there.

    python infer.py --surface surface_coarse_cloth --defect smear
    python infer.py --surface surface_fine_weave --defect scratch
    python infer.py --surface surface_brick_wall --clean
    python infer.py photo.jpg --defect blob

`--clean` plants no defect at all, so the true answer is an empty mask and
everything printed is a false alarm. That is the arm a benchmark built only on
defective parts never runs, and on this data the best detector still marks 3.7%
of a good surface.

Every run prints the detection and the false-alarm share together, because
neither means anything alone — and never a pixel accuracy, which a defect
covering one per cent of a surface makes meaningless.
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
from shared.io import ensure_rgb  # noqa: E402
from shared.report import init_console  # noqa: E402

import defects as df  # noqa: E402


def _overlay(image, pred, truth):
    out = ensure_rgb(image).copy()
    sel = pred > 0
    out[sel] = (0.45 * out[sel] + 0.55 * np.array([230, 60, 60],
                                                  np.float32)).astype(np.uint8)
    contours, _ = cv2.findContours((truth > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (60, 230, 60), 2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default=None,
                    help="your own image of a surface")
    ap.add_argument("--surface", default=None, choices=list(df.IMAGES))
    ap.add_argument("--defect", default="scratch", choices=list(df.DEFECTS))
    ap.add_argument("--clean", action="store_true",
                    help="plant nothing: the true answer is an empty mask")
    ap.add_argument("--severity", type=float, default=df.SEVERITY)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    init_console()

    if args.surface:
        image = df.load(args.surface)
        name = args.surface
    elif args.path:
        image = io.imread(args.path)
        name = Path(args.path).name
    else:
        ap.error("give an image path or --surface with one of the project's twelve")

    ceiling = df.noise_ceiling(image)
    print(f"{name}  {image.shape[1]}x{image.shape[0]}")
    print(f"  uniformity {df.uniformity(image):.2f}, noise ceiling {ceiling:.2f} "
          "grey levels")

    if args.clean:
        bad, mask = df.clean(image)
        print("  no defect planted — the true answer is an empty mask, so every "
              "pixel marked below is a false alarm")
    else:
        bad, mask = df.plant(image, args.defect, seed=args.seed,
                             severity=args.severity)
        applied = float(np.abs(df.to_gray(bad).astype(np.float32)
                               - df.to_gray(image).astype(np.float32))[mask > 0].mean())
        print(f"  planted a {args.defect} covering "
              f"{100 * float((mask > 0).mean()):.2f}% of the surface, "
              f"{applied:.1f} grey levels deep "
              f"({applied / max(ceiling, 1e-9):.1f}x the noise ceiling)")

    print(f"\n{'detector':30s} {'marked':>8s} {'IoU':>8s} {'found?':>8s}")
    panels = [("the surface", _overlay(bad, np.zeros_like(mask), mask))]
    results = {}
    for detector in df.REAL_DETECTORS:
        pred = df.fill(df.DETECTORS[detector](bad))
        marked = float((pred > 0).mean())
        value = df.iou(pred, mask)
        hit = df.found(pred, mask) if mask.any() else False
        results[detector] = (pred, marked, value, hit)
        verdict = ("found" if hit else "missed") if mask.any() else "—"
        print(f"{detector:30s} {100 * marked:7.2f}% {value:8.3f} {verdict:>8s}")
        panels.append((f"{detector}\n{100 * marked:.1f}% marked", _overlay(bad, pred, mask)))

    # ------------------------------------------------------------------ #
    if mask.any():
        finders = [d for d, v in results.items() if v[3]]
        if finders:
            print(f"\n{len(finders)} of {len(df.REAL_DETECTORS)} found it: "
                  f"{', '.join(finders)}")
        else:
            print(f"\nnone of the {len(df.REAL_DETECTORS)} found it")
        missed = [d for d, v in results.items() if not v[3]]
        if missed and args.defect == "smear":
            print("  a smear has the same mean as its surround and less texture, so "
                  "no intensity residual can see it — only a texture measure can")

        # what would it look like on a clean surface?
        ok, _ = df.clean(image)
        print("\nthe same detectors on this surface with nothing wrong with it:")
        for detector in df.REAL_DETECTORS:
            alarm = float((df.fill(df.DETECTORS[detector](ok)) > 0).mean())
            tag = "  <- found the defect" if results[detector][3] else ""
            print(f"  {detector:30s} marks {100 * alarm:6.2f}%{tag}")
        best_finder = [d for d in finders]
        if best_finder:
            alarms = {d: float((df.fill(df.DETECTORS[d](ok)) > 0).mean())
                      for d in best_finder}
            worst = max(alarms, key=alarms.get)
            print(f"  — {worst} found the defect and marks "
                  f"{100 * alarms[worst]:.1f}% of a good surface. Both numbers are "
                  "the result; neither alone is.")
    else:
        alarms = {d: v[1] for d, v in results.items()}
        best = min(alarms, key=alarms.get)
        worst = max(alarms, key=alarms.get)
        print(f"\nnothing is wrong with this surface, and every detector marked "
              f"some of it: {100 * alarms[best]:.2f}% ({best}) to "
              f"{100 * alarms[worst]:.2f}% ({worst})")
        print("  a pixel accuracy here would be 1 minus those numbers, which is why "
              "this project never reports one")

    out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "detected.png"
    figures.grid(panels, out_path, ncols=4,
                 suptitle=(f"{name}: "
                           + ("no defect" if args.clean
                              else f"a {args.defect} at severity {args.severity:g}")
                           + " — green is truth, red is what was marked"))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
