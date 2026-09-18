"""Estimate and remove a colour cast from an image of your own.

    python infer.py photo.jpg
    python infer.py photo.jpg --method "Grey-edge (p=6)" --out balanced.png
    python infer.py --photo red_chrysanthemums --cast "tungsten (warm)"

With `--photo` the cast is applied by this tool, so the true illuminant is known
and the **angular error is real**. On your own photograph there is no ground
truth, so no error is printed — what is printed instead is:

* how far your scene's mean already sits from the grey axis, which is
  grey-world's error *in advance* and says whether to trust it at all;
* how far apart the five estimates are. Five methods disagreeing by 15 degrees on
  your picture means the scene does not satisfy anybody's assumption, and the
  answer to pick is the one whose assumption your scene actually meets.

On this project's photographs grey-world's error runs from 1.99 degrees on a
beach to 21.68 on a frame full of red flowers — worse there than not correcting
at all.
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

import white_balance as wb  # noqa: E402

#: Above this, the scene's own mean is far enough from grey that grey-world is
#: reading the subject rather than the light. `red_chrysanthemums` is at 29.2.
GREY_WORLD_UNSAFE = 12.0

#: Spread between the five estimates above which they are not measuring the same
#: thing and no single answer should be trusted.
DISAGREEMENT = 8.0

#: Fraction of pixels clipped in all three channels above which white-patch is
#: reading a blown highlight rather than a white surface.
CLIPPED_WARN = 0.0005


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", default=None)
    ap.add_argument("--photo", default=None, choices=list(wb.IMAGES),
                    help="use one of the project's photographs and apply a known cast")
    ap.add_argument("--cast", default="tungsten (warm)", choices=list(wb.CASTS))
    ap.add_argument("--method", default=None, choices=list(wb.ESTIMATORS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    init_console()
    if args.photo:
        img, _, truth = wb.make_case(args.photo, gains=wb.CASTS[args.cast])
        source = f"{args.photo} with a {args.cast} cast applied"
    elif args.image:
        img = io.imread(args.image)
        truth = None
        source = args.image
    else:
        ap.error("give an image path, or --photo with one of the project's photographs")

    deviation = wb.grey_deviation(img)
    clipped = float((img.min(axis=2) >= 255).mean())
    print(f"{source}  {img.shape[1]}x{img.shape[0]}")
    print(f"the scene's mean sits {deviation:.2f} deg from the grey axis")
    if deviation > GREY_WORLD_UNSAFE:
        print(f"  that is above {GREY_WORLD_UNSAFE:g}, so grey-world is about to read your "
              "subject as if it were the light. Expect it to be the worst method here.")
    else:
        print("  close enough that grey-world's assumption is roughly satisfied")
    print(f"{100 * clipped:.4f}% of pixels are clipped in all three channels")
    if clipped > CLIPPED_WARN:
        print(f"  above {100 * CLIPPED_WARN:.2f}%, and a clipped pixel is (255,255,255) "
              "whatever the light was. White-patch (true max) will read one of those and "
              "report a white illuminant — on this project's test scene that puts it "
              "exactly on the do-nothing control.")

    header = f"\n{'method':24s} {'estimate (r, g)':>18s} {'gain R:G:B':>22s}"
    if truth is not None:
        header += f" {'error (deg)':>12s}"
    print(header)

    estimates, panels = {}, [("as given", img)]
    for name, fn in wb.ESTIMATORS.items():
        v = wb.normalise_illuminant(fn(img))
        estimates[name] = v
        chroma = (v[0] / v.sum(), v[1] / v.sum())
        line = (f"{name:24s} {f'({chroma[0]:.3f}, {chroma[1]:.3f})':>18s} "
                f"{f'{v[0]:.3f} : {v[1]:.3f} : {v[2]:.3f}':>22s}")
        if truth is not None:
            line += f" {wb.angular_error(v, truth):12.3f}"
        print(line)
        panels.append((f"{name}\n{wb.angular_error(v, truth):.2f} deg" if truth is not None
                       else name, wb.apply_correction(img, v)))

    real = [n for n in estimates if n != "Do nothing (control)"]
    spread = max(wb.angular_error(estimates[a], estimates[b])
                 for a in real for b in real)
    print(f"\nthe five estimates disagree by up to {spread:.2f} deg.")
    if spread > DISAGREEMENT:
        print(f"  above {DISAGREEMENT:g} deg: this scene does not satisfy everybody's "
              "assumption, so there is no single answer to take. Pick the method whose "
              "assumption your scene actually meets — grey-edge if one colour dominates, "
              "the 99th-percentile white patch if there is a genuine white surface.")
    else:
        print("  they agree, which is the case where any of them will do.")

    if truth is not None:
        errors = {n: wb.angular_error(v, truth) for n, v in estimates.items()}
        best = min(errors, key=errors.get)
        control = errors["Do nothing (control)"]
        worse = [n for n in real if errors[n] > control]
        print(f"\nbest here: {best} at {errors[best]:.3f} deg "
              f"(doing nothing scores {control:.3f})")
        if worse:
            print(f"  {len(worse)} of {len(real)} methods are WORSE than not correcting: "
                  f"{', '.join(n.split()[0] for n in worse)}")
    else:
        print("\nNo angular error is printed. There is no ground truth for your "
              "photograph, and the two numbers above — the deviation from grey and the "
              "spread between methods — are the measurements that are real.")

    if args.method:
        out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "balanced.png"
        io.imwrite(out_path, wb.apply_correction(img, estimates[args.method]))
    else:
        out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "estimates.png"
        figures.grid(panels, out_path, ncols=4,
                     suptitle=f"Five illuminant estimates for {Path(source).name}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
