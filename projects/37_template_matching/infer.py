"""Find a template in a scene of your own, with all five scoring functions.

    python infer.py scene.jpg template.png
    python infer.py scene.jpg template.png --multiscale
    python infer.py --photo two_at_a_railing --offset -0.4

Each method prints where it thinks the template is, its raw score, and the
**peak-to-mean ratio** of its response surface — which is the number that says
whether to believe the answer. On this project's photographs ZNCC and NCC
localise identically and their peaks differ by 357x, so the error alone does not
distinguish a confident match from a lucky one.

The runner-up peak is printed too. A second peak nearly as tall as the first
means the scene contains something else that looks like your template, and a
single-best-answer matcher cannot tell you that from its output.
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

import template_matching as tm  # noqa: E402

#: Peak-to-mean below which the response surface is too flat for the score to
#: mean anything. NCC sits at 1.22 on this project's photographs even when it is
#: exactly right, which is the point.
FLAT_SURFACE = 3.0

#: How close the runner-up peak may get before the scene is genuinely ambiguous.
AMBIGUOUS = 0.9


def _second_peak(result: np.ndarray, best_loc, size: int, lower_is_better: bool):
    """The best score outside a window around the winner."""
    r = result.astype(np.float64)
    if lower_is_better:
        r = r.max() - r
    masked = r.copy()
    y0 = max(0, best_loc[1] - size)
    x0 = max(0, best_loc[0] - size)
    masked[y0:best_loc[1] + size, x0:best_loc[0] + size] = -np.inf
    if not np.isfinite(masked).any():
        return None, None
    idx = int(np.argmax(masked))
    loc = (idx % masked.shape[1], idx // masked.shape[1])
    return float(masked.flat[idx]) / max(float(r.max()), tm.EPS), loc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", nargs="?", default=None)
    ap.add_argument("template", nargs="?", default=None)
    ap.add_argument("--photo", default=None, choices=list(tm.IMAGES),
                    help="use one of the project's photographs and crop a template from it")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="darken or brighten the scene (not the template)")
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--multiscale", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    init_console()
    truth = None
    if args.photo:
        scene, template, truth = tm.make_case(
            args.photo, brightness_gain=args.gain, brightness_offset=args.offset, seed=0)
        source = args.photo
        print(f"{args.photo}: template cropped from the scene, so the answer is known "
              f"exactly — it is at {truth}")
    elif args.scene and args.template:
        scene = io.imread(args.scene)
        template = io.imread(args.template)
        if args.gain != 1.0 or args.offset != 0.0:
            from shared.io import to_float, to_uint8
            scene = to_uint8(to_float(scene) * args.gain + args.offset)
        source = args.scene
    else:
        ap.error("give a scene and a template, or --photo with one of the project's images")

    if min(template.shape[:2]) >= min(scene.shape[:2]):
        raise SystemExit("the template must be smaller than the scene")

    print(f"scene {scene.shape[1]}x{scene.shape[0]}, template "
          f"{template.shape[1]}x{template.shape[0]}, "
          f"scene entropy {tm.entropy(scene):.2f} bits")
    if args.gain != 1.0 or args.offset != 0.0:
        print(f"  scene degraded by gain {args.gain:g}, offset {args.offset:+g} "
              "— the template is untouched, which is the realistic direction")

    header = (f"\n{'method':24s} {'found at':>14s} {'score':>10s} "
              f"{'peak/mean':>10s} {'2nd peak':>9s}")
    if truth is not None:
        header += f" {'error px':>9s}"
    print(header)

    results, panels = {}, [("scene", ensure_rgb(scene)), ("template", ensure_rgb(template))]
    for name, (flag, lower_is_better) in tm.METHODS.items():
        found, score, result = tm.match(scene, template, name)
        r = result.astype(np.float64)
        peak = r.min() if lower_is_better else r.max()
        if lower_is_better:
            r = r.max() - r
            peak = r.max()
        ratio = float(peak) / max(abs(float(r.mean())), tm.EPS)
        second, second_loc = _second_peak(result, found, template.shape[0], lower_is_better)

        line = (f"{name:24s} {str(found):>14s} {score:10.4f} {ratio:10.2f} "
                f"{(second if second is not None else float('nan')):9.3f}")
        if truth is not None:
            line += f" {tm.localisation_error(found, truth):9.1f}"
        print(line)
        results[name] = (found, ratio, second)

        out = ensure_rgb(scene).copy()
        cv2.rectangle(out, found, (found[0] + template.shape[1], found[1] + template.shape[0]),
                      (60, 200, 60), 3)
        if truth is not None:
            cv2.rectangle(out, truth, (truth[0] + template.shape[1],
                                       truth[1] + template.shape[0]), (255, 255, 255), 1)
        panels.append((f"{name}\npeak/mean {ratio:,.1f}x", out))

    # ------------------------------------------------------------------ #
    # what to make of it
    # ------------------------------------------------------------------ #
    print()
    locations = {tuple(v[0]) for v in results.values()}
    if len(locations) == 1:
        print("All five methods agree on the location.")
    else:
        print(f"The methods disagree — {len(locations)} different locations. On an "
              "undegraded scene four of the five should agree; raw cross-correlation "
              "usually does not, because it finds the brightest patch rather than "
              "the matching one.")

    flat = [n for n, (_, ratio, _) in results.items() if ratio < FLAT_SURFACE]
    if flat:
        print(f"\n{len(flat)} of {len(results)} have a nearly flat response surface "
              f"(peak/mean under {FLAT_SURFACE:g}): "
              f"{', '.join(n.split()[0] for n in flat)}.")
        print("  Those answers may be correct and are not confident — a score threshold "
              "on them cannot separate 'found it' from 'found nothing'.")

    ambiguous = [n for n, (_, _, second) in results.items()
                 if second is not None and second > AMBIGUOUS]
    if ambiguous:
        print(f"\n{len(ambiguous)} methods found a runner-up peak above "
              f"{AMBIGUOUS:g} of the winner: {', '.join(n.split()[0] for n in ambiguous)}.")
        print("  There is something else in this scene that looks like your template. "
              "A single-best-answer matcher will never mention it.")

    if args.multiscale:
        print("\n--- pyramid search ---")
        best = tm.match_multiscale(scene, template)
        if best is None:
            print("  no usable scale")
        else:
            loc, score, scale = best
            line = f"  best at {loc} score {score:.4f} at scale {scale:.3f}"
            if truth is not None:
                line += f", error {tm.localisation_error(loc, truth):.1f} px"
            print(line)
            if abs(scale - 1.0) > 0.05:
                print(f"  the target is {scale:.2f}x the template's size — plain matching "
                      "cannot find that, and on this project's photographs it drops to "
                      "0.083 at 1.25x while the pyramid holds 1.000")
            else:
                print("  the target is at the template's own scale, so the pyramid bought "
                      "nothing here beyond 21x the time")

    out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "matches.png"
    figures.grid(panels, out_path, ncols=4,
                 suptitle=f"Five scoring functions on {Path(source).name}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
