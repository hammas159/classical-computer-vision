"""Stitch two frames and score the join two ways — they will disagree.

    python infer.py --image tugboat_under_the_bridge
    python infer.py --image tugboat_under_the_bridge --exposure 1.0
    python infer.py --image brick_wall_courses
    python infer.py --real
    python infer.py photo.jpg

Two numbers are printed for every blender:

* **PSNR** against the original photograph — the default choice, and on this
  construction it is maximised by the control that **does not stitch at all**,
  because the first frame *is* the original in its own region.
* **seam step** — the brightness step straight across the join, in grey levels.
  This is what blending is for, and it ranks the methods the other way round.

`--exposure 1.0` is the control worth running: with the two frames at matched
exposure there is nothing to hide, and every method scores the same.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import numpy as np  # noqa: E402

from shared import figures, io  # noqa: E402
from shared.io import ensure_rgb  # noqa: E402
from shared.report import init_console  # noqa: E402

import panorama as pa  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default=None,
                    help="your own photograph; it is cut into two frames")
    ap.add_argument("--image", default=None, choices=list(pa.IMAGES))
    ap.add_argument("--real", action="store_true",
                    help="stitch the two graffiti photographs instead")
    ap.add_argument("--exposure", type=float, default=pa.EXPOSURE)
    ap.add_argument("--coverage", type=float, default=pa.COVERAGE)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    init_console()

    # ------------------------------------------------------------------ #
    if args.real:
        if not pa.graf_available():
            ap.error("graf1/graf3 are not cached. Run "
                     "`python tools/fetch_assets.py --set graf` first.")
        info = pa.real_pair()
        print(f"the real pair: graf1 / graf3, {info['shape'][1]}x{info['shape'][0]}")
        print(f"  {info['matches']} matches survive the ratio test, "
              f"{info['inliers']} are RANSAC inliers ({info['inlier_rate']:.1%})")
        print(f"  cycle error {info['cycle_error_px']:.3f} px — a consistency check, "
              "not an accuracy: H composed with its own inverse is the identity "
              "whatever H is")
        print("  the Oxford set's published homography for this pair 404s upstream, "
              "so there is no external truth here and none is claimed")

        print(f"\n{'blender':34s} {'seam step':>10s}")
        panels = [("graf1", ensure_rgb(pa.load_graf()[0])),
                  ("graf3", ensure_rgb(pa.load_graf()[1]))]
        for blender in pa.BLENDERS:
            r = pa.stitch_real(blender)
            step = pa.worst_seam_step(r["output"], r["overlap"])
            print(f"{blender:34s} {step:10.2f}")
            panels.append((f"{blender}\n{step:.0f}", ensure_rgb(r["output"])))
        out_path = (Path(args.out) if args.out
                    else PROJECT_DIR / "results" / "stitched.png")
        figures.grid(panels, out_path, ncols=4,
                     suptitle="The real pair, stitched five ways")
        print(f"\nwrote {out_path}")
        return 0

    # ------------------------------------------------------------------ #
    if args.image:
        image = pa.load(args.image)
        name = args.image
    elif args.path:
        image = io.imread(args.path)
        name = Path(args.path).name
    else:
        ap.error("give an image path, --image NAME, or --real")

    reg = pa.registrability(image)
    print(f"{name}  {image.shape[1]}x{image.shape[0]}")
    print(f"  registrability: {reg['matches']} matches, {reg['inliers']} inliers "
          f"({reg['inlier_rate']:.3f}), {reg['per_megapixel']:.0f} per megapixel")
    if reg["per_megapixel"] < 200:
        print("  — that is very low. A scene of repeated identical structure gives "
              "matches that are all equally good, so there is no consistent "
              "homography to find. No blender can rescue a panorama that was never "
              "registered.")
    print(f"  two frames covering {100 * args.coverage:.0f}% each; the second is "
          f"{100 * (args.exposure - 1):+.0f}% brighter")

    print(f"\n{'blender':34s} {'PSNR (dB)':>10s} {'seam step':>10s}")
    results = {}
    panels = []
    for blender in pa.BLENDERS:
        left, lc, right, rc, H, truth = pa.make_pair(
            image, exposure=args.exposure, coverage=args.coverage)
        aligned, ac = pa.align(right, rc, H, image.shape)
        overlap, right_only, covered = pa.regions(lc, ac)
        out = pa.BLENDERS[blender](left, aligned, overlap, right_only=right_only)
        db = pa.psnr_on(out, truth, covered)
        step = pa.worst_seam_step(out, overlap)
        results[blender] = (out, db, step)
        print(f"{blender:34s} {db:10.3f} {step:10.2f}")
        panels.append((f"{blender}\n{db:.1f} dB, step {step:.0f}", out))

    # ------------------------------------------------------------------ #
    by_psnr = max(results, key=lambda b: results[b][1])
    by_seam = min(results, key=lambda b: results[b][2])
    print(f"\nbest by PSNR:      {by_psnr}")
    print(f"best by seam step: {by_seam}")
    if by_psnr != by_seam:
        seam_rank = sorted(results, key=lambda b: results[b][2]).index(by_psnr) + 1
        print(f"  — they disagree. The PSNR winner ranks {seam_rank} of "
              f"{len(results)} on the seam.")
        if "Keep the first" in by_psnr:
            print("  and the PSNR winner is the control that does not stitch: the "
                  "first frame IS the original in its own region, so any blending "
                  "that mixes the second frame in can only move away from it.")

    steps = [results[b][2] for b in results]
    spread = max(steps) - min(steps)
    print(f"\nspread across the five blenders: {spread:.2f} grey levels")
    if spread < 2.0:
        print("  — essentially nothing. With the exposures matched there is nothing "
              "for a blender to hide, and every method leaves the scene's own "
              "texture at the join.")
    elif abs(args.exposure - 1.0) > 1e-9:
        matched = {}
        for blender in pa.BLENDERS:
            left, lc, right, rc, H, truth = pa.make_pair(
                image, exposure=1.0, coverage=args.coverage)
            aligned, ac = pa.align(right, rc, H, image.shape)
            overlap, right_only, _ = pa.regions(lc, ac)
            out = pa.BLENDERS[blender](left, aligned, overlap, right_only=right_only)
            matched[blender] = pa.worst_seam_step(out, overlap)
        flat = max(matched.values()) - min(matched.values())
        print(f"  at exposure 1.00 the spread is {flat:.2f} — the whole value of "
              "blending here is the exposure difference it has to hide")

    out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "stitched.png"
    figures.grid([("the original (truth)", ensure_rgb(image))] + panels,
                 out_path, ncols=3,
                 suptitle=f"{name}: two frames joined five ways "
                          f"(exposure ratio {args.exposure:.2f})")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
