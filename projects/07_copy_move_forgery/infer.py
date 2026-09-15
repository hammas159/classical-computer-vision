"""Check your own photo for a copy-move forgery — inference from the command line.

    python infer.py suspect.jpg
    python infer.py suspect.jpg --method "Block matching" --out flagged.png
    python infer.py suspect.jpg --all-methods
    python infer.py suspect.jpg --overlay matches.png       # draw the self-matches

**This tool cannot tell you a photograph is genuine, and a flag is not proof.**
Two things it can tell you, and they are the two worth having:

* **How many keypoint pairs agreed on one transform.** A consensus of hundreds
  at a distinctly non-zero rotation is what a rotated copy-move looks like. A
  handful at 0.0 degrees is what a flat sky looks like.
* **What the detector flags on images known to be clean.** Measured on the six
  untampered test photographs, `SIFT + similarity verify` flags 6.3% of pixels on
  average and 34% on a repeating texture, while `Block matching` flags nothing at
  all. Those are the priors your result has to be read against.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import forgery as fg  # noqa: E402
from shared.bench import timeit  # noqa: E402
from shared.io import imread, imwrite  # noqa: E402
from shared.report import init_console  # noqa: E402

MAX_SIDE = 1200


def load(path: str) -> np.ndarray:
    img = imread(path)
    if max(img.shape[:2]) > MAX_SIDE:
        s = MAX_SIDE / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("image", help="path to a photo to check")
    ap.add_argument("--method", default="SIFT + similarity verify", choices=list(fg.METHODS))
    ap.add_argument("--out", default="flagged.png")
    ap.add_argument("--all-methods", action="store_true", help="run every detector")
    ap.add_argument("--overlay", help="write an image with the self-matches drawn on it")
    args = ap.parse_args()

    if not Path(args.image).exists():
        print(f"error: no such file: {args.image}")
        return 2

    img = load(args.image)
    print(f"input   : {args.image}  {img.shape[1]}x{img.shape[0]}")

    fit = fg.describe_match(img)
    print(f"matches : {fit['pairs']} SIFT self-matches, {fit['inliers']} consistent with one transform")
    if fit["angle_deg"] is not None:
        print(f"transform: rotation {fit['angle_deg']:+.2f} deg, scale {fit['scale']:.4f}")

    if args.all_methods:
        print(f"\n{'Method':<28}{'Flagged':>10}{'Time (ms)':>11}")
        print("-" * 49)
        for name, fn in fg.METHODS.items():
            out, t = timeit(lambda f=fn: f(img), runs=1, warmup=0)
            print(f"{name:<28}{float((out > 0).mean()):>9.2%}{t.median_ms:>11.0f}")
        print(
            "\nNo IoU column: an image you supplied has no known forgery mask, so\n"
            "there is nothing to score against. Compare the flagged fractions to\n"
            "what each method flags on images known to be CLEAN -- 0.0% for block\n"
            "matching, 6.3% for SIFT + similarity verify. A method flagging less\n"
            "than its own clean-image baseline has found nothing."
        )
        return 0

    pred, timing = timeit(lambda: fg.METHODS[args.method](img), runs=1, warmup=0)
    flagged = float((pred > 0).mean())
    imwrite(args.out, pred)

    print(f"method  : {args.method}   {timing.median_ms:.0f} ms")
    print(f"flagged : {flagged:.2%} of pixels")
    print(f"wrote   : {args.out}")

    if args.overlay:
        pairs = fg.self_matches(img, fg.make_sift(), cv2.NORM_L2, 0.6, 30)
        overlay = img.copy()
        for p1, p2 in pairs:
            cv2.line(overlay, tuple(np.int32(p1)), tuple(np.int32(p2)), (255, 60, 60), 1)
            cv2.circle(overlay, tuple(np.int32(p1)), 3, (60, 255, 60), -1)
        imwrite(args.overlay, overlay)
        print(f"wrote   : {args.overlay}  ({len(pairs)} matches drawn)")

    # ------------------------------------------------------------------ #
    # how to read the result — each note from a measured baseline
    # ------------------------------------------------------------------ #
    print()
    if flagged == 0.0:
        print(
            "verdict : nothing flagged. That is NOT a clean bill of health.\n"
            f"          `{args.method}` has measured blind spots: block matching sees\n"
            "          nothing at 2 degrees of rotation or 0.95x scale, and every method\n"
            "          here scores 0.000 on a forgery below about 40x40 px. If you\n"
            "          suspect a rotated or small paste, try --all-methods."
        )
    elif args.method == "Block matching":
        print(
            f"verdict : {flagged:.2%} flagged by block matching, which flags 0.00% of\n"
            "          untampered images in this project's test set. This method does not\n"
            "          raise false alarms, so a non-empty result here is worth taking\n"
            "          seriously -- and it only ever fires on an UNROTATED, UNSCALED copy."
        )
    elif flagged > 0.34:
        print(
            f"verdict : {flagged:.2%} flagged -- above the 34% that `{args.method}`\n"
            "          reaches on the worst UNTAMPERED image tested. At this level the\n"
            "          detector is almost certainly responding to repeated texture\n"
            "          (foliage, brickwork, water, fabric) rather than to an edit.\n"
            '          Cross-check with --method "Block matching", which does not.'
        )
    elif flagged > 0.063:
        print(
            f"verdict : {flagged:.2%} flagged, against a 6.3% mean on untampered images\n"
            f"          for this method. Suggestive, not conclusive. The number to look at\n"
            f"          is the inlier count above ({fit['inliers']}) and the fitted transform."
        )
    else:
        print(
            f"verdict : {flagged:.2%} flagged, which is at or below what this method\n"
            "          flags on photographs known to be clean. Treat it as nothing found."
        )

    if fit["angle_deg"] is not None and abs(fit["angle_deg"]) < 0.5 and fit["inliers"] < 20:
        print(
            "\nnote    : the fitted transform is a near-identity on few inliers. That is\n"
            "          the signature of a smooth or self-similar region agreeing with\n"
            "          itself, not of a paste. A real copy-move usually produces a\n"
            "          large translation, and often a non-zero rotation."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
