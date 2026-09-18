"""Remove red-eye from a photograph of your own — and see what the detector marks.

    python infer.py photo.jpg
    python infer.py --portrait woman_in_red_scarf
    python infer.py --clean scattered_sweets          # a photo with no red-eye in it
    python infer.py photo.jpg --detector "Colour only (control)"

What is printed for every detector is **how much of the frame it marked**, not
just whether it found the eyes. That is the number this project exists to make
visible: on six photographs containing no eyes at all, the naive colour detector
marks 86,294 pixels and the face-constrained one marks 98.

With `--portrait` the red-eye is planted at recorded pupil positions in one of
this project's six portraits, so there is an exact truth mask and the IoU is
real. With `--clean` the photograph has no red-eye in it and truth is empty, so
every marked pixel is a mistake. On your own photograph there is no truth, so
what you get is each detector's mask, the corrected image, and your photograph's
**pupil-like blob count** — how many small round red things it already contains.
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
from shared.metrics import iou as iou_of  # noqa: E402
from shared.report import init_console  # noqa: E402

import red_eye as re58  # noqa: E402

#: Above this many small round red blobs, the geometric filter has plenty of
#: places to go wrong — measured across this project's twelve photographs,
#: which span 2 to 59.
CLUTTERED = 20


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", default=None)
    ap.add_argument("--portrait", default=None, choices=list(re58.PORTRAITS),
                    help="plant red-eye in one of the project's portraits")
    ap.add_argument("--clean", default=None, choices=list(re58.CLEAN_PHOTOGRAPHS),
                    help="a photograph with no red-eye in it at all")
    ap.add_argument("--detector", default="Face-constrained", choices=list(re58.DETECTORS))
    ap.add_argument("--correction", default="Desaturate (feathered)",
                    choices=list(re58.CORRECTIONS))
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--strength", type=float, default=0.75)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    init_console()

    truth = None
    if args.portrait:
        img, pre_flash, truth = re58.portrait_scene(args.portrait, args.strength)
        source = f"{args.portrait} with red-eye planted"
    elif args.clean:
        img = io.real_photo(args.clean)
        pre_flash = img
        truth = np.zeros(img.shape[:2], np.uint8)
        source = f"{args.clean} (no red-eye in it)"
    elif args.image:
        img = io.imread(args.image)
        pre_flash = None
        source = args.image
    else:
        ap.error("give an image path, --portrait NAME, or --clean NAME")

    blobs = re58.pupil_like_blobs(img, args.threshold)
    print(f"{source}  {img.shape[1]}x{img.shape[0]}")
    print(f"  {blobs} small round red blobs already in the frame")
    if blobs > CLUTTERED:
        print(f"  above {CLUTTERED} — the shape filter has plenty of places to go wrong "
              "here. A pupil is small, round and red; so is a sweet, a brake light and "
              "a poppy.")

    frame_pixels = img.shape[0] * img.shape[1]
    # With an empty truth mask IoU is 1.000 for marking nothing and 0.000 for
    # marking anything, which is not a scale — so the count is the whole story
    # and the column is left out rather than printed as a number to compare.
    scored = truth is not None and bool(truth.any())
    header = f"\n{'detector':24s} {'marked px':>10s} {'% of frame':>11s}"
    if scored:
        header += f" {'IoU':>8s} {'recall':>8s}"
    elif truth is not None:
        header += "   (truth is empty: every marked pixel is a mistake)"
    print(header)

    panels = [("the photograph", ensure_rgb(img))]
    if truth is not None and truth.any():
        marked = ensure_rgb(img).copy()
        marked[truth > 0] = (60, 220, 60)
        panels.append(("true pupils", marked))

    for name, fn in re58.DETECTORS.items():
        pred = fn(img, args.threshold)
        n = int((pred > 0).sum())
        line = f"{name:24s} {n:10d} {100 * n / frame_pixels:10.3f}%"
        if scored:
            line += (f" {iou_of(pred, truth):8.3f}"
                     f" {float((pred[truth > 0] > 0).mean()):8.3f}")
        print(line)

        overlay = ensure_rgb(img).copy()
        sel = pred > 0
        overlay[sel] = (0.35 * overlay[sel] + 0.65 * np.array([255, 40, 40],
                                                             np.float32)).astype(np.uint8)
        panels.append((f"{name}\n{n} px", overlay))

    # ------------------------------------------------------------------ #
    mask, fixed = re58.remove(img, args.detector, args.correction)
    panels.append((f"corrected\n{args.detector} + {args.correction}", fixed))

    print(f"\ncorrected with {args.detector} + {args.correction}")
    if truth is not None and truth.any():
        from shared.metrics import psnr

        sel = truth > 0
        before = float(psnr(img[sel].reshape(-1, 1, 3), pre_flash[sel].reshape(-1, 1, 3)))
        after = float(psnr(fixed[sel].reshape(-1, 1, 3), pre_flash[sel].reshape(-1, 1, 3)))
        print(f"  pupil PSNR {before:.2f} dB before, {after:.2f} dB after "
              f"({after - before:+.2f} dB)")
        if after <= before:
            print("  it made the pupils worse. That is not impossible — zeroing the red "
                  "channel scores below doing nothing on three of this project's six "
                  "portraits — which is why there is a do-nothing control.")
    elif truth is not None:
        changed = int((mask > 0).sum())
        print(f"  {changed} pixels changed in a photograph that had nothing wrong with "
              "it. Truth here is empty, so that is the entire result.")
    else:
        changed = int((mask > 0).sum())
        print(f"  {changed} pixels changed ({100 * changed / frame_pixels:.3f}% of the "
              "frame). There is no truth on your own photograph, so compare the panels: "
              "a correction that marked far more than two pupils has found something "
              "else that is red.")

    out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "removed.png"
    figures.grid(panels, out_path, ncols=3,
                 suptitle=f"Four detectors and one correction on {Path(source).name}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
