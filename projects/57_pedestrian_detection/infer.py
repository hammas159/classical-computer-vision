"""Detect pedestrians in an image of your own — and see the margin, not just the box.

    python infer.py photo.jpg
    python infer.py --frame 600                 # a frame of this project's clip
    python infer.py --frame 600 --threshold 1.0
    python infer.py --synthetic                 # the drawn scene, for comparison

What is printed for every detection is the **SVM margin**, because that is the
number the box hides. On this project's real footage the mean margin is 1.59 and
40 of 52 detections clear 0.5; on drawn silhouettes it is 0.51 and 5 of 13. If
your own image scores like the drawings, the detector is not seeing people in it
— it is returning whatever the window liked best.

With `--frame`, background subtraction over the preceding 40 frames gives
**independent evidence**: the camera is static, so a region that moves is an
object, and that is established without any appearance model. HOG cannot see
motion, so agreement between the two is not circular. Boxes are drawn green when
a moving region supports them and red when nothing does.
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

import pedestrian as ped  # noqa: E402

#: Below this, a detection looks like the drawn silhouettes rather than like a
#: person: measured, not chosen — see `synthetic_versus_real` in the source.
DRAWING_LIKE = 0.51
PERSON_LIKE = 1.59


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", default=None)
    ap.add_argument("--frame", type=int, default=None,
                    help="a frame index of this project's clip (motion evidence too)")
    ap.add_argument("--synthetic", action="store_true",
                    help="a drawn scene, to see what a non-person scores")
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--scale", type=float, default=1.05)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    init_console()

    blobs = None
    if args.frame is not None:
        if not ped.video_available():
            ap.error(f"{ped.VIDEO} is missing. Run "
                     "`python tools/fetch_assets.py --set video` first.")
        img = ped.load_frame(args.frame)
        blobs, mask = ped.moving_blobs(args.frame)
        source = f"frame {args.frame} of vtest.avi"
    elif args.synthetic:
        img, truth = ped.make_scene(n_people=3, seed=0)
        source = "a drawn scene (3 silhouettes)"
    elif args.image:
        img = io.imread(args.image)
        source = args.image
    else:
        ap.error("give an image path, --frame N, or --synthetic")

    boxes, scores = ped.detect(img, scale=args.scale, hit_threshold=args.threshold)
    tight = [ped.tighten_box(b) for b in boxes]

    print(f"{source}  {img.shape[1]}x{img.shape[0]}")
    print(f"  threshold {args.threshold:+.2f}, pyramid step {args.scale:g}")
    print(f"\n{len(tight)} detections")
    if not len(tight):
        print("  nothing above the threshold. Lower it with --threshold -0.5 to see "
              "what the detector was closest to accepting.")

    header = f"\n{'#':>2s} {'box (x,y,w,h)':>22s} {'margin':>8s}"
    if blobs is not None:
        header += f" {'moves?':>8s}"
    print(header)

    colours, supported_count = [], 0
    order = np.argsort(scores)[::-1] if len(scores) else []
    for rank, i in enumerate(order, start=1):
        box = tight[i]
        line = f"{rank:2d} {str(tuple(int(v) for v in box)):>22s} {scores[i]:8.3f}"
        colour = (220, 160, 60)
        if blobs is not None:
            iou = max((ped.box_iou(box, m) for m in blobs), default=0.0)
            hit = iou >= 0.3
            supported_count += int(hit)
            colour = (60, 200, 60) if hit else (220, 60, 60)
            line += f" {('yes' if hit else 'no'):>8s}"
        print(line)
        colours.append((box, colour))

    # ------------------------------------------------------------------ #
    # the number the box hides
    # ------------------------------------------------------------------ #
    if len(scores):
        mean = float(np.mean(scores))
        confident = int((scores > 0.5).sum())
        print(f"\nmean margin {mean:.3f}, {confident} of {len(scores)} above 0.5")
        if mean < (DRAWING_LIKE + PERSON_LIKE) / 2:
            print(f"  that is closer to the {DRAWING_LIKE:.2f} this project measures on "
                  f"drawn silhouettes than to the {PERSON_LIKE:.2f} it measures on real "
                  "pedestrians. Treat these boxes as the window's best guess rather "
                  "than as people.")
        else:
            print(f"  in the range this project measures on real pedestrians "
                  f"({PERSON_LIKE:.2f} mean over 52 detections).")

    if blobs is not None:
        print(f"\n{supported_count} of {len(tight)} detections sit on one of "
              f"{len(blobs)} person-sized moving regions")
        if supported_count < len(tight):
            print("  the unsupported ones are either false positives or people standing "
                  "still — background subtraction cannot tell those apart, which is why "
                  "this is called evidence and not truth.")

    # ------------------------------------------------------------------ #
    panels = [("the scene", ensure_rgb(img))]
    if blobs is not None:
        marked = ensure_rgb(mask).copy()
        for x, y, w, h in blobs:
            cv2.rectangle(marked, (x, y), (x + w, y + h), (60, 200, 60), 2)
        panels.append((f"{len(blobs)} moving regions", marked))

    drawn = ensure_rgb(img).copy()
    for (x, y, w, h), colour in colours:
        cv2.rectangle(drawn, (x, y), (x + w, y + h), colour, 3)
    panels.append((f"{len(tight)} detections", drawn))

    gray = cv2.cvtColor(ensure_rgb(img), cv2.COLOR_RGB2GRAY)
    panels.append(("what the SVM sees", ensure_rgb(ped.hog_visualisation(gray))))

    out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "detected.png"
    figures.grid(panels, out_path, ncols=2,
                 suptitle=f"HOG + linear SVM on {source}, threshold {args.threshold:+.2f}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
