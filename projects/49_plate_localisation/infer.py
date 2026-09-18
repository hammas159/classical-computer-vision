"""Locate the plate in one photograph with all six locators, and score them three ways.

    python infer.py --image eu4          # the largest plate in the set
    python infer.py --image eu1          # a dark garage; watch IoU and coverage part
    python infer.py --image eu10         # 78 pixels wide
    python infer.py car.jpg

Every run prints **IoU and coverage side by side**, because on this data they
disagree about which locator won. Where the photograph is one of the annotated
ones, the character count against the plate's typed text is printed too — the
only measure here that can tell a box that scores well from a box that is useful.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

from shared import figures, io  # noqa: E402
from shared.report import init_console  # noqa: E402

import plates as pl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default=None, help="your own photograph of a car")
    ap.add_argument("--image", default=None,
                    help="one of the project's annotated photographs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    init_console()

    if args.image:
        image, name = pl.load(args.image), args.image
        target, text = pl.truth(args.image)
    elif args.path:
        image, name = io.imread(args.path), Path(args.path).name
        target, text = None, None
    else:
        ap.error("give an image path, or --image with one of the project's photographs")

    h, w = image.shape[:2]
    print(f"{name}  {w}x{h}")
    if target is not None:
        print(f"  annotated plate '{text}': {target[2]}x{target[3]} px, "
              f"{100 * target[2] * target[3] / (w * h):.2f}% of the frame, "
              f"aspect {target[2] / target[3]:.1f}")
        proxy = pl.character_blobs(image, target)
        chars = sum(1 for c in text if c.isalnum())
        print(f"  on a crop of the annotation itself the blob counter finds {proxy} "
              f"characters against {chars} typed"
              + ("" if abs(proxy - chars) <= 1
                 else "  <- the proxy is wrong on this one, so read its column "
                      "sceptically below"))
    else:
        print("  no annotation for this photograph, so only the boxes are shown")

    panels = [("the photograph" if target is None else "the annotation",
               image if target is None else pl.draw_truth(image, target))]

    if target is not None:
        print(f"\n{'locator':24s} {'boxes':>6s} {'IoU':>6s} {'covers':>7s} "
              f"{'IoU>=.5':>8s} {'blobs':>6s}")
    else:
        print(f"\n{'locator':24s} {'boxes':>6s}  largest candidate")

    scored = {}
    for locator in pl.REAL_LOCATORS:
        boxes = pl.LOCATORS[locator](image)
        if target is None:
            biggest = max(boxes, key=lambda b: b[2] * b[3]) if boxes else None
            print(f"{locator:24s} {len(boxes):6d}  "
                  + (f"{biggest[2]}x{biggest[3]} at ({biggest[0]}, {biggest[1]})"
                     if biggest else "nothing found"))
            panels.append((f"{locator}\n{len(boxes)} candidates",
                           pl.draw(image, boxes)))
            continue

        best, v_iou = pl.best_by(boxes, target, pl.iou)
        cov = pl.coverage(best, target) if best is not None else 0.0
        blobs = pl.character_blobs(image, best) if best is not None else 0
        scored[locator] = (v_iou, cov, blobs)
        print(f"{locator:24s} {len(boxes):6d} {v_iou:6.2f} {cov:7.2f} "
              f"{('yes' if v_iou >= 0.5 else 'NO'):>8s} {blobs:6d}")
        drawn = pl.draw_truth(image, target)
        if best is not None:
            drawn = pl.draw(drawn, [best],
                            colour=(60, 220, 90) if v_iou >= 0.5 else (235, 70, 70))
        panels.append((f"{locator}\nIoU {v_iou:.2f}  covers {cov:.2f}"
                       if best is not None else f"{locator}\nnothing found", drawn))

    # ------------------------------------------------------------------ #
    if scored:
        chars = sum(1 for c in text if c.isalnum())
        by_iou = max(scored, key=lambda k: scored[k][0])
        by_cov = max(scored, key=lambda k: scored[k][1])
        if by_iou != by_cov:
            print(f"\nIoU says {by_iou} ({scored[by_iou][0]:.2f}); "
                  f"coverage says {by_cov} ({scored[by_cov][1]:.2f}).")
            print("  They disagree on this photograph, which is the project's whole "
                  "point.")
        else:
            print(f"\nBoth metrics agree here: {by_iou}.")

        readable = [k for k, (_, _, b) in scored.items() if abs(b - chars) <= 1]
        if readable:
            print(f"  Crops with a usable character count ({chars} expected): "
                  + ", ".join(readable))
        else:
            print(f"  No crop yields a usable character count ({chars} expected) — "
                  "including the\n  ones that pass IoU >= 0.5.")

        whole = pl.coverage((0, 0, w, h), target)
        print(f"  For scale: returning the whole photograph scores coverage "
              f"{whole:.2f} and IoU "
              f"{pl.iou((0, 0, w, h), target):.2f}.")

    out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "located.png"
    figures.grid(panels, out_path, ncols=4,
                 suptitle=f"{name}" + (" — amber is the box a person drew"
                                       if target is not None else ""))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
