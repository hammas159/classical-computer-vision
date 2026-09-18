"""Run all six cascades on one photograph and say what each one is doing.

    python infer.py --image class57
    python infer.py --image class57 --upscale 2
    python infer.py --image addams-family --rotate 20
    python infer.py --no-face 140075
    python infer.py photo.jpg

`--no-face` picks one of the eleven photographs with no human face in them, so
the true answer is no boxes and everything printed is a false alarm.

`--upscale` resamples before the search. It is the only thing that helps a
cascade whose training window is larger than the faces in front of it, and
`--rotate` is the thing that helps none of them.
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
from shared.report import init_console  # noqa: E402

import faces as fc  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default=None, help="your own photograph")
    ap.add_argument("--image", default=None,
                    help="one of the project's group photographs")
    ap.add_argument("--no-face", default=None, choices=sorted(fc.NO_FACE),
                    help="one of the photographs with no human face in it")
    ap.add_argument("--upscale", type=float, default=1.0)
    ap.add_argument("--rotate", type=float, default=0.0)
    ap.add_argument("--min-neighbors", type=int, default=fc.MIN_NEIGHBORS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    init_console()

    if args.image:
        image, name, empty = fc.load(args.image), args.image, False
    elif args.no_face:
        image, name, empty = fc.load_no_face(args.no_face), args.no_face, True
    elif args.path:
        image, name, empty = io.imread(args.path), Path(args.path).name, False
    else:
        ap.error("give an image path, --image, or --no-face")

    if args.rotate:
        M = fc.rotation_matrix(image.shape, args.rotate)
        h, w = image.shape[:2]
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    h, w = image.shape[:2]
    print(f"{name}  {w}x{h}"
          + (f", rotated {args.rotate:g} degrees" if args.rotate else "")
          + (f", searched at {args.upscale:g}x" if args.upscale != 1.0 else ""))
    if empty:
        print("  there is no human face in this photograph "
              f"({fc.NO_FACE[args.no_face]}), so every box below is a false alarm")

    print(f"\n{'cascade':16s} {'window':>7s} {'boxes':>6s} {'smallest':>9s} "
          f"{'largest':>8s}")
    panels = [("the photograph", image)]
    counts = {}
    for method in fc.DETECTORS:
        boxes = fc.detect(image, method, min_neighbors=args.min_neighbors,
                          upscale=args.upscale)
        counts[method] = len(boxes)
        sizes = [b[2] for b in boxes]
        window = fc.training_window(method)
        print(f"{method:16s} {window:3d}x{window:<3d} {len(boxes):6d} "
              f"{(f'{min(sizes):.0f}px' if sizes else '—'):>9s} "
              f"{(f'{max(sizes):.0f}px' if sizes else '—'):>8s}")
        panels.append((f"{method}\n{len(boxes)} box" + ("" if len(boxes) == 1 else "es"),
                       fc.draw(image, boxes,
                               colour=(235, 70, 70) if empty else (60, 220, 90))))

    # ------------------------------------------------------------------ #
    blind = [m for m, c in counts.items() if c == 0]
    if blind and not empty:
        biggest = max(counts, key=counts.get)
        for method in blind:
            window = fc.training_window(method)
            hint = (f"  — its training window is {window}x{window}; try --upscale 2"
                    if window > 24 else "")
            print(f"\n{method} found nothing where {biggest} found "
                  f"{counts[biggest]}.{hint}")

    if empty:
        total = sum(counts.values())
        if total == 0:
            print("\nAll six returned nothing, which is the right answer here.")
        else:
            worst = max(counts, key=counts.get)
            print(f"\n{total} false alarms across six cascades at "
                  f"minNeighbors={args.min_neighbors}, worst is {worst} "
                  f"with {counts[worst]}.")
            if args.min_neighbors > 1:
                print("  Run again with --min-neighbors 1 to see the other end of the "
                      "trade; that\n  sweep is the point of the project's "
                      "operating-point figure.")
    elif args.rotate:
        print(f"\nAt {args.rotate:g} degrees. Run again without --rotate to see what "
              "each of these\nfound before the photograph was tilted — the cascades "
              "are trained upright.")

    out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "detected.png"
    figures.grid(panels, out_path, ncols=4,
                 suptitle=f"{name}" + (" — no human face here, so every box is a "
                                       "false alarm" if empty else ""))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
