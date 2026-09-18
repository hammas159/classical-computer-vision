"""Run every background subtraction method on one frame — with both controls.

    python infer.py --frame 620
    python infer.py --frame 620 --oracle-multiple 4
    python infer.py --frame 200 --history 240
    python infer.py --composite 3

The controls are printed with the real methods rather than left out, because the
point of this project is that **pixel accuracy is beaten by doing nothing**:
foreground is under 4% of the frame, so calling every pixel background scores
0.96 and beats two of the five real methods.

`--oracle-multiple` is the knob that changes the answer. At 1.5× the measured
noise floor the median background wins; at 4× it is last and KNN wins. Nothing
about the data changes — only where the threshold on *truth* goes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import numpy as np  # noqa: E402

from shared import figures  # noqa: E402
from shared.io import ensure_rgb  # noqa: E402
from shared.report import init_console  # noqa: E402

import backsub as bs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", type=int, default=None,
                    help="a frame index of the clip, scored against the oracle")
    ap.add_argument("--composite", type=int, default=None,
                    help=f"one of the {len(bs.SWAPS)} region-swap composites, by index")
    ap.add_argument("--history", type=int, default=bs.HISTORY)
    ap.add_argument("--oracle-multiple", type=float, default=bs.ORACLE_MULTIPLE)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    init_console()
    if not bs.video_available():
        ap.error(f"{bs.VIDEO} is missing. Run "
                 "`python tools/fetch_assets.py --set video` first.")

    if args.composite is not None:
        target, donor, rect = bs.SWAPS[args.composite % len(bs.SWAPS)]
        frame, truth, _ = bs.swap_composite(target, donor, rect)
        index = target
        source = (f"composite: frame {target} with a {rect[2]}x{rect[3]} region of "
                  f"frame {donor} pasted at ({rect[0]},{rect[1]})")
        truth_note = (f"truth is the rectangle intersected with where the pixels "
                      f"actually changed: {(truth > 0).sum():,} of "
                      f"{rect[2] * rect[3]:,} px "
                      f"({(truth > 0).sum() / (rect[2] * rect[3]):.1%})")
    else:
        index = args.frame if args.frame is not None else bs.FRAMES[0]
        frame = bs.load_frame(index)
        truth = bs.oracle_mask(index, multiple=args.oracle_multiple)
        source = f"frame {index} of vtest.avi"
        truth_note = (f"truth is the empty-scene oracle at {args.oracle_multiple:g}x the "
                      f"measured noise floor ({bs.noise_floor():.1f} grey levels)")

    past = bs.load_frames(max(0, index - args.history), index)

    print(f"{source}  {frame.shape[1]}x{frame.shape[0]}")
    print(f"  {len(past)} frames of history for every model")
    print(f"  {truth_note}")
    print(f"  foreground is {100 * float((truth > 0).mean()):.2f}% of the frame")

    print(f"\n{'method':26s} {'IoU':>8s} {'precision':>10s} {'recall':>8s} "
          f"{'pixel acc':>10s}")

    panels = [("the frame", frame), ("truth", ensure_rgb(truth))]
    rows = []
    for name, fn in bs.METHODS.items():
        pred = fn(past, frame)
        s = bs.score(pred, truth)
        rows.append((name, s))
        print(f"{name:26s} {s['iou']:8.4f} {s['precision']:10.4f} {s['recall']:8.4f} "
              f"{s['pixel_accuracy']:10.4f}")
        if "control" not in name:
            panels.append((f"{name}\nIoU {s['iou']:.3f}", ensure_rgb(pred)))

    # ------------------------------------------------------------------ #
    nothing = dict(rows)["All background (control)"]
    beaten = [n for n, s in rows
              if "control" not in n and s["pixel_accuracy"] < nothing["pixel_accuracy"]]
    print(f"\ncalling every pixel background scores {nothing['pixel_accuracy']:.4f} "
          f"pixel accuracy")
    if beaten:
        print(f"  which beats {len(beaten)} real method(s): {', '.join(beaten)}")
        print("  its IoU is 0.0000. That is the number that notices.")

    best = max((r for r in rows if "control" not in r[0]), key=lambda r: r[1]["iou"])
    print(f"\nbest here: {best[0]} at IoU {best[1]['iou']:.4f}")
    if args.composite is None:
        other = 4.0 if args.oracle_multiple < 3.0 else 1.5
        alt_truth = bs.oracle_mask(index, multiple=other)
        alt = max(((n, bs.score(bs.METHODS[n](past, frame), alt_truth))
                   for n in bs.REAL_METHODS), key=lambda r: r[1]["iou"])
        print(f"  with the oracle at {other:g}x instead of {args.oracle_multiple:g}x, "
              f"the best is {alt[0]} at IoU {alt[1]['iou']:.4f}")
        if alt[0] != best[0]:
            print("  — a different method, on the same frame, with the same history. "
                  "Only the threshold on truth moved.")

    out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "masks.png"
    figures.grid(panels, out_path, ncols=4,
                 suptitle=f"Five background models on {source}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
