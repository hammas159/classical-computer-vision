"""Run every tracker on one run of the clip — and see the three metrics disagree.

    python infer.py --start 210
    python infer.py --start 550 --length 100
    python infer.py --start 210 --tracker "MOSSE (from scratch)"

Mean IoU, centre error and survival rate are printed side by side, with the
do-nothing control, because this project's result is that they do not agree:
over twelve runs template matching wins on mean IoU, MOSSE on survival and
optical flow on centre error.

`alive IoU` is measured from **frame 1**, because frame 0 is handed to every
tracker and scores 1.0 by construction. Counting it made CamShift look like the
most accurate tracker in the project on the strength of the one frame it was
given — the kind of number that reads as a finding and is an artefact.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from shared import figures  # noqa: E402
from shared.io import ensure_rgb  # noqa: E402
from shared.report import init_console  # noqa: E402

import tracking as tk  # noqa: E402


def _drawn(frame, box, truth):
    out = ensure_rgb(frame).copy()
    if truth is not None:
        x, y, w, h = (int(v) for v in truth)
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 255), 4)
    if box is not None:
        alive = truth is not None and tk.box_iou(box, truth) >= tk.SURVIVAL_IOU
        x, y, w, h = (int(v) for v in box)
        cv2.rectangle(out, (x, y), (x + w, y + h),
                      (60, 200, 60) if alive else (220, 60, 60), 2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=tk.STARTS[3],
                    help=f"first frame of the run (measured good starts: {tk.STARTS})")
    ap.add_argument("--length", type=int, default=tk.RUN_LENGTH)
    ap.add_argument("--tracker", default=None, choices=list(tk.TRACKERS),
                    help="draw only this one, frame by frame")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    init_console()
    if not tk.video_available():
        ap.error(f"{tk.VIDEO} is missing. Run "
                 "`python tools/fetch_assets.py --set video` first.")

    frames, truth = tk.truth_chain(args.start, args.length)
    if not frames or truth[0] is None:
        ap.error(f"no trackable target at frame {args.start}; "
                 f"measured good starts are {tk.STARTS}")

    alive_frames = sum(1 for b in truth if b is not None)
    print(f"run from frame {args.start}, {len(frames)} frames, "
          f"{frames[0].shape[1]}x{frames[0].shape[0]}")
    print(f"  initial box {truth[0]}")
    print(f"  the truth chain has a box in {alive_frames} of {len(frames)} frames")

    hsv = cv2.cvtColor(tk._crop(frames[0], truth[0]), cv2.COLOR_RGB2HSV)
    saturation = float(hsv[..., 1].mean())
    print(f"  target mean saturation {saturation:.0f} of 255")
    if saturation < 60:
        print("    — too little hue for the colour trackers to work with; expect "
              "mean shift and CamShift to wander")

    print(f"\n{'tracker':24s} {'mean IoU':>9s} {'alive IoU':>10s} {'centre px':>10s} "
          f"{'survived':>9s}")

    results = {}
    for name, fn in tk.TRACKERS.items():
        boxes = fn(frames, truth[0])
        s = tk.score_run(boxes, truth)
        alive = []
        # from frame 1: frame 0 is given to every tracker and its IoU is 1.0
        for p, t in zip(boxes[1:], truth[1:]):
            if t is None:
                continue
            value = tk.box_iou(p, t)
            if value < tk.SURVIVAL_IOU:
                break
            alive.append(value)
        alive_iou = float(np.mean(alive)) if alive else 0.0
        results[name] = (boxes, s, alive_iou)
        print(f"{name:24s} {s['mean_iou']:9.4f} {alive_iou:10.4f} "
              f"{s['centre_error_px']:10.2f} {s['frames_survived']:6d}/{len(frames)}")

    # ------------------------------------------------------------------ #
    real = {n: v for n, v in results.items() if "control" not in n}
    by_iou = max(real, key=lambda n: real[n][1]["mean_iou"])
    by_survival = max(real, key=lambda n: real[n][1]["frames_survived"])
    by_centre = min(real, key=lambda n: real[n][1]["centre_error_px"])
    print(f"\nbest by mean IoU:     {by_iou}")
    print(f"best by survival:     {by_survival}")
    print(f"best by centre error: {by_centre}")
    if len({by_iou, by_survival, by_centre}) > 1:
        print("  — the three metrics do not agree on this run either.")

    accurate_but_short = max(
        real, key=lambda n: real[n][2] - real[n][1]["survival_rate"])
    boxes, s, alive_iou = real[accurate_but_short]
    if alive_iou - s["survival_rate"] > 0.2:
        print(f"\n{accurate_but_short} scores {alive_iou:.3f} IoU while it is on the "
              f"target and stays there for {s['frames_survived']} of {len(frames)} "
              "frames. A mean over the whole run reports neither.")

    static = results["Static box (control)"][1]
    worse = [n for n in real if real[n][1]["mean_iou"] < static["mean_iou"]]
    if worse:
        print(f"\nthe do-nothing control scores {static['mean_iou']:.3f} here and "
              f"beats: {', '.join(worse)}")

    # ------------------------------------------------------------------ #
    if args.tracker:
        boxes = results[args.tracker][0]
        steps = np.linspace(0, len(frames) - 1, 6).round().astype(int)
        panels = [(f"frame {int(i)}", _drawn(frames[i], boxes[i], truth[i]))
                  for i in steps]
        suptitle = f"{args.tracker} on the run from frame {args.start}"
        ncols = 3
    else:
        step = min(len(frames) - 1, 45)
        panels = [(f"{n}\nIoU {tk.box_iou(v[0][step], truth[step]):.3f}"
                   if truth[step] is not None else n,
                   _drawn(frames[step], v[0][step], truth[step]))
                  for n, v in results.items()]
        suptitle = (f"Seven trackers at frame {step} of the run from {args.start}. "
                    "White is truth.")
        ncols = 4

    out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "tracked.png"
    figures.grid(panels, out_path, ncols=ncols, suptitle=suptitle)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
