"""Stabilise one segment — and see what it cost, and what it invented.

    python infer.py
    python infer.py --estimator "Block matching" --smoother "Kalman (causal)"
    python infer.py --shake 8
    python infer.py --no-shake

Three numbers are printed together because none of them means much alone:

* **residual jitter** — how steady the output is;
* **crop** — how much of the frame had to be thrown away to get it;
* **estimation error** — how well the motion itself was recovered, which on this
  footage is almost never the limiting factor.

`--no-shake` is the one worth running. It stabilises footage that never moved, so
the true motion is exactly zero and everything the estimator reports is invented.
Because a camera path is an integral, a small per-frame bias becomes a visible
drift.
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

import stabilise as st  # noqa: E402


def _edges(frames, indices):
    stack = [cv2.Canny(cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY), 80, 160)
             .astype(np.float32) for i in indices]
    return ensure_rgb(np.clip(np.mean(stack, axis=0) * 1.6, 0, 255).astype(np.uint8))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=st.STARTS[0],
                    help=f"first frame of the segment (this project uses {st.STARTS})")
    ap.add_argument("--frames", type=int, default=st.SEGMENT)
    ap.add_argument("--estimator", default="Features + LK + RANSAC",
                    choices=list(st.ESTIMATORS))
    ap.add_argument("--smoother", default="Gaussian (sigma=8)",
                    choices=list(st.SMOOTHERS))
    ap.add_argument("--shake", type=float, default=st.SHAKE_TRANSLATION,
                    help="translation step standard deviation, in pixels")
    ap.add_argument("--no-shake", action="store_true",
                    help="stabilise the untouched clip: the true motion is zero")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    init_console()
    if not st.video_available():
        ap.error(f"{st.VIDEO} is missing. Run "
                 "`python tools/fetch_assets.py --set video` first.")

    if args.no_shake:
        original = st.load_frames(args.start, args.frames)
        shaken = original
        truth = np.zeros((len(original), 3))
        print(f"frames {args.start}-{args.start + len(original)} of vtest.avi, "
              "UNSHAKEN")
        print("  the camera never moved, so every motion reported below is invented")
    else:
        original, shaken, truth = st.shaken_segment(
            args.start, seed=args.seed, count=args.frames,
            translation=args.shake,
            rotation=st.SHAKE_ROTATION * args.shake / st.SHAKE_TRANSLATION)
        steps = st.path_steps(truth)
        print(f"frames {args.start}-{args.start + len(original)} of vtest.avi, "
              f"shaken by a known path")
        print(f"  shake: {steps[:, :2].std():.2f} px and {steps[:, 2].std():.3f} deg "
              f"per frame; the path wanders up to "
              f"{np.abs(truth[:, :2]).max():.1f} px")

    print(f"  estimator: {args.estimator}")
    print(f"  smoother:  {args.smoother}")

    out, estimated, smoothed = st.stabilise(shaken, args.estimator, args.smoother)
    effective = st.effective_path(truth, estimated, smoothed)
    crop = st.required_crop(smoothed - estimated, shaken[0].shape)

    error = np.abs(st.path_steps(estimated) - st.path_steps(truth))
    print(f"\n  estimation error   {error[:, :2].mean():.4f} px/frame, "
          f"{error[:, 2].mean():.4f} deg/frame")
    print(f"  input jitter       {st.jitter(truth):.4f}")
    print(f"  residual jitter    {st.jitter(effective):.4f}")
    if st.jitter(truth) > 1e-6:
        print(f"  reduction          "
              f"{st.jitter(truth) / max(st.jitter(effective), 1e-9):.1f}x")
    print(f"  field of view lost {100 * crop:.2f}%")

    # ------------------------------------------------------------------ #
    if args.no_shake:
        invented = float(np.mean(np.abs(st.path_steps(estimated)[:, :2])))
        drift = float(np.max(np.abs(estimated[:, :2])))
        print(f"\n  it reported {invented:.4f} px of motion per frame and "
              f"{drift:.2f} px of accumulated drift")
        if drift > 1.0:
            print(f"  — that is {drift:.0f} px of wander introduced into a shot that "
                  "never moved. A camera path is an integral, so a small consistent "
                  "bias never cancels.")
        else:
            print("  — small enough not to matter here, but it is not zero, and the "
                  "true answer is.")
    else:
        alt = ("Kalman (causal)" if args.smoother != "Kalman (causal)"
               else "Gaussian (sigma=8)")
        _, est2, sm2 = st.stabilise(shaken, args.estimator, alt)
        other = st.jitter(st.effective_path(truth, est2, sm2))
        print(f"\n  with '{alt}' instead of '{args.smoother}': residual jitter "
              f"{other:.4f}")
        ratio = max(other, st.jitter(effective)) / max(min(other, st.jitter(effective)),
                                                       1e-9)
        print(f"  — a factor of {ratio:.1f} from the smoother alone, on identical "
              "motion estimates")

        worst = max(st.ESTIMATORS, key=lambda n: 0 if "control" in n else 1)
        del worst
        alt_est = ("Phase correlation" if args.estimator != "Phase correlation"
                   else "Features + LK + RANSAC")
        _, est3, sm3 = st.stabilise(shaken, alt_est, args.smoother)
        other_est = st.jitter(st.effective_path(truth, est3, sm3))
        print(f"  with '{alt_est}' instead of '{args.estimator}': residual jitter "
              f"{other_est:.4f}")

    indices = [i for i in (12, 24, 36, 48) if i < len(original)]
    panels = [("the input", ensure_rgb(shaken[indices[0]])),
              ("stabilised", ensure_rgb(out[indices[0]])),
              ("input: 4 frames of edges", _edges(shaken, indices)),
              ("stabilised: 4 frames of edges", _edges(out, indices))]

    out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "stabilised.png"
    figures.grid(panels, out_path, ncols=2,
                 suptitle=(f"{args.estimator} + {args.smoother}, "
                           f"{100 * crop:.1f}% of frame lost"))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
