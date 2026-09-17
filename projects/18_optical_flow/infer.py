"""Compute dense optical flow on your own frames with every method at once.

    python infer.py frame1.jpg frame2.jpg
    python infer.py frame1.jpg frame2.jpg --method DIS --out flow.png
    python infer.py photo.jpg --simulate --px 4      # warp by a known field, score it

Two real frames have no ground-truth flow, so **no endpoint error is printed** —
any number there would be invented. What is printed instead is each method's
median displacement and **how far the methods disagree with each other**, which
is the only signal available without a truth and is labelled as exactly that.

Disagreement is worth having. On this project's generated data the methods
diverge by orders of magnitude once the displacement passes plain LK's 1 px
limit, so a large spread between them is a reliable sign that the motion is
larger than the weakest method can handle.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import numpy as np  # noqa: E402

from shared import io, synth  # noqa: E402
from shared.metrics import endpoint_error  # noqa: E402
from shared.report import init_console  # noqa: E402

import optical_flow as of  # noqa: E402

MARGIN = 16

#: Spread between the methods' median displacements, in pixels, above which the
#: frames are probably moving more than the weakest method can follow.
DISAGREEMENT_WARN = 1.0


def median_displacement(flow: np.ndarray) -> float:
    return float(np.median(np.linalg.norm(flow[MARGIN:-MARGIN, MARGIN:-MARGIN], axis=-1)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("first")
    ap.add_argument("second", nargs="?", default=None,
                    help="the second frame; omit it and pass --simulate instead")
    ap.add_argument("--method", default=None, choices=list(of.METHODS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--simulate", action="store_true",
                    help="ignore the second frame and warp the first by a known "
                         "field, so every method can be scored against a truth")
    ap.add_argument("--px", type=float, default=4.0,
                    help="with --simulate, the displacement to warp by")
    ap.add_argument("--kind", default="sinusoidal", choices=["sinusoidal", "translation"])
    args = ap.parse_args()

    init_console()
    first = io.imread(args.first)
    print(f"{args.first}  {first.shape[1]}x{first.shape[0]}")

    truth = None
    if args.simulate:
        truth = synth.synthetic_flow(first.shape[:2], magnitude=args.px, kind=args.kind)
        # A backward warp moves content by the NEGATIVE of the field, so the
        # forward flow -- what every method predicts -- is `truth`. See
        # `optical_flow.make_pair`; getting this backwards made every method in
        # the project score worse than predicting zero.
        second = synth.warp_by_flow(first, -truth)
        print(f"warped by a {args.kind} field of up to {args.px:g} px")
    else:
        if args.second is None:
            ap.error("give a second frame, or pass --simulate")
        second = io.imread(args.second)
        if second.shape != first.shape:
            ap.error(f"frames differ in size: {first.shape} vs {second.shape}")
        print(f"{args.second}  {second.shape[1]}x{second.shape[0]}")

    print(f"texture of frame 1: {of.texture_of(first):.1f} "
          "(low means little structure for any method to lock onto)")

    names = [args.method] if args.method else list(of.METHODS)
    header = f"\n{'method':24s} {'median px':>10s} {'max px':>9s} {'ms':>8s}"
    if truth is not None:
        header += f" {'EPE':>8s} {'EPE/disp':>9s}"
    print(header)

    flows, medians = {}, {}
    for name in names:
        import time

        t0 = time.perf_counter()
        flow = of.METHODS[name](first, second)
        ms = (time.perf_counter() - t0) * 1000
        flows[name] = flow
        med = median_displacement(flow)
        medians[name] = med
        line = (f"{name:24s} {med:10.3f} "
                f"{float(np.linalg.norm(flow, axis=-1).max()):9.3f} {ms:8.1f}")
        if truth is not None:
            e = endpoint_error(flow[MARGIN:-MARGIN, MARGIN:-MARGIN],
                               truth[MARGIN:-MARGIN, MARGIN:-MARGIN])
            line += f" {e:8.3f} {e / max(args.px, 1e-9):9.3f}"
        print(line)

    if truth is not None:
        control = endpoint_error(
            np.zeros_like(truth)[MARGIN:-MARGIN, MARGIN:-MARGIN],
            truth[MARGIN:-MARGIN, MARGIN:-MARGIN])
        best = min(((endpoint_error(f[MARGIN:-MARGIN, MARGIN:-MARGIN],
                                    truth[MARGIN:-MARGIN, MARGIN:-MARGIN]), n)
                    for n, f in flows.items()))
        print(f"\npredicting zero scores {control:.3f} px — any method above that "
              "line has found nothing.")
        print(f"best here: {best[1]} at {best[0]:.3f} px "
              f"({control / max(best[0], 1e-9):.0f}x better than doing nothing).")
    else:
        spread = max(medians.values()) - min(medians.values()) if len(medians) > 1 else 0.0
        print("\nNo endpoint error is reported. Two real frames have no "
              "ground-truth flow, so any number here would be invented.")
        if len(medians) > 1:
            print(f"The methods' median displacements span {spread:.2f} px "
                  f"({min(medians.values()):.2f} to {max(medians.values()):.2f}).")
            if spread > DISAGREEMENT_WARN:
                print("  -> They disagree substantially. On this project's "
                      "generated data that happens once the motion passes plain "
                      "LK's 1 px limit — trust DIS and the pyramid, not the "
                      "single-scale methods.")
            else:
                print("  -> They agree closely, which is what small motion looks "
                      "like. Any of them is probably fine here.")
        print("Run with --simulate to see them scored against a known field.")

    out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "flow.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scale = max(float(np.linalg.norm(f, axis=-1).max()) for f in flows.values())
    panels = [of.flow_to_colour(flows[n], scale) for n in names]
    io.imwrite(out_path, panels[0] if args.method else np.hstack(panels))
    print(f"\nwrote {out_path}  (hue = direction, brightness = magnitude, "
          "one shared scale)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
