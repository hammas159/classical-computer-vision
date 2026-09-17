"""Segment your own image with every method at once.

    python infer.py photo.jpg
    python infer.py photo.jpg --method "SLIC superpixels" --out regions.png
    python infer.py photo.jpg --annotated tiger_in_shade   # score against humans

On your own photograph there is no annotation, so **no score is printed** — only
how many regions each method produced and how much of the frame the largest
region covers, which together say whether a method has over- or under-segmented.

`--annotated` takes the name of one of this project's twelve photographs, which
do have human segmentations, and scores every method against what at least two
of the five annotators agreed was a boundary. That is the metric this project
reports; the IoU column in its README exists to show what the obvious
alternative gets wrong.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import numpy as np  # noqa: E402

from shared import bsds, io  # noqa: E402
from shared.report import init_console  # noqa: E402

import segmentation as sg  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", default=None)
    ap.add_argument("--method", default=None, choices=list(sg.METHODS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--annotated", default=None, choices=list(sg.IMAGES),
                    help="use one of this project's annotated photographs and "
                         "score every method against the human consensus")
    args = ap.parse_args()

    init_console()
    if args.annotated:
        img = sg.load_scene(args.annotated)
        target = bsds.consensus_boundaries(args.annotated)
        ceiling = sg.human_boundary_ceiling(args.annotated)
        print(f"{args.annotated}  {img.shape[1]}x{img.shape[0]}  "
              f"{ceiling['annotators']} human annotators")
        print(f"the best annotator scores F {ceiling['best']:.3f} against the "
              f"consensus of the others — that is the ceiling here, not 1.0")
    elif args.image:
        img = io.imread(args.image)
        target, ceiling = None, None
        print(f"{args.image}  {img.shape[1]}x{img.shape[0]}")
    else:
        ap.error("give an image path, or --annotated with one of the project's photographs")

    names = [args.method] if args.method else list(sg.METHODS)
    header = f"\n{'method':26s} {'regions':>8s} {'largest':>9s} {'ms':>9s}"
    if target is not None:
        header += f" {'F':>7s} {'prec':>7s} {'recall':>7s}"
    print(header)

    outs = {}
    for name in names:
        t0 = time.perf_counter()
        labels = sg.METHODS[name](img)
        ms = (time.perf_counter() - t0) * 1000
        outs[name] = labels
        ids, counts = np.unique(labels[labels > 0], return_counts=True)
        largest = float(counts.max() / labels.size) if len(counts) else 0.0
        line = f"{name:26s} {len(ids):8d} {100 * largest:8.1f}% {ms:9.1f}"
        if target is not None:
            s = bsds.boundary_f_measure(sg.labels_to_boundaries(labels), target)
            line += f" {s['f']:7.3f} {s['precision']:7.3f} {s['recall']:7.3f}"
        print(line)

    if target is not None:
        scored = {n: bsds.boundary_f_measure(sg.labels_to_boundaries(l), target)["f"]
                  for n, l in outs.items()}
        best = max(scored, key=scored.get)
        print(f"\nbest: {best} at F {scored[best]:.3f}, against a human ceiling of "
              f"{ceiling['best']:.3f} — {ceiling['best'] / max(scored[best], 1e-9):.1f}x higher.")
        print("Read precision and recall together. A method that draws boundaries "
              "everywhere has near-perfect recall and says nothing; the grid "
              "control in the README has neither and sits last, which is correct.")
    else:
        sizes = {n: len(np.unique(l[l > 0])) for n, l in outs.items()}
        print("\nNo score is reported. This photograph has no annotation, and a "
              "segmentation has no single right answer to compare against — on "
              "this project's annotated images two people agree at F 0.90, not 1.0.")
        print(f"Region counts here span {min(sizes.values())} to "
              f"{max(sizes.values())}. That spread is the thing to look at: a "
              "method returning thousands of regions has not segmented the "
              "picture, it has tiled it.")
        print("Pass --annotated <name> to score the methods against human "
              "boundaries on one of the twelve photographs that has them.")

    out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "segmented.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panels = [io.ensure_rgb((sg.labels_to_boundaries(outs[n]) * 255).astype(np.uint8))
              for n in names]
    io.imwrite(out_path, panels[0] if args.method else np.hstack(panels))
    print(f"\nwrote {out_path}  (boundary maps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
