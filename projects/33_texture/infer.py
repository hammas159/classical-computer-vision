"""Classify a texture patch of your own against this project's twelve surfaces.

    python infer.py my_surface.jpg
    python infer.py my_surface.jpg --descriptor "LBP (uniform)"
    python infer.py my_surface.jpg --relight 0.6      # see which one survives it
    python infer.py --plate brick_paving              # a held-out crop of a known plate

There is no training and no model file. Your image is cut into patches, each is
described, and each is matched to the nearest of 144 patches cut from the twelve
plates. The "prediction" is a distance computation.

**Read the margin, not the label.** With twelve classes, chance is 0.083 and
something always wins — a photograph of a surface that is not in the set will
still be assigned one of the twelve. The margin printed next to each answer is
the ratio between the nearest wrong-class distance and the nearest right-class
one; below about 1.05 the answer is a coin toss with extra steps.

`--relight` is the experiment worth running on your own picture: it dims the
patch before describing it and leaves the gallery alone, which is the protocol
the project's invariance table uses. LBP should barely move and everything else
should collapse.
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
from shared.io import to_float, to_gray, to_uint8  # noqa: E402
from shared.report import init_console  # noqa: E402

import texture as tx  # noqa: E402

#: Below this ratio between the best wrong class and the best right one, the
#: answer carries no information worth printing as a label.
MARGIN_FLOOR = 1.05


def patches_from(img: np.ndarray, patch: int, limit: int = 12) -> list[np.ndarray]:
    """Cut a grid of patches from the middle of an arbitrary image."""
    gray = to_gray(img)
    h, w = gray.shape
    if min(h, w) < patch:
        raise SystemExit(f"image is {w}x{h}; it must be at least {patch} px on both sides")
    step = max(patch, min(h, w) // 4)
    coords = [(y, x)
              for y in range(0, h - patch + 1, step)
              for x in range(0, w - patch + 1, step)]
    return [gray[y : y + patch, x : x + patch].copy() for y, x in coords[:limit]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", default=None)
    ap.add_argument("--plate", default=None, choices=list(tx.TEXTURES),
                    help="use held-out crops of one of the project's own plates")
    ap.add_argument("--descriptor", default=None, choices=list(tx.DESCRIPTORS))
    ap.add_argument("--patch", type=int, default=tx.PATCH)
    ap.add_argument("--relight", type=float, default=0.0,
                    help="dim the probe by this fraction before describing it")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    init_console()

    gallery, _, labels = tx.build_split(patch=args.patch)
    print(f"gallery: {len(gallery)} patches, {args.patch}x{args.patch}, "
          f"{len(tx.TEXTURES)} surfaces — chance is {1 / len(tx.TEXTURES):.3f}")

    if args.plate:
        _, probes, probe_labels = tx.build_split(patch=args.patch)
        keep = [i for i, v in enumerate(probe_labels) if tx.TEXTURES[v] == args.plate]
        probes = [probes[i] for i in keep]
        truth = args.plate
        source = f"held-out crops of {args.plate}"
    elif args.image:
        probes = patches_from(io.imread(args.image), args.patch)
        truth = None
        source = args.image
    else:
        ap.error("give an image path, or --plate with one of the project's surfaces")

    if args.relight:
        probes = [to_uint8(to_float(p) * (1.0 - args.relight) + args.relight * 0.1)
                  for p in probes]
        print(f"probes dimmed by {args.relight:g} — the gallery is untouched, which is "
              "the protocol the invariance table uses")

    print(f"{source}: {len(probes)} patches\n")

    descriptors = [args.descriptor] if args.descriptor else list(tx.DESCRIPTORS)
    header = f"{'descriptor':26s} {'verdict':28s} {'votes':>7s} {'margin':>8s}"
    if truth:
        header += f" {'correct':>8s}"
    print(header)

    verdicts = {}
    for name in descriptors:
        fn = tx.DESCRIPTORS[name]
        g = tx._features(gallery, fn)
        p = tx._features(probes, fn)
        mean = g.mean(0, keepdims=True)
        std = np.maximum(g.std(0, keepdims=True), tx.EPS)
        d = np.linalg.norm(((p - mean) / std)[:, None, :] - ((g - mean) / std)[None, :, :],
                           axis=2)

        predicted = labels[np.argmin(d, axis=1)]
        counts = np.bincount(predicted, minlength=len(tx.TEXTURES))
        winner = int(np.argmax(counts))

        # margin: nearest distance to the winning class against nearest to any
        # other class, averaged over the patches. One patch's answer is noise.
        best_in, best_out = [], []
        for row in d:
            same = row[labels == winner]
            other = row[labels != winner]
            best_in.append(same.min())
            best_out.append(other.min())
        margin = float(np.mean(best_out) / max(np.mean(best_in), tx.EPS))

        label = tx.TEXTURES[winner].replace("_", " ")
        if margin < MARGIN_FLOOR:
            label += " (no margin)"
        line = f"{name:26s} {label:28s} {counts[winner]:3d}/{len(probes):<3d} {margin:8.3f}"
        if truth:
            line += f" {'yes' if tx.TEXTURES[winner] == truth else 'NO':>8s}"
        print(line)
        verdicts[name] = (tx.TEXTURES[winner], margin, counts[winner] / len(probes))

    # ------------------------------------------------------------------ #
    # what the numbers mean for this particular picture
    # ------------------------------------------------------------------ #
    print()
    agreed = {v[0] for v in verdicts.values()}
    if len(agreed) == 1:
        print(f"All {len(verdicts)} descriptors agree on "
              f"{next(iter(agreed)).replace('_', ' ')}.")
    elif truth:
        print(f"The descriptors disagree — {len(agreed)} different answers on a surface "
              "that IS in the set. Each of them is invariant to something different, so "
              "a condition that breaks one leaves another untouched.")
    else:
        print(f"The descriptors disagree — {len(agreed)} different answers. That is the "
              "normal outcome for a surface that is not one of the twelve, because "
              "each of them is measuring a different thing.")

    weak = [n for n, (_, m, _) in verdicts.items() if m < MARGIN_FLOOR]
    if weak:
        print(f"{len(weak)} of {len(verdicts)} have no margin worth the name "
              f"(<{MARGIN_FLOOR:g}): {', '.join(w.split()[0] for w in weak)}. "
              "A nearest-neighbour classifier always returns something.")

    if args.relight:
        print("\nCompare this run against one without --relight. LBP keeps only the sign "
              "of local differences, so its answer should barely move; the others read "
              "absolute intensity and should not survive.")

    if truth:
        right = [n for n, (v, _, _) in verdicts.items() if v == truth]
        print(f"\n{len(right)} of {len(verdicts)} correct on a surface that IS in the set.")

    out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "infer.png"
    panels = [("your patch" if not args.plate else "held-out crop",
               cv2.resize(probes[0], None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))]
    for name in descriptors[:4]:
        winner = verdicts[name][0]
        idx = int(np.flatnonzero(labels == tx.TEXTURES.index(winner))[0])
        panels.append((f"{name.split()[0]} says\n{winner.replace('_', ' ')}",
                       cv2.resize(gallery[idx], None, fx=6, fy=6,
                                  interpolation=cv2.INTER_NEAREST)))
    figures.grid(panels, out_path, ncols=len(panels),
                 suptitle="What each descriptor matched your patch to")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
