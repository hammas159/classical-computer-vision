"""Align two images of your own, with all four methods.

    python infer.py reference.jpg moving.jpg
    python infer.py reference.jpg moving.jpg --method "Mutual information"
    python infer.py --photo stone_guardian --modality inverted

With `--photo` the transform is applied here, so the error printed is real. On
your own pair there is no ground truth, so what is printed instead is the
**mutual information between the two images before and after each alignment**.
MI rises when an alignment is right whatever the intensities are doing, which
makes it the one self-check available without truth.

It also reports whether your two images have a monotonic intensity relationship,
because that single fact decides whether you need mutual information's 426x cost:
on this project's photographs everything handles a gamma remap, and only MI
survives a non-monotonic one.
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
from shared.io import to_gray  # noqa: E402
from shared.report import init_console  # noqa: E402

import registration as rg  # noqa: E402

#: Spearman correlation below which the two images do not have a monotonic
#: intensity relationship, and only mutual information will align them.
MONOTONIC = 0.5

#: Texture energy below which there may not be enough structure to align at all.
FLAT = 8.0


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation — positive or negative, any monotonic map scores +-1."""
    x = to_gray(a).ravel().astype(np.float64)
    y = to_gray(b).ravel().astype(np.float64)
    step = max(1, x.size // 50_000)
    x, y = x[::step], y[::step]
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference", nargs="?", default=None)
    ap.add_argument("moving", nargs="?", default=None)
    ap.add_argument("--photo", default=None, choices=list(rg.IMAGES))
    ap.add_argument("--modality", default="same",
                    choices=("same", "gamma", "inverted", "synthetic_mri"))
    ap.add_argument("--dx", type=float, default=7.0)
    ap.add_argument("--dy", type=float, default=4.0)
    ap.add_argument("--method", default=None, choices=list(rg.METHODS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    init_console()
    if args.photo:
        reference, moving, truth = rg.make_pair(args.photo, dx=args.dx, dy=args.dy,
                                                modality=args.modality)
        source = f"{args.photo} ({args.modality})"
    elif args.reference and args.moving:
        reference = io.imread(args.reference)
        moving = io.imread(args.moving)
        truth = None
        source = f"{Path(args.reference).name} / {Path(args.moving).name}"
        if reference.shape[:2] != moving.shape[:2]:
            h = min(reference.shape[0], moving.shape[0])
            w = min(reference.shape[1], moving.shape[1])
            print(f"cropping both to {w}x{h} — phase correlation needs matching sizes")
            reference, moving = reference[:h, :w], moving[:h, :w]
    else:
        ap.error("give a reference and a moving image, or --photo")

    energy = rg.texture_energy(reference)
    rank = _spearman(reference, moving)
    print(f"{source}  {reference.shape[1]}x{reference.shape[0]}")
    print(f"texture energy {energy:.1f}" +
          ("  — that is flat enough that there may be nothing to align to"
           if energy < FLAT else ""))
    print(f"rank correlation between the two: {rank:+.3f}")
    if abs(rank) < MONOTONIC:
        print("  the intensity relationship is NOT monotonic. Phase correlation and ECC "
              "both assume one; expect only mutual information to work, and expect it "
              "to cost about 400x as much.")
    elif rank < 0:
        print("  monotonic but INVERTED. Phase correlation handles this only without a "
              "Hanning window (this project: 0.06 px without, 151 px with) and ECC does "
              "not converge at all.")
    else:
        print("  monotonic and positive — every method here should work, so use the "
              "fastest.")

    before = rg.mutual_information(reference, moving)
    print(f"\nmutual information before alignment: {before:.4f} bits")

    header = f"\n{'method':26s} {'dx':>9s} {'dy':>9s} {'MI after':>10s} {'gain':>8s}"
    if truth is not None:
        header += f" {'error px':>9s}"
    print(header)

    panels, results = [("reference", reference), ("moving", moving)], {}
    for name, fn in rg.METHODS.items():
        estimate = rg.extract_shift(name, fn(reference, moving)[0])
        if not np.all(np.isfinite(estimate)):
            print(f"{name:26s} {'-':>9s} {'-':>9s} {'-':>10s} {'-':>8s}"
                  + ("      n/a" if truth is not None else "")
                  + "   did not converge")
            continue

        m = np.float32([[1, 0, -estimate[0]], [0, 1, -estimate[1]]])
        aligned = cv2.warpAffine(moving, m, (moving.shape[1], moving.shape[0]),
                                 borderMode=cv2.BORDER_REFLECT)
        after = rg.mutual_information(reference, aligned)
        results[name] = (estimate, aligned, after)

        line = (f"{name:26s} {estimate[0]:9.3f} {estimate[1]:9.3f} "
                f"{after:10.4f} {after - before:+8.4f}")
        if truth is not None:
            line += f" {rg.shift_error(estimate, truth):9.3f}"
        print(line)
        panels.append((f"{name}\n({estimate[0]:.2f}, {estimate[1]:.2f})", aligned))

    # ------------------------------------------------------------------ #
    # what to make of it
    # ------------------------------------------------------------------ #
    if not results:
        print("\nNothing converged. Check that the two images overlap at all.")
        return 1

    print()
    shifts = {n: (round(v[0][0], 1), round(v[0][1], 1)) for n, v in results.items()}
    if len(set(shifts.values())) == 1:
        print(f"All {len(shifts)} methods agree on {next(iter(shifts.values()))}.")
    else:
        print(f"The methods disagree — {len(set(shifts.values()))} different answers. "
              "On matched intensities they should agree to a hundredth of a pixel, so "
              "a disagreement means the intensity relationship is the problem, not the "
              "alignment.")

    improved = [n for n, (_, _, after) in results.items() if after > before + 0.01]
    print(f"\n{len(improved)} of {len(results)} alignments raised the mutual information.")
    if improved:
        best = max(results, key=lambda n: results[n][2])
        print(f"  the largest rise is {best} "
              f"({results[best][2] - before:+.4f} bits). Without ground truth that is "
              "the best available evidence an alignment is right — MI rises on a correct "
              "alignment whatever the intensities are doing.")
    else:
        print("  None did. Either the pair is already aligned, or none of these methods "
              "found it.")

    if truth is not None:
        errors = {n: rg.shift_error(v[0], truth) for n, v in results.items()}
        best = min(errors, key=errors.get)
        print(f"\nbest against the truth: {best} at {errors[best]:.4f} px")

    if args.method and args.method in results:
        out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "aligned.png"
        io.imwrite(out_path, results[args.method][1])
    else:
        out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "alignments.png"
        figures.grid(panels, out_path, ncols=3,
                     suptitle=f"Four registrations of {source}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
