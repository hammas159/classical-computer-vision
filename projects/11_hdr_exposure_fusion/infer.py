"""Fuse your own exposure bracket from the command line.

    python infer.py shot_-2.jpg shot_0.jpg shot_+2.jpg
    python infer.py bracket/*.jpg --method "Mertens fusion" --out fused.png
    python infer.py photo.jpg --simulate          # make a bracket and fuse it

With real photographs there is no reference, so **no accuracy is printed**. What
is printed instead is what needs no reference: how much of each frame is
saturated or crushed, how much of the scene no frame recorded, and which frame
supplied each region.

That last number is the useful one. If `unrecoverable` is large, the bracket is
too narrow and no fusion method will fix it — take another exposure instead of
trying another algorithm.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import numpy as np  # noqa: E402

from shared import io, synth  # noqa: E402

import hdr  # noqa: E402


def estimate_times(frames: list[np.ndarray]) -> np.ndarray:
    """Guess relative exposure times from the frames themselves.

    A real bracket carries its times in EXIF, which this does not read. Ordering
    the frames by mean brightness and assuming an even spacing in stops is a
    guess, and it is stated as one — but it is only used by the Debevec
    pipelines. Mertens needs no times at all, which is a genuine practical
    advantage of it and the reason it is the default here.
    """
    order = np.argsort([float(f.mean()) for f in frames])
    times = np.zeros(len(frames), np.float32)
    spacing = 2.0  # stops, assumed
    mid = len(frames) // 2
    for rank, idx in enumerate(order):
        times[idx] = 2.0 ** (spacing * (rank - mid))
    return times


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", help="the bracket, in any order")
    ap.add_argument("--method", default="Mertens fusion", choices=list(hdr.METHODS))
    ap.add_argument("--out", default=None, help="where to write the fused image")
    ap.add_argument("--simulate", action="store_true",
                    help="treat a single image as a scene and generate a bracket from it")
    args = ap.parse_args()

    if args.simulate:
        if len(args.images) != 1:
            print("--simulate takes exactly one image", file=sys.stderr)
            return 2
        frames, times, reference = synth.hdr_bracket(io.imread(args.images[0]))
        print(f"simulated a {synth.HDR_SCENE_STOPS:.0f}-stop scene and bracketed it "
              f"at {list(synth.EXPOSURE_STOPS)} stops")
    else:
        frames = [io.imread(p) for p in args.images]
        shapes = {f.shape for f in frames}
        if len(shapes) != 1:
            print(f"the frames are different sizes: {sorted(shapes)}", file=sys.stderr)
            return 2
        if len(frames) < 2:
            print("a bracket needs at least two exposures; with one there is "
                  "nothing to fuse", file=sys.stderr)
            return 2
        times = estimate_times(frames)
        reference = None
        print(f"{len(frames)} frames, exposure times ASSUMED 2 stops apart: "
              f"{np.round(times, 4).tolist()}")

    print("\nper frame:")
    for i, f in enumerate(frames):
        print(f"  {i}  mean {f.mean():6.1f}   saturated {100 * (f >= 254).mean():5.1f}%"
              f"   crushed {100 * (f <= 2).mean():5.1f}%")

    loss = synth.bracket_loss(frames)
    print(f"\nblown in EVERY frame   : {100 * loss['blown_everywhere']:.2f}%")
    print(f"crushed in EVERY frame : {100 * loss['crushed_everywhere']:.2f}%")
    print(f"unrecoverable          : {100 * loss['unrecoverable']:.2f}%")
    if loss["unrecoverable"] > 0.02:
        print("  -> the bracket is too narrow. Another exposure will help; "
              "another algorithm will not.")

    fused = hdr.METHODS[args.method](frames, times)
    contrib = hdr.contribution_map(frames)
    used = np.bincount(contrib.ravel(), minlength=len(frames))
    print(f"\nfused with: {args.method}")
    print("best-exposed frame, by area:")
    for i, n in enumerate(used):
        print(f"  frame {i}: {100 * n / contrib.size:5.1f}%")

    if reference is not None:
        s = hdr.score(fused, reference)
        print(f"\nagainst the oracle: {s['psnr_db']} dB, "
              f"{s['psnr_matched_db']} dB matched, SSIM {s['ssim']}")
    else:
        print("\nNo accuracy is reported. These are real photographs and nobody "
              "recorded what the scene truly looked like, so any number here "
              "would be invented.")

    out = Path(args.out) if args.out else PROJECT_DIR / "results" / "fused.png"
    io.imwrite(out, fused)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
