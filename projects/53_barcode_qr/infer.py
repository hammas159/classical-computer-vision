"""Find and decode a code in an image of your own — and see the gap between the two.

    python infer.py photo.jpg
    python infer.py --background zebra_herd --blur 4
    python infer.py --background coiled_rope --scale 0.5

With `--background` a QR code of known payload is placed on one of this project's
photographs, so both the localisation IoU and the decode are checked against
truth. On your own photograph there is no truth, so what is printed is each
localiser's box, whether anything decoded, and your image's **barcode-like
share** — the fraction of the frame that already responds to a 1-D barcode cue.

That last number predicts the gradient localiser's false positives before it
runs. On this project's twelve photographs it reaches 0.021 localisation against
1.000 on generated clutter, because random rectangles have no repeating vertical
structure and zebras, saguaro ribs and coiled rope do.

And the whole point: **found is not decoded**. At blur sigma 5, at 6.7 pixels per
module and at noise sigma 80, the code is located every time and read none.
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
from shared.io import ensure_rgb  # noqa: E402
from shared.report import init_console  # noqa: E402

import barcode as bc  # noqa: E402

LOCATORS = {
    "Gradient + morphology": bc.locate_gradient_morphology,
    "Local variance": bc.locate_variance,
    "QRCodeDetector": bc.locate_qr_detector,
}

#: Barcode-like share above which the gradient localiser will be finding the
#: background rather than the code.
CLUTTERED = 20.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", default=None)
    ap.add_argument("--background", default=None, choices=list(bc.IMAGES),
                    help="place a known code on one of the project's photographs")
    ap.add_argument("--payload", default="CLASSICAL")
    ap.add_argument("--blur", type=float, default=0.0)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--rotation", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    init_console()
    truth = None
    if args.background:
        code = bc.make_qr(args.payload)
        img, truth = bc.place_on_background(code, seed=0, rotation=args.rotation,
                                            scale=args.scale, background=args.background)
        source = f"{args.payload} on {args.background}"
    elif args.image:
        img = io.imread(args.image)
        source = args.image
    else:
        ap.error("give an image path, or --background with one of the project's photographs")

    if args.blur or args.noise:
        img = bc.degrade(img, blur=args.blur, noise_sigma=args.noise, seed=0)

    share = bc.barcode_like_share(img)
    print(f"{source}  {img.shape[1]}x{img.shape[0]}")
    print(f"  barcode-like share {share:.1f}% of the frame")
    if share > CLUTTERED:
        print(f"  above {CLUTTERED:g}% — the gradient localiser is going to find the "
              "background. Repeating vertical structure is what it looks for, and what "
              "a 1-D barcode is.")
    degradations = [f"blur {args.blur:g}" if args.blur else "",
                    f"noise {args.noise:g}" if args.noise else "",
                    f"scale {args.scale:g}" if args.scale != 1.0 else "",
                    f"rotation {args.rotation:g}deg" if args.rotation else ""]
    degradations = [d for d in degradations if d]
    if degradations:
        print(f"  degraded by: {', '.join(degradations)}")

    header = f"\n{'localiser':24s} {'found a box':>12s}"
    if truth is not None:
        header += f" {'IoU':>8s} {'hit?':>6s}"
    print(header)

    panels = [("the scene", ensure_rgb(img))]
    if truth is not None:
        marked = ensure_rgb(img).copy()
        cv2.polylines(marked, [np.int32(truth).reshape(-1, 1, 2)], True, (60, 200, 60), 3)
        panels.append(("truth", marked))

    hits = {}
    for name, fn in LOCATORS.items():
        corners = fn(img)
        line = f"{name:24s} {('yes' if corners is not None else 'no'):>12s}"
        colour = (200, 200, 60)
        if truth is not None:
            value = bc.localisation_iou(corners, truth, img.shape)
            hit = value >= bc.FOUND_IOU
            hits[name] = hit
            colour = (60, 200, 60) if hit else (220, 60, 60)
            line += f" {value:8.3f} {('HIT' if hit else 'miss'):>6s}"
        print(line)

        drawn = ensure_rgb(img).copy()
        if corners is not None:
            cv2.polylines(drawn, [np.int32(corners).reshape(-1, 1, 2)], True, colour, 3)
        panels.append((name, drawn))

    # ------------------------------------------------------------------ #
    # the only objective answer
    # ------------------------------------------------------------------ #
    decoded = bc.decode_qr(img)
    print(f"\ndecoded: {decoded or 'nothing'}")
    if truth is not None:
        correct = decoded == args.payload
        print(f"  expected {args.payload!r} — {'correct' if correct else 'WRONG'}")

        located = sum(1 for v in hits.values() if v)
        if located and not correct:
            print(f"\n{located} of {len(hits)} localisers found the code and it did NOT "
                  "decode. That gap is this project's whole point: a detection rate says "
                  "nothing about whether the pipeline worked.")
        elif correct and located < len(hits):
            print(f"\nit decoded while {len(hits) - located} of {len(hits)} localisers "
                  "missed it — decoding does not depend on the boxes above, which is why "
                  "the two are measured separately.")
    elif decoded:
        print("  no expected payload to check it against, but a decode that returns a "
              "string is self-validating: the error correction would have failed rather "
              "than returned the wrong text.")
    else:
        print("  nothing decoded. On your own photograph that may mean there is no code, "
              "or that there is one below the readable limit — about 7 pixels per module "
              "on this project's measurements.")

    out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "scan.png"
    figures.grid(panels, out_path, ncols=3,
                 suptitle=f"Three localisers and one decode on {Path(source).name}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
