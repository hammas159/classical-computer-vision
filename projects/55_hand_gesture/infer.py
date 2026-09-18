"""Put one hand on one background and see which segmenters find it.

    python infer.py --hand 5_P__5_P_hgr1_id04_2 --background 314016
    python infer.py --hand 3_P__3_P_hgr1_id02_2 --background 81090
    python infer.py --hand 1_P__1_P_hgr1_id01_3 --background 293029
    python infer.py --list

Every run prints the IoU against the **exact** matte and the finger count beside
the one the dataset recorded, because a good mask and a right answer are not the
same thing here: the shape rule manages three of the five numbered gestures even
when it is handed a perfect mask.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import numpy as np  # noqa: E402

from shared import figures  # noqa: E402
from shared.report import init_console  # noqa: E402

import gestures as gs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hand", default=None, help="one of the project's hands")
    ap.add_argument("--background", default=None, help="one of the backgrounds")
    ap.add_argument("--list", action="store_true", help="show what is available")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    init_console()

    if args.list:
        print(f"{len(gs.hand_names())} hands:")
        for h in gs.hand_names():
            want = gs.fingers_of(h)
            print(f"  {h:34s} gesture {gs.gesture_of(h):2s}  {gs.person_of(h)}"
                  + (f"  {want} fingers" if want else ""))
        print(f"\n{len(gs.BACKGROUNDS)} backgrounds, by how much of each a skin rule "
              "already accepts:")
        for b in sorted(gs.BACKGROUNDS, key=gs.skin_likeness):
            print(f"  {b:8s} {100 * gs.skin_likeness(b):5.1f}%  {gs.BACKGROUNDS[b]}")
        return 0

    hand = args.hand or gs.hand_names()[0]
    background = args.background or list(gs.BACKGROUNDS)[0]
    if hand not in gs.hand_names():
        ap.error(f"unknown hand {hand!r} — run with --list")
    if background not in gs.BACKGROUNDS:
        ap.error(f"unknown background {background!r} — run with --list")

    frame, truth = gs.composite(hand, background)
    want = gs.fingers_of(hand)
    likeness = gs.skin_likeness(background)

    print(f"gesture {gs.gesture_of(hand)} ({gs.person_of(hand)}) on {background} — "
          f"{gs.BACKGROUNDS[background]}")
    print(f"  the background is {100 * likeness:.1f}% skin-coloured before the hand "
          "is put on it")
    print(f"  the hand covers {100 * (truth > 0).mean():.1f}% of the frame; "
          f"its matte halo is {100 * gs.matte_is_clean(hand):.2f}%")
    if want is None:
        print("  the dataset records this as a letter, so there is no finger count "
              "to check against")
    if likeness > 0.6:
        print("  — expect every colour rule to fail here")

    print(f"\n{'segmenter':26s} {'IoU':>6s} {'fingers':>8s} {'feature gap':>12s}")
    panels = [("the exact matte", gs.overlay(frame, np.zeros_like(truth), truth))]
    true_features = gs.shape_features(truth)
    scores = {}
    for name in gs.SEGMENTERS:
        mask = gs.SEGMENTERS[name](frame, truth=truth)
        v = gs.iou(mask, truth)
        feats = gs.shape_features(mask)
        gap = gs.feature_gap(feats, true_features)
        scores[name] = v
        mark = ""
        if want is not None:
            mark = " ok" if feats["fingers"] == want else ""
        print(f"{name:26s} {v:6.2f} {feats['fingers']:8d}{mark:3s} {gap:12.3f}")
        panels.append((f"{name}\nIoU {v:.2f}, {feats['fingers']} fingers",
                       gs.overlay(frame, mask, truth,
                                  colour=(60, 220, 90) if v >= 0.5 else (235, 70, 70))))

    # ------------------------------------------------------------------ #
    real = {k: v for k, v in scores.items() if k in gs.REAL_SEGMENTERS}
    best = max(real, key=real.get)
    ellipse = scores["Centre ellipse (control)"]
    print(f"\nbest here: {best} at {real[best]:.2f}")
    beaten = [k for k, v in real.items() if v < ellipse]
    if beaten:
        print(f"  an ellipse drawn without reading the image scores {ellipse:.2f} "
              f"and beats {len(beaten)} of them: {', '.join(beaten)}")
    if want is not None:
        oracle = gs.shape_features(truth)["fingers"]
        print(f"  with a perfect mask the shape rule says {oracle} finger"
              + ("" if oracle == 1 else "s")
              + f"; the dataset says {want}."
              + ("" if oracle == want
                 else " It is wrong before any segmenter is involved."))

    out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "segmented.png"
    figures.grid(panels, out_path, ncols=4,
                 suptitle=f"gesture {gs.gesture_of(hand)} on {gs.BACKGROUNDS[background]} "
                          f"({100 * likeness:.0f}% skin-coloured) — amber is the truth")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
