"""Run the alerting rules over the clip and print when each one would call you.

    python infer.py
    python infer.py --rule "Any motion in the zone"
    python infer.py --cooldown 30 --persistence 8 --area 600
    python infer.py --causal

Every run reports **events**, not pixels: how many times the alarm went off, how
many of the seven planted intrusions it caught, and how long after each one
entered the zone it did so.
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

import alert as al  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rule", default=None, choices=list(al.RULES),
                    help="one of the project's six rules")
    ap.add_argument("--area", type=int, default=None)
    ap.add_argument("--persistence", type=int, default=None)
    ap.add_argument("--cooldown", type=int, default=None)
    ap.add_argument("--causal", action="store_true",
                    help="use a mask a camera could compute live")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    init_console()

    if not al.available():
        ap.error("the clip is missing — run `python tools/fetch_assets.py --set video`")

    activity = al.zone_activity()
    quiet = al.quiet_frames(activity)
    seq, present = al.planted_clip(activity=activity)
    spans = al.present_spans(present)

    print(f"{len(seq)} frames at {al.FPS:g} fps = {len(seq) / al.FPS:.1f} s")
    print(f"  the zone is verified still in {len(quiet)} frames and has real, "
          f"unlabelled activity in {len(seq) - len(quiet)}")
    print(f"  {len(spans)} intrusions planted on a recorded schedule: "
          + ", ".join(f"{a}-{b}" for a, b in spans))

    if args.causal:
        print("\n  using a CAUSAL mask — the median of the last 60 frames, no future")
        signal = np.array([al.largest_blob_in_zone(al.causal_mask(i, seq=seq))
                           for i in range(len(seq))])
    else:
        signal = np.array([al.largest_blob_in_zone(al.oracle_mask(f)) for f in seq])

    if args.rule:
        chosen = {args.rule: al.RULES[args.rule]}
    elif any(v is not None for v in (args.area, args.persistence, args.cooldown)):
        chosen = {"custom": (args.area or 600, args.persistence or 1,
                             args.cooldown or 0)}
    else:
        chosen = dict(al.RULES)

    print(f"\n{'rule':26s} {'alerts':>7s} {'found':>6s} {'missed':>7s} "
          f"{'latency s':>10s} {'false':>6s} {'per min':>8s}")
    results = {}
    for name, (area, persistence, cooldown) in chosen.items():
        fired = al.alerts_from(signal, area, persistence, cooldown)
        s = al.score_alerts(fired, present, quiet, spans)
        results[name] = (s, fired)
        print(f"{name:26s} {s['alerts']:7d} {s['detected']:6d} {s['missed']:7d} "
              f"{s['median_latency_seconds']:10.2f} {s['false_alerts']:6d} "
              f"{s['alerts_per_minute']:8.1f}")

    # ------------------------------------------------------------------ #
    if len(results) == 1:
        name, (s, fired) = next(iter(results.items()))
        print(f"\nalerts at frames: {', '.join(str(i) for i in fired[:40])}"
              + (" ..." if len(fired) > 40 else ""))
        for first, last in spans:
            hits = [i for i in fired if first <= i <= last]
            if hits:
                print(f"  intrusion {first}-{last}: first alert at {min(hits)} "
                      f"(+{(min(hits) - first) / al.FPS:.1f} s), {len(hits)} alert"
                      + ("" if len(hits) == 1 else "s"))
            else:
                print(f"  intrusion {first}-{last}: MISSED")
        if s["unscored"]:
            print(f"  {s['unscored']} alert(s) fell on frames with real, unlabelled "
                  "activity — counted as neither right nor wrong")
    else:
        clean = [(k, v[0]) for k, v in results.items() if v[0]["missed"] == 0]
        if clean:
            best = min(clean, key=lambda kv: kv[1]["alerts"])
            naive = results["Any motion in the zone"][0]
            print(f"\nquietest rule that misses nothing: {best[0]} — "
                  f"{best[1]['alerts']} alerts against the naive rule's "
                  f"{naive['alerts']},")
            print(f"  {naive['alerts'] / max(best[1]['alerts'], 1):.0f}x fewer, for "
                  f"{best[1]['median_latency_seconds']:.1f} s of latency.")
        print("\n  The mask is identical in every row. Only the decision changed.")

    # ------------------------------------------------------------------ #
    series = {}
    for name, (s, fired) in list(results.items())[:3]:
        marks = np.full(len(signal), np.nan)
        marks[fired] = 1.0
        series[f"{name} ({s['alerts']})"] = marks
    series["an intruder is present"] = np.where(present, 0.5, np.nan)
    out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "alerts.png"
    figures.lines(list(range(len(signal))), series, out_path,
                  xlabel="frame", ylabel="alert fired",
                  title="When the alarm went off"
                        + (" (causal mask)" if args.causal else ""))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
