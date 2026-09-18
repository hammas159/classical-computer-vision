"""Run the motion-alert comparison and write results + figures.

    python run.py

Regenerates `results/results.json`, `results/tables.md` and every figure in
`docs/images/`. The README's numbers are copied from those files rather than
typed, so this script is the single source of truth for every claim made.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from shared import figures  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import alert as al  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"

COOLDOWNS = (0, 10, 20, 30, 50, 75, 100, 150)


def main() -> None:
    init_console()
    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    if not al.available():
        raise SystemExit("Run `python tools/fetch_assets.py --set video` first.")

    n = al.frame_count()
    print(f"{n} frames at {al.FPS:g} fps = {n / al.FPS:.1f} s, "
          f"noise floor {al.noise_floor():.2f} grey levels")

    # ------------------------------------------------------------------ #
    # 1. the truth has to be earned
    # ------------------------------------------------------------------ #
    print("\nIs the restricted zone actually empty? ...")
    activity = al.zone_activity()
    quiet = al.quiet_frames(activity)
    busy = al.busy_frames(activity)
    print(f"  {len(quiet)} frames verified still, {len(busy)} with real activity "
          f"({100 * len(busy) / n:.1f}%)")
    print(f"  real activity at frames {busy.min()}-{busy.max()}, in "
          f"{len(al.present_spans(activity >= al.QUIET_AREA))} stretches")
    print("  So 'no intrusion' is checked frame by frame rather than assumed.")

    print("\nPlanting intruders on a recorded schedule ...")
    plan = al.plan_intrusions(activity=activity)
    for r in plan:
        mark = "" if r["usable"] else (f"  <- REFUSED: overlaps real activity on "
                                       f"{r['overlaps_real_activity']} frames")
        print(f"  frames {r['first']:3d}-{r['last']:3d} ({r['seconds']:.1f} s)"
              f"{mark}")
    refused = [r for r in plan if not r["usable"]]
    print(f"  {len(plan) - len(refused)} of {len(plan)} planted; {len(refused)} "
          "refused rather than silently overlapped.")

    seq, present = al.planted_clip(activity=activity)
    spans = al.present_spans(present)
    print(f"  {len(spans)} intrusions, "
          f"{sum(b - a + 1 for a, b in spans)} frames of intrusion in total")

    # ------------------------------------------------------------------ #
    # 2. the decision, with a perfect mask
    # ------------------------------------------------------------------ #
    print("\nThe in-zone signal from the ORACLE mask — the exact per-pixel truth ...")
    signal = np.array([al.largest_blob_in_zone(al.oracle_mask(f)) for f in seq])

    print(f"\n  {'rule':26s} {'alerts':>7s} {'found':>6s} {'missed':>7s} "
          f"{'latency s':>10s} {'false':>6s} {'false/min':>10s}")
    rows = {}
    for name, (area, persistence, cooldown) in al.RULES.items():
        fired = al.alerts_from(signal, area, persistence, cooldown)
        s = al.score_alerts(fired, present, quiet, spans)
        rows[name] = s
        print(f"  {name:26s} {s['alerts']:7d} {s['detected']:6d} {s['missed']:7d} "
              f"{s['median_latency_seconds']:10.2f} {s['false_alerts']:6d} "
              f"{s['false_per_minute']:10.1f}")

    controls = {}
    for name, fn in (("Always alert (control)", al.control_always),
                     ("Never alert (control)", al.control_never)):
        s = al.score_alerts(fn(signal), present, quiet, spans)
        controls[name] = s
        print(f"  {name:26s} {s['alerts']:7d} {s['detected']:6d} {s['missed']:7d} "
              f"{s['median_latency_seconds']:10.2f} {s['false_alerts']:6d} "
              f"{s['false_per_minute']:10.1f}")

    best = min((r for r in rows.values() if r["missed"] == 0),
               key=lambda r: r["alerts"])
    matched = al.score_alerts(al.control_random(signal, count=best["alerts"]),
                              present, quiet, spans)
    controls["Random, rate-matched (control)"] = matched
    print(f"  {'Random, rate-matched':26s} {matched['alerts']:7d} "
          f"{matched['detected']:6d} {matched['missed']:7d} "
          f"{matched['median_latency_seconds']:10.2f} {matched['false_alerts']:6d} "
          f"{matched['false_per_minute']:10.1f}")

    naive = rows["Any motion in the zone"]
    print(f"\n  The naive rule fires {naive['alerts']} times for {len(spans)} "
          f"intrusions — {naive['alerts'] / len(spans):.0f} calls per intruder —")
    print("  and the mask it is reading is exact. Nothing about the pixels is "
          "wrong.")
    print(f"  The best rule that misses nothing fires {best['alerts']} times, "
          f"{naive['alerts'] / max(best['alerts'], 1):.0f}x fewer, for "
          f"{best['median_latency_seconds']:.1f} s of latency.")
    print(f"\n  Firing the right *number* of times is not enough: the rate-matched "
          f"random control\n  fires {matched['alerts']} times too and catches "
          f"{matched['detected']} of {len(spans)}.")

    # ------------------------------------------------------------------ #
    # 3. the cooldown sweep
    # ------------------------------------------------------------------ #
    print("\nSweeping the cooldown, everything else fixed ...")
    sweep = []
    for cooldown in COOLDOWNS:
        s = al.score_alerts(al.alerts_from(signal, 600, 8, cooldown),
                            present, quiet, spans)
        sweep.append({"cooldown": cooldown, "seconds": cooldown / al.FPS, **s})
        print(f"  cooldown {cooldown:3d} ({cooldown / al.FPS:4.1f} s): "
              f"{s['alerts']:3d} alerts, {s['detected']}/{len(spans)} found, "
              f"latency {s['median_latency_seconds']:.1f} s")
    perfect = [r for r in sweep if r["missed"] == 0]
    knee = min(perfect, key=lambda r: r["alerts"]) if perfect else sweep[0]
    print(f"\n  The knee is at cooldown {knee['cooldown']} "
          f"({knee['seconds']:.1f} s): {knee['alerts']} alerts and nothing missed.")
    over = [r for r in sweep if r["missed"] > 0]
    if over:
        print(f"  Beyond {min(r['cooldown'] for r in over)} the system starts "
              "missing intrusions, which is the point at which quieter stops "
              "being better.")

    # ------------------------------------------------------------------ #
    # 4. what a live system could actually compute
    # ------------------------------------------------------------------ #
    print("\nThe same rules on a CAUSAL mask — no frame from the future ...")
    causal = np.array([al.largest_blob_in_zone(al.causal_mask(i, seq=seq))
                       for i in range(len(seq))])
    causal_rows = {}
    for name, (area, persistence, cooldown) in al.RULES.items():
        s = al.score_alerts(al.alerts_from(causal, area, persistence, cooldown),
                            present, quiet, spans)
        causal_rows[name] = s
        print(f"  {name:26s} {s['alerts']:5d} alerts, {s['detected']}/{len(spans)} "
              f"found, {s['false_alerts']:3d} false")
    print("\n  The causal mask is what a camera could compute live; the oracle "
          "uses the whole\n  clip including the future. The gap between them is "
          "the price of not knowing\n  what happens next.")

    # ------------------------------------------------------------------ #
    # figures
    # ------------------------------------------------------------------ #
    print("\nFigures ...")

    # the scene and the zone
    show = seq[spans[0][0] + 10]
    zone = al.zone_mask(show.shape)
    outlined = show.copy()
    contours, _ = cv2.findContours(zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(outlined, contours, -1, (250, 190, 40), 3)
    plain = al.frames()[spans[0][0] + 10].copy()
    cv2.drawContours(plain, contours, -1, (250, 190, 40), 3)
    figures.grid([("the clip as it is", plain),
                  ("with an intruder planted in the zone", outlined),
                  ("the oracle mask of that frame",
                   cv2.cvtColor(al.oracle_mask(show), cv2.COLOR_GRAY2RGB))],
                 IMAGES / "scene.png", ncols=3,
                 suptitle="Amber is the restricted zone: the only part of this scene "
                          "that is still in 96% of frames. The paving is busy in 94%.")

    # the timeline — the project's signature figure
    fig_rules = ["Any motion in the zone", "+ persistence 8", "+ cooldown 50"]
    figures.lines(
        list(range(len(signal))),
        {"in-zone blob area (oracle)": signal.astype(float)},
        IMAGES / "signal.png",
        xlabel="frame", ylabel="largest blob in the zone, pixels",
        title="The signal every rule reads. The seven intrusions are the tall "
              "blocks; everything else is the scene being still.")

    alert_series = {}
    for i, name in enumerate(fig_rules):
        area, persistence, cooldown = al.RULES[name]
        fired = al.alerts_from(signal, area, persistence, cooldown)
        marks = np.zeros(len(signal))
        marks[fired] = 1.0 + i
        alert_series[f"{name} ({len(fired)} alerts)"] = marks
    truth_series = np.where(present, 0.5, np.nan)
    alert_series["an intruder is present"] = truth_series
    figures.lines(
        list(range(len(signal))), alert_series, IMAGES / "timeline.png",
        xlabel="frame", ylabel="alert fired (offset per rule)",
        title="When each rule calls you. Same mask, same signal, three decisions.")

    figures.lines(
        [r["seconds"] for r in sweep],
        {"alerts raised": [float(r["alerts"]) for r in sweep],
         "intrusions missed": [float(r["missed"]) for r in sweep],
         "latency, seconds": [r["median_latency_seconds"] if r["detected"] else 0.0
                              for r in sweep]},
        IMAGES / "cooldown.png",
        xlabel="cooldown, seconds", ylabel="count / seconds",
        title="One knob. Quieter is better until it starts missing people.")

    real = [(k, v) for k, v in rows.items()]
    figures.metric_bars([k.replace("+ ", "") for k, _ in real],
                        [v["alerts"] for _, v in real],
                        IMAGES / "alerts.png",
                        ylabel=f"alerts raised for {len(spans)} intrusions",
                        title="Same perfect mask, six decisions",
                        highlight_best="min")

    # ------------------------------------------------------------------ #
    write_results(RESULTS, "55_motion_alert", {
        "frames": n, "fps": al.FPS, "zone": al.ZONE,
        "noise_floor": al.noise_floor(),
        "quiet_frames": int(len(quiet)), "busy_frames": int(len(busy)),
        "plan": plan, "spans": [[int(a), int(b)] for a, b in spans],
        "rules": {k: {kk: vv for kk, vv in v.items()} for k, v in rows.items()},
        "controls": {k: v for k, v in controls.items()},
        "cooldown_sweep": sweep,
        "causal": {k: v for k, v in causal_rows.items()},
    })

    write_tables(RESULTS, [
        ("Six decisions on one perfect mask", markdown_table(
            [{"Rule": k, "Alerts": v["alerts"],
              "Found": f"{v['detected']}/{len(spans)}",
              "Missed": v["missed"],
              "Latency (s)": (f"{v['median_latency_seconds']:.2f}"
                              if v["detected"] else "—"),
              "False alerts": v["false_alerts"],
              "False per minute": f"{v['false_per_minute']:.1f}"}
             for k, v in list(rows.items()) + list(controls.items())],
            [(c, c) for c in ("Rule", "Alerts", "Found", "Missed", "Latency (s)",
                              "False alerts", "False per minute")])),
        ("The cooldown sweep", markdown_table(
            [{"Cooldown (s)": f"{r['seconds']:.1f}", "Alerts": r["alerts"],
              "Found": f"{r['detected']}/{len(spans)}", "Missed": r["missed"],
              "Latency (s)": (f"{r['median_latency_seconds']:.2f}"
                              if r["detected"] else "—")} for r in sweep],
            [(c, c) for c in ("Cooldown (s)", "Alerts", "Found", "Missed",
                              "Latency (s)")])),
        ("The same rules on a causal mask", markdown_table(
            [{"Rule": k, "Alerts": v["alerts"],
              "Found": f"{v['detected']}/{len(spans)}",
              "False alerts": v["false_alerts"]} for k, v in causal_rows.items()],
            [(c, c) for c in ("Rule", "Alerts", "Found", "False alerts")])),
        ("Planting, and what was refused", markdown_table(
            [{"Frames": f"{r['first']}-{r['last']}",
              "Seconds": f"{r['seconds']:.1f}",
              "Overlaps real activity": r["overlaps_real_activity"],
              "Planted": "yes" if r["usable"] else "REFUSED"} for r in plan],
            [(c, c) for c in ("Frames", "Seconds", "Overlaps real activity",
                              "Planted")])),
    ])
    print(f"wrote {RESULTS / 'results.json'} and five figures")


if __name__ == "__main__":
    main()
