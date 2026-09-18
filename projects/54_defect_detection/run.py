"""Run the defect detection study and write results + figures.

    python run.py

Regenerates `results/results.json`, `results/tables.md` and every figure in
`docs/images/`. Every number in the README comes out of this script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import numpy as np  # noqa: E402

from shared import figures  # noqa: E402
from shared.io import ensure_rgb  # noqa: E402
from shared.report import init_console, markdown_table, write_results, write_tables  # noqa: E402

import defects as df  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"


def _overlay(image, pred, truth):
    """Truth outlined in green, detection filled red, so both are visible."""
    import cv2

    out = ensure_rgb(image).copy()
    sel = pred > 0
    out[sel] = (0.45 * out[sel] + 0.55 * np.array([230, 60, 60],
                                                  np.float32)).astype(np.uint8)
    contours, _ = cv2.findContours((truth > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (60, 230, 60), 2)
    return out


def main() -> None:
    init_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--severity", type=float, default=df.SEVERITY)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    print(f"{len(df.IMAGES)} surfaces, {len(df.DEFECTS)} defect kinds, "
          f"severity {args.severity:g} x each surface's own noise ceiling")

    surfaces = df.per_surface(severity=args.severity)
    print("\nThe surfaces, in order of uniformity ...")
    for r in surfaces:
        print(f"  {r['surface']:26s} uniformity {r['uniformity']:6.2f}  "
              f"ceiling {r['noise_ceiling']:6.2f}  best detection "
              f"{r['best_detection_rate']:.2f}  false alarms "
              f"{100 * r['lowest_false_alarm']:6.3f}%–{100 * r['worst_false_alarm']:6.3f}%")

    print("\nEvery detector, on defective surfaces and on clean ones ...")
    overall = df.evaluate(severity=args.severity)
    for r in overall:
        tag = "  (control)" if r["is_control"] else ""
        print(f"  {r['detector']:30s} detection {r['detection_rate']:.4f}  "
              f"IoU {r['iou']:.4f}  false alarms "
              f"{100 * r['false_alarm_share']:7.3f}%{tag}")

    defect_table = df.per_defect(severity=args.severity)
    disagree = df.arms_disagree(severity=args.severity)
    fills = df.filling_helps(severity=args.severity)
    severities = df.sweep_severity()
    imbalance = df.pixel_accuracy_is_broken(severity=args.severity)

    print("\nWhich detector finds which kind of defect ...")
    for r in defect_table:
        print(f"  {r['defect']:8s} " + "  ".join(
            f"{n.split()[0][:6]}:{r[n]:.2f}" for n in df.REAL_DETECTORS))

    # ------------------------------------------------------------------ #
    # the comparison at the top of the README
    # ------------------------------------------------------------------ #
    surface = "surface_coarse_cloth"
    image = df.load(surface)

    rows, notes = [], []
    table_rows = [{"Sr": sr, "Defect": k} for sr, k in enumerate(df.DEFECTS, start=1)]

    planted = {k: df.plant(image, k, severity=args.severity) for k in df.DEFECTS}
    rows.append(("the defect\n(truth outlined)",
                 [_overlay(planted[k][0], np.zeros_like(planted[k][1]), planted[k][1])
                  for k in df.DEFECTS]))
    notes.append([f"{100 * float((planted[k][1] > 0).mean()):.2f}% of the surface"
                  for k in df.DEFECTS])

    for name in df.REAL_DETECTORS:
        panels, cell = [], []
        for pos, k in enumerate(df.DEFECTS):
            bad, mask = planted[k]
            pred = df.fill(df.DETECTORS[name](bad))
            panels.append(_overlay(bad, pred, mask))
            hit = df.found(pred, mask)
            cell.append("found" if hit else "missed")
            table_rows[pos][name] = "✅" if hit else "—"
        rows.append((name, panels))
        notes.append(cell)
        print(f"  {name:30s} {cell}")

    figures.gallery(
        list(df.DEFECTS), rows, IMAGES / "compare_detection.png",
        cell_notes=notes,
        suptitle=(f"Four defect kinds planted in {surface.replace('_', ' ')}. "
                  "Green outlines the truth, red is what the detector marked. "
                  "No detector finds all four."),
    )
    gallery_table = markdown_table(
        table_rows, [("Sr", "Sr"), ("Defect", "Defect")]
        + [(n, n) for n in df.REAL_DETECTORS])
    print("\n--- found or missed, on one surface ---\n" + gallery_table)

    # ------------------------------------------------------------------ #
    # the signature figures
    # ------------------------------------------------------------------ #
    real = [r for r in overall if not r["is_control"]]
    figures.scatter_plane(
        {r["detector"]: [(100 * r["false_alarm_share"], r["detection_rate"])]
         for r in real},
        target=None,
        out_path=IMAGES / "detection_vs_alarm.png",
        xlabel="false alarms on a clean surface (% of it marked)",
        ylabel="detection rate on defective surfaces",
        title=("The two arms of the problem. A benchmark built only on defective "
               "parts sees the vertical axis."),
    )

    figures.metric_bars(
        [r["defect"] for r in defect_table for _ in df.REAL_DETECTORS],
        [r[n] for r in defect_table for n in df.REAL_DETECTORS],
        IMAGES / "per_defect.png",
        ylabel="detection rate",
        title=("Six detectors within each of four defect kinds. They are "
               "complementary, not competing."),
    )

    figures.metric_bars(
        [r["detector"] for r in real] * 2,
        [r["detection_rate"] for r in real]
        + [100 * r["false_alarm_share"] for r in real],
        IMAGES / "arms.png",
        ylabel="detection rate  /  false alarms (%)",
        title=("Left six: detection rate, higher is better. Right six: false "
               "alarms, lower is better. The orders differ."),
    )

    figures.lines(
        [r["severity"] for r in severities],
        {n: [r[n] for r in severities] for n in df.REAL_DETECTORS},
        IMAGES / "severity.png",
        xlabel="defect severity (multiples of the surface's own noise ceiling)",
        ylabel="detection rate", logx=True,
        title="How far outside the texture a defect has to be")

    figures.metric_bars(
        [r["surface"].replace("surface_", "").replace("_", " ") for r in surfaces] * 2,
        [r["best_detection_rate"] for r in surfaces]
        + [r["worst_false_alarm"] for r in surfaces],
        IMAGES / "surfaces.png",
        ylabel="best detection rate  /  worst false-alarm share",
        title=("Left twelve: the best any detector manages. Right twelve: the "
               "worst false-alarm rate. The surface decides both."))

    figures.metric_bars(
        [r["detector"] for r in fills] * 2,
        [r["iou_raw"] for r in fills] + [r["iou_filled"] for r in fills],
        IMAGES / "filling.png",
        ylabel="IoU with the true defect mask",
        title=("Left six: the raw response. Right six: after filling. A residual "
               "detector finds a smooth defect's boundary, not its area."))

    # ------------------------------------------------------------------ #
    results_path = write_results(
        RESULTS, "54_defect_detection",
        {
            "surfaces": list(df.IMAGES),
            "defects": list(df.DEFECTS),
            "severity": args.severity,
            "per_surface": surfaces,
            "overall": overall,
            "per_defect": defect_table,
            "arms_disagree": disagree,
            "filling": fills,
            "severity_sweep": severities,
            "class_imbalance": imbalance,
        },
    )

    overall_table = markdown_table(
        overall, [("Detector", "detector"), ("Detection rate", "detection_rate"),
                  ("IoU", "iou"), ("False alarm share", "false_alarm_share"),
                  ("Control?", "is_control")])
    defect_md = markdown_table(
        defect_table, [("Defect", "defect")] + [(n, n) for n in df.REAL_DETECTORS])
    surface_md = markdown_table(
        surfaces, [("Surface", "surface"), ("Uniformity", "uniformity"),
                   ("Noise ceiling", "noise_ceiling"),
                   ("Best detection", "best_detection_rate"),
                   ("Lowest false alarm", "lowest_false_alarm"),
                   ("Worst false alarm", "worst_false_alarm")])
    fill_md = markdown_table(
        fills, [("Detector", "detector"), ("IoU raw", "iou_raw"),
                ("IoU filled", "iou_filled"), ("Gain", "gain")])
    severity_md = markdown_table(
        severities, [("Severity", "severity")] + [(n, n) for n in df.REAL_DETECTORS])

    write_tables(
        RESULTS,
        [
            ("Every detector, both arms", overall_table),
            ("Which detector finds which defect", defect_md),
            ("The surface decides", surface_md),
            ("What filling is worth", fill_md),
            ("How large a defect has to be", severity_md),
            ("Found or missed, on one surface", gallery_table),
        ],
    )

    # ------------------------------------------------------------------ #
    print("\n--- HEADLINE NUMBERS ---")
    print(f"a defect covers {100 * imbalance['mean_defect_share']:.2f}% of a surface, "
          f"so flagging nothing scores "
          f"{imbalance['flag_nothing_pixel_accuracy']:.4f} on pixel accuracy")
    print("  which is why this project reports a detection rate and a false-alarm "
          "share instead, and never an accuracy")

    print(f"\nthe detectors are complementary, not competing:")
    for r in defect_table:
        best = max(df.REAL_DETECTORS, key=lambda n: r[n])
        others = sorted((r[n] for n in df.REAL_DETECTORS), reverse=True)
        print(f"  {r['defect']:8s} best is {best} ({r[best]:.2f}); "
              f"second best {others[1]:.2f}")
    smear = next(r for r in defect_table if r["defect"] == "smear")
    zeros = [n for n in df.REAL_DETECTORS if smear[n] == 0.0]
    print(f"  and a smear is found by the local standard deviation "
          f"({smear['Local standard deviation']:.2f}) while "
          f"{len(zeros)} of the six find it exactly never")
    print("  a smear has the same mean as its surround and less texture, so no "
          "intensity residual can see it at all")

    print(f"\nthe two arms rank the detectors differently "
          f"({'they agree' if disagree['rankings_agree'] else 'they do not'}):")
    print(f"  best detection: {disagree['detection_winner']}, which ranks "
          f"{disagree['detection_winner_alarm_rank']} of {len(df.REAL_DETECTORS)} "
          "on false alarms")
    print(f"  fewest false alarms: {disagree['alarm_winner']}")
    worst_alarm = max(real, key=lambda r: r["false_alarm_share"])
    best_alarm = min(real, key=lambda r: r["false_alarm_share"])
    print(f"  the spread is {100 * best_alarm['false_alarm_share']:.3f}% to "
          f"{100 * worst_alarm['false_alarm_share']:.3f}% of a surface marked when "
          "nothing is wrong with it")

    spread = max(r["best_detection_rate"] for r in surfaces) - \
        min(r["best_detection_rate"] for r in surfaces)
    hardest = min(surfaces, key=lambda r: r["best_detection_rate"])
    easiest = max(surfaces, key=lambda r: r["best_detection_rate"])
    print(f"\nthe surface decides more than the detector: the best achievable "
          f"detection rate spans {spread:.2f} across the twelve")
    print(f"  {easiest['surface']} reaches {easiest['best_detection_rate']:.2f}, "
          f"{hardest['surface']} only {hardest['best_detection_rate']:.2f}")

    gains = [r["gain"] for r in fills]
    print(f"\nfilling the detection is worth {min(gains):+.3f} to {max(gains):+.3f} "
          "IoU — a residual detector responds to a smooth defect's boundary, not "
          "its area")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
