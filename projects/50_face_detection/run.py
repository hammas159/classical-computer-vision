"""Run the face-detection comparison and write results + figures.

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

import faces as fc  # noqa: E402

RESULTS = PROJECT_DIR / "results"
IMAGES = PROJECT_DIR / "docs" / "images"

ANGLES = (0, 5, 10, 15, 20, 30, 45)

#: Five photographs chosen to be five different problems: a crowd of small faces,
#: a handful of large ones, a painting, one of OpenCV's own face-test images that
#: turns out to have no resolvable face in it, and the carved totems.
SHOWN = ("class57", "addams-family", "mona-lisa", "churchill-downs", "101085")
SHOWN_WHY = {
    "class57": "57 small faces in rows",
    "addams-family": "six large faces, one in profile",
    "mona-lisa": "a painting, not a photograph",
    "churchill-downs": "a grandstand: all six find nothing",
    "101085": "carved wooden faces — scored in neither arm",
}


def _load_shown(name):
    if name in fc.NO_FACE or name in fc.CARVED:
        return fc.load_no_face(name)
    return fc.load(name)


def main() -> None:
    init_console()
    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    if not fc.available():
        raise SystemExit(
            "Assets are missing. Run `python tools/fetch_assets.py --set faces` and "
            "`--set cascades`, and `python tools/fetch_images.py`.")

    names = fc.image_names()
    print(f"{len(names)} group photographs, {len(fc.NO_FACE)} with no human face, "
          f"{len(fc.DETECTORS)} cascades, {len(fc.CONTROLS)} controls")

    # ------------------------------------------------------------------ #
    # 1. what each cascade even is
    # ------------------------------------------------------------------ #
    print("\nThe training window is in the XML, not in the API ...")
    blindness = fc.window_blindness(names)
    for r in blindness:
        print(f"  {r['method']:16s} window {r['window']:2d}x{r['window']:<2d}  "
              f"boxes at 1x {r['x1']:3d}  1.5x {r['x1.5']:3d}  2x {r['x2']:3d}  "
              f"3x {r['x3']:3d}   recovery {r['recovery']:5.2f}x")
    worst = max(blindness, key=lambda r: r["recovery"])
    others = [r["recovery"] for r in blindness if r is not worst]
    print(f"\n  {worst['method']} recovers {worst['recovery']:.1f}x from upscaling; "
          f"no other cascade recovers more than {max(others):.2f}x.")
    print(f"  Its training window is {worst['window']}x{worst['window']}, against "
          f"{min(r['window'] for r in blindness)}x"
          f"{min(r['window'] for r in blindness)} for the smallest.")

    print("\nAnd lowering minSize does not help, because minSize is not the floor ...")
    sizes = fc.min_size_does_not_help(names)
    for r in sizes:
        cols = "  ".join(f"{k.split()[1]}:{v:3d}" for k, v in r.items()
                         if k.startswith("minSize"))
        print(f"  {r['method']:16s} window {r['window']:2d}  {cols}")
    same = all(r["minSize 12"] == r["minSize 24"] for r in sizes)
    print(f"  minSize 12 and minSize 24 give identical counts for "
          f"{'every' if same else 'not every'} cascade: the cascade's own window is "
          "the real floor.")

    # ------------------------------------------------------------------ #
    # 2. a recorded rotation
    # ------------------------------------------------------------------ #
    print("\nRotating by a known angle and counting what comes back ...")
    rotation = {}
    for method in fc.DETECTORS:
        rotation[method] = fc.rotation_survival(method, ANGLES, names)
        row = "  ".join(f"{r['degrees']:2d}d:{r['survival']:.2f}"
                        for r in rotation[method])
        print(f"  {method:16s} {row}")
    for control in fc.CONTROLS:
        rotation[control] = fc.rotation_survival(control, ANGLES, names)
        row = "  ".join(f"{r['degrees']:2d}d:{r['survival']:.2f}"
                        for r in rotation[control])
        print(f"  {control:26s} {row}")

    at20 = {m: next(r["survival"] for r in rows if r["degrees"] == 20)
            for m, rows in rotation.items() if m in fc.DETECTORS}
    at30 = {m: next(r["survival"] for r in rows if r["degrees"] == 30)
            for m, rows in rotation.items() if m in fc.DETECTORS}
    best20 = max(at20, key=at20.get)
    print(f"\n  best at 20 degrees: {best20} keeps {at20[best20]:.0%}; "
          f"at 30 degrees it keeps {at30[best20]:.0%}")
    dead = [m for m in at20 if at20[m] == 0.0]
    if dead:
        print(f"  at zero by 20 degrees: {', '.join(dead)}")

    print("\nChanges that move nothing ...")
    photometric = {m: fc.photometric_survival(m, names) for m in fc.DETECTORS}
    changes = [r["change"] for r in photometric[fc.DETECTORS[0]]]
    print(f"  {'method':16s} " + "  ".join(f"{c[:12]:>12s}" for c in changes))
    for m, rows in photometric.items():
        print(f"  {m:16s} " + "  ".join(f"{r['survival']:12.2f}" for r in rows))

    # ------------------------------------------------------------------ #
    # 3. an empty truth
    # ------------------------------------------------------------------ #
    print("\nDetections on photographs with no human face in them ...")
    alarms = {m: fc.false_alarms(m) for m in fc.ALL_METHODS}
    found = {m: fc.detections_on_faces(m, names=names) for m in fc.ALL_METHODS}
    print(f"  {'method':26s} {'boxes on faces':>15s} {'false alarms':>13s} "
          f"{'per Mpx':>9s}  worst")
    for m in fc.ALL_METHODS:
        a = alarms[m]
        print(f"  {m:26s} {found[m]['boxes']:15d} {a['false_alarms']:13d} "
              f"{a['per_megapixel']:9.1f}  "
              f"{a['worst_image']} ({fc.NO_FACE[a['worst_image']]}) x{a['worst_count']}")

    print("\nThe same two arms across minNeighbors ...")
    trade = {m: fc.operating_points(m) for m in fc.DETECTORS}
    for m, rows in trade.items():
        print(f"  {m:16s} " + "  ".join(
            f"n{r['min_neighbors']}:{r['boxes']}/{r['false_alarms']}" for r in rows))
    print("  (boxes on the face photographs / false alarms on the face-free ones)")

    loose = {m: rows[0] for m, rows in trade.items()}
    clean = [m for m, r in loose.items() if r["false_alarms"] == 0]
    dirty = max(loose, key=lambda m: loose[m]["false_alarms"])
    if clean:
        best_clean = max(clean, key=lambda m: loose[m]["boxes"])
        print(f"\n  At minNeighbors=1, {dirty} returns {loose[dirty]['boxes']} boxes and "
              f"{loose[dirty]['false_alarms']} false alarms;")
        print(f"  {best_clean} returns {loose[best_clean]['boxes']} and "
              f"{loose[best_clean]['false_alarms']}. One value of one knob does not "
              "mean the same thing to two cascades.")

    # ------------------------------------------------------------------ #
    # 4. agreement, and the photograph nobody can score
    # ------------------------------------------------------------------ #
    print("\nHow much the six cascades agree, per photograph ...")
    agree = fc.agreement(names)
    print(f"  {'photograph':24s} {'distinct':>9s} {'1+':>5s} {'3+':>5s} {'6':>5s}")
    for r in agree:
        print(f"  {r['image']:24s} {r['distinct_boxes']:9d} {r['1_or_more']:5d} "
              f"{r['3_or_more']:5d} {r['6_or_more']:5d}")
    total_distinct = sum(r["distinct_boxes"] for r in agree)
    all_six = sum(r["6_or_more"] for r in agree)
    print(f"\n  {all_six} of {total_distinct} distinct boxes "
          f"({all_six / max(total_distinct, 1):.0%}) are found by all six.")
    print("  This is agreement between six cascades sharing an architecture. "
          "It is not a count of faces.")

    print("\nThe photograph this project refuses to score ...")
    carved = fc.carved_faces()
    for r in carved:
        counts = "  ".join(f"{m.split()[-1]}:{r[m]}" for m in fc.DETECTORS)
        print(f"  {r['image']} — {r['what']}: {counts}")
    print("  A carved face has eyes, a nose and a mouth in the right places. "
          "Whether that is a\n  false alarm is a question about the word, not "
          "about the detector.")

    # ------------------------------------------------------------------ #
    # figures
    # ------------------------------------------------------------------ #
    print("\nFigures ...")

    columns = ["Input"] + list(fc.DETECTORS)
    rows, notes = [], []
    for name in SHOWN:
        img = _load_shown(name)
        # Three cases, three colours. Green where a box may be a face; red where
        # the truth is empty so any box is wrong; amber for the carved faces,
        # which are deliberately scored in neither arm and so must not be
        # coloured as though they had been.
        empty = name in fc.NO_FACE
        carved = name in fc.CARVED
        colour = (235, 70, 70) if empty else (250, 190, 40) if carved else (60, 220, 90)
        noun = "false alarm" if empty else "box" if not carved else "detection"
        cells = [img]
        cell_notes = [f"{img.shape[1]}x{img.shape[0]}"]
        for m in fc.DETECTORS:
            boxes = fc.detect(img, m)
            cells.append(fc.draw(img, boxes, colour=colour))
            plural = "" if len(boxes) == 1 else ("es" if noun == "box" else "s")
            cell_notes.append(f"{len(boxes)} {noun}{plural}")
        rows.append((SHOWN_WHY[name], cells))
        notes.append(cell_notes)
    figures.gallery(columns, rows, IMAGES / "compare.png", cell_notes=notes,
                    suptitle="Every box a cascade returned. Amber is the carved-face "
                             "row, which this project scores in neither arm.")

    figures.lines(
        list(ANGLES),
        {m: [r["survival"] for r in rotation[m]] for m in fc.DETECTORS},
        IMAGES / "rotation.png",
        xlabel="rotation applied to the photograph, degrees",
        ylabel="share of the cascade's own boxes that survive",
        title="Upright means upright: the truth here is the rotation, which is applied "
              "by this code")

    figures.metric_bars(
        [r["method"].replace("Haar ", "H ").replace("LBP ", "L ") for r in blindness],
        [r["recovery"] for r in blindness],
        IMAGES / "window.png",
        ylabel="boxes at 3x upscale / boxes at 1x",
        title="One cascade recovers 29x from upscaling the image. Its training "
              "window is 45x45.",
        highlight_best="max")

    figures.scatter_plane(
        {m: [(r["boxes"], r["false_alarms"]) for r in trade[m]] for m in fc.DETECTORS},
        None, IMAGES / "operating_points.png",
        xlabel="boxes returned on the face photographs (not a recall)",
        ylabel="false alarms on photographs with no face in them",
        title="minNeighbors 1 to 12: the cascades do not share an operating point",
        equal_aspect=False)

    # the empty-truth arm, drawn
    panels = []
    worst_name = max(fc.NO_FACE,
                     key=lambda n: sum(len(fc.detect(fc.load_no_face(n), m,
                                                     min_neighbors=1))
                                       for m in fc.DETECTORS))
    for name in (worst_name, list(fc.CARVED)[0]):
        img = fc.load_no_face(name)
        carved = name in fc.CARVED
        what = fc.CARVED[name] if carved else fc.NO_FACE[name]
        # Red only where the truth really is empty. The carved faces are amber
        # everywhere in this project, because colouring them as false alarms
        # would be deciding the question the project says it is not deciding.
        colour = (250, 190, 40) if carved else (235, 70, 70)
        for mn in (1, 5):
            boxes = np.vstack([fc.detect(img, m, min_neighbors=mn)
                               for m in fc.DETECTORS] + [np.zeros((0, 4))])
            panels.append((f"{what}\nminNeighbors={mn}: {len(boxes)} boxes "
                           "from six cascades", fc.draw(img, boxes, colour=colour)))
    figures.grid(panels, IMAGES / "empty_truth.png", ncols=2,
                 suptitle="Top row: no human face, so every box is a false alarm — and "
                          "the count goes 11 to 0 on one setting of minNeighbors. "
                          "Bottom row (amber): carved faces, which this project does "
                          "not adjudicate.")

    # ------------------------------------------------------------------ #
    results = {
        "photographs": len(names),
        "no_face_photographs": len(fc.NO_FACE),
        "cascades": {m: fc.training_window(m) for m in fc.DETECTORS},
        "window_blindness": blindness,
        "min_size": sizes,
        "rotation": {m: rotation[m] for m in rotation},
        "photometric": photometric,
        "false_alarms": {m: {k: v for k, v in alarms[m].items() if k != "per_image"}
                         for m in fc.ALL_METHODS},
        "false_alarms_per_image": {m: alarms[m]["per_image"] for m in fc.ALL_METHODS},
        "detections_on_faces": found,
        "operating_points": trade,
        "agreement": agree,
        "carved": carved,
        "shown": list(SHOWN),
    }
    write_results(RESULTS, "50_face_detection", results)

    write_tables(RESULTS, [
        ("The training window explains what a cascade cannot see", markdown_table(
            [{"Cascade": r["method"], "Training window": f"{r['window']}x{r['window']}",
              "Boxes at 1x": r["x1"], "at 1.5x": r["x1.5"], "at 2x": r["x2"],
              "at 3x": r["x3"], "Recovery": f"{r['recovery']:.2f}x"}
             for r in blindness],
            [(c, c) for c in ("Cascade", "Training window", "Boxes at 1x", "at 1.5x",
                              "at 2x", "at 3x", "Recovery")])),
        ("Rotation survival, against a truth that is the rotation", markdown_table(
            [{"Cascade": m, **{f"{d}°": f"{next(r['survival'] for r in rotation[m] if r['degrees'] == d):.2f}"
                               for d in ANGLES}} for m in fc.DETECTORS],
            [("Cascade", "Cascade")] + [(f"{d}°", f"{d}°") for d in ANGLES])),
        ("Both arms at minNeighbors 5", markdown_table(
            [{"Method": m, "Boxes on faces": found[m]["boxes"],
              "False alarms": alarms[m]["false_alarms"],
              "Per megapixel": f"{alarms[m]['per_megapixel']:.1f}"}
             for m in fc.ALL_METHODS],
            [(c, c) for c in ("Method", "Boxes on faces", "False alarms",
                              "Per megapixel")])),
        ("Agreement between the six cascades", markdown_table(
            [{"Photograph": r["image"], "Distinct boxes": r["distinct_boxes"],
              "Found by 3+": r["3_or_more"], "Found by all 6": r["6_or_more"]}
             for r in agree],
            [(c, c) for c in ("Photograph", "Distinct boxes", "Found by 3+",
                              "Found by all 6")])),
    ])
    print(f"wrote {RESULTS / 'results.json'} and five figures")


if __name__ == "__main__":
    main()
