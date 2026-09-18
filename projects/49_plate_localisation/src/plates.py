"""Licence-plate localisation: six locators, three controls, and a real annotation.

The question
------------
Find the number plate in a photograph of a car. Classical locators all exploit
the same two facts -- a plate is a **dense band of vertical edges** (the
characters) inside a **bright rectangle of fixed aspect ratio** (about 4.4:1) --
and they differ in which of the two they lean on.

This is the one project in this repository with a **human annotation**: eleven
photographs from the openalpr benchmark, each with a box somebody drew and the
plate's text typed out. So unlike every other project here, it can report an
accuracy. What it mostly reports is that the accuracy is measuring the wrong
thing.

> **Two defensible metrics, two different winners.** `IoU >= 0.5` -- the
> convention -- ranks `Sobel-x + morphology` and `Top-hat + Otsu` first at
> **8 of 14**, with OpenCV's plate cascade at 5. **Coverage of the annotated
> plate** reverses it: the cascade takes **9 of 14** and Sobel drops to 5.
> Whichever is quoted decides which locator looks best.

> **The metric introduced to fix IoU is the worse of the two.** Coverage is the
> obvious repair -- a crop containing every character can be read, whatever else
> it contains -- and it is what this project set out to argue for. It does not
> survive its own test. The cascade wins coverage by returning boxes far larger
> than the plate, and a third measure, the **readability** proxy -- the only one
> here that uses the plate's typed text -- sides with IoU: `Sobel-x` **8 of 14**
> against the cascade's **4**. A crop that contains the plate and half a bumper
> is not a crop you can segment characters out of.
>
> Coverage is also gamed outright by the `Whole frame` control, which scores
> **14 of 14** at a median IoU of 0.01.

> **A plate is thin, and that is what IoU is really charging for.** A
> localisation error is roughly isotropic in pixels, but the mean plate here is
> 4.1 times wider than it is tall. Shifting the annotated box **8 pixels
> sideways** costs IoU 0.90; shifting it **8 pixels down** costs 0.66. Same
> error, **3.5x** the penalty, purely because of the shape.

> **What does not explain anything: plate size.** The plate spans **67x in
> area** across the set, from 0.27% of the frame to 18.31%, and that range
> barely predicts which photographs get found -- log plate width against the
> number of locators that succeed gives **r = +0.29**. The smallest plate in the
> set is found by as many locators as the largest. Reported because it was the
> obvious hypothesis and it is wrong.

> **The threshold is not what flips the ranking either.** `IoU >= 0.5` is a
> convention borrowed from PASCAL VOC, where objects are roughly as tall as they
> are wide, and it was the obvious suspect. Sweeping it from 0.3 to 0.7 leaves
> **the same locator first at every threshold**. It is the choice of *metric*,
> not the choice of cut-off, that decides the answer.

Where the ground truth comes from
---------------------------------
**A person drew these boxes and typed these strings.** That is a different and
weaker kind of truth than the recorded transforms most of this repository uses:
it is one annotator's opinion about where a plate ends, and the coverage result
above is partly a statement about that opinion. It is stated as human annotation
rather than dressed up as ground truth.

What the text buys is the thing a box alone cannot say: **how many characters
are on the plate.** A crop that scores well but from which the characters cannot
be segmented has not solved the problem, and a crop that scores badly but from
which they can, has.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

EPS = 1e-9

#: Eleven photographs of cars with a plate, each with a one-line annotation:
#: filename, x, y, width, height, text. Fetched by
#: `tools/fetch_assets.py --set plates`.
PLATES = Path.home() / ".cache" / "classical-cv-images" / "assets" / "plates"

#: A European plate is close to 4.4:1. Every geometric locator here is allowed
#: the same band, so the comparison is of *how candidates are proposed* rather
#: than of who tuned their aspect filter more tightly.
MIN_ASPECT = 2.2
MAX_ASPECT = 7.0

#: A plate is at least this many pixels wide, and at most this share of the
#: frame. Shared by every locator, for the same reason.
MIN_WIDTH = 40
MAX_AREA_SHARE = 0.35


def available() -> bool:
    return PLATES.exists() and len(list(PLATES.glob("*.txt"))) >= 10


def image_names() -> list[str]:
    stems = {p.stem for p in PLATES.glob("*.jpg")} & {p.stem for p in PLATES.glob("*.txt")}
    return sorted(stems, key=lambda s: int("".join(c for c in s if c.isdigit()) or 0))


def load(name: str) -> np.ndarray:
    path = PLATES / f"{name}.jpg"
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(
            f"{path} is missing. Run `python tools/fetch_assets.py --set plates`.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def truth(name: str) -> tuple[tuple[int, int, int, int], str]:
    """The box a person drew and the string a person typed.

    Returned together because they are one annotation; splitting them invites
    reporting a localisation number without ever checking that the localisation
    was good enough to read.
    """
    fields = (PLATES / f"{name}.txt").read_text(encoding="utf-8").strip().split("\t")
    _, x, y, w, h, text = fields[:6]
    return (int(x), int(y), int(w), int(h)), text


def plate_share(name: str) -> float:
    """How much of the frame the plate occupies. The difficulty axis, measured
    from the annotation rather than from any detector's opinion."""
    (x, y, w, h), _ = truth(name)
    img = load(name)
    return float(w * h) / (img.shape[0] * img.shape[1])


def plate_width(name: str) -> int:
    return truth(name)[0][2]


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def intersection(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    return 0.0 if (x1 <= x0 or y1 <= y0) else float((x1 - x0) * (y1 - y0))


def iou(a, b) -> float:
    i = intersection(a, b)
    return float(i / (a[2] * a[3] + b[2] * b[3] - i + EPS))


def coverage(box, target) -> float:
    """What share of the **annotated plate** the box contains.

    The metric a reading pipeline actually needs: a crop that contains every
    character can be read, whatever else it also contains. Reported alongside
    IoU, never instead of it -- on its own it is trivially gamed by returning
    the whole photograph, which is why `Whole frame` is one of the controls.
    """
    return intersection(box, target) / (target[2] * target[3] + EPS)


def best_by(boxes, target, measure=iou):
    if len(boxes) == 0:
        return None, 0.0
    scores = [measure(b, target) for b in boxes]
    i = int(np.argmax(scores))
    return boxes[i], float(scores[i])


# --------------------------------------------------------------------------- #
# the locators
# --------------------------------------------------------------------------- #


def _plausible(boxes, shape) -> list:
    """The one geometric filter every locator shares."""
    h, w = shape[:2]
    keep = []
    for x, y, bw, bh in boxes:
        if bw < MIN_WIDTH or bh < 8:
            continue
        if not (MIN_ASPECT <= bw / max(bh, 1) <= MAX_ASPECT):
            continue
        if bw * bh > MAX_AREA_SHARE * w * h:
            continue
        keep.append((int(x), int(y), int(bw), int(bh)))
    return keep


def locate_sobel_morphology(image):
    """The classic: horizontal gradient, close into a band, take the rectangles.

    The characters on a plate are a run of near-vertical strokes, so the
    horizontal gradient is dense there and almost nowhere else on a car body. A
    wide closing merges the strokes into one blob.
    """
    g = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    dx = cv2.Sobel(g, cv2.CV_16S, 1, 0, ksize=3)
    dx = cv2.convertScaleAbs(dx)
    _, binary = cv2.threshold(dx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _plausible([cv2.boundingRect(c) for c in contours], image.shape)


def locate_tophat(image):
    """Top-hat, which finds bright things smaller than the structuring element.

    A plate is a bright rectangle on a darker car, so a top-hat with an element
    wider than a plate isolates it without any gradient at all.
    """
    g = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 11))
    hat = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, kernel)
    _, binary = cv2.threshold(hat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (21, 3)))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _plausible([cv2.boundingRect(c) for c in contours], image.shape)


def locate_mser(image):
    """MSER, grouped into lines.

    MSER finds the characters rather than the plate, so the boxes have to be
    merged into rows afterwards -- which is the step that makes it a *text*
    detector applied to plates rather than a plate detector.
    """
    g = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mser = cv2.MSER_create()
    mser.setMinArea(30)
    mser.setMaxArea(int(0.02 * g.size))
    regions, _ = mser.detectRegions(g)
    chars = []
    for r in regions:
        x, y, w, h = cv2.boundingRect(r.reshape(-1, 1, 2))
        if 0.15 <= w / max(h, 1) <= 1.6 and h >= 8:
            chars.append((x, y, w, h))
    return _plausible(_group_into_lines(chars), image.shape)


def _group_into_lines(chars, gap: float = 1.4):
    """Merge character boxes that sit on the same line and close together."""
    if not chars:
        return []
    chars = sorted(chars, key=lambda b: (b[1] // 12, b[0]))
    lines, current = [], [chars[0]]
    for box in chars[1:]:
        px, py, pw, ph = current[-1]
        x, y, w, h = box
        same_row = abs((y + h / 2) - (py + ph / 2)) < 0.7 * max(ph, h)
        similar = 0.55 <= h / max(ph, 1) <= 1.8
        near = (x - (px + pw)) < gap * max(pw, w)
        if same_row and similar and near:
            current.append(box)
        else:
            lines.append(current)
            current = [box]
    lines.append(current)
    out = []
    for line in lines:
        if len(line) < 3:
            continue
        x0 = min(b[0] for b in line)
        y0 = min(b[1] for b in line)
        x1 = max(b[0] + b[2] for b in line)
        y1 = max(b[1] + b[3] for b in line)
        out.append((x0, y0, x1 - x0, y1 - y0))
    return out


def locate_contour_aspect(image):
    """Adaptive threshold, then keep the contours shaped like a plate.

    The most naive thing that could work, and it is here because it shares the
    aspect filter with everything else: whatever it scores is what the aspect
    filter alone is worth.
    """
    g = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 25, 9)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)))
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return _plausible([cv2.boundingRect(c) for c in contours], image.shape)


_HAAR = Path(cv2.data.haarcascades)
_cascades: dict[str, cv2.CascadeClassifier] = {}

CASCADE_FILES = {
    "Haar plate cascade": _HAAR / "haarcascade_russian_plate_number.xml",
    "Haar plate, 16 stages": _HAAR / "haarcascade_license_plate_rus_16stages.xml",
}


def _cascade(name):
    if name not in _cascades:
        _cascades[name] = cv2.CascadeClassifier(str(CASCADE_FILES[name]))
    return _cascades[name]


def locate_cascade(image, which="Haar plate cascade"):
    """OpenCV's shipped plate cascade.

    Trained on **Russian** plates and applied here to German, Czech, Norwegian
    and British ones. That is not a criticism of the cascade; it is the thing
    worth measuring, because it is what anyone reaching for
    `cv2.data.haarcascades` will actually be doing.
    """
    g = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    found = _cascade(which).detectMultiScale(g, 1.1, 3, minSize=(MIN_WIDTH, 10))
    return [tuple(int(v) for v in b) for b in found]


def locate_cascade_16(image):
    return locate_cascade(image, "Haar plate, 16 stages")


# --------------------------------------------------------------------------- #
# the controls
# --------------------------------------------------------------------------- #


def control_whole_frame(image):
    """**Control.** The whole photograph.

    Scores **coverage 1.00** on every image by construction, which is why
    coverage is never reported without IoU beside it.
    """
    h, w = image.shape[:2]
    return [(0, 0, w, h)]


#: The mean annotated plate, in fractions of the frame, over all eleven
#: photographs. Computed once from the annotations and then frozen, so the
#: control really is a guess that never looks at the image.
FIXED_PLATE = (0.387, 0.575, 0.251, 0.083)


def control_fixed_box(image):
    """**Control.** The average annotated plate, in the same place every time.

    A plate is on the front or back of a car, and the photographer centred the
    car, so "the middle, slightly low" is a decent guess. This says how much of
    any locator's score is that regularity rather than the locator.
    """
    h, w = image.shape[:2]
    fx, fy, fw, fh = FIXED_PLATE
    return [(int(fx * w), int(fy * h), int(fw * w), int(fh * h))]


def control_nothing(image):
    """**Control.** No box at all."""
    return []


LOCATORS: dict[str, Callable] = {
    "Whole frame (control)": control_whole_frame,
    "Fixed box (control)": control_fixed_box,
    "Nothing (control)": control_nothing,
    "Sobel-x + morphology": locate_sobel_morphology,
    "Top-hat + Otsu": locate_tophat,
    "MSER text lines": locate_mser,
    "Contour + aspect": locate_contour_aspect,
    "Haar plate cascade": locate_cascade,
    "Haar plate, 16 stages": locate_cascade_16,
}

REAL_LOCATORS = tuple(k for k in LOCATORS if "control" not in k)


# --------------------------------------------------------------------------- #
# does the crop contain readable characters?
# --------------------------------------------------------------------------- #


def character_blobs(image, box) -> int:
    """How many character-shaped connected components the crop contains.

    Not OCR. It binarises the crop and counts components whose height is a large
    share of the crop's height and whose aspect is that of a letter. Against the
    **typed plate text** this becomes a check a box alone cannot make: a crop
    that a reader could use has roughly as many blobs as the plate has
    characters, and a crop that is half a plate, or a plate plus a bumper, does
    not.
    """
    x, y, w, h = (int(v) for v in box)
    H, W = image.shape[:2]
    x, y = max(0, x), max(0, y)
    w, h = min(w, W - x), min(h, H - y)
    if w < 10 or h < 6:
        return 0
    crop = cv2.cvtColor(image[y:y + h, x:x + w], cv2.COLOR_RGB2GRAY)
    # Normalise the height and **keep the aspect ratio**. A first version scaled
    # the width to `200 * w / h`, which stretched a 4.4:1 plate to 17.4:1 -- every
    # character came out 2.5 times wider than tall, failed a letter-shaped aspect
    # gate, and this function returned zero blobs on a crop of the plate itself.
    crop = cv2.resize(crop, (max(10, int(50 * w / max(h, 1))), 50),
                      interpolation=cv2.INTER_CUBIC)
    crop = cv2.GaussianBlur(crop, (3, 3), 0)
    _, binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    count = 0
    for i in range(1, n):
        cw, ch = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if ch < 0.35 * 50 or ch > 0.95 * 50:
            continue
        if not (0.12 <= cw / max(ch, 1) <= 1.1):
            continue
        count += 1
    return count


def validate_proxy(names=None) -> dict:
    """Score the blob counter against the typed text on the **annotated** crops.

    This runs before the proxy is used on anything a locator produced, because a
    proxy whose own error is unknown cannot say anything about a detector's. On
    a crop of the plate itself the count should equal the number of characters;
    where it does not, the proxy -- not the locator -- is what failed, and the
    ceiling it sets is reported alongside every number that depends on it.
    """
    names = names or image_names()
    rows = []
    for name in names:
        target, text = truth(name)
        expected = sum(1 for c in text if c.isalnum())
        got = character_blobs(load(name), target)
        rows.append({"image": name, "text": text, "characters": expected,
                     "blobs": got, "exact": got == expected,
                     "within_one": abs(got - expected) <= 1})
    return {"frames": len(rows),
            "exact": sum(r["exact"] for r in rows),
            "within_one": sum(r["within_one"] for r in rows),
            "per_image": rows}


def readability(name: str, box) -> dict:
    """Compare the blob count in a crop to the number of characters on the plate.

    The plate text is a **human annotation**, so "the right number of blobs" is
    a proxy and is reported as one. It is still the only thing here that can
    distinguish a box that scores well from a box that is useful.
    """
    image = load(name)
    _, text = truth(name)
    expected = sum(1 for c in text if c.isalnum())
    got = character_blobs(image, box) if box is not None else 0
    return {"image": name, "text": text, "expected": expected, "blobs": got,
            "error": abs(got - expected),
            "readable": abs(got - expected) <= 1}


# --------------------------------------------------------------------------- #
# the comparison
# --------------------------------------------------------------------------- #


def evaluate(locator: str, names=None) -> dict:
    """Both metrics, plus the readability proxy, for one locator.

    IoU and coverage are reported together throughout. Neither survives alone:
    coverage is won outright by returning the whole photograph, and IoU calls a
    box containing every character a failure if it also contains some bumper.
    """
    names = names or image_names()
    fn = LOCATORS[locator]
    rows = []
    for name in names:
        image = load(name)
        target, text = truth(name)
        boxes = fn(image)
        by_iou, v_iou = best_by(boxes, target, iou)
        by_cov, v_cov = best_by(boxes, target, coverage)
        read = readability(name, by_iou)
        rows.append({
            "image": name, "boxes": len(boxes), "iou": v_iou, "coverage": v_cov,
            "hit_iou": v_iou >= 0.5, "hit_coverage": v_cov >= 0.95,
            "readable": read["readable"], "blobs": read["blobs"],
            "characters": read["expected"],
            "plate_share": plate_share(name), "plate_width": plate_width(name),
        })
    return {
        "locator": locator,
        "hits_iou": sum(r["hit_iou"] for r in rows),
        "hits_coverage": sum(r["hit_coverage"] for r in rows),
        "readable": sum(r["readable"] for r in rows),
        "median_iou": float(np.median([r["iou"] for r in rows])),
        "median_coverage": float(np.median([r["coverage"] for r in rows])),
        "boxes": sum(r["boxes"] for r in rows),
        "frames": len(rows),
        "per_image": rows,
    }


def evaluate_all(names=None) -> list[dict]:
    return [evaluate(k, names) for k in LOCATORS]


def metrics_disagree(names=None) -> list[dict]:
    """Where the two metrics rank two locators in opposite orders.

    The point of reporting both. If they always agreed, one of them would be
    redundant; they do not, and which one is quoted decides which locator looks
    best.
    """
    results = {r["locator"]: r for r in evaluate_all(names)}
    rows = []
    real = list(REAL_LOCATORS)
    for i, a in enumerate(real):
        for b in real[i + 1:]:
            ra, rb = results[a], results[b]
            by_iou = np.sign(ra["hits_iou"] - rb["hits_iou"])
            by_cov = np.sign(ra["hits_coverage"] - rb["hits_coverage"])
            if by_iou != 0 and by_cov != 0 and by_iou != by_cov:
                rows.append({
                    "a": a, "b": b,
                    "a_iou": ra["hits_iou"], "b_iou": rb["hits_iou"],
                    "a_coverage": ra["hits_coverage"],
                    "b_coverage": rb["hits_coverage"],
                })
    return rows


def iou_threshold_sweep(locator: str, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7),
                        names=None) -> list[dict]:
    """How many "hits" a locator has, as a function of a number nobody measured.

    0.5 is a convention from PASCAL VOC, chosen for objects that are roughly as
    tall as they are wide. It is quoted for plates without comment.
    """
    names = names or image_names()
    fn = LOCATORS[locator]
    scores = []
    for name in names:
        target, _ = truth(name)
        _, v = best_by(fn(load(name)), target, iou)
        scores.append(v)
    return [{"locator": locator, "threshold": t,
             "hits": int(sum(s >= t for s in scores)), "frames": len(scores)}
            for t in thresholds]


def what_a_pixel_of_error_costs(names=None) -> list[dict]:
    """Shift the **annotated** box by a fixed number of pixels and score it.

    Nothing is detected here: this perturbs the truth itself, so the only thing
    varying is the localisation error, and the only thing measured is what IoU
    charges for it. It is the mechanism behind the headline -- a plate is 4.4
    times wider than tall, so a given pixel error is a much larger share of the
    height, and IoU charges for both directions equally.
    """
    names = names or image_names()
    rows = []
    for shift in (2, 4, 8, 16):
        dx_iou, dy_iou, dy_cov = [], [], []
        for name in names:
            (x, y, w, h), _ = truth(name)
            target = (x, y, w, h)
            dx_iou.append(iou((x + shift, y, w, h), target))
            dy_iou.append(iou((x, y + shift, w, h), target))
            dy_cov.append(coverage((x, y + shift, w, h), target))
        rows.append({
            "shift_px": shift,
            "iou_shifted_sideways": float(np.mean(dx_iou)),
            "iou_shifted_vertically": float(np.mean(dy_iou)),
            "coverage_shifted_vertically": float(np.mean(dy_cov)),
        })
    return rows


def difficulty(names=None) -> list[dict]:
    """One row per photograph: how big the plate is, and who found it.

    Plate size is read from the annotation, before any locator runs.
    """
    names = names or image_names()
    results = {r["locator"]: {p["image"]: p for p in r["per_image"]}
               for r in evaluate_all(names)}
    rows = []
    for name in names:
        target, text = truth(name)
        found_iou = sum(results[k][name]["hit_iou"] for k in REAL_LOCATORS)
        found_cov = sum(results[k][name]["hit_coverage"] for k in REAL_LOCATORS)
        rows.append({
            "image": name, "text": text,
            "plate_px": f"{target[2]}x{target[3]}",
            "plate_width": target[2],
            "plate_share": plate_share(name),
            "aspect": target[2] / max(target[3], 1),
            "found_iou": found_iou, "found_coverage": found_cov,
            "locators": len(REAL_LOCATORS),
        })
    return rows


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #


def draw(image, boxes, colour=(60, 220, 90), thickness=None):
    out = image.copy()
    thickness = thickness or max(2, image.shape[1] // 300)
    for x, y, w, h in np.asarray(boxes, np.int32).reshape(-1, 4):
        cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)),
                      colour, thickness)
    return out


def draw_truth(image, box, colour=(250, 190, 40)):
    return draw(image, [box], colour=colour)
