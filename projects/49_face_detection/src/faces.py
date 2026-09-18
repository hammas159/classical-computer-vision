"""Face detection: six cascades, three controls, and no annotation anywhere.

The question
------------
Viola-Jones, 2001. A cascade of boosted classifiers slid over every position and
every scale. Twenty-five years later it still ships with OpenCV, in six variants,
and every tutorial picks one of them with a sentence like "``alt2`` is usually
better". This project asks what "better" could mean when nobody has labelled the
photographs.

> **A cascade is trained at one face size and cannot see below it.** OpenCV's
> `lbpcascade_frontalface_improved` has a **45x45** training window where every
> other cascade here has 20x20 or 24x24. On these eleven group photographs it
> finds **3 faces** where the cascade it replaced finds **90**. Upscaling the
> photographs by 2x recovers it to **81** -- so it is not broken, it is blind
> below its own window, and `minSize` cannot fix that because `minSize` is a
> floor on the search, not a resampling of the image.

> **Upright means upright.** Rotate a class photograph by 20 degrees and the best
> cascade here loses **28% of its detections**; by 30 degrees, **88%**. One
> cascade (`alt_tree`) is at **zero by 20 degrees**. Nothing in the API suggests
> a tilt of a head is outside the model.

> **`minNeighbors` is not a quality knob, it is an operating point**, and the
> cascades do not share one. At `minNeighbors=1`, `haar default` returns 158
> boxes on the face photographs and **25 on photographs with no face in them**;
> `haar alt` returns 109 and **0**. The extra 49 detections cost 25 false alarms,
> and a comparison at a single setting cannot see that.

Where the ground truth comes from
---------------------------------
**Nobody has labelled a face in this repository, and nothing here pretends
otherwise.** There is no box drawn by hand, so there is no recall and no
precision. Three things stand in, and each is honest about what it is:

1. **A recorded transform.** Rotate, scale, brighten, blur or re-encode a
   photograph by a known amount; a box found in the original must reappear where
   the transform puts it. The thing being measured *is* the thing that was
   applied, so this is exact.
2. **An empty truth.** Eleven photographs with no human face in them -- animals,
   textures, architecture. Every detection is a false alarm, with no annotation
   needed to say so.
3. **Agreement between cascades**, reported as agreement and never as accuracy.
   Six detectors agreeing on a box is evidence about the detectors, not about
   the photograph.

The three controls
------------------
`Nothing` returns no boxes: it scores a perfect zero false alarms and finds
nothing, which is what an empty-truth arm alone would reward. `One centre box`
returns a single box in the middle of the frame. `Every box` returns a dense grid
at every scale, which "finds" every face and is useless. Between them they fence
in what any real number here can mean.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

EPS = 1e-9

#: Group photographs from OpenCV's own cascade test data, fetched by
#: `tools/fetch_assets.py --set faces`. Several faces per frame at different
#: scales, some in profile, one of them a painting.
FACES = Path.home() / ".cache" / "classical-cv-images" / "assets" / "faces"

#: The LBP cascades, which do not ship inside the `opencv-python` wheel.
CASCADE_DIR = Path.home() / ".cache" / "classical-cv-images" / "assets" / "cascades"

#: Photographs with **no human face in them**, from the BSDS500 images no other
#: project in this repository uses. Chosen adversarially rather than
#: conveniently: six of the eleven contain an *animal* looking straight at the
#: camera, and one is a rack of wooden clogs, which is the kind of repeated dark
#: blob a Haar cascade is supposed to like.
BSDS = Path.home() / ".cache" / "classical-cv-images" / "bsds"

NO_FACE = {
    "100098": "a bear in snow, head down",
    "106024": "a penguin on rock, head up",
    "108005": "a tiger facing the camera",
    "108036": "a tiger facing the camera, at water",
    "12003": "a starfish on green weed",
    "134035": "a leopard in bare branches",
    "138078": "a rowing boat on a wooden dock",
    "140075": "racks of wooden clogs",
    "159045": "a bobcat among ferns",
    "16068": "three zebras grazing",
    "161062": "the pyramids at Giza",
}

#: Two photographs of **carved human faces**. They are kept out of both arms and
#: reported on their own, because "is a wooden totem a face?" is a question about
#: what the word means and not about the detector. Adjudicating it silently --
#: in either direction -- would put an opinion into a false-alarm rate.
CARVED = {"101085": "three carved wooden totems"}

_HAAR = Path(cv2.data.haarcascades)

#: The six cascades, with where each one comes from.
CASCADE_FILES = {
    "Haar default": _HAAR / "haarcascade_frontalface_default.xml",
    "Haar alt": _HAAR / "haarcascade_frontalface_alt.xml",
    "Haar alt2": _HAAR / "haarcascade_frontalface_alt2.xml",
    "Haar alt_tree": _HAAR / "haarcascade_frontalface_alt_tree.xml",
    "LBP frontal": CASCADE_DIR / "lbpcascade_frontalface.xml",
    "LBP improved": CASCADE_DIR / "lbpcascade_frontalface_improved.xml",
}

#: One scale factor and one minimum size for every cascade, so the comparison is
#: of models rather than of tuning. `minNeighbors` is swept separately, because
#: it is the one knob where a single value genuinely does mean different things
#: to different cascades -- see `operating_points`.
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5
MIN_SIZE = 24

_loaded: dict[str, cv2.CascadeClassifier] = {}
_windows: dict[str, int] = {}


def available() -> bool:
    return (FACES.exists() and len(list(FACES.glob("*.png"))) >= 10
            and all(p.exists() for p in CASCADE_FILES.values())
            and all((BSDS / f"{n}.jpg").exists() for n in NO_FACE))


def cascade(name: str) -> cv2.CascadeClassifier:
    if name not in _loaded:
        path = CASCADE_FILES[name]
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Run `python tools/fetch_assets.py --set cascades`.")
        c = cv2.CascadeClassifier(str(path))
        if c.empty():
            raise RuntimeError(f"{path} did not load as a cascade")
        _loaded[name] = c
    return _loaded[name]


def training_window(name: str) -> int:
    """The width of the window the cascade was trained on, read from its own XML.

    This is the number that explains `LBP improved`, and it is not in the
    documentation, the function signature or any tutorial -- it is in the file.
    """
    if name not in _windows:
        text = CASCADE_FILES[name].read_text(errors="ignore")
        _windows[name] = int(re.search(r"<width>(\d+)</width>", text).group(1))
    return _windows[name]


def image_names() -> list[str]:
    return sorted(p.stem for p in FACES.glob("*.png"))


def load(name: str) -> np.ndarray:
    path = FACES / f"{name}.png"
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(
            f"{path} is missing. Run `python tools/fetch_assets.py --set faces`.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_no_face(name: str) -> np.ndarray:
    path = BSDS / f"{name}.jpg"
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"{path} is missing. Run `python tools/fetch_images.py`.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def prepare(image: np.ndarray) -> np.ndarray:
    """Grey, then histogram-equalise. Every cascade gets the identical input."""
    g = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    return cv2.equalizeHist(g)


# --------------------------------------------------------------------------- #
# the detectors
# --------------------------------------------------------------------------- #


def detect(image: np.ndarray, name: str, min_neighbors: int = MIN_NEIGHBORS,
           scale_factor: float = SCALE_FACTOR, min_size: int = MIN_SIZE,
           upscale: float = 1.0) -> np.ndarray:
    """Boxes as an (N, 4) array of x, y, w, h in the **original** coordinates.

    ``upscale`` resamples the image before the search and divides the boxes back
    down afterwards. It exists because `minSize` cannot make a cascade see below
    its own training window -- it is a floor on the search, not a resampling --
    and resampling is the only thing that can.
    """
    g = prepare(image)
    if upscale != 1.0:
        g = cv2.resize(g, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    found = cascade(name).detectMultiScale(
        g, scale_factor, min_neighbors, minSize=(min_size, min_size))
    if len(found) == 0:
        return np.zeros((0, 4), np.float64)
    boxes = np.asarray(found, np.float64)
    return boxes / upscale if upscale != 1.0 else boxes


def detect_nothing(image, **kw) -> np.ndarray:
    """**Control.** No boxes, ever. Perfect on the empty-truth arm."""
    return np.zeros((0, 4), np.float64)


def detect_one_centre_box(image, **kw) -> np.ndarray:
    """**Control.** One box in the middle of the frame, the image never read."""
    h, w = image.shape[:2]
    side = 0.25 * min(h, w)
    return np.array([[w / 2 - side / 2, h / 2 - side / 2, side, side]], np.float64)


def detect_every_box(image, step: float = 0.5, **kw) -> np.ndarray:
    """**Control.** A dense grid of boxes at three scales.

    This "finds" nearly every face in the set, which is the point: a measure that
    rewards finding faces without charging for the ones invented has to rank this
    first.
    """
    h, w = image.shape[:2]
    boxes = []
    for side in (0.10 * min(h, w), 0.20 * min(h, w), 0.35 * min(h, w)):
        stride = max(1.0, step * side)
        for y in np.arange(0, h - side, stride):
            for x in np.arange(0, w - side, stride):
                boxes.append([x, y, side, side])
    return np.asarray(boxes, np.float64) if boxes else np.zeros((0, 4), np.float64)


CONTROLS: dict[str, Callable] = {
    "Nothing (control)": detect_nothing,
    "One centre box (control)": detect_one_centre_box,
    "Every box (control)": detect_every_box,
}

DETECTORS = list(CASCADE_FILES)


def run(image: np.ndarray, method: str, **kw) -> np.ndarray:
    """One entry point for a cascade name or a control name."""
    if method in CONTROLS:
        return CONTROLS[method](image, **kw)
    return detect(image, method, **kw)


ALL_METHODS = DETECTORS + list(CONTROLS)


# --------------------------------------------------------------------------- #
# box arithmetic
# --------------------------------------------------------------------------- #


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    return float(inter / (aw * ah + bw * bh - inter + EPS))


def iou_matrix(boxes, targets) -> np.ndarray:
    """All-pairs IoU as a (len(targets), len(boxes)) array.

    Vectorised because the `Every box` control returns eight hundred boxes per
    frame, and the obvious nested-loop version turned the rotation sweep into
    forty-nine million Python-level IoU calls. The control is the reason: it
    exists precisely to be absurd, so it has to be cheap to score.
    """
    b = np.asarray(boxes, np.float64).reshape(-1, 4)
    t = np.asarray(targets, np.float64).reshape(-1, 4)
    if len(b) == 0 or len(t) == 0:
        return np.zeros((len(t), len(b)))
    bx0, by0 = b[None, :, 0], b[None, :, 1]
    bx1, by1 = bx0 + b[None, :, 2], by0 + b[None, :, 3]
    tx0, ty0 = t[:, None, 0], t[:, None, 1]
    tx1, ty1 = tx0 + t[:, None, 2], ty0 + t[:, None, 3]
    iw = np.clip(np.minimum(bx1, tx1) - np.maximum(bx0, tx0), 0, None)
    ih = np.clip(np.minimum(by1, ty1) - np.maximum(by0, ty0), 0, None)
    inter = iw * ih
    union = b[None, :, 2] * b[None, :, 3] + t[:, None, 2] * t[:, None, 3] - inter
    return inter / (union + EPS)


def matched(boxes, targets, threshold: float = 0.5) -> int:
    """How many targets have a box over them, each box used at most once.

    Greedy over the best remaining pair rather than in target order: taking the
    targets in whatever order they arrive lets an early target claim a box that
    was a much better fit for a later one, which undercounts.
    """
    if len(boxes) == 0 or len(targets) == 0:
        return 0
    m = iou_matrix(boxes, targets)
    hits = 0
    while True:
        i, j = np.unravel_index(np.argmax(m), m.shape)
        if m[i, j] < threshold:
            return hits
        hits += 1
        m[i, :] = -1.0
        m[:, j] = -1.0


# --------------------------------------------------------------------------- #
# arm 1: a recorded transform
# --------------------------------------------------------------------------- #


def rotation_matrix(shape, degrees: float):
    h, w = shape[:2]
    return cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degrees, 1.0)


def map_box(M, box):
    """Push a box through an affine transform and take the axis-aligned hull.

    The hull of a rotated square is larger than the square, which **lowers** the
    IoU a correct detector can reach. That is stated rather than corrected: at 30
    degrees the hull of a square has 1.37x its area, so an IoU of 0.73 is the
    ceiling even for a detector that is exactly right. The threshold used here is
    0.5, comfortably below that ceiling up to 45 degrees.
    """
    x, y, w, h = box
    corners = np.array([[x, y, 1.0], [x + w, y, 1.0],
                        [x, y + h, 1.0], [x + w, y + h, 1.0]])
    pts = corners @ M.T
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    return np.array([x0, y0, x1 - x0, y1 - y0])


def rotation_survival(method: str, degrees=(0, 5, 10, 15, 20, 30, 45),
                      names=None, **kw) -> list[dict]:
    """Rotate by a known angle and count how many original boxes come back.

    The truth is the rotation, which is applied here and therefore exact. What is
    measured is not accuracy -- a box this loses may never have been a face -- but
    **whether the detector's own answer survives a tilt of the camera**, which is
    the property Viola-Jones is famous for not having and which no tutorial
    quantifies.
    """
    names = names or image_names()
    rows = []
    for angle in degrees:
        kept = base = 0
        for n in names:
            img = load(n)
            before = run(img, method, **kw)
            if len(before) == 0:
                continue
            M = rotation_matrix(img.shape, angle)
            h, w = img.shape[:2]
            turned = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            after = run(turned, method, **kw)
            targets = [map_box(M, b) for b in before]
            kept += matched(after, targets)
            base += len(before)
        rows.append({"method": method, "degrees": angle, "kept": kept,
                     "of": base, "survival": kept / max(base, 1)})
    return rows


def photometric_survival(method: str, names=None, **kw) -> list[dict]:
    """The same idea for changes that do not move anything.

    Brightness, contrast, blur and JPEG leave every box exactly where it was, so
    the mapped truth is the identity and any loss is the detector giving up
    rather than the geometry moving. Worth separating from rotation for that
    reason: they fail for different reasons and a single "robustness" number
    would merge them.
    """
    names = names or image_names()

    def brighter(img, k):
        return np.clip(img.astype(np.float32) * k, 0, 255).astype(np.uint8)

    def blurred(img, s):
        return cv2.GaussianBlur(img, (0, 0), s)

    def jpeg(img, q):
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                               [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
        return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    changes = [("brightness x0.5", lambda im: brighter(im, 0.5)),
               ("brightness x1.5", lambda im: brighter(im, 1.5)),
               ("blur sigma 2", lambda im: blurred(im, 2.0)),
               ("blur sigma 4", lambda im: blurred(im, 4.0)),
               ("JPEG quality 30", lambda im: jpeg(im, 30)),
               ("JPEG quality 10", lambda im: jpeg(im, 10))]

    rows = []
    for label, fn in changes:
        kept = base = 0
        for n in names:
            img = load(n)
            before = run(img, method, **kw)
            if len(before) == 0:
                continue
            after = run(fn(img), method, **kw)
            kept += matched(after, before)
            base += len(before)
        rows.append({"method": method, "change": label, "kept": kept,
                     "of": base, "survival": kept / max(base, 1)})
    return rows


# --------------------------------------------------------------------------- #
# arm 2: an empty truth
# --------------------------------------------------------------------------- #


def false_alarms(method: str, min_neighbors: int = MIN_NEIGHBORS, **kw) -> dict:
    """Detections on photographs with no human face: every one is wrong.

    Reported **per megapixel**, because a detector is slid over every position
    and a bigger photograph therefore offers more chances to be wrong. A raw
    count would rank the detectors by the sizes of the images they happened to
    be given.
    """
    total = 0
    megapixels = 0.0
    per_image = {}
    for name in NO_FACE:
        img = load_no_face(name)
        found = run(img, method, min_neighbors=min_neighbors, **kw) \
            if method in DETECTORS else run(img, method)
        total += len(found)
        mp = img.shape[0] * img.shape[1] / 1e6
        megapixels += mp
        per_image[name] = len(found)
    return {"method": method, "false_alarms": total,
            "megapixels": megapixels, "per_megapixel": total / max(megapixels, EPS),
            "worst_image": max(per_image, key=per_image.get),
            "worst_count": max(per_image.values()), "per_image": per_image}


def detections_on_faces(method: str, min_neighbors: int = MIN_NEIGHBORS,
                        names=None, **kw) -> dict:
    """Boxes returned on the face photographs. **Not a recall** -- nothing here
    knows how many faces there are."""
    names = names or image_names()
    total = 0
    megapixels = 0.0
    for n in names:
        img = load(n)
        found = run(img, method, min_neighbors=min_neighbors, **kw) \
            if method in DETECTORS else run(img, method)
        total += len(found)
        megapixels += img.shape[0] * img.shape[1] / 1e6
    return {"method": method, "boxes": total, "megapixels": megapixels,
            "per_megapixel": total / max(megapixels, EPS)}


def operating_points(method: str, neighbors=(1, 2, 3, 5, 8, 12)) -> list[dict]:
    """The two arms at every setting of the one knob everybody turns.

    `minNeighbors` is documented as "how many neighbours each candidate rectangle
    should have to retain it", which sounds like a quality setting. It is an
    operating point, and the same value buys wildly different trades from
    different cascades.
    """
    rows = []
    for mn in neighbors:
        if method in CONTROLS:
            boxes = detections_on_faces(method)
            alarms = false_alarms(method)
        else:
            boxes = detections_on_faces(method, min_neighbors=mn)
            alarms = false_alarms(method, min_neighbors=mn)
        rows.append({"method": method, "min_neighbors": mn,
                     "boxes": boxes["boxes"],
                     "false_alarms": alarms["false_alarms"],
                     "false_per_megapixel": alarms["per_megapixel"]})
    return rows


# --------------------------------------------------------------------------- #
# arm 3: the training window
# --------------------------------------------------------------------------- #


def window_blindness(names=None, scales=(1.0, 1.5, 2.0, 3.0)) -> list[dict]:
    """Does a cascade's own training window explain what it fails to find?

    The prediction is specific: a cascade trained at 45x45 cannot see a face of
    30 pixels no matter what `minSize` says, and resampling the image up is the
    only thing that can help. If the prediction is right, upscaling recovers the
    45x45 cascade and barely moves the 24x24 one.
    """
    names = names or image_names()
    rows = []
    for method in DETECTORS:
        window = training_window(method)
        counts = {}
        for s in scales:
            counts[s] = sum(len(detect(load(n), method, upscale=s)) for n in names)
        rows.append({"method": method, "window": window,
                     **{f"x{s:g}": counts[s] for s in scales},
                     "recovery": counts[max(scales)] / max(counts[1.0], 1)})
    return rows


def min_size_does_not_help(names=None, sizes=(12, 24, 45, 80)) -> list[dict]:
    """The thing a reader will try first, measured so they do not have to.

    `minSize` is a floor on the search window, so lowering it can only add
    smaller candidates -- it cannot make a 45x45 classifier evaluate a 30-pixel
    face, because there is no 30-pixel classifier to evaluate.
    """
    names = names or image_names()
    rows = []
    for method in DETECTORS:
        row = {"method": method, "window": training_window(method)}
        for s in sizes:
            row[f"minSize {s}"] = sum(
                len(detect(load(n), method, min_size=s)) for n in names)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# arm 4: agreement, reported as agreement
# --------------------------------------------------------------------------- #


def consensus(image: np.ndarray, votes: int = 3, min_neighbors: int = MIN_NEIGHBORS):
    """Boxes that at least `votes` of the six cascades agree on.

    **This is not ground truth and is never used as one.** Six detectors sharing
    a training set and an architecture agree about their shared blind spots as
    readily as about faces, so a consensus box is evidence about the cascades.
    It is here to say how much of the disagreement between them is real.
    """
    pools = [detect(image, m, min_neighbors=min_neighbors) for m in DETECTORS]
    everything = [b for pool in pools for b in pool]
    clusters: list[list[np.ndarray]] = []
    owners: list[set[int]] = []
    for i, pool in enumerate(pools):
        for box in pool:
            for c, o in zip(clusters, owners):
                if iou(c[0], box) >= 0.4:
                    c.append(box)
                    o.add(i)
                    break
            else:
                clusters.append([box])
                owners.append({i})
    keep = [np.mean(c, axis=0) for c, o in zip(clusters, owners) if len(o) >= votes]
    return (np.asarray(keep) if keep else np.zeros((0, 4), np.float64),
            len(clusters), len(everything))


def agreement(names=None, min_neighbors: int = MIN_NEIGHBORS) -> list[dict]:
    names = names or image_names()
    rows = []
    for n in names:
        img = load(n)
        for votes in (1,):
            boxes, clusters, total = consensus(img, votes=votes,
                                               min_neighbors=min_neighbors)
        agreed = {v: len(consensus(img, votes=v, min_neighbors=min_neighbors)[0])
                  for v in (1, 2, 3, 4, 5, 6)}
        rows.append({"image": n, "distinct_boxes": clusters, "total_boxes": total,
                     **{f"{v}_or_more": agreed[v] for v in agreed}})
    return rows


def carved_faces(min_neighbors: int = MIN_NEIGHBORS) -> list[dict]:
    """The photographs this project refuses to score.

    A wooden totem has eyes, a nose and a mouth in the right arrangement. Whether
    a detection there is a false alarm is a question about the word "face", and
    putting an answer to it inside a false-alarm rate would be smuggling an
    opinion into a number.
    """
    rows = []
    for name, what in CARVED.items():
        img = load_no_face(name)
        row = {"image": name, "what": what}
        for method in DETECTORS:
            row[method] = len(detect(img, method, min_neighbors=min_neighbors))
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #


def draw(image, boxes, colour=(60, 220, 90), thickness=None):
    out = image.copy()
    thickness = thickness or max(2, image.shape[1] // 320)
    for x, y, w, h in np.asarray(boxes).reshape(-1, 4):
        cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)),
                      colour, thickness)
    return out
