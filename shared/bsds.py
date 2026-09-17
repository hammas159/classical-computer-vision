"""Human segmentations from BSDS500 — the only real ground truth in this repo.

Everywhere else in these projects the truth is either **generated** (a flow
field, a blur kernel, a synthetic scene) or **defined by construction** (Otsu's
binarisation declared to be the target). Both are exact, and both are answers to
a question a person never asked.

BSDS500 ships something different: **five human annotators per image**, each
giving a complete region labelling and a boundary map. That brings two things
generated truth cannot.

1. A segmentation target that reflects what a person actually considers one
   object, including the parts that are semantic rather than photometric — a
   tiger and its shadow are one region to a human and two to any clustering.
2. **The annotators disagree with each other.** Scoring one human against
   another gives a ceiling that is not an assumption: no algorithm has any
   business scoring above the agreement between two people looking at the same
   picture, and an algorithm that appears to is being scored wrongly.

The `.mat` files are cached by ``tools/fetch_images.py cache --set bsds_gt`` and
are keyed by the BSDS image id. `shared.io` knows images by descriptive name, so
`gt_for_name` walks the manifest that ``tools/check_image_reuse.py`` writes to
map one to the other.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "assets" / "real" / "manifest.json"
CACHE_DIR = Path.home() / ".cache" / "classical-cv-images" / "bsds_gt"


class GroundTruthMissing(RuntimeError):
    """Raised with an actionable message rather than a KeyError three frames down."""


@lru_cache(maxsize=1)
def _name_to_id() -> dict[str, str]:
    """Map a `shared.io` photo name to its BSDS id, via the perceptual-hash manifest.

    The manifest exists because images are renamed to something descriptive on
    the way into the repository; without it there is no link between
    ``tiger_in_shade.jpg`` and ``108082.mat``.
    """
    if not MANIFEST.exists():
        return {}
    out = {}
    for stem, entry in json.loads(MANIFEST.read_text()).items():
        source = entry.get("source") or ""
        if source.startswith("bsds/"):
            out[stem] = source.split("/", 1)[1]
    return out


def bsds_id(name: str) -> str | None:
    """The BSDS id behind a `shared.io` photo name, or None if it is not BSDS."""
    return _name_to_id().get(name)


def has_ground_truth(name: str) -> bool:
    ident = bsds_id(name)
    return bool(ident) and (CACHE_DIR / f"{ident}.mat").exists()


def load_annotations(name: str) -> list[dict[str, np.ndarray]]:
    """Every annotator's segmentation and boundary map for one photograph.

    Returns a list of ``{"segmentation": (H, W) int, "boundaries": (H, W) bool}``,
    one per annotator. Typically five, but it varies by image and that variation
    is itself worth not averaging away.
    """
    ident = bsds_id(name)
    if not ident:
        raise GroundTruthMissing(
            f"{name!r} is not a BSDS photograph, or the manifest is stale. "
            "Run `python tools/check_image_reuse.py` to rebuild it."
        )
    path = CACHE_DIR / f"{ident}.mat"
    if not path.exists():
        raise GroundTruthMissing(
            f"no cached ground truth for {name!r} (BSDS {ident}) at {path}. "
            "Run `python tools/fetch_images.py cache --set bsds_gt`."
        )

    from scipy.io import loadmat

    raw = loadmat(str(path))["groundTruth"]
    out = []
    for i in range(raw.shape[1]):
        entry = raw[0, i]
        out.append({
            "segmentation": np.asarray(entry["Segmentation"][0, 0]).astype(np.int32),
            "boundaries": np.asarray(entry["Boundaries"][0, 0]).astype(bool),
        })
    return out


def consensus_boundaries(name: str, min_annotators: int = 2) -> np.ndarray:
    """Boundary pixels at least ``min_annotators`` people agreed on.

    A single annotator's boundary map is one person's opinion and includes marks
    nobody else made. Requiring agreement is what turns five opinions into a
    target, and the threshold is a stated choice rather than a hidden default:
    at 1 the map is the union of every stray line, at 5 it is almost empty.
    """
    ann = load_annotations(name)
    votes = np.sum([a["boundaries"] for a in ann], axis=0)
    return votes >= min_annotators


def boundary_f_measure(predicted: np.ndarray, target: np.ndarray,
                       tolerance: int = 2) -> dict[str, float]:
    """Precision, recall and F against a boundary map, with a matching tolerance.

    The metric BSDS exists for. A boundary is a one-pixel-wide curve, so an exact
    pixel comparison scores a perfect contour drawn one pixel to the left at
    zero; every published number on this dataset uses a tolerance, and 2 px is
    the usual choice.

    Precision and recall are both reported and neither is optional. A method
    that draws boundaries everywhere has perfect recall, and one that draws a
    single confident edge has perfect precision -- the pair is what says
    anything, which is exactly the lesson the IoU column of project 24 teaches
    the hard way.
    """
    import cv2

    pred = np.asarray(predicted) > 0
    targ = np.asarray(target) > 0
    if not pred.any() or not targ.any():
        return {"precision": 0.0, "recall": 0.0, "f": 0.0}

    k = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
    pred_near = cv2.dilate(pred.astype(np.uint8), k) > 0
    targ_near = cv2.dilate(targ.astype(np.uint8), k) > 0

    precision = float((pred & targ_near).sum() / max(pred.sum(), 1))
    recall = float((targ & pred_near).sum() / max(targ.sum(), 1))
    f = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f": round(f, 4)}


def dominant_foreground(name: str, annotator: int = 0) -> np.ndarray:
    """The largest region in one annotator's labelling, inverted to a foreground mask.

    **A lossy reading of a rich annotation, and on some images a wrong one.** A
    human segmentation is a *labelling*, not a figure/ground split, and the
    assumption here -- that the largest region is the background -- fails
    whenever it is not. On `memorial_arch` the largest annotated region covers
    24.5% of the frame, so this returns a 75.5% "foreground" that corresponds to
    nothing anyone drew, and methods returning sensible regions score 0.000
    against it.

    Kept because the binary framing is what several projects need and because
    the failure is instructive. Project 24 reports boundary F-measure against
    `consensus_boundaries` as its headline for exactly this reason, and keeps
    the IoU column only to show what it does wrong.
    """
    seg = load_annotations(name)[annotator]["segmentation"]
    labels, counts = np.unique(seg, return_counts=True)
    background = labels[int(np.argmax(counts))]
    return (seg != background).astype(np.uint8) * 255
