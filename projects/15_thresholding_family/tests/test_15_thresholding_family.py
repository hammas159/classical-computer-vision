"""Tests for project 15, the thresholding family.

Most of these pin a *finding* rather than a number, so that a later change which
quietly reverses one of the project's conclusions fails loudly instead of
rewriting the README by accident.
"""

from __future__ import annotations

import numpy as np
import pytest

import thresholding as th
from shared.metrics import iou


def score(mask: np.ndarray, truth: np.ndarray) -> float:
    """Best of the two polarities — see `run.best_polarity` for why."""
    return max(iou(mask > 0, truth > 0), iou(mask == 0, truth > 0))


# --------------------------------------------------------------------------- #
# the scene
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", ["solid", "thin"])
def test_the_scene_has_the_foreground_fraction_it_was_asked_for(kind):
    _, truth = th.synthetic_scene(kind=kind, fg_fraction=0.12, seed=0)
    assert 0.06 < float((truth > 0).mean()) < 0.20


def test_an_unknown_scene_kind_raises():
    with pytest.raises(ValueError, match="unknown scene kind"):
        th.synthetic_scene(kind="squiggles")


def test_thin_strokes_are_narrower_than_the_local_window():
    """The relationship the whole project turns on.

    Local thresholding compares a pixel to its own window. A stroke narrower
    than that window always has background in it; the interior of a filled
    circle never does.
    """
    _, truth = th.synthetic_scene(kind="thin", fg_fraction=0.12, seed=0)
    import cv2

    dist = cv2.distanceTransform((truth > 0).astype(np.uint8), cv2.DIST_L2, 3)
    assert float(dist.max()) * 2 < 31  # the default window used by every local method


def test_solid_shapes_are_wider_than_the_local_window():
    _, truth = th.synthetic_scene(kind="solid", fg_fraction=0.12, seed=0)
    import cv2

    dist = cv2.distanceTransform((truth > 0).astype(np.uint8), cv2.DIST_L2, 3)
    assert float(dist.max()) * 2 > 31


# --------------------------------------------------------------------------- #
# the oracle
# --------------------------------------------------------------------------- #


def test_the_oracle_is_a_ceiling_over_every_global_method():
    """It searches every cut *and both polarities*, so nothing global can beat it.

    It used to search only `gray > t`. The scene is dark-on-light, so that
    formulation cannot express the right answer at any threshold: the oracle
    scored **0.122 IoU** and sat below every method it exists to bound. A
    ceiling under the thing it bounds is worse than no ceiling — it reads as
    "no global threshold was available" when a perfect one was.
    """
    for kind in ("solid", "thin"):
        img, truth = th.synthetic_scene(kind=kind, fg_fraction=0.12, illum_min=0.6, seed=0)
        oracle, _t = th.thresh_best_global(img, truth)
        ceiling = score(oracle, truth)
        for name in ("Fixed 127 (control)", "Otsu", "Triangle", "Multi-Otsu (3 class)"):
            assert score(th.METHODS[name](img), truth) <= ceiling + 1e-6, name


def test_a_perfect_global_cut_exists_under_even_light():
    img, truth = th.synthetic_scene(kind="solid", fg_fraction=0.12, seed=0)
    assert score(th.thresh_best_global(img, truth)[0], truth) > 0.99


# --------------------------------------------------------------------------- #
# the findings
# --------------------------------------------------------------------------- #


def test_local_thresholding_wins_on_thin_strokes_and_loses_on_solid_shapes():
    """The project's headline, pinned.

    Under *identical* bad lighting, Sauvola beats Otsu by a wide margin on thin
    strokes and loses to it on solid shapes. The received wisdom — "use adaptive
    thresholding when the light is uneven" — is missing its other half: it
    depends on the shape of the foreground, not on the illumination.
    """
    thin_img, thin_truth = th.synthetic_scene(kind="thin", fg_fraction=0.12,
                                              illum_min=0.3, seed=0)
    solid_img, solid_truth = th.synthetic_scene(kind="solid", fg_fraction=0.12,
                                                illum_min=0.3, seed=0)

    thin_otsu = score(th.METHODS["Otsu"](thin_img), thin_truth)
    thin_sauvola = score(th.METHODS["Sauvola"](thin_img), thin_truth)
    solid_otsu = score(th.METHODS["Otsu"](solid_img), solid_truth)
    solid_sauvola = score(th.METHODS["Sauvola"](solid_img), solid_truth)

    assert thin_sauvola > thin_otsu + 0.5      # local wins, hugely
    assert solid_sauvola > solid_otsu          # ... and on solids it is closer
    assert thin_sauvola > 0.95
    assert solid_sauvola < 0.6                 # but never actually good


def test_local_thresholding_hollows_out_a_solid_shape():
    """Why it loses there, shown directly rather than asserted.

    The interior of a filled region is marked background, so what survives is a
    ring. Measured as: the detected foreground overlaps the *boundary* of the
    truth far better than the truth itself.
    """
    import cv2

    img, truth = th.synthetic_scene(kind="solid", fg_fraction=0.12, seed=0)
    mask = th.METHODS["Adaptive mean"](img)
    mask = mask if iou(mask > 0, truth > 0) >= iou(mask == 0, truth > 0) else (mask == 0)

    eroded = cv2.erode((truth > 0).astype(np.uint8), np.ones((9, 9), np.uint8))
    interior = eroded > 0
    boundary = (truth > 0) & ~interior

    covered_interior = float((np.asarray(mask) > 0)[interior].mean())
    covered_boundary = float((np.asarray(mask) > 0)[boundary].mean())
    assert covered_boundary > covered_interior + 0.2


def test_otsu_can_fail_where_a_perfect_global_cut_exists():
    """The finding project 01 reached on documents, on a controlled scene.

    Under a strong gradient on solid shapes the oracle still reaches 1.000 — a
    single global threshold separates the two populations perfectly — and Otsu
    scores about 0.3. Otsu's criterion picked the wrong cut; the cut was there.
    """
    img, truth = th.synthetic_scene(kind="solid", fg_fraction=0.12, illum_min=0.3, seed=0)
    oracle = score(th.thresh_best_global(img, truth)[0], truth)
    otsu = score(th.METHODS["Otsu"](img), truth)
    assert oracle > 0.95
    assert otsu < 0.5
    assert oracle - otsu > 0.5


def test_sometimes_no_global_cut_exists_at_all():
    """The other half, which makes the first half meaningful.

    On thin strokes under a severe gradient the oracle itself only reaches about
    0.74, so no global threshold can do better — and Sauvola reaches 1.000 by
    not being global. Without this case, "Otsu failed" would always look like
    Otsu's fault.
    """
    img, truth = th.synthetic_scene(kind="thin", fg_fraction=0.12, illum_min=0.15, seed=0)
    oracle = score(th.thresh_best_global(img, truth)[0], truth)
    sauvola = score(th.METHODS["Sauvola"](img), truth)
    assert oracle < 0.95
    assert sauvola > oracle


def test_niblack_speckles_where_sauvola_does_not():
    """Sauvola's dynamic-range term exists for exactly this, and it is measurable."""
    img, truth = th.synthetic_scene(kind="solid", fg_fraction=0.12, seed=0)
    assert score(th.METHODS["Sauvola"](img), truth) > score(th.METHODS["Niblack"](img), truth) + 0.2
