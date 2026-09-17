"""Tests for project 22, morphology.

The algebraic tests are laws: a failure there is a bug, not a result. The rest
pin *findings*, so that a later change which quietly reverses one of the
project's conclusions fails loudly instead of rewriting the README by accident.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import morphology as mo
from shared.metrics import iou


# --------------------------------------------------------------------------- #
# the laws
# --------------------------------------------------------------------------- #


def test_the_algebraic_identities_hold_for_every_element():
    """Opening is idempotent and anti-extensive, closing is extensive, and
    erosion and dilation are duals. These are theorems; a failure is a bug."""
    for row in mo.verify_algebraic_identities():
        for key, value in row.items():
            if key != "element":
                assert value is True, f"{row['element']} failed {key}"


def test_opening_removes_everything_smaller_than_the_element():
    """The definition, made measurable: the fraction kept falls monotonically."""
    kept = [r["fraction_kept"] for r in mo.sweep_size()]
    assert kept == sorted(kept, reverse=True)
    assert kept[0] > 0.98 and kept[-1] < kept[0]


def test_hit_or_miss_finds_every_corner_exactly():
    """Not a statistical method — it either matches the template or it does not."""
    rows = mo.evaluate_hit_or_miss()
    assert len(rows) == 4
    for r in rows:
        assert r["detections"] == r["expected"], r["corner"]


# --------------------------------------------------------------------------- #
# the structuring element
# --------------------------------------------------------------------------- #


def test_the_element_shape_decides_what_survives_a_diagonal():
    """The whole reason morphology has a shape parameter.

    A 9 px erosion annihilates diagonal lines with a rectangle or an ellipse
    (0.0003 of them left) and keeps 0.108 with a cross — 360 times more. Every
    element treats the axis-aligned rectangle identically, so the difference is
    entirely about orientation, not about strength.
    """
    rows = {r["element"]: r for r in mo.compare_elements()}
    assert rows["cross"]["diagonal"] > 100 * rows["rect"]["diagonal"]
    assert rows["cross"]["diagonal"] > 100 * rows["ellipse"]["diagonal"]
    # ... while all three do the same thing to an axis-aligned block
    axis = [r["axis_aligned"] for r in rows.values()]
    assert max(axis) - min(axis) < 0.01


def test_no_element_preserves_structures_thinner_than_itself():
    """A 9 px erosion of a 1-3 px line leaves nothing, whatever its shape."""
    for r in mo.compare_elements():
        assert r["thin"] == 0.0, r["element"]


# --------------------------------------------------------------------------- #
# the photographs -- Otsu's binarisation is the ground truth by construction
# --------------------------------------------------------------------------- #


def test_the_photo_truth_is_exact_and_the_noise_really_damages_it():
    truth, noisy = mo.photo_binary("parthenon_columns")
    assert set(np.unique(truth)) <= {0, 255}
    assert set(np.unique(noisy)) <= {0, 255}
    assert 0.85 < iou(noisy > 0, truth > 0) < 0.99


def test_nothing_beats_doing_nothing_on_a_binarised_photograph():
    """The project's most uncomfortable result, and it survives every kernel size.

    Otsu's binarisation of a real photograph has structure at *every* scale,
    including isolated single pixels that are genuinely part of the image. Any
    operation that removes salt-and-pepper noise removes those too, and on
    average the removal costs more than the noise did.

    This is the opposite of what the synthetic scene says, where opening and
    closing clearly help — because a scene made of rectangles and 5 px lines has
    no structure small enough to lose.
    """
    rows = mo.evaluate_photo_cleaning(images=mo.IMAGES[:6], sizes=(3, 5, 7))
    for size in (3, 5, 7):
        at_size = {r["operation"]: r["iou"] for r in rows if r["size"] == size}
        control = at_size["Do nothing (control)"]
        morphological = {k: v for k, v in at_size.items()
                         if k not in ("Do nothing (control)", "Median filter")}
        assert max(morphological.values()) < control, f"size {size}"


def test_the_synthetic_scene_says_the_opposite():
    """And that disagreement is the point of running both.

    On the generated scene an open-then-close clearly beats the noisy input. A
    project that only ran the synthetic experiment would have concluded that
    morphological denoising works.
    """
    rows = mo.evaluate_denoising(density=0.05, sizes=(3,))
    assert max(r["iou"] for r in rows) > 0.9


def test_opening_and_closing_are_the_only_operations_that_denoise_at_all():
    """Gradient, top-hat and black-hat are not denoisers and score like it.

    They are in the table because they are morphology, not because they are
    candidates. Reporting them beside opening without saying so would be a
    comparison that nobody could read.
    """
    rows = {r["operation"]: r["iou"]
            for r in mo.evaluate_photo_cleaning(images=mo.IMAGES[:4], sizes=(3,))}
    for name in ("Gradient", "Top-hat", "Black-hat"):
        assert rows[name] < 0.4, name
    for name in ("Open", "Close"):
        assert rows[name] > 0.8, name


# --------------------------------------------------------------------------- #
# skeletons
# --------------------------------------------------------------------------- #


def test_the_morphological_skeleton_fragments_the_shape():
    """The classic failure, measured rather than described.

    The open-and-subtract skeleton is not guaranteed to be connected, and it is
    not: it leaves 61 disconnected components where Zhang-Suen thinning leaves
    9. Zhang-Suen pays for that with a pure-Python iteration that is hundreds of
    times slower.
    """
    rows = {r["method"]: r for r in mo.evaluate_skeletons()}
    morph = next(v for k, v in rows.items() if k.startswith("Morph"))
    thin = next(v for k, v in rows.items() if k.startswith("Zhang"))

    assert morph["components"] > 3 * thin["components"]
    assert thin["median_ms"] > 50 * morph["median_ms"]


@pytest.mark.parametrize("shape", ["rect", "ellipse", "cross"])
def test_an_element_is_the_size_it_was_asked_for(shape):
    se = mo.element(shape, 7)
    assert se.shape == (7, 7)
    assert se.dtype == np.uint8
    assert se.max() == 1


def test_an_unknown_element_shape_raises():
    with pytest.raises(ValueError, match="unknown"):
        mo.element("hexagon", 5)
