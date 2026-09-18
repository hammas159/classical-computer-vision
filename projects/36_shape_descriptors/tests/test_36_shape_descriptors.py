"""Tests for project 36, the shape descriptors.

Most of these pin a *finding* rather than a number, so that a later change which
quietly reverses one of the project's conclusions fails loudly instead of
rewriting the README by accident.

`test_the_textbook_log_hu_recipe_has_a_hole_at_exactly_zero` is the one worth
reading first. It pins a defect in a formula that appears in every tutorial, and
it is the reason this project reports two Hu rows instead of one.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import shapes_desc as sd

CHANCE = 1.0 / len(sd.REAL_IMAGES)


@pytest.fixture(scope="module")
def human_spread():
    """How far apart two people are, per descriptor — the scale for everything else."""
    return {r["descriptor"]: r["mean_relative_change"] for r in sd.annotator_variation()}


@pytest.fixture(scope="module")
def real_rotation():
    return {r["descriptor"]: r["mean_relative_change"]
            for r in sd.real_invariance_table("rotation", sd.ROTATIONS)}


# --------------------------------------------------------------------------- #
# the log-Hu defect
# --------------------------------------------------------------------------- #


def test_the_textbook_log_hu_recipe_has_a_hole_at_exactly_zero():
    """`sign(h) * log10(|h| + eps)` is wrong, and symmetric shapes fall in.

    A drawn circle's 5th to 7th Hu moments are **exactly 0.0**. `np.sign(0)` is
    `0`, so those components come out as 0 instead of at the floor -12. Rotate
    the circle by any angle at all and they become 4.7e-62 — still numerically
    zero — but now `sign` is 1 and the component jumps to -12.

    The descriptor moves 12 units because a quantity went from zero to 1e-62.
    """
    circle = sd.make_shape("circle")
    turned = sd.make_shape("circle", rotation=30.0)

    raw = cv2.HuMoments(cv2.moments(circle, binaryImage=True)).ravel()
    assert (raw[4:] == 0.0).all(), "the premise: a circle's high Hu moments are exactly zero"

    textbook = np.abs(sd.desc_hu_textbook(circle) - sd.desc_hu_textbook(turned))
    assert textbook[4:].max() > 10.0

    # flooring the magnitude leaves the affected components untouched; the first
    # component still moves a little, because h1 genuinely does change when a
    # shape is resampled onto a rotated pixel grid.
    fixed = np.abs(sd.desc_hu(circle) - sd.desc_hu(turned))
    assert fixed[4:].max() == 0.0
    assert fixed.max() < 1e-4


def test_a_sign_flip_below_the_epsilon_doubles_the_damage():
    """The star's h5-h7 sit at 1e-16. Rotation flips their sign, not their size."""
    star, turned = sd.make_shape("star"), sd.make_shape("star", rotation=30.0)
    raw_a = cv2.HuMoments(cv2.moments(star, binaryImage=True)).ravel()
    raw_b = cv2.HuMoments(cv2.moments(turned, binaryImage=True)).ravel()

    tiny = np.abs(raw_a) < sd.EPS
    assert tiny.sum() >= 2, "h5 and h7 are at 1e-16, numerically zero"
    assert (np.sign(raw_a[tiny]) != np.sign(raw_b[tiny])).any()

    textbook = np.abs(sd.desc_hu_textbook(star) - sd.desc_hu_textbook(turned))
    assert textbook[tiny].max() > 20.0, "sign flip across the floor swings it by 2 x 12"
    assert np.abs(sd.desc_hu(star) - sd.desc_hu(turned))[tiny].max() == 0.0


def test_fixing_the_zero_makes_hu_moments_look_invariant_again():
    """34x on rotation, 14x on scale — and none of it was Hu's fault."""
    for transform, levels, factor in (("rotation", sd.ROTATIONS, 20.0),
                                      ("scale", sd.SCALES, 8.0)):
        rows = {r["descriptor"]: r["mean_relative_change"]
                for r in sd.invariance_table(transform, levels)}
        assert rows["Hu moments (textbook log)"] > factor * rows["Hu moments (log)"], transform


def test_the_two_hu_variants_agree_on_shapes_people_traced(real_rotation):
    """Because a traced outline almost never lands on the floor.

    A collie's smallest Hu moment is 1.5e-9, a thousand times the epsilon. The
    defect above is invisible on real data, which is exactly why it survives in
    tutorials.

    One of the twelve is an exception and it is the near-circular one: the
    basket of grain has h5 = 3.4e-14, below the floor. The textbook recipe
    survives it only by luck — the moment stays on the same side of zero at every
    angle, so `sign` never flips.
    """
    assert real_rotation["Hu moments (log)"] == pytest.approx(
        real_rotation["Hu moments (textbook log)"], abs=0.001)

    below_floor = []
    for name in sd.REAL_IMAGES:
        hu = cv2.HuMoments(cv2.moments(sd.load_silhouette(name), binaryImage=True)).ravel()
        if np.abs(hu).min() < sd.EPS:
            below_floor.append(name)
    assert below_floor == ["basket_of_grain"], below_floor


# --------------------------------------------------------------------------- #
# the silhouettes
# --------------------------------------------------------------------------- #


def test_every_silhouette_is_one_person_s_tracing_of_one_region():
    from shared import bsds

    for name, (annotator, label) in sd.SILHOUETTES.items():
        assert bsds.has_ground_truth(name), name
        annotations = bsds.load_annotations(name)
        assert annotator < len(annotations), name
        assert label in np.unique(annotations[annotator]["segmentation"]), name

        mask = sd.load_silhouette(name)
        assert set(np.unique(mask)) <= {0, 255}
        assert 0.05 < (mask > 0).mean() < 0.6, name


def test_at_least_two_people_traced_every_object():
    """Or there is no human spread to compare the invariances against."""
    for name in sd.REAL_IMAGES:
        assert len(sd.silhouettes_by_annotator(name)) >= 2, name


def test_transforming_a_silhouette_does_not_clip_it():
    """A traced shape is not centred in its frame; rotating in place would cut it."""
    mask = sd.load_silhouette("collie_standing")
    area = (mask > 0).sum()
    for angle in (0.0, 30.0, 45.0, 90.0, 180.0):
        moved = sd.transform_mask(mask, rotation=angle)
        assert (moved > 0).sum() == pytest.approx(area, rel=0.02), angle
        assert moved[0].sum() == 0 and moved[-1].sum() == 0, angle


# --------------------------------------------------------------------------- #
# the findings
# --------------------------------------------------------------------------- #


def test_an_invariance_is_only_meaningful_against_how_well_the_shape_is_defined(
        human_spread, real_rotation):
    """The project's headline.

    Two people tracing the same object move Hu moments by 0.753. A 180 degree
    rotation moves them by 0.001. Quoting that invariance is quoting a shape to
    a thousand times the precision the people who drew it agreed on.

    The chain code histogram is the other end: its rotation error is three times
    the human spread, which makes it a real failure rather than arithmetic.
    """
    assert real_rotation["Hu moments (log)"] < 0.01 * human_spread["Hu moments (log)"]
    assert real_rotation["Fourier descriptors"] < human_spread["Fourier descriptors"]

    for name in ("Chain code histogram", "Simple geometry"):
        assert real_rotation[name] > human_spread[name], name


def test_the_descriptor_that_wins_upright_is_the_one_that_cannot_survive_a_turn():
    """An invariance is a trade, not a quality.

    The chain code histogram keeps the *directions* of the boundary steps, which
    is a strong signature of a particular outline and is destroyed by rotating
    it. Best upright at 0.750, barely above chance at 0.114 once the shape is
    turned 30 degrees.
    """
    upright = {r["descriptor"]: r["accuracy"] for r in sd.cross_annotator_classification()}
    turned = {r["descriptor"]: r["accuracy"]
              for r in sd.cross_annotator_classification(rotation=30.0)}

    assert max(upright, key=upright.get) == "Chain code histogram"
    assert turned["Chain code histogram"] < 0.2
    assert turned["Chain code histogram"] < 0.25 * upright["Chain code histogram"]

    assert max(turned, key=turned.get).startswith("Hu moments")
    assert turned["Hu moments (log)"] > 0.9 * upright["Hu moments (log)"]


def test_recognising_another_person_s_tracing_is_a_genuinely_hard_task():
    """And classifying a shape against transformed copies of itself is not.

    Every descriptor here scores 0.97 or better when the probes are rotated
    copies of the gallery shapes, because a rotated copy of a silhouette is
    still that silhouette. Across annotators nothing reaches 0.76.
    """
    easy = {r["descriptor"]: r["accuracy"] for r in sd.real_classification_accuracy()}
    hard = {r["descriptor"]: r["accuracy"] for r in sd.cross_annotator_classification()}

    assert min(easy.values()) > 0.95
    assert max(hard.values()) < 0.8
    assert min(hard.values()) > 3 * CHANCE, "and it is still well above chance"


def test_only_the_seventh_hu_moment_can_tell_a_shape_from_its_mirror():
    """Every other descriptor here is exactly blind to a reflection."""
    rows = sd.real_reflection_test()
    flipped = [r for r in rows if r["hu7_sign_flipped"]]
    assert len(flipped) >= 10

    for r in rows:
        assert r["other_hu_flipped"] == 0, r["image"]
        assert r["geometry_change"] == 0.0, r["image"]
        assert r["fourier_change"] < 0.05, r["image"]


def test_translation_invariance_is_the_one_claim_that_is_exactly_true():
    for rows in (sd.invariance_table("translation", sd.TRANSLATIONS),
                 sd.real_invariance_table("translation", sd.TRANSLATIONS)):
        for r in rows:
            assert r["max_relative_change"] == 0.0, r["descriptor"]


def test_a_bigger_raster_does_not_rescue_the_log_hu_instability():
    """The comfortable explanation, checked and found false.

    "It is only discretisation, use a bigger image" predicts a decreasing curve.
    The measured one bounces — 0.236 at 64 px, 0.042 at 96, 0.391 at 128 — and a
    drawn ellipse gets seventeen times *worse* from 64 px to 256. A finer raster
    computes a symmetric shape's near-zero moments more accurately, which pushes
    them closer to the floor where the log is least stable.

    What does help is fixing the floor: the textbook variant is worse at every
    single raster size.
    """
    rows = sorted(sd.discretisation_error(), key=lambda r: r["raster_size"])
    fixed = [r["Hu moments (log)"] for r in rows]

    assert fixed != sorted(fixed, reverse=True), "it is not a decreasing curve"
    assert max(fixed[1:]) > fixed[0], "and the largest rasters are not the best"

    for r in rows:
        assert r["Hu moments (textbook log)"] > r["Hu moments (log)"], r["raster_size"]

    ellipse = [sd.relative_change(sd.desc_hu(sd.make_shape("ellipse", size=n)),
                                  sd.desc_hu(sd.make_shape("ellipse", size=n, rotation=30.0)))
               for n in (64, 256)]
    assert ellipse[1] > 10 * ellipse[0]


def test_every_descriptor_returns_a_finite_fixed_length_vector():
    masks = [sd.load_silhouette(n) for n in sd.REAL_IMAGES[:4]]
    for name, fn in sd.DESCRIPTORS.items():
        vectors = [fn(m) for m in masks]
        assert len({v.shape for v in vectors}) == 1, name
        assert all(np.isfinite(v).all() for v in vectors), name


def test_flooring_the_magnitude_also_floors_the_reflection_signal():
    """The fix is a trade, and one of the twelve silhouettes pays for it.

    `desc_hu` maps everything below EPS to one value regardless of sign — and the
    sign of h7 is the only reflection detector here. The basket of grain has
    h7 = +3.7e-13 upright and exactly -3.7e-13 mirrored: structured, since it
    negates to four significant figures, but below the 1e-12 floor. The floored
    variant therefore reports **no** change under reflection where the textbook
    one reports 1.151.

    For a nearly mirror-symmetric shape, rotation stability and reflection
    sensitivity pull against each other and no single epsilon resolves both.
    """
    base = sd.transform_mask(sd.load_silhouette("basket_of_grain"))
    mirrored = cv2.flip(base, 1)

    raw = cv2.HuMoments(cv2.moments(base, binaryImage=True)).ravel()
    mirror_raw = cv2.HuMoments(cv2.moments(mirrored, binaryImage=True)).ravel()
    assert abs(raw[6]) < sd.EPS
    assert raw[6] == pytest.approx(-mirror_raw[6], rel=1e-3), "structured, not noise"

    assert sd.relative_change(sd.desc_hu(base), sd.desc_hu(mirrored)) < 1e-9
    assert sd.relative_change(sd.desc_hu_textbook(base),
                              sd.desc_hu_textbook(mirrored)) > 1.0

    # and it is the only one of the twelve where the two variants disagree here
    disagreeing = []
    for name in sd.REAL_IMAGES:
        b = sd.transform_mask(sd.load_silhouette(name))
        m = cv2.flip(b, 1)
        fixed = sd.relative_change(sd.desc_hu(b), sd.desc_hu(m))
        textbook = sd.relative_change(sd.desc_hu_textbook(b), sd.desc_hu_textbook(m))
        if abs(textbook - fixed) > 0.01:
            disagreeing.append(name)
    assert disagreeing == ["basket_of_grain"], disagreeing
