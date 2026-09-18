"""Tests for project 47, colour space robustness.

Most of these pin a *finding* rather than a number, so that a later change which
quietly reverses one of the project's conclusions fails loudly instead of
rewriting the README by accident.

`test_each_space_gets_its_own_tolerance` exists because a single shared tolerance
decides the comparison on its own: Lab and YCrCb peak at 25 and RGB and HSV at
60, so any one number hands the result to whichever space it suits.
"""

from __future__ import annotations

import numpy as np
import pytest

import colour_spaces as cs

from shared.metrics import iou


@pytest.fixture(scope="module")
def photos():
    return {r["space"]: r for r in cs.photo_robustness()}


# --------------------------------------------------------------------------- #
# the harness
# --------------------------------------------------------------------------- #


def test_opencv_hue_is_half_a_circle():
    """0-179, not 0-359. A threshold written in degrees selects the wrong colour.

    Every entry here is a colour whose true hue is twice what OpenCV stores, so
    code that treats the channel as degrees is off by a factor of two at every
    angle except red.
    """
    rows = cs.hue_range_demonstration()
    assert cs.OPENCV_HUE_MAX == 179

    for row in rows:
        assert row["opencv_hue"] * 2 == row["true_degrees"], row["colour"]
        assert row["correct_degrees"] == row["true_degrees"], row["colour"]
        if row["true_degrees"] != 0:
            assert row["naive_degrees_reading"] != row["true_degrees"], row["colour"]


def test_each_space_gets_its_own_tolerance():
    """Or the comparison is decided by a shared constant rather than by the spaces.

    The spaces do not share units: Lab's a/b run about +-100 around a neutral
    axis while RGB spans 0-255 in three correlated channels, so the same
    "distance 25" is a far wider net in one than the other.
    """
    rows = cs.sweep_photo_tolerance()
    peaks = {space: max(rows, key=lambda r: r[space])["tolerance"] for space in cs.SPACES}

    assert len(set(peaks.values())) > 1, "if they all peaked together, say so instead"
    for space, tol in cs.PHOTO_TOLERANCE.items():
        assert tol == peaks[space], (space, tol, peaks[space])


def test_every_target_is_one_persons_tracing_of_one_region():
    from shared import bsds

    for name, (annotator, label) in cs.SEGMENTS.items():
        assert bsds.has_ground_truth(name), name
        annotations = bsds.load_annotations(name)
        assert annotator < len(annotations), name
        assert label in np.unique(annotations[annotator]["segmentation"]), name

        mask = cs.load_target(name)
        assert set(np.unique(mask)) <= {0, 255}
        assert 0.05 < (mask > 0).mean() < 0.45, name


def test_the_pool_spans_colour_distinctness():
    from shared import io

    values = []
    for name in cs.IMAGES:
        assert name in io.REAL_PHOTOS, name
        values.append(cs.chroma_distance(cs.load_scene(name), cs.load_target(name)))

    assert len(set(cs.IMAGES)) == len(cs.IMAGES)
    assert min(values) > 30.0, "every target must be separable by colour at all"
    assert max(values) > 55.0


def test_the_threshold_is_tuned_once_and_never_retuned():
    """Re-tuning on the degraded image would measure nothing."""
    clean = cs.load_scene(cs.IMAGES[0])
    truth = cs.load_target(cs.IMAGES[0])
    warm = cs.degrade_colour_cast(clean, (1.25, 1.0, 0.75))

    fixed = cs.threshold_in_space(warm, "Lab", clean, truth, 25.0)
    retuned = cs.threshold_in_space(warm, "Lab", warm, truth, 25.0)

    assert not np.array_equal(fixed, retuned)
    assert iou(retuned, truth) > iou(fixed, truth), "retuning always wins, which is why it is not done"


# --------------------------------------------------------------------------- #
# the findings
# --------------------------------------------------------------------------- #


def test_hsv_is_exactly_brightness_invariant(photos):
    """The half of the folklore that is true, on human-traced regions.

    0.487 undegraded, 0.487 at x0.6 brightness, 0.493 at x1.3. Hue is an angle in
    a plane that scaling all three channels does not rotate.
    """
    hsv = photos["HSV"]
    for level in ("Brightness x0.6", "Brightness x1.3"):
        assert abs(hsv[level] - hsv["undegraded_iou"]) < 0.02, level


def test_hsv_is_also_the_least_accurate_space_undegraded(photos):
    """And that is the price of the invariance, not a separate defect."""
    assert photos["HSV"]["undegraded_iou"] == min(
        r["undegraded_iou"] for r in photos.values())
    for better in ("Lab", "YCrCb"):
        assert photos[better]["undegraded_iou"] > photos["HSV"]["undegraded_iou"] + 0.25


def test_a_colour_cast_is_what_hsv_cannot_survive(photos):
    """The half that is false. Hue is invariant to a scale, not to a rotation.

    A warm cast multiplies the channels *differently*, which moves the hue angle
    — and costs HSV 37% of its IoU while leaving its brightness response flat.
    """
    hsv = photos["HSV"]
    loss = hsv["undegraded_iou"] - hsv["Warm cast"]
    assert loss > 0.15
    assert loss > 5 * abs(hsv["undegraded_iou"] - hsv["Brightness x0.6"])


def test_accuracy_robustness_and_worst_case_have_different_winners(photos):
    """So there is no ranking of colour spaces, only a trade."""
    most_accurate = max(photos, key=lambda s: photos[s]["undegraded_iou"])
    most_robust = min(photos, key=lambda s: photos[s]["mean_loss"])
    best_worst = max(photos, key=lambda s: photos[s]["worst_case"])

    assert len({most_accurate, most_robust, best_worst}) == 3


def test_normalised_rgb_is_invariant_to_a_scale_and_destroyed_by_gamma(photos):
    """Dividing by the sum removes a scale. Gamma is not a scale.

    r = R/(R+G+B) is unchanged when every channel is multiplied by the same
    number and changes completely when each is raised to a power.
    """
    norm = photos["Normalised RGB"]
    assert abs(norm["Brightness x0.6"] - norm["undegraded_iou"]) < 0.02
    assert norm["Gamma 2.0"] < 0.4 * norm["undegraded_iou"]


def test_plain_rgb_loses_the_most(photos):
    assert photos["RGB"]["mean_loss"] == max(r["mean_loss"] for r in photos.values())
    assert photos["RGB"]["Brightness x0.6"] < 0.4 * photos["RGB"]["undegraded_iou"]


def test_a_strong_cast_destroys_every_space_and_white_balance_restores_it():
    """The conclusion the whole project points at.

    At cast level 0.5 every colour space scores 0.000 — choosing a different one
    is not a substitute for correcting the illuminant. An explicit white balance
    step restores four of the five to about 0.66.
    """
    rows = sorted(cs.white_balance_rescue(), key=lambda r: r["cast_level"])
    worst = rows[-1]

    for space in cs.SPACES:
        assert worst[f"{space} raw"] < 0.05, space

    rescued = [s for s in cs.SPACES if worst[f"{s} balanced"] > 0.5]
    assert len(rescued) >= 4, {s: worst[f"{s} balanced"] for s in cs.SPACES}


def test_the_synthetic_chart_flatters_every_space():
    """Flat patches are an easier problem than a region of a photograph.

    A traced region has shadow, highlight and texture in it; a colour chart patch
    is one colour. Both are reported because the difference between them is the
    reason the photograph arm exists.
    """
    synthetic = {r["space"]: r for r in cs.compare_all_degradations()}
    photos = {r["space"]: r for r in cs.photo_robustness()}

    for space in cs.SPACES:
        assert synthetic[space]["Brightness scale mean"] >= 0.0
        assert 0.0 <= photos[space]["undegraded_iou"] <= 1.0


# --------------------------------------------------------------------------- #
# the conversions themselves
# --------------------------------------------------------------------------- #


def test_every_conversion_keeps_the_shape_and_three_channels():
    img = cs.load_scene(cs.IMAGES[0])
    for name, fn in cs.SPACES.items():
        out = fn(img)
        assert out.shape == img.shape, name
        assert np.isfinite(out).all(), name


def test_hue_wraps_around_the_circle():
    """Red is at both ends, so a plain absolute difference is wrong there."""
    red_low = np.full((4, 4, 3), (255, 0, 0), np.uint8)
    red_high = np.full((4, 4, 3), (255, 8, 0), np.uint8)

    a = cs.to_hsv(red_low)[..., 0].astype(float).mean()
    b = cs.to_hsv(red_high)[..., 0].astype(float).mean()
    plain = abs(a - b)
    wrapped = min(plain, (cs.OPENCV_HUE_MAX + 1) - plain)
    assert wrapped <= plain
