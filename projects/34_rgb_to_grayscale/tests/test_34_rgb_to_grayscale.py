"""Tests for project 34, RGB to grayscale.

Most of these pin a *finding* rather than a number, so that a later change which
quietly reverses one of the project's conclusions fails loudly instead of
rewriting the README by accident.

`test_the_isoluminant_colour_stays_in_gamut` exists because the first version of
this project solved for one channel algebraically, got a negative red for BT.709,
and clipped it — which silently destroyed the isoluminance the scene was built
to have and made BT.709 score 29.7 grey levels against a scene it was supposed
to be blind to.
"""

from __future__ import annotations

import numpy as np
import pytest

import grayscale as gs

LEVEL = 1.0 / 255.0


# --------------------------------------------------------------------------- #
# the adversarial scene
# --------------------------------------------------------------------------- #


def test_the_isoluminant_colour_stays_in_gamut():
    """Clipping a colour changes its luma, which is the one thing that must not happen."""
    bg = np.array([0.85, 0.20, 0.20], np.float32)
    for name, weights in gs.LINEAR_WEIGHTINGS.items():
        target = float(bg @ np.array(weights, np.float32))
        fg = gs.matched_luma_colour(bg, target, weights)

        assert fg.min() >= 0.0 and fg.max() <= 1.0, name
        assert float(fg @ np.array(weights, np.float32)) == pytest.approx(
            target, abs=LEVEL), name
        # and it has to be a genuinely different colour, or the scene is blank
        assert np.linalg.norm(fg - bg) > 0.5, name


def test_each_scene_is_invisible_to_the_weighting_it_was_built_against():
    """Under a fraction of a grey level — which is the definition, not a result."""
    for built_for, weights in gs.LINEAR_WEIGHTINGS.items():
        img, mask = gs.isoluminant_scene(weights=weights)
        separation = gs.region_separation(gs.METHODS[built_for](img), mask)
        assert separation < 1.0, f"{built_for} sees {separation:.2f} levels of its own plane"


def test_the_luma_offset_does_what_it_says():
    for offset in (0.0, 4.0, 16.0):
        img, mask = gs.isoluminant_scene(luma_offset=offset)
        separation = gs.region_separation(gs.gray_bt601(img), mask)
        assert separation == pytest.approx(offset, abs=1.0), offset


# --------------------------------------------------------------------------- #
# the findings
# --------------------------------------------------------------------------- #


def test_every_fixed_weighting_is_blind_somewhere_including_the_correct_one():
    """The project's headline.

    Weights are a projection onto a line, so each set of them has a whole plane
    of colours it maps to a single grey. Choosing "better" weights moves the
    blind plane; it does not remove it. BT.709 is the modern, correct choice and
    it is exactly as blind as BT.601 — just about different colours.
    """
    rows = {r["built_against"]: r for r in gs.blind_spot_matrix()}
    assert set(rows) == set(gs.LINEAR_WEIGHTINGS)

    for built_for, row in rows.items():
        assert row[built_for] < 1.0, built_for
        others = [row[m] for m in gs.METHODS if m != built_for]
        assert min(others) > 5.0, f"nothing else could see {built_for}'s blind plane"


def test_only_the_per_image_method_escapes_every_blind_plane():
    """Because it is the only one allowed to look at the image before choosing."""
    for row in gs.blind_spot_matrix():
        assert row["Contrast-preserving"] > 100.0, row["built_against"]


def test_on_photographs_the_conversion_choice_is_irrelevant():
    """The negative result, and it is the useful half of this project.

    Across twelve photographs spanning 5.8 to 80.3 mean chroma, the projection
    keeps almost all of the colour contrast at almost every strong colour edge.
    The catastrophe above is real and essentially absent from natural images.
    """
    rows = gs.luma_retention()
    assert len(rows) == len(gs.IMAGES)

    assert min(r["worst_retained"] for r in rows) > 0.7
    assert min(r["minimum_retained"] for r in rows) > 0.4
    assert all(0.95 < r["median_retained"] < 1.1 for r in rows)

    # the one photograph that comes closest to the synthetic case is the one
    # with an iridescent blue-green subject on green
    worst = min(rows, key=lambda r: r["minimum_retained"])
    assert worst["image"] == "damselfly_on_leaf"


def test_the_whole_weights_argument_is_worth_a_couple_of_grey_levels():
    rows = {r["method"]: r for r in gs.evaluate_on_photos(runs=1)}
    assert rows["BT.709 (HDTV)"]["mean_diff_vs_bt601"] < 4.0
    assert rows["Value (max channel)"]["mean_diff_vs_bt601"] > 3 * \
        rows["BT.709 (HDTV)"]["mean_diff_vs_bt601"]

    # and every one of them retains essentially all the structure
    for name, row in rows.items():
        assert row["contrast_retained"] > 0.98, name
        assert row["worst_image_recall"] > 0.99, name


def test_the_per_image_method_costs_two_orders_of_magnitude_for_nothing_here():
    rows = {r["method"]: r for r in gs.evaluate_on_photos(runs=3)}
    fast = rows["BT.601 (OpenCV default)"]["median_ms"]
    slow = rows["Contrast-preserving"]["median_ms"]
    assert slow > 50 * fast
    # and it buys no measurable advantage on a photograph
    assert rows["Contrast-preserving"]["contrast_retained"] - \
        rows["BT.601 (OpenCV default)"]["contrast_retained"] < 0.01


def test_canny_and_otsu_disagree_about_how_much_separation_is_enough():
    """Two stages reading the same grayscale, differing by about 7x.

    Otsu compares two populations of pixels and recovers the shape once the
    boundary is 4 grey levels; Canny thresholds a *gradient* and needs the step
    to clear its hysteresis high threshold, which on this edge takes 28. A
    conversion good enough for one is not automatically good enough for the
    other, and neither number is a property of the conversion alone.
    """
    rows = sorted(gs.sweep_isoluminance(seeds=(0,)), key=lambda r: r["luma_offset"])
    key = "BT.601 (OpenCV default)"

    otsu_needs = next(r["luma_offset"] for r in rows if r[f"{key}__otsu_iou"] >= 0.98)
    canny_needs = next(r["luma_offset"] for r in rows if r[key] >= 0.99)

    assert otsu_needs <= 8.0
    assert canny_needs >= 4 * otsu_needs
    # at the isoluminant end both fail
    assert rows[0][key] == 0.0
    assert rows[0][f"{key}__otsu_iou"] < 0.5


def test_recovering_the_boundary_is_not_the_same_as_the_next_stage_finding_it():
    """"Good enough" is a property of the pair, not of the conversion.

    On the scene BT.601 is blind to, four of the six conversions recover the
    boundary — 9.5 to 37.9 grey levels of it. Otsu then finds the shape in five
    of the six. Canny finds it in **two**: its hysteresis high threshold needs
    around 28 levels on this edge, so BT.709's 18.7 is a real recovery that the
    next stage still throws away.
    """
    seen = {r["method"]: r["region_separation"] for r in gs.evaluate_isoluminant()}
    rows = {r["method"]: r for r in gs.downstream_effect()}

    blind = "BT.601 (OpenCV default)"
    assert seen[blind] < 1.0
    assert rows[blind]["canny_edge_recall"] == 0.0
    assert rows[blind]["otsu_iou"] < 0.5

    recovered = [m for m in gs.METHODS if m != blind and seen[m] > 5.0]
    assert len(recovered) >= 4

    found_by_otsu = [m for m in recovered if rows[m]["otsu_iou"] > 0.9]
    found_by_canny = [m for m in recovered if rows[m]["canny_edge_recall"] > 0.9]
    assert len(found_by_otsu) == len(recovered), "Otsu finds every recovery"
    assert len(found_by_canny) < len(found_by_otsu), "Canny does not"

    # and the ones Canny finds are exactly the ones clearing its threshold
    for name in recovered:
        assert (rows[name]["canny_edge_recall"] > 0.9) == (seen[name] > 28.0), name


# --------------------------------------------------------------------------- #
# the conversions themselves
# --------------------------------------------------------------------------- #


def test_every_conversion_returns_a_single_channel_image_of_the_same_size():
    img = gs.load_scene(gs.IMAGES[0])
    for name, fn in gs.METHODS.items():
        out = fn(img)
        assert out.shape == img.shape[:2], name
        assert out.dtype == np.uint8, name


def test_the_weights_sum_to_one_so_grey_maps_to_itself():
    for name, weights in (("BT.601", gs.BT601), ("BT.709", gs.BT709)):
        assert sum(weights) == pytest.approx(1.0, abs=1e-4), name

    grey = np.full((8, 8, 3), 137, np.uint8)
    for name in ("Average (R+G+B)/3", "BT.601 (OpenCV default)", "BT.709 (HDTV)",
                 "Value (max channel)"):
        assert abs(int(gs.METHODS[name](grey).mean()) - 137) <= 1, name


def test_value_is_not_a_luminance_at_all():
    """max(R, G, B) ignores two thirds of every pixel, and it shows.

    On a saturated photograph it lands 15.9 grey levels from BT.601 on average —
    seven times further than the BT.601-to-BT.709 difference that gets argued
    about.
    """
    blue = np.zeros((8, 8, 3), np.uint8)
    blue[..., 2] = 255
    assert int(gs.gray_value(blue).mean()) == 255
    assert int(gs.gray_bt601(blue).mean()) < 40


def test_every_photograph_exists_and_the_pool_spans_the_axis():
    from shared import io

    chromas = []
    for name in gs.IMAGES:
        assert name in io.REAL_PHOTOS, name
        chromas.append(gs.chroma(gs.load_scene(name)))

    assert len(set(gs.IMAGES)) == len(gs.IMAGES)
    assert min(chromas) < 10.0, "no near-monochrome control in the pool"
    assert max(chromas) > 60.0, "nothing saturated enough to stress the conversion"
