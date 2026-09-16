"""Tests for project 09, coin counting and measurement.

Most of these pin a *finding* rather than a number, so that a later "improvement"
which quietly reverses one of the project's conclusions fails loudly instead of
rewriting the README by accident.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import coins as cn
from shared import io, synth


# --------------------------------------------------------------------------- #
# the mask
# --------------------------------------------------------------------------- #


def test_plain_otsu_classifies_background_as_foreground():
    """The bug the top-hat exists to fix, asserted rather than described.

    The plate is lit unevenly: the background at the top of the frame is brighter
    than Otsu's global threshold, so a band of empty table joins the top row of
    coins into one enormous component.
    """
    img = io.sample("coins")
    plain = cn._foreground_mask(img, flatten=False)
    n, _, stats, _ = cv2.connectedComponentsWithStats(plain, 8)
    biggest = int(stats[1:, cv2.CC_STAT_AREA].max())
    # a single coin is about 1000-2500 px; anything near 10k is the merged band
    assert biggest > 8000


def test_flattening_the_illumination_breaks_that_band_up():
    img = io.sample("coins")
    flat = cn._foreground_mask(img, flatten=True)
    n, _, stats, _ = cv2.connectedComponentsWithStats(flat, 8)
    assert int(stats[1:, cv2.CC_STAT_AREA].max()) < 8000


def test_the_background_kernel_must_exceed_the_largest_coin():
    """A top-hat removes whatever the structuring element can contain.

    With a kernel smaller than a coin, the opening keeps the coin and the top-hat
    subtracts it away, leaving only its rim. The failure is silent — you still
    get a mask, it is just made of rings — so it is asserted through the thing
    that actually breaks: the count.
    """
    img = io.sample("coins")
    gray = cv2.GaussianBlur(io.to_gray(img), (5, 5), 0)
    _, bad = cv2.threshold(
        cn._flatten_illumination(gray, kernel=15), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    bad_mask = cn._clean_binary(bad)
    good_mask = cn._foreground_mask(img, flatten=True)

    # measured: foreground area jumps from ~0.21 to ~0.32 of the frame as the
    # kernel crosses the largest coin, because below it the top-hat is
    # subtracting the coins' own interiors and leaving only their rims
    assert float((good_mask > 0).mean()) > 1.4 * float((bad_mask > 0).mean())


def test_filling_holes_leaves_a_solid_region():
    """A 6 px rim of a radius-30 circle has ~1131 px; the filled disc has ~2827."""
    ring = np.zeros((80, 80), np.uint8)
    cv2.circle(ring, (40, 40), 30, 255, 6)
    filled = cn._fill_holes(ring)
    assert filled[40, 40] == 255
    assert filled.sum() > 2.0 * ring.sum()
    assert int((filled > 0).sum()) == pytest.approx(np.pi * 33**2, rel=0.15)


# --------------------------------------------------------------------------- #
# counting
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", list(cn.METHODS))
def test_every_method_returns_an_int_label_image(name):
    labels = cn.METHODS[name](io.sample("coins"))
    assert labels.dtype in (np.int32, np.int64)
    assert labels.shape == io.sample("coins").shape[:2]
    assert labels.min() == 0


def test_local_maxima_watershed_counts_every_coin():
    labels = cn.segment_watershed(io.sample("coins"))
    assert cn.count_coins(labels) == cn.TRUE_COIN_COUNT


def test_hough_counts_every_coin():
    assert cn.count_coins(cn.segment_hough_circles(io.sample("coins"))) == cn.TRUE_COIN_COUNT


def test_connected_components_must_undercount():
    """The naive baseline's failure is structural, not a tuning problem.

    Two touching coins are one connected component. No threshold fixes that,
    which is the entire reason watershed exists.
    """
    n = cn.count_coins(cn.segment_otsu_components(io.sample("coins")))
    assert n < cn.TRUE_COIN_COUNT - 2


def test_the_area_floor_must_not_discard_real_regions():
    """Pins the bug where a round-number cutoff threw away two correct basins.

    Correctly-seeded watershed produces exactly one basin per coin. A `min_area`
    of 250 px discarded two of them, turning 24/24 into 22/24 *after* the
    segmentation had already got the answer right.
    """
    labels = cn.segment_watershed(io.sample("coins"))
    assert int(labels.max()) == cn.TRUE_COIN_COUNT
    assert len(cn.region_properties(labels, min_area=cn.MIN_COIN_AREA_PX)) == cn.TRUE_COIN_COUNT
    assert len(cn.region_properties(labels, min_area=250)) < cn.TRUE_COIN_COUNT


# --------------------------------------------------------------------------- #
# the findings
# --------------------------------------------------------------------------- #


def test_local_maxima_seeding_survives_a_broken_mask_and_the_global_rule_does_not():
    """The project's headline, pinned.

    A two-factor ablation with a lopsided result: a lighting artefact that
    merges the mask reduces the tutorial's seeding rule to **one** object, and
    leaves local-maxima seeding at the correct count.

    This test previously also asserted that flattening the mask *fails* to fully
    rescue the tutorial rule — it reached 23 of 24. That turned out to be an
    artefact of a hard-coded top-hat kernel of 61 px, which is not comfortably
    larger than the coins in this image. With the kernel sized from the image
    (:func:`coins.background_kernel_for`) the tutorial rule reaches 24 as well.

    The finding is sharper for it. The tutorial's rule is not *wrong*; it is
    **entirely dependent on the mask being clean**, and that dependency is
    invisible until the mask is not. Local-maxima seeding does not have it.
    """
    rows = cn.ablate_mask_and_seeding()
    tutorial = next(r for r in rows if r["seeding"].startswith("Global"))
    local = next(r for r in rows if r["seeding"].startswith("Local"))

    # with a broken mask the two rules disagree completely
    assert tutorial["plain_otsu"] <= 2
    assert local["plain_otsu"] == cn.TRUE_COIN_COUNT
    # with a good mask they agree, which is the point: the difference between
    # them is robustness, not accuracy
    assert tutorial["tophat_otsu"] == cn.TRUE_COIN_COUNT
    assert local["tophat_otsu"] == cn.TRUE_COIN_COUNT


def test_the_tophat_kernel_must_exceed_the_largest_object():
    """The silent failure that made every method undercount by five.

    `_flatten_illumination` subtracts an opening, and only erases the
    illumination if the structuring element is larger than the coins. Below
    that it subtracts the coin's own interior and leaves a ring, which the
    cleanup then discards — so the count is simply wrong and nothing raises.

    Pinned because the requirement lived in a docstring and a constant sized for
    one particular image, and the first scene with larger coins lost five of
    twenty on a frame where they were not even touching.
    """
    scene, truth = synth.coin_scene(seed=0, mm_per_px=0.42)
    gray = io.to_gray(scene)
    largest_px = max(c["diameter_px"] for c in truth)

    # the kernel the image asks for clears the biggest coin
    assert cn.background_kernel_for(gray) > largest_px
    # and with it, the coins all survive
    assert cn.count_coins(cn.segment_watershed(scene)) == len(truth)

    # the old fixed 61 is smaller than these coins, so the top-hat eats them:
    # the flattened image keeps far less of the coin than a correct kernel does
    kept_correct = float(cn._flatten_illumination(gray).mean())
    kept_starved = float(cn._flatten_illumination(gray, kernel=61).mean())
    assert kept_starved < 0.75 * kept_correct


def test_the_tutorials_knob_has_no_safe_setting():
    """Pins that `fg_ratio` is not tunable, only trade-able."""
    rows = cn.sweep_watershed_seed()
    counts = [r["global_seed_count"] for r in rows]
    assert max(counts) - min(counts) > 10  # wildly sensitive
    assert counts == sorted(counts, reverse=True)  # monotonically collapsing
    # the local-maxima count does not depend on it at all
    assert len({r["local_maxima_count"] for r in rows}) == 1


def test_counting_correctly_does_not_mean_measuring_correctly():
    """The project's second finding, pinned.

    Several methods find exactly 24 objects. Fewer of them produce 24 regions
    that could be coins — a watershed boundary can squeeze a basin to a fraction
    of its coin without changing the count.
    """
    rows = cn.evaluate_methods(runs=1)
    exact = [r for r in rows if r["count_error"] == 0]
    assert len(exact) >= 3
    plausible = [r for r in exact if r["implausible"] == 0]
    assert len(plausible) < len(exact), "if every exact method also measures, the finding is gone"
    assert plausible[0]["method"] == "Hough circles"


def test_the_shape_prior_measures_best_and_a_region_method_does_not():
    rows = cn.evaluate_methods(runs=1)
    hough = next(r for r in rows if r["method"] == "Hough circles")
    water = next(r for r in rows if r["method"] == "Watershed (local maxima)")
    assert hough["count"] == water["count"] == cn.TRUE_COIN_COUNT
    assert hough["diameter_cv"] < water["diameter_cv"]
    assert hough["min_diameter_mm"] > 2.0 * water["min_diameter_mm"]


def test_calibration_error_propagates_exactly_one_to_one():
    """A single multiplication cannot attenuate an error — asserted, not assumed."""
    rows = cn.calibration_sensitivity()
    for r in rows:
        assert r["measured_error_pct"] == pytest.approx(r["reference_error_pct"], abs=0.02)


def test_the_reference_is_the_largest_object_by_construction():
    labels = cn.segment_watershed(io.sample("coins"))
    props = cn.region_properties(labels)
    scale = cn.calibrate_mm_per_px(props, cn.REFERENCE_DIAMETER_MM)
    mm = cn.measure_mm(props, scale)
    assert max(mm) == pytest.approx(cn.REFERENCE_DIAMETER_MM, rel=1e-6)


# --------------------------------------------------------------------------- #
# entry points
# --------------------------------------------------------------------------- #


def test_analyse_returns_labels_props_and_a_scale():
    labels, props, mm_per_px = cn.analyse(io.sample("coins"))
    assert len(props) == cn.TRUE_COIN_COUNT
    assert mm_per_px > 0
    assert all("diameter_mm" in p for p in props)


def test_equivalent_diameter_matches_a_known_circle():
    disc = np.zeros((200, 200), np.int32)
    cv2.circle(disc, (100, 100), 40, 1, -1)
    props = cn.region_properties(disc)
    assert len(props) == 1
    assert props[0]["diameter_px"] == pytest.approx(80.0, rel=0.02)


def test_empty_input_does_not_crash_the_calibration():
    assert cn.calibrate_mm_per_px([], 24.25) == 0.0
    assert cn.measure_mm([], 0.0) == []


def test_analyse_rejects_an_unknown_method():
    with pytest.raises(KeyError):
        cn.analyse(io.sample("coins"), method="not a method")


# --------------------------------------------------------------------------- #
# naming the coins
# --------------------------------------------------------------------------- #


def test_ten_and_twenty_rupees_are_the_same_diameter():
    """Not a bug in the classifier. A fact about the coinage.

    The two are told apart in the hand by the 20's twelve-sided edge, which an
    overhead photograph of a flat disc does not record. Pinned so that nobody
    later "fixes" the confusion matrix by nudging a number in the table.
    """
    assert synth.RUPEE_COINS_MM[10] == synth.RUPEE_COINS_MM[20]
    assert 20 not in cn.IDENTIFIABLE


def test_the_hard_pair_is_one_millimetre_apart():
    """1 and 5 rupees differ by 1.07 mm, which sets the measurement budget."""
    gap = abs(synth.RUPEE_COINS_MM[5] - synth.RUPEE_COINS_MM[1])
    assert gap == pytest.approx(1.07, abs=0.01)
    # at the scale the scenes are generated, that is a couple of pixels
    assert gap / 0.42 < 3.0


def test_identification_names_most_coins_and_never_says_twenty():
    scenes = []
    for i, bg in enumerate(("felt", "slate", "navy")):
        scene, truth = synth.coin_scene(seed=i, background=bg, mm_per_px=0.42)
        scenes.append((scene, truth, 0.42))
    rows, confusion = cn.evaluate_identification(scenes)
    exact = next(r for r in rows if r["calibration"] == "exact scale")

    assert exact["coins_matched"] > 50
    assert exact["accuracy"] > 0.55
    # every 20-rupee coin is read as something else, because it must be
    assert all(read != 20 for _true, read in confusion)
    # and the errors are self-consistent with the accuracy beside them
    errors = sum(n for (t, r), n in confusion.items() if t != r)
    assert errors == exact["coins_matched"] - exact["identified"]


def test_a_wrong_reference_moves_the_whole_reading():
    """Self-calibration cannot detect its own failure.

    Assuming the largest coin present is 27 mm is what a user can actually do
    without a ruler in shot. When it is true the reading is good; when it is
    not, every diameter scales by the same wrong factor and nothing downstream
    can tell.
    """
    scene, truth = synth.coin_scene(seed=0, mm_per_px=0.42)
    labels = cn.METHODS[cn.IDENT_METHOD](scene)
    props = cn.region_properties(labels)

    honest = cn.calibrate_from_largest(props, 27.0)
    wrong = cn.calibrate_from_largest(props, 21.93)
    assert wrong < honest
    assert wrong / honest == pytest.approx(21.93 / 27.0, rel=1e-6)
