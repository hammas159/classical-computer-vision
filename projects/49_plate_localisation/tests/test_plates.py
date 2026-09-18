"""Tests for project 49: licence-plate localisation.

They pin the result (two metrics, two orders; the whole-frame control wins
coverage outright; the readability proxy sides with IoU; plate size does not
predict difficulty; the winner is stable across thresholds), the mechanism (a
vertical shift costs more IoU than a horizontal one on a shape wider than it is
tall), the construction (the annotation parses, the geometric filter is shared,
the proxy is validated before use) and the box arithmetic.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import plates as pl  # noqa: E402

pytestmark = pytest.mark.skipif(
    not pl.available(),
    reason="run `python tools/fetch_assets.py --set plates` to fetch the photographs")


# --------------------------------------------------------------------------- #
# the annotation
# --------------------------------------------------------------------------- #


def test_there_are_at_least_ten_annotated_photographs():
    assert len(pl.image_names()) >= 10


def test_every_annotation_parses_into_a_box_and_a_string():
    for name in pl.image_names():
        (x, y, w, h), text = pl.truth(name)
        assert w > 0 and h > 0
        assert x >= 0 and y >= 0
        assert len(text) >= 4
        assert any(c.isalnum() for c in text)


def test_every_annotated_box_is_inside_its_photograph():
    for name in pl.image_names():
        (x, y, w, h), _ = pl.truth(name)
        img = pl.load(name)
        assert x + w <= img.shape[1] and y + h <= img.shape[0], name


def test_the_plates_really_are_wider_than_they_are_tall():
    """The shape the whole IoU argument rests on, taken from the annotation."""
    aspects = []
    for name in pl.image_names():
        (_, _, w, h), _ = pl.truth(name)
        aspects.append(w / h)
        assert 2.0 < w / h < 6.0, (name, w / h)
    assert 3.5 < float(np.mean(aspects)) < 5.0


def test_the_set_spans_a_wide_range_of_plate_sizes():
    shares = [pl.plate_share(n) for n in pl.image_names()]
    assert max(shares) / min(shares) > 20


# --------------------------------------------------------------------------- #
# box arithmetic
# --------------------------------------------------------------------------- #


def test_iou_of_a_box_with_itself_is_one():
    assert pl.iou((10, 10, 40, 10), (10, 10, 40, 10)) == pytest.approx(1.0)


def test_iou_of_a_half_overlap_is_arithmetic():
    """Two 10x10 boxes offset by 5 share 50 of a 150 union."""
    assert pl.iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)


def test_coverage_of_a_contained_box_is_exactly_one():
    """A box twice the size of the target, containing it, covers 1.00 and scores
    IoU 0.25. Both halves of the project's argument in one assertion."""
    target = (10, 10, 40, 10)
    big = (0, 5, 80, 20)
    assert pl.coverage(big, target) == pytest.approx(1.0)
    assert pl.iou(big, target) == pytest.approx(400 / 1600)


def test_coverage_of_a_disjoint_box_is_zero():
    assert pl.coverage((0, 0, 10, 10), (500, 500, 10, 10)) == 0.0


def test_best_by_on_nothing_returns_nothing():
    box, score = pl.best_by([], (0, 0, 10, 10))
    assert box is None and score == 0.0


# --------------------------------------------------------------------------- #
# the controls
# --------------------------------------------------------------------------- #


def test_the_whole_frame_control_covers_every_plate():
    """The reason coverage is never reported on its own.

    Returning the entire photograph scores a perfect coverage on every image in
    the set, at an IoU of essentially zero.
    """
    r = pl.evaluate("Whole frame (control)")
    assert r["hits_coverage"] == r["frames"]
    assert r["median_iou"] < 0.05


def test_the_whole_frame_control_wins_coverage_outright():
    results = {r["locator"]: r for r in pl.evaluate_all()}
    best = max(results.values(), key=lambda r: r["hits_coverage"])
    assert best["locator"] == "Whole frame (control)"


def test_the_nothing_control_scores_zero_on_everything():
    r = pl.evaluate("Nothing (control)")
    assert r["hits_iou"] == 0 and r["hits_coverage"] == 0 and r["boxes"] == 0


def test_the_fixed_box_control_ignores_the_image():
    a = np.zeros((400, 600, 3), np.uint8)
    b = np.full((400, 600, 3), 255, np.uint8)
    assert pl.control_fixed_box(a) == pl.control_fixed_box(b)


def test_the_fixed_box_control_does_not_solve_the_problem():
    """Unlike the lane project, where the average answer is nearly the answer,
    a plate is not in the same place twice."""
    r = pl.evaluate("Fixed box (control)")
    assert r["hits_iou"] == 0


# --------------------------------------------------------------------------- #
# the result
# --------------------------------------------------------------------------- #


def test_two_metrics_rank_the_locators_in_opposite_orders():
    """The headline. At least one pair must swap places, or reporting both
    metrics would be redundant."""
    assert len(pl.metrics_disagree()) >= 1


def test_the_cascade_wins_coverage_and_loses_iou():
    results = {r["locator"]: r for r in pl.evaluate_all()}
    cascade = results["Haar plate cascade"]
    sobel = results["Sobel-x + morphology"]
    assert cascade["hits_coverage"] > sobel["hits_coverage"]
    assert cascade["hits_iou"] < sobel["hits_iou"]


def test_the_readability_proxy_sides_with_iou():
    """What settles it. The only metric that uses the plate's typed text agrees
    with IoU, not with the coverage metric introduced to replace it."""
    results = {r["locator"]: r for r in pl.evaluate_all()}
    assert (results["Sobel-x + morphology"]["readable"]
            > results["Haar plate cascade"]["readable"])


def test_the_cascade_returns_boxes_much_larger_than_the_plate():
    """The mechanism behind that: high coverage with low IoU means one thing."""
    r = pl.evaluate("Haar plate cascade")
    assert r["median_coverage"] > 0.9
    assert r["median_iou"] < 0.55


def test_the_aspect_filter_alone_is_worth_nothing():
    """`Contour + aspect` shares the geometric filter with every other locator
    and proposes candidates badly. It scores zero on both metrics while
    returning more boxes than anything else, which is what says the filter is
    not the hard part."""
    r = pl.evaluate("Contour + aspect")
    assert r["hits_iou"] == 0 and r["hits_coverage"] == 0
    assert r["boxes"] > 50


# --------------------------------------------------------------------------- #
# the mechanism
# --------------------------------------------------------------------------- #


def test_a_vertical_shift_costs_more_iou_than_a_horizontal_one():
    """On a shape 4.4 times wider than it is tall, the same pixel error eats a
    much larger share of the short side, and IoU charges for both."""
    rows = {r["shift_px"]: r for r in pl.what_a_pixel_of_error_costs()}
    for shift, r in rows.items():
        assert r["iou_shifted_vertically"] < r["iou_shifted_sideways"], shift
    eight = rows[8]
    assert (1 - eight["iou_shifted_vertically"]) > 2 * (1 - eight["iou_shifted_sideways"])


def test_coverage_is_less_sensitive_to_a_vertical_shift_than_iou():
    for r in pl.what_a_pixel_of_error_costs():
        assert r["coverage_shifted_vertically"] >= r["iou_shifted_vertically"]


def test_a_zero_shift_scores_one():
    """A regression test for the perturbation itself: with nothing moved, the
    annotated box must score perfectly against itself."""
    for name in pl.image_names():
        target, _ = pl.truth(name)
        assert pl.iou(target, target) == pytest.approx(1.0)
        assert pl.coverage(target, target) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# two hypotheses that came out flat, kept as tests so they stay flat
# --------------------------------------------------------------------------- #


def test_plate_size_does_not_predict_difficulty():
    """The obvious explanation, tested and rejected.

    If this ever starts passing strongly the finding has changed and the README
    is wrong, which is why it is asserted rather than left as prose.
    """
    rows = pl.difficulty()
    widths = np.log([r["plate_width"] for r in rows])
    found = np.array([r["found_iou"] for r in rows], float)
    assert abs(float(np.corrcoef(widths, found)[0, 1])) < 0.6


def test_the_winner_is_the_same_at_every_iou_threshold():
    """It is the choice of metric that flips the ranking, not the threshold."""
    sweeps = {k: pl.iou_threshold_sweep(k) for k in pl.REAL_LOCATORS}
    thresholds = [r["threshold"] for r in sweeps[pl.REAL_LOCATORS[0]]]
    winners = set()
    for i in range(len(thresholds)):
        winners.add(max(sweeps, key=lambda k: sweeps[k][i]["hits"]))
    assert len(winners) == 1, winners


def test_fewer_photographs_pass_as_the_threshold_rises():
    """A monotonicity check on the sweep itself."""
    for k in pl.REAL_LOCATORS:
        hits = [r["hits"] for r in pl.iou_threshold_sweep(k)]
        assert hits == sorted(hits, reverse=True), (k, hits)


# --------------------------------------------------------------------------- #
# the readability proxy is scored before it is used
# --------------------------------------------------------------------------- #


def test_the_proxy_is_right_on_most_of_the_annotated_crops():
    """Given a crop of the plate itself the blob count should equal the number
    of characters. Where it does not, the proxy failed, not the locator -- and
    that error bounds everything the proxy is used for."""
    v = pl.validate_proxy()
    assert v["exact"] >= 0.6 * v["frames"], v
    assert v["within_one"] >= 0.8 * v["frames"], v


def test_the_proxy_finds_nothing_in_an_empty_crop():
    img = np.full((200, 400, 3), 200, np.uint8)
    assert pl.character_blobs(img, (10, 10, 200, 46)) == 0


def test_the_proxy_rejects_a_degenerate_box():
    img = pl.load(pl.image_names()[0])
    assert pl.character_blobs(img, (0, 0, 3, 2)) == 0


# --------------------------------------------------------------------------- #
# the shared filter
# --------------------------------------------------------------------------- #


def test_every_locator_is_held_to_the_same_geometry():
    """The comparison is of how candidates are proposed, not of who tuned their
    aspect filter more tightly."""
    for name in pl.image_names():
        img = pl.load(name)
        for locator in pl.REAL_LOCATORS:
            if "cascade" in locator:
                continue  # the cascade proposes and filters internally
            for x, y, w, h in pl.LOCATORS[locator](img):
                assert w >= pl.MIN_WIDTH
                assert pl.MIN_ASPECT <= w / h <= pl.MAX_ASPECT
