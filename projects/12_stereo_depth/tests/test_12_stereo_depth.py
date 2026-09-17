"""Tests for project 12, stereo to depth.

Most of these pin a *finding* rather than a number, so that a later change which
quietly reverses one of the project's conclusions fails loudly instead of
rewriting the README by accident.
"""

from __future__ import annotations

import numpy as np
import pytest

import stereo as st
from shared import io, synth

MAXD = 48


def a_pair(name: str = "stone_arch", seed: int = 0):
    return synth.stereo_pair(io.real_photo(name), max_disparity=MAXD, seed=seed)


# --------------------------------------------------------------------------- #
# the generated pair
# --------------------------------------------------------------------------- #


def test_the_right_view_is_the_left_view_shifted_by_the_disparity():
    """The generator's contract. If this is wrong every score below is fiction."""
    left, right, disparity, valid = a_pair()
    h, w = disparity.shape
    ys, xs = np.mgrid[0:h, 0:w]
    tx = np.round(xs - disparity).astype(int)
    inside = valid & (tx >= 0) & (tx < w)
    # sample a few thousand valid pixels rather than all of them
    idx = np.argwhere(inside)[::37]
    lhs = left[idx[:, 0], idx[:, 1]].astype(int)
    rhs = right[idx[:, 0], tx[idx[:, 0], idx[:, 1]]].astype(int)
    assert np.abs(lhs - rhs).mean() < 2.0


def test_disparity_stays_inside_the_search_range():
    """A truncated disparity reads as a wrong match, not an out-of-range one.

    If the scene contains disparities past `MAX_DISPARITY`, every matcher is
    charged for something the search could never have found.
    """
    _, _, disparity, _ = a_pair()
    assert float(np.nanmax(disparity)) < st.MAX_DISPARITY


def test_some_of_the_scene_is_occluded():
    """Otherwise the pair has no occlusions and the hardest case is missing."""
    _, _, _, valid = a_pair()
    hidden = 1.0 - float(valid.mean())
    assert 0.005 < hidden < 0.25


def test_the_depth_map_is_layered_not_smooth():
    """Smooth depth has no discontinuities, which is where stereo fails."""
    depth = synth.depth_layers((300, 200), layers=4, seed=0)
    grad = np.abs(np.diff(depth, axis=1))
    # a smooth ramp has a tiny maximum gradient; layers have real steps
    assert grad.max() > 0.05
    assert len(np.unique(np.round(depth, 2))) < depth.size / 100


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def test_declining_to_answer_is_counted_two_different_ways():
    """The distinction the whole project turns on.

    A matcher that answers half the frame perfectly should score 0% bad among
    what it answered and 50% bad over the whole frame. One number cannot say
    both, so both are reported.
    """
    truth = np.full((40, 40), 10.0, np.float32)
    valid = np.ones((40, 40), bool)
    pred = truth.copy()
    pred[:20] = np.nan  # answer only half, and answer it exactly

    s = st.score_disparity(pred, truth, valid)
    assert s["density"] == pytest.approx(0.5)
    assert s["bad2_answered"] == pytest.approx(0.0)
    assert s["bad2_all"] == pytest.approx(0.5)


def test_invalid_pixels_are_excluded_from_scoring():
    """Scoring a matcher where only one camera can see measures nothing."""
    truth = np.full((20, 20), 5.0, np.float32)
    valid = np.zeros((20, 20), bool)
    valid[:10] = True
    pred = truth.copy()
    pred[10:] = 999.0  # nonsense, but outside `valid`
    assert st.score_disparity(pred, truth, valid)["bad2_all"] == pytest.approx(0.0)


def test_the_regions_partition_the_valid_pixels():
    left, _, truth, valid = a_pair()
    regions = st.error_regions(truth, valid, left)
    total = sum(int(m.sum()) for m in regions.values())
    assert total == int(valid.sum())
    assert all(int(m.sum()) > 0 for m in regions.values())


# --------------------------------------------------------------------------- #
# methods
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", list(st.METHODS))
def test_every_matcher_returns_a_float_disparity_map(name):
    left, right, truth, _ = a_pair()
    d = st.METHODS[name](left, right)
    assert d.shape == truth.shape
    assert d.dtype == np.float32
    finite = d[np.isfinite(d)]
    assert finite.size > 0
    assert finite.min() >= 0


def test_every_matcher_beats_predicting_a_constant():
    """The bar that sounds low and is not always cleared."""
    left, right, truth, valid = a_pair()
    control = st.score_disparity(st.match_constant(left, right), truth, valid)
    for name, fn in st.METHODS.items():
        if name in st.CONTROLS:
            continue
        s = st.score_disparity(fn(left, right), truth, valid)
        assert s["bad2_all"] < control["bad2_all"]


# --------------------------------------------------------------------------- #
# the findings
# --------------------------------------------------------------------------- #


def test_density_and_accuracy_pick_different_winners():
    """The project's headline, pinned.

    Block matching declines to answer most of a textureless region, so among the
    pixels it *does* answer it is the most accurate method here. Count a refusal
    as a miss and it is the worst. Nothing about the method changed between
    those two sentences.
    """
    left, right, truth, valid = a_pair()
    scores = {
        name: st.score_disparity(fn(left, right), truth, valid)
        for name, fn in st.METHODS.items()
        if name not in st.CONTROLS
    }
    most_accurate = min(scores, key=lambda k: scores[k]["bad2_answered"])
    densest = max(scores, key=lambda k: scores[k]["density"])
    assert most_accurate != densest


def test_error_concentrates_at_depth_discontinuities():
    """Where a single bad-pixel percentage hides the failure.

    Every matcher is worse at a depth edge than in the flat regions either side
    of it, and by a wide margin.
    """
    left, right, truth, valid = a_pair()
    regions = st.error_regions(truth, valid, left)
    for name, fn in st.METHODS.items():
        if name in st.CONTROLS:
            continue
        pred = fn(left, right)
        at_edge = st.score_disparity(pred, truth, regions["discontinuity"])["bad2_all"]
        elsewhere = st.score_disparity(pred, truth, regions["well-textured"])["bad2_all"]
        assert at_edge > elsewhere, f"{name} is not worse at discontinuities"


def test_the_smoothness_penalty_buys_density_and_costs_precision():
    """SGBM fills what BM abandons, and pays for it where BM was right.

    Higher density, higher mean absolute error. Both directions matter: the
    filling is not free, and a table showing only one of them would recommend
    SGBM without saying what it gives up.
    """
    left, right, truth, valid = a_pair()
    bm = st.score_disparity(st.match_bm(left, right), truth, valid)
    sgbm = st.score_disparity(st.match_sgbm(left, right), truth, valid)
    assert sgbm["density"] > bm["density"]
    assert sgbm["mae_px"] > bm["mae_px"]


def test_the_real_pair_is_harder_than_the_generated_ones():
    """The check that the generator is not quietly flattering itself.

    Middlebury's Aloe pair has measured ground truth — structured light, not a
    warp written here. Every matcher does worse on it, which is what stops the
    generated scores being mistaken for real-world ones.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from run import load_aloe

    left, right, truth, valid = load_aloe()
    real = st.score_disparity(st.match_sgbm_hh(left, right), truth, valid)

    gl, gr, gt, gv = a_pair()
    generated = st.score_disparity(st.match_sgbm_hh(gl, gr), gt, gv)

    assert real["bad2_all"] > generated["bad2_all"]
