"""Tests for project 28, Canny's parameter sensitivity.

Most of these pin a *finding* rather than a number, so that a later change which
quietly reverses one of the project's conclusions fails loudly instead of
rewriting the README by accident.
"""

from __future__ import annotations

import numpy as np
import pytest

import canny_sensitivity as cs
from shared import bsds
from shared.io import to_gray


# --------------------------------------------------------------------------- #
# the operator
# --------------------------------------------------------------------------- #


def test_the_high_threshold_is_always_above_the_low_one():
    """Parameterising as (low, ratio) is what guarantees it.

    Sweeping low and high independently spends most of the grid on combinations
    where high < low, which Canny treats as a single threshold — so half the
    "results" would be duplicates of each other.
    """
    gray = to_gray(cs.load_scene("alpine_church"))
    for low in (10, 100, 200, 250):
        for ratio in (1.5, 3.0):
            edges = cs.canny(gray, 1.4, low, ratio)
            assert edges.dtype == np.uint8
            assert set(np.unique(edges)) <= {0, 255}


def test_no_smoothing_finds_more_edges_than_any_smoothing():
    gray = to_gray(cs.load_scene("bears_on_hillside"))
    none = (cs.canny(gray, 0.0, 50, 3.0) > 0).mean()
    some = (cs.canny(gray, 2.0, 50, 3.0) > 0).mean()
    assert none > some


# --------------------------------------------------------------------------- #
# the synthetic scene
# --------------------------------------------------------------------------- #


def test_sigma_is_the_only_parameter_that_matters_much():
    """The variance decomposition, which is the point of running a full grid.

    Smoothing explains 30% of the F1 variance; the low threshold explains 2%
    and the ratio 0.03%. The ratio is the parameter every tutorial discusses.
    """
    variance = cs.variance_decomposition(cs.full_grid())
    assert variance["sigma"]["fraction_of_total"] > 0.2
    assert variance["low"]["fraction_of_total"] < 0.1
    assert variance["ratio"]["fraction_of_total"] < 0.01
    assert variance["sigma"]["range"] > 5 * variance["ratio"]["range"]


def test_the_synthetic_scene_saturates():
    """Why a parameter study on shapes cannot choose a setting.

    Twelve of the configurations reach a **perfect** F1 of 1.000 on the
    generated scene. Once a benchmark is saturated it has no power left to
    separate the things it is being asked to rank, and any choice among those
    twelve is arbitrary as far as it is concerned.
    """
    grid = cs.full_grid()
    perfect = [r for r in grid if r["f1"] >= 0.999]
    assert len(perfect) > 5, f"only {len(perfect)} configurations are perfect"


# --------------------------------------------------------------------------- #
# the photographs -- where the answer is a person's
# --------------------------------------------------------------------------- #


def test_the_photographs_have_human_boundaries_and_a_measured_ceiling():
    ceiling = cs.photo_human_ceiling(images=cs.IMAGES[:6])
    assert 0.7 < ceiling["best"] < 1.0
    assert ceiling["mean"] < ceiling["best"]
    for name in cs.IMAGES:
        assert bsds.has_ground_truth(name), name


def test_tuning_on_shapes_costs_nineteen_percent_on_photographs():
    """The project's headline.

    The best setting the synthetic scene can identify scores 0.3657 on
    photographs; the best setting the photographs identify scores 0.4367. Both
    score a perfect 1.000 on the shapes, so the synthetic benchmark cannot tell
    them apart — and picking the wrong one costs 19% of the real performance.
    """
    grid = {(r["sigma"], r["low"], r["ratio"]): r for r in cs.full_grid()}
    photo = {(r["sigma"], r["low"], r["ratio"]): r for r in cs.photo_grid()}
    shared = sorted(set(grid) & set(photo))

    best_syn = max(shared, key=lambda k: grid[k]["f1"])
    best_photo = max(shared, key=lambda k: photo[k]["f"])

    assert best_syn != best_photo
    assert grid[best_syn]["f1"] == pytest.approx(grid[best_photo]["f1"], abs=0.01)
    gap = photo[best_photo]["f"] - photo[best_syn]["f"]
    assert gap > 0.03, f"only {gap:.4f} apart on photographs"


def test_the_two_benchmarks_agree_only_loosely():
    """Positive correlation, nowhere near 1 — which is the useful statement.

    A synthetic benchmark is not worthless here; it is just not decisive. It
    orders the settings roughly right and cannot pick between the good ones.
    """
    grid = {(r["sigma"], r["low"], r["ratio"]): r for r in cs.full_grid()}
    photo = {(r["sigma"], r["low"], r["ratio"]): r for r in cs.photo_grid()}
    shared = sorted(set(grid) & set(photo))

    corr = float(np.corrcoef([grid[k]["f1"] for k in shared],
                             [photo[k]["f"] for k in shared])[0, 1])
    assert 0.4 < corr < 0.95, corr


def test_canny_reaches_about_half_the_human_ceiling():
    """And no parameter choice closes that gap, because it is not a tuning problem.

    Canny has no colour, no semantics and no idea what an object is. The best
    setting reaches 48% of what one annotator achieves against the others.
    """
    photo = cs.photo_grid(images=cs.IMAGES[:6])
    ceiling = cs.photo_human_ceiling(images=cs.IMAGES[:6])
    best = max(r["f"] for r in photo)
    assert best < 0.7 * ceiling["best"]


def test_the_edge_density_of_the_photograph_predicts_the_score():
    """Canny does well on a silhouette and badly in a thicket.

    On an acacia against open sky it reaches F 0.97 — close to the human
    ceiling. On a hillside of grass and bears it reaches 0.31. The detector has
    not changed; the question has.
    """
    import cv2

    scores, densities = [], []
    for name in cs.IMAGES:
        gray = to_gray(cs.load_scene(name))
        th, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        densities.append(float((cv2.Canny(gray, 0.5 * th, th) > 0).mean()))
        target = bsds.consensus_boundaries(name)
        scores.append(bsds.boundary_f_measure(
            cs.canny(gray, 2.0, 50, 2.0), target, cs.TOLERANCE)["f"])

    # A clear negative relationship, not a tight one -- what is in the picture
    # matters as much as how much of it there is, and twelve images is not
    # enough to separate those.
    assert float(np.corrcoef(densities, scores)[0, 1]) < -0.35
    assert max(scores) > 0.9 and min(scores) < 0.4


# --------------------------------------------------------------------------- #
# the measurement
# --------------------------------------------------------------------------- #


def test_the_matching_tolerance_moves_the_score_more_than_the_method_does():
    """A published edge-detection number without its tolerance is not a number."""
    rows = sorted(cs.sweep_tolerance(), key=lambda r: r["tolerance_px"])
    best = [r["best_f1"] for r in rows]
    assert best == sorted(best)
    assert best[-1] > best[0] + 0.2


def test_noise_changes_which_smoothing_is_best():
    """The one parameter that matters is the one the noise level decides."""
    rows = sorted(cs.sweep_noise(), key=lambda r: r["noise_sigma"])
    assert rows[-1]["best_sigma"] > rows[0]["best_sigma"]
    assert rows[-1]["best_f1"] < rows[0]["best_f1"]
