"""Tests for project 25, descriptor matching and robust estimation.

Most of these pin a *finding* rather than a number, so that a later change which
quietly reverses one of the project's conclusions fails loudly instead of
rewriting the README by accident.
"""

from __future__ import annotations

import numpy as np
import pytest

import matching as mt
from shared.io import to_gray


# --------------------------------------------------------------------------- #
# the setup
# --------------------------------------------------------------------------- #


def test_the_homography_is_known_so_every_match_can_be_labelled():
    """What the whole project rests on.

    With H known, each match is labelled a true inlier or a true outlier
    *before* any estimator runs. An estimator's own inlier mask is its opinion;
    scoring against it would let a method that keeps four points and fits them
    perfectly report zero error.
    """
    a, b, H = mt.make_pair("leopard_in_tree", seed=0)
    assert H.shape == (3, 3)
    kp_a, desc_a, norm = mt.DESCRIPTORS["SIFT"](to_gray(a))
    kp_b, desc_b, _ = mt.DESCRIPTORS["SIFT"](to_gray(b))
    src, dst, truth = mt.label_matches(kp_a, kp_b, mt.match_ratio(desc_a, desc_b, norm), H)

    assert len(src) == len(dst) == len(truth) > 50
    assert truth.mean() > 0.9          # a clean pair is mostly inliers
    assert mt.reprojection_error(H, src[truth], dst[truth]) < 3.0


def test_injecting_outliers_really_injects_them():
    a, _, H = mt.make_pair("station_platform", seed=1)
    kp_a, desc_a, norm = mt.DESCRIPTORS["SIFT"](to_gray(a))
    kp_b, desc_b, _ = mt.DESCRIPTORS["SIFT"](to_gray(mt.make_pair("station_platform", seed=1)[1]))
    src, dst, truth = mt.label_matches(kp_a, kp_b, mt.match_ratio(desc_a, desc_b, norm), H)

    _, _, spoiled = mt.inject_outliers(src, dst, truth, 0.6, a.shape, seed=0)
    assert 0.35 < float(spoiled.mean()) < 0.45   # about 40% left as inliers


# --------------------------------------------------------------------------- #
# breakdown points -- a theorem, checked
# --------------------------------------------------------------------------- #


def test_lmeds_breaks_at_its_theoretical_fifty_percent():
    """Least Median of Squares minimises a median, so it cannot survive more than
    half the data being wrong. That is a theorem, and here it is met exactly.

    Measured: 0.222 px at 40% outliers, 2.00 px at 50%, 190.9 px at 60%.
    """
    rows = {r["outlier_fraction"]: r for r in
            mt.sweep_outliers(images=mt.IMAGES[:4], fractions=(0.4, 0.5, 0.6))}

    assert rows[0.4]["LMEDS"] < 1.0        # comfortably fine below the limit
    assert rows[0.6]["LMEDS"] > 50.0       # destroyed above it
    assert rows[0.5]["LMEDS"] > rows[0.4]["LMEDS"]   # and already degrading at it


def test_ransac_survives_far_past_lmeds_and_still_has_a_limit():
    rows = {r["outlier_fraction"]: r for r in
            mt.sweep_outliers(images=mt.IMAGES[:4], fractions=(0.6, 0.8, 0.9))}

    assert rows[0.6]["RANSAC"] < 1.0
    assert rows[0.8]["RANSAC"] < 1.0       # still fine where LMEDS is ruined
    assert rows[0.9]["RANSAC"] > 5.0       # but not unbreakable


def test_least_squares_is_destroyed_by_a_fifth_of_the_data():
    """The control, and the reason robust estimation exists at all.

    A single outlier pulls a least-squares fit, and 20% of them move the
    reprojection error from 1.5 px to 26 px — a factor of 17.
    """
    rows = {r["outlier_fraction"]: r for r in
            mt.sweep_outliers(images=mt.IMAGES[:4], fractions=(0.0, 0.2))}
    control = "Least squares (control)"

    assert rows[0.0][control] < 5.0
    assert rows[0.2][control] > 10 * rows[0.0][control]


def test_the_iteration_count_matches_the_closed_form():
    """Reported alongside the measured breakdown so the theory is checked, not quoted."""
    assert mt.ransac_iterations_needed(0.0) < 1
    assert 500 < mt.ransac_iterations_needed(0.7) < 700
    assert mt.ransac_iterations_needed(0.9) > 40000
    # monotone in the outlier fraction
    fractions = (0.0, 0.2, 0.4, 0.6, 0.8)
    counts = [mt.ransac_iterations_needed(f) for f in fractions]
    assert counts == sorted(counts)


# --------------------------------------------------------------------------- #
# descriptors -- the half of SIFT that project 19 could not see
# --------------------------------------------------------------------------- #


def test_sift_wins_on_descriptor_accuracy_where_orb_won_on_detection():
    """The complement of project 19, and together they are the honest answer.

    Project 19 measured *detector* repeatability and found ORB ahead of SIFT.
    This project measures what the descriptors are for: SIFT's matches
    reprojection at 0.222 px against ORB's 0.927, a factor of four, at four to
    five times the cost.

    Neither project alone supports "SIFT is better" or "ORB is better". Both
    together support "they are better at different things", which is the useful
    statement.
    """
    rows = {r["descriptor"]: r for r in mt.evaluate_descriptors(images=mt.IMAGES[:6])}
    sift, orb = rows["SIFT"], rows["ORB"]

    assert sift["reprojection_px"] < 0.5 * orb["reprojection_px"]
    assert sift["inlier_precision"] > orb["inlier_precision"]
    assert orb["median_ms"] < sift["median_ms"]       # and ORB is still faster
    assert orb["matches"] > sift["matches"]           # and still finds more


def test_the_ratio_test_buys_precision_with_matches():
    """Monotonically, in both directions, which is what makes it a dial.

    At ratio 1.0 the test is off and precision is 0.635; at 0.5 it is 0.999 and
    half the matches are gone.
    """
    rows = sorted(mt.sweep_ratio(images=mt.IMAGES[:4]), key=lambda r: r["ratio"])
    precision = [r["inlier_precision"] for r in rows]
    kept = [r["matches_kept"] for r in rows]

    assert precision == sorted(precision, reverse=True)
    assert kept == sorted(kept)
    assert precision[0] > 0.99 and precision[-1] < 0.75


def test_cross_check_sits_between_no_filter_and_the_ratio_test():
    rows = {r["filter"]: r for r in mt.evaluate_filters(images=mt.IMAGES[:4])}
    none, cross, ratio = (rows["All nearest neighbours"], rows["Cross-check"],
                          rows["Ratio test 0.75"])

    assert none["inlier_recall"] == pytest.approx(1.0)
    assert none["inlier_precision"] < cross["inlier_precision"] < ratio["inlier_precision"]
    assert none["matches_kept"] > cross["matches_kept"] > ratio["matches_kept"]


def test_the_drawing_helper_reports_error_against_the_truth_not_its_own_inliers():
    """An estimator that keeps four points and fits them perfectly must not score 0."""
    a, b, H = mt.make_pair("gilded_stupa", seed=3)
    err, inliers, canvas = mt.estimate_and_draw(a, b, H, "RANSAC", outlier_fraction=0.5)

    assert canvas.shape == a.shape
    assert inliers > 4
    assert 0.0 < err < 5.0

    bad, _, _ = mt.estimate_and_draw(a, b, H, "Least squares (control)",
                                     outlier_fraction=0.5)
    assert bad > err * 5
