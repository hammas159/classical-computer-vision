"""Tests for project 32, the Hough transforms.

Most of these pin a *finding* rather than a number, so that a later change which
quietly reverses one of the project's conclusions fails loudly instead of
rewriting the README by accident.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import hough as hg
from shared import bsds
from shared.io import to_gray


# --------------------------------------------------------------------------- #
# the generated scene, where the answer is exact
# --------------------------------------------------------------------------- #


def test_both_line_methods_find_every_line_on_a_clean_scene():
    for row in hg.evaluate_lines(runs=1):
        assert row["recall"] == pytest.approx(1.0), row["method"]
        assert row["angle_error_deg"] < 1.0, row["method"]
        assert row["rho_error_px"] < 3.0, row["method"]


def test_circles_are_found_with_sub_pixel_centres():
    for row in hg.evaluate_circles(runs=1):
        assert row["recall"] > 0.8, row["method"]
        assert row["centre_error_px"] < 5.0, row["method"]


def test_clutter_costs_precision_and_not_recall():
    """Hough finds the lines *and more*, which is the shape of its failure.

    Voting is why it survives noise: an accumulator peak needs many pixels to
    agree, so scattered noise never builds one. The same property means a few
    collinear clutter pixels do build one, and the extra lines are real peaks.
    """
    rows = sorted(hg.sweep_clutter(), key=lambda r: r["clutter"])
    assert rows[-1]["standard_recall"] >= rows[0]["standard_recall"] - 0.05
    assert rows[-1]["standard_precision"] < rows[0]["standard_precision"]


def test_standard_hough_is_faster_here_than_the_probabilistic_one():
    """Which is the opposite of the usual claim, and needs its caveat stated.

    On this scene standard Hough runs in 1.76 ms against probabilistic's 4.03,
    and is more precise. The two are **not** running at the same operating
    point — standard needs 120 votes and probabilistic 60 — so this is a
    comparison of the defaults people actually use rather than of the algorithms
    at matched sensitivity.
    """
    rows = {r["method"]: r for r in hg.evaluate_lines(clutter=5, runs=3)}
    std, prob = rows["Standard Hough"], rows["Probabilistic Hough"]
    assert std["median_ms"] < prob["median_ms"]
    assert std["precision"] > prob["precision"]
    # the probabilistic one does buy something: tighter angles, from segment fits
    assert prob["angle_error_deg"] < std["angle_error_deg"]


# --------------------------------------------------------------------------- #
# the photographs, where the answer is a person's
# --------------------------------------------------------------------------- #


def test_rasterising_puts_infinite_lines_and_segments_on_one_footing():
    """A (rho, theta) line and an (x1,y1,x2,y2) segment cannot be compared until
    both are drawn. Drawing them is also what exposes the difference: an infinite
    line crosses the whole frame including where nothing supported it."""
    gray = to_gray(hg.load_scene("hotel_rossiya"))
    inf_lines = hg.rasterise_lines(hg.detect_lines_standard(gray), gray.shape)
    segments = hg.rasterise_lines(hg.detect_lines_probabilistic(gray)[0], gray.shape)

    assert inf_lines.shape == gray.shape
    assert set(np.unique(inf_lines)) <= {0, 255}
    assert (inf_lines > 0).mean() > (segments > 0).mean()


def test_hough_is_less_precise_than_the_canny_it_is_built_on():
    """The project's headline, and it is uncomfortable.

    Hough votes on Canny's output, so its lines are a *subset* of what Canny
    found — in principle a cleaner one. Measured against human boundaries it is
    not: Canny scores 0.194 precision and the best Hough 0.117, while standard
    Hough marks **55% of the frame** as line against Canny's 18%.

    The voting stage adds error on real photographs, by the only measure
    available.
    """
    rows = {r["method"]: r for r in hg.evaluate_photo_lines(images=hg.IMAGES[:6])}
    canny = rows["Canny edges (control)"]

    for name in ("Standard Hough", "Probabilistic Hough"):
        assert rows[name]["precision"] < canny["precision"], name
    assert rows["Standard Hough"]["pixels_drawn"] > 2 * canny["pixels_drawn"]


def test_the_man_made_scene_is_where_hough_does_best_relative_to_canny():
    """Straight structure is what it is for, and photographs mostly have none."""
    gains = {}
    for name in hg.IMAGES:
        gray = to_gray(hg.load_scene(name))
        target = bsds.consensus_boundaries(name)
        canny = bsds.boundary_f_measure(cv2.Canny(gray, 50, 150), target, 2)["precision"]
        hough = bsds.boundary_f_measure(
            hg.rasterise_lines(hg.detect_lines_probabilistic(gray)[0], gray.shape),
            target, 2)["precision"]
        gains[name] = hough - canny

    best = max(gains, key=gains.get)
    assert best in ("hotel_rossiya", "glass_roof_trees", "barges_and_blocks",
                    "two_women_street", "ocelot_on_rock"), best


def test_every_photograph_has_human_boundaries():
    for name in hg.IMAGES:
        assert bsds.has_ground_truth(name), name


# --------------------------------------------------------------------------- #
# the parameter
# --------------------------------------------------------------------------- #


def test_the_accumulator_threshold_trades_recall_against_precision():
    """It is the whole method: how many pixels have to agree to call it a line."""
    rows = sorted(hg.sweep_threshold(), key=lambda r: r["threshold"])
    recalls = [r["recall"] for r in rows]
    precisions = [r["precision"] for r in rows]

    assert recalls == sorted(recalls, reverse=True)
    assert precisions[-1] > precisions[0]


def test_voting_is_what_makes_hough_survive_noise():
    """Recall holds up far better than a per-pixel method would.

    An accumulator peak needs many pixels to agree, and scattered noise never
    builds one — which is the entire argument for the transform.
    """
    rows = sorted(hg.sweep_noise(), key=lambda r: r["noise_sigma"])
    assert rows[0]["line_recall"] == pytest.approx(1.0)
    assert rows[-1]["line_recall"] > 0.5
    assert rows[-1]["line_angle_error_deg"] < 5.0
