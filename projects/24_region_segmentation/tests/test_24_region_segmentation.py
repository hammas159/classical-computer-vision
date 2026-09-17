"""Tests for project 24, region segmentation.

Most of these pin a *finding* rather than a number, so that a later change which
quietly reverses one of the project's conclusions fails loudly instead of
rewriting the README by accident.
"""

from __future__ import annotations

import numpy as np
import pytest

import segmentation as sg
from shared import bsds
from shared.metrics import iou


# --------------------------------------------------------------------------- #
# the ground truth -- real people, not a thresholding
# --------------------------------------------------------------------------- #


def test_every_image_has_several_human_annotations():
    """The truth here is annotated, not constructed, and there is more than one.

    An earlier version used Otsu's binarisation as the target, which for a
    segmentation project is close to circular: Otsu is itself a segmentation
    method, so every row was being scored on how well it reproduces one
    particular competitor.
    """
    for name in sg.IMAGES:
        assert bsds.has_ground_truth(name), name
        ann = bsds.load_annotations(name)
        assert len(ann) >= 4, f"{name} has only {len(ann)} annotators"
        for a in ann:
            assert a["segmentation"].shape == a["boundaries"].shape


def test_the_human_ceiling_is_measured_and_is_not_one():
    """People do not agree with each other, and the disagreement is the ceiling.

    Averaged over the twelve photographs the best-agreeing pair of annotators
    reaches about 0.87 IoU, not 1.0. On one image — a train on a viaduct — the
    best pair agrees **0.184**: they disagree about what the background even is.
    No algorithm has any business being scored against a single annotator's
    opinion as though it were the answer.
    """
    ceilings = [sg.human_ceiling(n) for n in sg.IMAGES]
    best = [c["best"] for c in ceilings]

    assert 0.7 < float(np.mean(best)) < 0.95
    assert min(best) < 0.3, "the hard case has gone; the ceiling may be stale"
    for c in ceilings:
        assert c["mean"] <= c["best"] + 1e-9


def test_consensus_boundaries_need_more_than_one_vote():
    """One annotator's boundary map includes marks nobody else made."""
    name = sg.IMAGES[1]
    one = bsds.consensus_boundaries(name, min_annotators=1)
    three = bsds.consensus_boundaries(name, min_annotators=3)
    assert one.sum() > three.sum() > 0


# --------------------------------------------------------------------------- #
# the metric -- and why the table needs a control that does nothing
# --------------------------------------------------------------------------- #


def test_a_grid_of_rectangles_beats_most_real_methods_on_iou():
    """The project's headline, and it is about the metric rather than the methods.

    `labels_to_foreground` assigns each region to foreground or background by
    which it overlaps more — using the truth. That is deliberately generous so a
    low score cannot be blamed on the conversion, and the consequence is that
    **the more regions a method returns, the more the oracle assignment can do
    for it.**

    A grid of rectangles that has never looked at the image scores 0.914 and
    beats five of the six real methods. Any row above it is measuring region
    count.
    """
    rows = sg.evaluate_methods(images=sg.IMAGES[:6])
    scored = {r["method"]: r for r in rows}
    grid = scored["Grid tiles (control)"]

    beaten = [r["method"] for r in rows
              if r["method"] != "Grid tiles (control)" and r["iou"] < grid["iou"]]
    assert len(beaten) >= 4, f"the grid only beat {beaten}"
    assert grid["iou"] > 0.8


def test_boundary_recall_separates_what_iou_cannot():
    """The resolution: a metric the grid cannot game.

    SLIC beats the grid by 0.016 on IoU and by roughly a factor of two on
    boundary recall. The second number is the one that says SLIC is doing real
    work, because a rectangle grid's edges fall where the grid is, not where the
    picture is.
    """
    rows = {r["method"]: r for r in sg.evaluate_methods(images=sg.IMAGES[:6])}
    slic, grid = rows["SLIC superpixels"], rows["Grid tiles (control)"]

    assert slic["iou"] - grid["iou"] < 0.1           # IoU barely separates them
    assert slic["boundary_recall"] > 1.5 * grid["boundary_recall"]   # this does


def test_more_superpixels_is_always_more_boundary_recall():
    """Which is exactly the problem with quoting recall alone.

    Boundary recall rises monotonically with the number of superpixels
    requested, all the way up, because more boundaries means more of the true
    ones are covered. Under-segmentation error falls at the same time. Neither
    number on its own has a stopping point; only the F-measure does.
    """
    rows = sorted(sg.sweep_slic_count(images=sg.IMAGES[:4]), key=lambda r: r["requested"])
    recalls = [r["boundary_recall"] for r in rows]
    errors = [r["underseg_error"] for r in rows]

    assert recalls == sorted(recalls)
    assert errors == sorted(errors, reverse=True)
    assert rows[-1]["actual_regions"] > rows[0]["actual_regions"]


def test_the_most_over_segmented_method_has_near_perfect_boundary_recall():
    """And that is not a good result either.

    Watershed without markers returns thousands of regions, so its boundaries
    include every true boundary almost by accident — 0.996 recall. Boundary
    recall alone is as gameable as IoU; the pair, read against the region count,
    is what says something.
    """
    rows = {r["method"]: r for r in sg.evaluate_methods(images=sg.IMAGES[:4])}
    ws = rows["Watershed (no markers)"]
    assert ws["regions"] > 1000
    assert ws["boundary_recall"] > 0.95
    assert ws["iou"] < rows["SLIC superpixels"]["iou"]


# --------------------------------------------------------------------------- #
# the operators
# --------------------------------------------------------------------------- #


def test_every_method_returns_a_label_image():
    img, truth = sg.scene(sg.IMAGES[0])
    for name, fn in sg.METHODS.items():
        labels = fn(img)
        assert labels.shape[:2] == img.shape[:2], name
        assert labels.max() >= 1, name


def test_the_grid_control_looks_at_nothing():
    """Two completely different pictures must give the same tiles."""
    a, _ = sg.scene(sg.IMAGES[0])
    b = np.zeros_like(a)
    assert np.array_equal(sg.seg_grid(a), sg.seg_grid(b))


def test_noise_makes_the_unmarked_watershed_explode():
    """`sweep_noise` reports region COUNTS, and that is the interesting quantity.

    Every local minimum in the gradient becomes a catchment basin, and noise
    manufactures local minima. Unmarked watershed goes from about 6900 regions
    to 11400 — it does not get less accurate so much as stop being a
    segmentation at all. The marker-controlled version is what markers are for.
    """
    rows = sorted(sg.sweep_noise(images=sg.IMAGES[:4]), key=lambda r: r["noise_sigma"])
    unmarked = "Watershed (no markers)"

    assert rows[-1][unmarked] > 1.3 * rows[0][unmarked]
    assert rows[-1]["Watershed + markers"] < 0.1 * rows[-1][unmarked]
    # the grid cannot react to noise at all, which is what makes it a control
    assert rows[-1]["Grid tiles (control)"] == rows[0]["Grid tiles (control)"]


def test_asking_for_a_missing_ground_truth_says_what_to_do():
    with pytest.raises(bsds.GroundTruthMissing, match="check_image_reuse|fetch_images"):
        bsds.load_annotations("not_a_real_photograph_name")
