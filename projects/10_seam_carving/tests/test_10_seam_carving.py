"""Tests for project 10, seam carving.

Most of these pin a *finding* rather than a number, so that a later "improvement"
which quietly reverses one of the project's conclusions fails loudly instead of
rewriting the README by accident.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import seam_carving as sc
from shared import io

IMAGES = ("coffee", "chelsea")


# --------------------------------------------------------------------------- #
# the algorithm
# --------------------------------------------------------------------------- #


def test_the_dp_finds_the_true_minimum_path_on_a_hand_made_case():
    """A 3x3 grid with one obviously cheap column, checked against arithmetic."""
    e = np.array(
        [
            [1.0, 9.0, 9.0],
            [9.0, 1.0, 9.0],
            [9.0, 9.0, 1.0],
        ],
        np.float32,
    )
    M, _ = sc.cumulative_energy(e)
    # the cheap diagonal costs 1 + 1 + 1
    assert float(M[-1].min()) == pytest.approx(3.0)
    assert list(sc.find_seam(e)) == [0, 1, 2]


def test_a_seam_is_connected():
    """Adjacent rows may differ by at most one column — that is what makes it a seam."""
    img = io.sample("coffee")
    seam = sc.find_seam(sc.energy_gradient(img))
    assert len(seam) == img.shape[0]
    assert np.abs(np.diff(seam)).max() <= 1


def test_a_seam_stays_inside_the_image():
    img = io.sample("chelsea")
    seam = sc.find_seam(sc.energy_gradient(img))
    assert seam.min() >= 0 and seam.max() < img.shape[1]


def test_remove_seam_matches_the_obvious_slow_implementation():
    """Guards the vectorised rewrite against the per-row loop it replaced."""
    img = io.sample("coffee")[:60, :80]
    seam = sc.find_seam(sc.energy_gradient(img))
    fast = sc.remove_seam(img, seam)
    slow = np.stack(
        [np.concatenate([img[i, : seam[i]], img[i, seam[i] + 1 :]]) for i in range(img.shape[0])]
    )
    assert np.array_equal(fast, slow)
    assert fast.shape == (img.shape[0], img.shape[1] - 1, 3)


def test_remove_seam_works_on_a_2d_mask_too():
    """The tracked mask goes through the identical removals, so it must."""
    mask = np.zeros((40, 50), np.uint8)
    mask[10:30, 10:40] = 255
    seam = np.clip(np.arange(40) % 3, 0, 49).astype(np.int32)
    out = sc.remove_seam(mask, seam)
    assert out.shape == (40, 49)


def test_carving_removes_exactly_the_requested_columns():
    img = io.sample("coffee")
    out, _ = sc.carve(img, img.shape[1] - 7)
    assert out.shape[1] == img.shape[1] - 7
    assert out.shape[0] == img.shape[0]


def test_carving_a_mask_alongside_keeps_them_the_same_width():
    img = io.sample("chelsea")
    mask, _ = sc.subject_region(img)
    out, out_mask = sc.carve(img, img.shape[1] - 20, track=mask)
    assert out.shape[1] == out_mask.shape[1]


# --------------------------------------------------------------------------- #
# the scene
# --------------------------------------------------------------------------- #


def test_the_region_of_interest_is_found_not_pasted():
    """Nothing is composited into the photograph — asserted directly."""
    img = io.sample("coffee")
    mask, (x, y, bw, bh) = sc.subject_region(img)
    assert mask.shape == img.shape[:2]
    assert 0 <= x and x + bw <= img.shape[1]
    assert 0 <= y and y + bh <= img.shape[0]
    assert float((mask > 0).mean()) == pytest.approx(0.16, abs=0.02)


def test_the_region_really_is_the_highest_energy_box():
    """The ROI's mean energy must exceed the image's, or it is not a subject."""
    img = io.sample("astronaut")
    mask, _ = sc.subject_region(img)
    e = sc.energy_gradient(img)
    assert float(e[mask > 0].mean()) > 1.5 * float(e.mean())


def test_a_uniform_region_would_be_invisible_to_a_gradient_energy():
    """Why the pasted solid square this project started with measured nothing.

    A flat patch has zero gradient inside it, so a gradient energy cannot see it
    and seams run straight through the middle. The experiment was asking whether
    seam carving protects a region it is structurally blind to.
    """
    img = io.sample("coffee").copy()
    img[100:200, 100:200] = (220, 40, 40)
    e = sc.energy_gradient(img)
    interior = e[120:180, 120:180]
    assert float(interior.max()) == pytest.approx(0.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# the findings
# --------------------------------------------------------------------------- #


def test_the_energy_function_matters_far_less_than_carving_at_all():
    """The project's first finding, pinned — and rescaled by better data.

    The direction survives: the one-line choice write-ups agonise over is
    smaller than the choice they do not discuss. **The size of it did not.**

    On the original image set — four scikit-image samples of roughly the same
    layout — the four energy functions spanned 0.5 points while carving-vs-not
    spanned 11.6, a ratio of 23x. On six photographs chosen for *shape* the
    energies span 5.1 points against 16.4, a ratio of **3.2x**. The energy
    choice matters ten times more than the old images suggested, because the
    Laplacian does badly on architecture and there was no architecture in the
    old set.

    The threshold here is 2x rather than 10x, and that is a weakening of the
    claim, recorded rather than quietly deleted.
    """
    rows = sc.evaluate_energies(reduction=0.20, images=sc.IMAGES)
    carving = [r for r in rows if not r["energy"].startswith("Plain")]
    control = next(r for r in rows if r["energy"].startswith("Plain"))

    kept = [r["subject_kept"] for r in carving]
    spread = max(kept) - min(kept)
    gap = max(kept) - control["subject_kept"]
    assert gap > 2.0 * spread


def test_carving_wins_both_objectives_on_every_shape():
    """A finding that REVERSED when the images were chosen properly.

    This used to assert that carving wins its own objective (retained gradient
    energy) on every image and the perceptual one (region retention) on only
    *most* — 3 of 4 — and that gap was written up as the honest summary of the
    method.

    It was an artefact of the image set. `coffee`, `rocket`, `chelsea` and
    `astronaut` are all roughly centre-weighted and similarly proportioned, and
    one of them happened to have its subject spread wide enough that carving
    lost. On six photographs chosen for shape, carving wins **both** objectives
    on **all six**, by +12.2 to +19.8 points.

    The honest summary changes with it: seam carving is not unreliable at
    protecting a region. It is expensive, and its advantage depends on how much
    low-energy space the photograph contains — which the shape sweep shows
    directly.
    """
    rows = sc.per_image(reduction=0.20)
    energy_wins = sum(1 for r in rows if r["carved_energy_kept"] > r["rescale_energy_kept"])
    region_wins = sum(1 for r in rows if r["advantage"] > 0)
    assert energy_wins == len(rows)
    assert region_wins == len(rows)


def test_the_advantage_varies_by_a_factor_across_shapes():
    """What replaced "an image where carving loses".

    There is no longer an image where it loses, but the spread is large and it
    tracks shape rather than subject: a boat along a sea wall leaves a lot of
    low-energy water to remove, a stone arch leaves very little.
    """
    rows = sc.per_image(reduction=0.20)
    advantages = [r["advantage"] for r in rows]
    assert min(advantages) > 0
    assert max(advantages) > 1.5 * min(advantages)


def test_carving_costs_orders_of_magnitude_more_than_a_rescale():
    rows = sc.compare_cost(reduction=0.20, images=IMAGES)
    carve_row, plain_row, diff = rows
    assert carve_row["median_ms"] > 100 * plain_row["median_ms"]
    assert diff["subject_kept"] > 0


def test_the_plain_rescale_control_is_exactly_one_minus_the_reduction():
    """The control is arithmetic, not a measurement — which is what makes it a control."""
    for red in (0.10, 0.20, 0.45):
        rows = sc.sweep_reduction(images=IMAGES, levels=(red,))
        assert rows[0]["rescale_aspect"] == pytest.approx(1 - red, abs=0.01)
        assert rows[0]["rescale_subject_kept"] == pytest.approx(1 - red, abs=0.01)


def test_carving_degrades_as_the_reduction_grows():
    rows = sc.sweep_reduction(images=IMAGES, levels=(0.05, 0.30, 0.60))
    kept = [r["carved_subject_kept"] for r in rows]
    assert kept == sorted(kept, reverse=True)
    # and it stays ahead of the rescale the whole way
    for r in rows:
        assert r["carved_energy_kept"] > r["rescale_energy_kept"]


# --------------------------------------------------------------------------- #
# entry points
# --------------------------------------------------------------------------- #


def test_resize_pair_returns_matching_widths():
    img = io.sample("coffee")
    carved, rescaled, mask, cm, rm = sc.resize_pair(img, 0.2)
    assert carved.shape[1] == rescaled.shape[1] == cm.shape[1] == rm.shape[1]


def test_region_shape_of_an_empty_mask_is_not_a_crash():
    bw, bh, aspect = sc.region_shape(np.zeros((10, 10), np.uint8))
    assert bw == 0 and bh == 0 and np.isnan(aspect)


def test_seam_overlay_marks_one_pixel_per_row_per_seam():
    img = io.sample("chelsea")
    overlay = sc.seam_overlay(img, n=5)
    assert overlay.shape == img.shape
    marked = np.all(overlay == (255, 0, 0), axis=2)
    # 5 seams x one pixel per row, allowing for pixels that were already red
    assert marked.sum() >= 4 * img.shape[0]
