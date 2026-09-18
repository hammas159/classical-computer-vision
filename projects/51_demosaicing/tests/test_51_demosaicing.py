"""Tests for project 51, demosaicing.

Most of these pin a *finding* rather than a number, so that a later change which
quietly reverses one of the project's conclusions fails loudly instead of
rewriting the README by accident.
"""

from __future__ import annotations

import numpy as np
import pytest

import demosaicing as dm

from shared.metrics import psnr


@pytest.fixture(scope="module")
def scored():
    return {r["method"]: r for r in dm.evaluate_methods(runs=1)}


@pytest.fixture(scope="module")
def green():
    return {r["method"]: r for r in dm.green_density_argument()}


# --------------------------------------------------------------------------- #
# the mosaic
# --------------------------------------------------------------------------- #


def test_the_mosaic_keeps_exactly_one_channel_per_pixel():
    """Two thirds of the colour is not captured — that is the whole problem."""
    img = dm.load_scene(dm.IMAGES[0])
    for pattern in dm.PATTERNS:
        raw = dm.mosaic(img, pattern)
        assert raw.ndim == 2, pattern
        assert raw.shape == img.shape[:2], pattern

        masks = dm.channel_masks(img.shape[:2], pattern)
        covered = sum(m.astype(int) for m in masks.values())
        assert (covered == 1).all(), f"{pattern}: every pixel has exactly one filter"


def test_the_bayer_pattern_is_half_green():
    for pattern in dm.PATTERNS:
        masks = dm.channel_masks((64, 64), pattern)
        assert masks["G"].mean() == pytest.approx(0.5)
        assert masks["R"].mean() == pytest.approx(0.25)
        assert masks["B"].mean() == pytest.approx(0.25)


def test_the_raw_value_at_each_site_is_the_true_value():
    """The mosaic samples, it does not alter — so a perfect method scores infinity."""
    img = dm.load_scene(dm.IMAGES[0])
    raw = dm.mosaic(img, "RGGB")
    masks = dm.channel_masks(img.shape[:2], "RGGB")

    for channel, key in enumerate("RGB"):
        m = masks[key]
        assert np.array_equal(raw[m], img[..., channel][m]), key


def test_the_pool_spans_the_axis_it_was_ordered_by():
    from shared import io

    values = []
    for name in dm.IMAGES:
        assert name in io.REAL_PHOTOS, name
        values.append(dm.saturated_edge_share(dm.load_scene(name)))

    assert values == sorted(values), "IMAGES should be ordered by saturated-edge share"
    assert values[0] < 0.5, "no control with essentially no saturated edge"
    assert values[-1] > 50.0


# --------------------------------------------------------------------------- #
# the findings
# --------------------------------------------------------------------------- #


def test_the_error_concentrates_on_edges(scored):
    """The project's headline, and the reason whole-image PSNR is not enough.

    Every method scores 2.1 to 2.8 dB worse on edge pixels than over the whole
    frame. Averaging over the flat regions — where interpolating a missing colour
    is trivial — hides the failure almost completely.
    """
    for name, row in scored.items():
        assert row["edge_psnr_db"] < row["psnr_db"], name
        assert row["edge_penalty_db"] > 1.5, name


def test_cross_channel_interpolation_is_what_actually_helps(scored):
    """Using green to guide red and blue is worth nearly 5 dB.

    Green is sampled twice as densely, so its gradients are known twice as well.
    Malvar and VNG use them to place the red and blue estimates; the bilinear
    methods interpolate each channel in isolation and cannot.
    """
    bilinear = scored["Bilinear (OpenCV)"]
    for name in ("Malvar (cross-channel)", "VNG (gradient)"):
        assert scored[name]["psnr_db"] > bilinear["psnr_db"] + 4.0, name
        assert scored[name]["colour_fringing"] < 0.7 * bilinear["colour_fringing"], name


def test_the_edge_aware_flag_does_not_help_on_edges(scored):
    """It produces different pixels from bilinear and scores the same on both metrics.

    Reported because the name promises otherwise, and because the difference
    between it and the cross-channel methods — about 5 dB — is where the real
    gain lives.
    """
    ea = scored["Edge-aware"]
    bilinear = scored["Bilinear (OpenCV)"]

    assert abs(ea["psnr_db"] - bilinear["psnr_db"]) < 0.3
    assert abs(ea["edge_psnr_db"] - bilinear["edge_psnr_db"]) < 0.3
    assert scored["Malvar (cross-channel)"]["edge_psnr_db"] > ea["edge_psnr_db"] + 4.0


def test_the_edge_aware_flag_is_nonetheless_a_different_algorithm():
    """So the equality above is a result and not a mistaken duplicate call."""
    img = dm.load_scene(dm.IMAGES[-1])
    raw = dm.mosaic(img, "RGGB")
    plain = dm.demosaic_opencv(raw)
    aware = dm.demosaic_opencv_ea(raw)

    assert not np.array_equal(plain, aware)
    assert int(np.abs(plain.astype(int) - aware.astype(int)).max()) > 10


def test_every_interpolating_method_collects_the_green_density(green):
    """Twice as many green photosites, and 3.2-3.9 dB for it.

    Nearest neighbour is the control: it copies a neighbour rather than
    interpolating, so having more of them buys it almost nothing.
    """
    for name, row in green.items():
        assert row["G_psnr_db"] > row["R_psnr_db"], name
        assert row["G_psnr_db"] > row["B_psnr_db"], name

    interpolating = [r for n, r in green.items() if n != "Nearest neighbour"]
    assert min(r["green_advantage_db"] for r in interpolating) > 3.0
    assert green["Nearest neighbour"]["green_advantage_db"] < 2.0


def test_interpolation_is_worth_nearly_ten_decibels(scored):
    best = max(scored.values(), key=lambda r: r["psnr_db"])
    nearest = scored["Nearest neighbour"]
    assert best["psnr_db"] - nearest["psnr_db"] > 8.0


def test_the_four_bayer_phases_are_the_same_problem():
    rows = dm.compare_patterns()
    values = [r["psnr_db"] for r in rows]
    assert len(rows) == 4
    assert max(values) - min(values) < 0.5


def test_noise_on_the_mosaic_hurts_before_any_colour_exists():
    rows = sorted(dm.sweep_noise(), key=lambda r: r["noise_sigma"])
    for method in dm.METHODS:
        if method in rows[0]:
            assert rows[-1][method] < rows[0][method], method


# --------------------------------------------------------------------------- #
# the methods themselves
# --------------------------------------------------------------------------- #


def test_every_method_returns_a_full_colour_image_of_the_right_size():
    img = dm.load_scene(dm.IMAGES[0])
    raw = dm.mosaic(img, "RGGB")
    for name, fn in dm.METHODS.items():
        out = fn(raw, "RGGB")
        assert out.shape == img.shape, name
        assert out.dtype == np.uint8, name
        assert psnr(out, img) > 18.0, name


def test_the_ground_truth_is_the_image_the_mosaic_was_made_from():
    """A rare case where the 'before' image is exactly what is being recovered."""
    for name in dm.IMAGES[:3]:
        img = dm.load_scene(name)
        raw = dm.mosaic(img, "RGGB")
        assert psnr(dm.demosaic_malvar(raw, "RGGB"), img) > 25.0, name
