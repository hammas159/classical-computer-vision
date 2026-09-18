"""Tests for project 42, image registration.

Most of these pin a *finding* rather than a number, so that a later change which
quietly reverses one of the project's conclusions fails loudly instead of
rewriting the README by accident.

Two exist because the project's own text was wrong before it was measured:
`test_ecc_returns_the_shift_and_not_its_negative` (the same sign inversion that
project 40 had) and `test_the_hanning_window_is_what_breaks_inverted_pairs`,
which contradicts the warning this module's docstring used to carry.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import registration as rg


@pytest.fixture(scope="module")
def modality():
    return {r["modality"]: r for r in rg.sweep_modality()}


@pytest.fixture(scope="module")
def clean():
    return {r["method"]: r for r in rg.evaluate_methods(runs=1)}


# --------------------------------------------------------------------------- #
# the harness
# --------------------------------------------------------------------------- #


def test_ecc_returns_the_shift_and_not_its_negative():
    """A negated translation is exactly twice the truth away, and looks plausible.

    Before this, ECC reported 16.12 px of error on a (7, 4) shift of an image
    against *itself* — which reads as "ECC cannot align identical images" rather
    than as a sign convention.
    """
    reference, moving, truth = rg.make_pair(rg.IMAGES[-1], dx=7.0, dy=4.0)
    warp, cc = rg.register_ecc(reference, moving)

    assert cc > 0.99
    dx, dy = rg.extract_shift("ECC", warp)
    assert dx == pytest.approx(truth[0], abs=0.1)
    assert dy == pytest.approx(truth[1], abs=0.1)


def test_the_truth_is_the_transform_that_was_applied():
    for name in rg.IMAGES[:3]:
        reference, moving, truth = rg.make_pair(name, dx=5.0, dy=-3.0)
        assert truth == (5.0, -3.0)
        assert reference.shape == moving.shape


def test_a_crop_pair_really_has_different_content_at_its_borders():
    """Which is the condition the Hanning window is supposed to be for."""
    reference, moving, truth = rg.crop_pair(rg.IMAGES[0], dx=17, dy=11)
    assert reference.shape == moving.shape
    assert truth == (17.0, 11.0)
    assert not np.array_equal(reference[:, :5], moving[:, :5])


def test_the_pool_spans_the_axis_it_was_selected_on():
    from shared import io

    values = []
    for name in rg.IMAGES:
        assert name in io.REAL_PHOTOS, name
        values.append(rg.texture_energy(rg.load_scene(name)))

    assert len(set(rg.IMAGES)) == len(rg.IMAGES)
    assert values == sorted(values), "IMAGES should be ordered by texture energy"
    assert max(values) > 2.5 * min(values)


# --------------------------------------------------------------------------- #
# the findings
# --------------------------------------------------------------------------- #


def test_on_matched_intensities_the_methods_do_not_differ(clean):
    """Which is why the comparison cannot stop there.

    All four land inside 0.013 px on an image aligned against a shifted copy of
    itself. Any ranking drawn from this table is noise.
    """
    for name, row in clean.items():
        assert row["success_rate"] == 1.0, name
        assert row["mean_error_px"] < 0.05, name
    assert max(r["mean_error_px"] for r in clean.values()) < 0.05


def test_only_mutual_information_survives_a_non_monotonic_remap(modality):
    """The multi-modal case, and the only one MI is needed for.

    Gamma is monotonic and everything handles it. A non-monotonic map destroys
    any linear *or* rank relationship, and MI is still exact where the others are
    tens of pixels out.
    """
    for row in modality.values():
        assert row["Mutual information"] == pytest.approx(0.0, abs=1e-6), row["modality"]

    gamma = modality["gamma"]
    for name in rg.METHODS:
        assert gamma[name] < 0.1, name

    hard = modality["synthetic_mri"]
    others = [hard[m] for m in rg.METHODS if m != "Mutual information"]
    assert min(v for v in others if np.isfinite(v)) > 10.0


def test_mutual_information_costs_two_orders_of_magnitude_for_that(clean):
    mi = clean["Mutual information"]["median_ms"]
    phase = clean["Phase correlation"]["median_ms"]
    assert mi > 100 * phase


def test_ecc_does_not_converge_at_all_on_inverted_intensities(modality):
    """A normalised *linear* correlation reads a perfect negative as no match."""
    assert not np.isfinite(modality["inverted"]["ECC"])
    assert np.isfinite(modality["same"]["ECC"])


def test_the_hanning_window_is_what_breaks_inverted_pairs():
    """The opposite of what this project's docstring used to claim.

    Multiplying by a window is multiplication by a shape: ``(255 - I) * w`` is
    ``255*w - I*w``, so the window's own smooth profile enters the spectrum at
    255 times the amplitude of anything in the picture. Remove it and phase
    correlation handles inversion almost perfectly, because a sign flip is only
    a phase flip.
    """
    rows = rg.inversion_per_image()
    assert len(rows) == len(rg.IMAGES)

    # without the window it works on every photograph, to a tenth of a pixel
    for row in rows:
        assert row["without_hanning_px"] < 0.2, row["image"]

    # with it, most fail by hundreds of pixels — and which ones is unpredictable,
    # which is worse than failing consistently. Three of the twelve come through.
    catastrophic = [r for r in rows if r["with_hanning_px"] > 10.0]
    assert len(catastrophic) >= 8
    assert max(r["with_hanning_px"] for r in rows) > 100.0
    assert float(np.mean([r["without_hanning_px"] for r in rows])) < 0.1


def test_the_hanning_window_earns_nothing_on_ordinary_pairs():
    """Including on crop pairs, where the textbook says it should earn most."""
    ordinary = rg.windowing_effect()
    for row in ordinary:
        assert abs(row["with_hanning_px"] - row["without_hanning_px"]) < 0.5, row

    crops = rg.windowing_on_crop_pairs()
    with_w = float(np.mean([r["with_hanning_px"] for r in crops]))
    without_w = float(np.mean([r["without_hanning_px"] for r in crops]))
    assert with_w < 1.0 and without_w < 1.0
    assert abs(with_w - without_w) < 0.1


def test_phase_correlation_cannot_represent_rotation_and_ecc_recovers_it():
    rows = sorted(rg.rotation_capability(), key=lambda r: r["rotation_deg"])
    for row in rows:
        assert row["phase_can_represent_rotation"] is False
        assert row["ecc_angle_error_deg"] < 0.05, row["rotation_deg"]

    assert rows[-1]["phase_translation_error_px"] > 10 * rows[1]["phase_translation_error_px"]


def test_noise_is_not_what_limits_any_of_these_methods():
    rows = sorted(rg.sweep_noise(), key=lambda r: r["noise_sigma"])
    for name in rg.METHODS:
        assert rows[-1][name] < 0.1, name


# --------------------------------------------------------------------------- #
# mutual information itself
# --------------------------------------------------------------------------- #


def test_mutual_information_is_blind_to_the_form_of_the_relationship():
    """An image against its own inverse has the same MI as against itself."""
    img = rg.load_scene(rg.IMAGES[0])
    same = rg.mutual_information(img, img)
    inverted = rg.mutual_information(img, 255 - img)

    assert same > 4.0
    assert inverted == pytest.approx(same, rel=1e-6)

    unrelated = rg.mutual_information(img, rg.load_scene(rg.IMAGES[-1]))
    assert unrelated < same / 2


def test_the_mi_landscape_peaks_at_the_true_alignment():
    grid, truth = rg.mi_landscape(image=rg.IMAGES[-1], modality="inverted", search=10)
    iy, ix = np.unravel_index(int(np.argmax(grid)), grid.shape)
    dy, dx = iy - 10, ix - 10
    assert abs(dx - truth[0]) <= 1.0 and abs(dy - truth[1]) <= 1.0
