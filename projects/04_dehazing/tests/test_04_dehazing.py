"""Tests for project 04, dehazing.

The claims here are physical, so most of these check that the code obeys the
model rather than that it produces a particular number.
"""

from __future__ import annotations

import numpy as np
import pytest

import dehazing as dz
from shared import io, synth
from shared.metrics import psnr, rms_contrast, ssim

# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #


def test_haze_moves_every_pixel_toward_the_airlight():
    """The scattering model is a weighted average with A, so nothing can move away."""
    clean = io.sample("rocket")
    hazy, t = synth.add_haze(clean, beta=1.4, airlight=dz.AIRLIGHT)
    a = dz.AIRLIGHT * 255.0
    before = np.abs(clean.astype(np.float64) - a)
    after = np.abs(hazy.astype(np.float64) - a)
    # allow a small tolerance for uint8 rounding
    assert float((after <= before + 1.5).mean()) > 0.99


def test_transmission_falls_as_beta_rises():
    clean = io.sample("coffee")
    mins = []
    for beta in (0.4, 1.0, 2.0, 3.0):
        _, t = synth.add_haze(clean, beta=beta, airlight=dz.AIRLIGHT)
        mins.append(float(t.min()))
    assert mins == sorted(mins, reverse=True)
    # t = exp(-beta*d) with d in [0, 1], so the minimum is exp(-beta)
    _, t = synth.add_haze(clean, beta=2.0, airlight=dz.AIRLIGHT)
    assert float(t.min()) == pytest.approx(np.exp(-2.0), abs=1e-4)


def test_the_oracle_inverts_the_model_almost_exactly():
    """Handed the true transmission, the inversion is arithmetic and should be near-perfect.

    "Near" rather than "exactly" because the hazy image was quantised to uint8
    and the t_min floor clips the deepest haze.
    """
    clean = io.sample("rocket")
    hazy, t = synth.add_haze(clean, beta=1.4, airlight=dz.AIRLIGHT)
    recovered = dz.dehaze_oracle(hazy, dz.AIRLIGHT, t)
    assert psnr(recovered, clean) > 40.0
    assert ssim(recovered, clean) > 0.98


def test_no_method_beats_the_oracle():
    """The oracle is a ceiling. Anything above it means a bug, not a better method."""
    rows, _ = dz.evaluate_methods(beta=1.4, images=("rocket", "coffee"), runs=1)
    oracle = next(r for r in rows if r["method"] == dz.ORACLE_NAME)
    for r in rows:
        if r["method"] != dz.ORACLE_NAME:
            assert r["psnr_db"] < oracle["psnr_db"]


def test_t_min_floor_prevents_the_division_exploding():
    """As t approaches zero the recovery divides by it. The floor is load-bearing."""
    hazy = np.full((32, 32, 3), 200, np.uint8)
    t = np.full((32, 32), 1e-6, np.float32)
    out = dz.recover_scene(hazy, np.array([0.9, 0.9, 0.9], np.float32), t)
    assert np.isfinite(out.astype(np.float64)).all()
    assert out.dtype == np.uint8


# --------------------------------------------------------------------------- #
# the methods
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", list(dz.METHODS))
def test_every_method_returns_rgb_uint8_of_the_same_shape(name):
    clean = io.sample("chelsea")
    hazy, _ = synth.add_haze(clean, beta=1.4, airlight=dz.AIRLIGHT)
    out = dz.METHODS[name](hazy)
    assert out.dtype == np.uint8 and out.shape == hazy.shape


def test_the_physical_methods_beat_the_contrast_control_on_psnr():
    """The project's control, asserted: contrast is not dehazing."""
    rows, _ = dz.evaluate_methods(beta=1.4, images=("rocket", "coffee"), runs=1)
    by_name = {r["method"]: r for r in rows}
    dcp = by_name["DCP + guided refine"]
    clahe = by_name["CLAHE (contrast only)"]
    assert dcp["psnr_db"] > clahe["psnr_db"] + 3.0
    # ...and yet CLAHE wins on contrast, which is exactly why it is in the table
    assert clahe["rms_contrast"] > dcp["rms_contrast"]


def test_guided_refinement_improves_the_transmission_estimate():
    clean = io.sample("rocket")
    hazy, true_t = synth.add_haze(clean, beta=1.4, airlight=dz.AIRLIGHT)
    a = dz.estimate_airlight(hazy)
    blocky = dz.transmission_dcp(hazy, a)
    refined = dz.refine_transmission_guided(hazy, blocky)
    assert dz.transmission_error(refined, true_t) <= dz.transmission_error(blocky, true_t)


def test_retinex_cannot_model_an_additive_veil():
    """Not an implementation failure — a model mismatch, asserted as one.

    Retinex assumes image = illumination x reflectance (multiplicative). Haze is
    additive. It should lose badly to a method that uses the right model.
    """
    rows, _ = dz.evaluate_methods(beta=1.4, images=("rocket", "coffee"), runs=1)
    by_name = {r["method"]: r for r in rows}
    assert by_name["Multi-scale Retinex"]["psnr_db"] < by_name["DCP + guided refine"]["psnr_db"] - 5.0


# --------------------------------------------------------------------------- #
# the finding
# --------------------------------------------------------------------------- #


def test_a_more_accurate_airlight_produces_a_worse_image():
    """The project's central finding, pinned so it cannot be silently "fixed".

    The default estimator is worse at estimating both the airlight and the
    transmission, and better at producing the output. The errors cancel in
    J = (I-A)/t + A. If someone later "improves" the default to the more
    accurate estimator, this test explains what that costs.
    """
    rows = dz.compare_airlight_estimators(images=("rocket", "coffee", "chelsea"))
    by_name = {r["estimator"]: r for r in rows}
    default = by_name["Brightest candidate (default)"]
    median = by_name["Median of candidates"]

    assert median["airlight_error"] < default["airlight_error"]
    assert median["transmission_mae"] < default["transmission_mae"]
    assert median["psnr_db"] < default["psnr_db"]


def test_the_default_airlight_saturates_on_at_least_one_image():
    """The known wart, asserted rather than hidden.

    On `rocket` the estimator picks a launch-pad floodlight and returns
    A = 1.000 against a true 0.88. It is kept because correcting it makes the
    output worse — see the test above.
    """
    clean = io.sample("rocket")
    hazy, _ = synth.add_haze(clean, beta=1.4, airlight=dz.AIRLIGHT)
    a = dz.estimate_airlight(hazy)
    assert float(np.max(a)) > 0.99


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #


def test_sweep_beta_shows_the_ceiling_falling():
    rows = dz.sweep_beta(images=("rocket",), levels=(0.4, 1.4, 3.0))
    ceilings = [r[dz.ORACLE_NAME] for r in rows]
    assert ceilings == sorted(ceilings, reverse=True)
    assert rows[0]["min_transmission"] > rows[-1]["min_transmission"]


def test_dehaze_dispatches_by_name():
    hazy, _ = synth.add_haze(io.sample("coffee"), beta=1.4, airlight=dz.AIRLIGHT)
    assert np.array_equal(dz.dehaze(hazy, "CLAHE (contrast only)"), dz.dehaze_clahe(hazy))
    with pytest.raises(KeyError):
        dz.dehaze(hazy, "not a method")
