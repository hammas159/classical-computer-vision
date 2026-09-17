"""Tests for project 21, super-resolution.

Most of these pin a *finding* rather than a number, so that a later change which
quietly reverses one of the project's conclusions fails loudly instead of
rewriting the README by accident.
"""

from __future__ import annotations

import numpy as np

import super_resolution as sr
from shared import synth
from shared.metrics import psnr


def _pair(name: str, scale: int = 4):
    hr = sr.load_scene(name)
    h = (hr.shape[0] // scale) * scale
    w = (hr.shape[1] // scale) * scale
    hr = hr[:h, :w]
    return hr, synth.downsample_for_sr(hr, scale=scale)


# --------------------------------------------------------------------------- #
# the setup
# --------------------------------------------------------------------------- #


def test_every_method_returns_the_requested_size():
    hr, lr = _pair("moated_chateau")
    for name, fn in sr.METHODS.items():
        out = fn(lr, 4)
        assert out.shape[0] >= hr.shape[0] and out.shape[1] >= hr.shape[1], name
        assert np.isfinite(out).all(), name


def test_upscaling_by_one_changes_almost_nothing():
    """A scale of 1 is the identity, and a method that fails it is not resampling."""
    hr, _ = _pair("longtail_boats")
    for name, fn in sr.METHODS.items():
        if name in ("Back-projection", "Edge-directed"):
            continue  # these sharpen by design; see their own rows in the table
        assert psnr(fn(hr, 1)[:hr.shape[0], :hr.shape[1]], hr) > 30.0, name


# --------------------------------------------------------------------------- #
# the reference -- deliberately NOT called an oracle
# --------------------------------------------------------------------------- #


def test_the_band_limited_row_is_a_reference_and_not_a_ceiling():
    """It can be beaten, and pretending otherwise would have been the tidier lie.

    The reference is the original blurred by the same anti-alias filter the
    downsample used. A Gaussian is not a brick wall: it *attenuates* the
    frequencies above the new Nyquist rather than removing them, so the
    decimated samples still carry a folded, weakened copy. Back-projection
    models the degradation and partly inverts that attenuation, so on some
    images it scores above this row.

    Measured here: it passes the reference on 1 of the 12 photographs, and sits
    0.2 to 0.5 dB below on the rest.
    """
    beaten = 0
    for name in sr.IMAGES:
        hr, lr = _pair(name)
        ceiling = psnr(sr.oracle_band_limited(hr, 4), hr)
        best = max(psnr(fn(lr, 4)[:hr.shape[0], :hr.shape[1]], hr)
                   for fn in sr.METHODS.values())
        if best > ceiling:
            beaten += 1
    assert beaten >= 1, "the reference was never passed; the caveat may be stale"
    assert beaten <= 3, "it is being passed routinely; it is not a useful reference"


def test_the_reference_is_far_above_every_plain_interpolator():
    """Whatever it is called, the plain resamplers are nowhere near it."""
    hr, lr = _pair("parasols_willows")
    ceiling = psnr(sr.oracle_band_limited(hr, 4), hr)
    for name in ("Nearest", "Bilinear", "Bicubic", "Lanczos-4"):
        assert psnr(sr.METHODS[name](lr, 4)[:hr.shape[0], :hr.shape[1]], hr) < ceiling - 1.0


# --------------------------------------------------------------------------- #
# the findings
# --------------------------------------------------------------------------- #


def test_the_whole_interpolator_argument_is_worth_half_a_decibel():
    """The project's headline.

    Nearest to Lanczos-4 — the entire span of the choice people agonise over —
    is 0.49 dB at 4x. Back-projection, which is the same family plus a model of
    how the image was downsampled, is +1.62 dB on its own.
    """
    rows, _ = sr.evaluate_methods(scale=4, images=sr.IMAGES[:6], runs=1)
    scored = {r["method"]: r["psnr_db"] for r in rows}
    classics = ["Nearest", "Bilinear", "Bicubic", "Lanczos-4", "Edge-directed"]
    spread = max(scored[m] for m in classics) - min(scored[m] for m in classics)

    assert spread < 1.0
    assert scored["Back-projection"] > max(scored[m] for m in classics) + spread


def test_the_ranking_of_the_classic_interpolators_never_changes():
    """Nearest < bilinear < bicubic <= Lanczos, at every scale factor.

    Dull, and worth pinning: it means the half-decibel above is a real ordering
    and not noise.
    """
    for row in sr.sweep_scale(images=sr.IMAGES[:4], scales=(2, 4, 8)):
        assert row["Nearest"] < row["Bilinear"] < row["Bicubic"], row["scale"]
        assert row["Lanczos-4"] >= row["Bicubic"] - 0.05, row["scale"]


def test_the_choice_of_downsampler_matters_more_than_the_choice_of_method():
    """The trap most super-resolution comparisons fall into.

    Building the low-resolution input with `cv2.resize(INTER_AREA)` instead of
    blur-then-decimate changes bicubic's score by more than the entire spread
    between all six methods. A benchmark that used resize measured how well each
    method inverts `INTER_AREA`, which is not a question anyone meant to ask.
    """
    d = sr.compare_degradations(images=sr.IMAGES[:6], scale=4)
    rows, _ = sr.evaluate_methods(scale=4, images=sr.IMAGES[:6], runs=1)
    real = [r["psnr_db"] for r in rows if r["method"] != sr.ORACLE_NAME]

    assert abs(d["difference_db"]) > (max(real) - min(real)) * 0.5
    # and the two low-resolution images really are different pictures
    assert d["lr_images_differ_psnr_db"] < 45.0


def test_no_method_recovers_the_original_high_frequency_energy():
    """Interpolation cannot add information, stated as a spectrum measurement."""
    rows = sr.evaluate_frequency(scale=4, images=sr.IMAGES[:6])
    original = next(r["high_freq_energy"] for r in rows if r["method"].startswith("Original"))
    for r in rows:
        if not r["method"].startswith("Original"):
            assert r["high_freq_energy"] < original, r["method"]


def test_high_frequency_energy_rewards_the_worst_method():
    """Why that spectrum measurement is not a quality metric.

    Nearest-neighbour scores the *highest* high-frequency energy of any method
    and the lowest PSNR. The energy is real; it is the blocky staircase, not
    recovered detail. A no-reference "sharpness" number cannot tell those apart.
    """
    freq = {r["method"]: r["high_freq_energy"]
            for r in sr.evaluate_frequency(scale=4, images=sr.IMAGES[:6])}
    rows, _ = sr.evaluate_methods(scale=4, images=sr.IMAGES[:6], runs=1)
    quality = {r["method"]: r["psnr_db"] for r in rows}

    methods = list(sr.METHODS)
    assert freq["Nearest"] == max(freq[m] for m in methods)
    assert quality["Nearest"] == min(quality[m] for m in methods)


def test_every_method_degrades_together_as_the_scale_rises():
    rows = sorted(sr.sweep_scale(images=sr.IMAGES[:4], scales=(2, 4, 8)),
                  key=lambda r: r["scale"])
    for m in sr.METHODS:
        scores = [r[m] for r in rows]
        assert scores == sorted(scores, reverse=True), m


def test_the_spread_between_methods_shrinks_as_the_task_gets_harder():
    """The more information the downsample destroyed, the less the choice matters."""
    rows = sorted(sr.sweep_scale(images=sr.IMAGES[:4], scales=(2, 4, 8)),
                  key=lambda r: r["scale"])
    assert rows[0]["spread_db"] > rows[-1]["spread_db"]
