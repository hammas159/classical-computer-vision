"""Tests for project 23, frequency-domain filtering.

Most of these pin a *finding* rather than a number, so that a later change which
quietly reverses one of the project's conclusions fails loudly instead of
rewriting the README by accident.
"""

from __future__ import annotations

import numpy as np
import pytest

import fft_filtering as ff
from shared.io import to_gray
from shared.metrics import psnr

MASKS = {
    "Ideal": lambda shape, c: ff.mask_ideal(shape, c),
    "Butterworth (n=2)": lambda shape, c: ff.mask_butterworth(shape, c, 2),
    "Butterworth (n=8)": lambda shape, c: ff.mask_butterworth(shape, c, 8),
    "Gaussian": lambda shape, c: ff.mask_gaussian(shape, c),
}


# --------------------------------------------------------------------------- #
# the transform
# --------------------------------------------------------------------------- #


def test_the_transform_round_trips():
    """If the FFT and its inverse do not return the image, nothing else counts."""
    img = to_gray(ff.load_scene("stone_wellhead"))
    assert psnr(ff.ifft(ff.fft(img)), img) > 50.0


def test_an_all_pass_mask_changes_nothing():
    img = to_gray(ff.load_scene("fjord_harbour"))
    ones = np.ones(img.shape, np.float32)
    assert psnr(ff.apply_mask(img, ones), img) > 50.0


@pytest.mark.parametrize("name", list(MASKS))
def test_every_mask_is_bounded_and_centred(name):
    mask = MASKS[name]((128, 128), 20.0)
    assert 0.0 <= mask.min() and mask.max() <= 1.0 + 1e-6
    assert mask[64, 64] == pytest.approx(mask.max(), abs=1e-3)  # DC passes


# --------------------------------------------------------------------------- #
# ringing -- measured where it can be measured
# --------------------------------------------------------------------------- #


def test_ringing_rises_with_how_sharp_the_cutoff_is():
    """The textbook ordering, on a step edge: ideal rings, Gaussian does not.

    Butterworth is the knob between them and its ringing grows with order —
    0.0336 at n=2 and 0.0841 at n=8, approaching the ideal filter's 0.0908.
    """
    scores = {name: ff.ringing_on_step(fn, 20.0) for name, fn in MASKS.items()}
    assert scores["Gaussian"]["overshoot"] == 0.0
    assert scores["Butterworth (n=2)"]["overshoot"] > 0.0
    assert scores["Butterworth (n=8)"]["overshoot"] > scores["Butterworth (n=2)"]["overshoot"]
    assert scores["Ideal"]["overshoot"] > scores["Butterworth (n=8)"]["overshoot"]


def test_the_oscillation_count_needs_its_amplitude_floor():
    """Without it the count ranked the non-ringing filter as the worst ringer.

    A Gaussian-filtered step is almost perfectly flat away from the edge, and
    its floating-point wiggle produced 147 turns against the ideal filter's 42.
    `RINGING_FLOOR` is one uint8 level — below that it could not be seen anyway.
    """
    assert ff.RINGING_FLOOR == pytest.approx(1 / 255)
    assert ff.ringing_on_step(MASKS["Gaussian"], 20.0)["oscillations"] == 0
    assert ff.ringing_on_step(MASKS["Ideal"], 20.0)["oscillations"] > 0


def test_the_photograph_ringing_score_does_not_separate_the_filters():
    """Kept as a documented failure, not as a result.

    On a natural image the band beside one edge is full of other edges, so the
    score is dominated by ordinary blur error. It ranks Gaussian worst, which is
    backwards. This test pins that it is still backwards, so nobody quotes it.
    """
    rows = {r["filter"]: r["ringing"]
            for r in ff.evaluate_lowpass(cutoff=40.0, images=ff.IMAGES[:4], runs=1)}
    assert rows["Gaussian"] > rows["Ideal"]


# --------------------------------------------------------------------------- #
# the findings
# --------------------------------------------------------------------------- #


def test_the_notch_filter_beats_the_spatial_median_by_a_mile():
    """The project's headline, and the one thing only the frequency domain can do.

    Periodic interference is a handful of isolated spikes in the spectrum and a
    spatially unbounded mess in the image. A notch removes 14.6 dB of it; a 5x5
    median — the obvious spatial answer — recovers 1.7 dB.
    """
    rows, _ = ff.evaluate_notch(images=ff.IMAGES[:6])
    scored = {r["method"]: r["psnr_db"] for r in rows}
    noisy = scored["Noisy input"]

    assert scored["Notch, blind peaks"] > noisy + 10.0
    assert scored["Median filter (spatial)"] < noisy + 4.0
    assert scored["Notch, blind peaks"] > scored["Median filter (spatial)"] + 10.0


def test_the_blind_peak_finder_matches_its_own_oracle():
    """Rare enough in this repository to be worth a test of its own.

    The blind notch locates the interference peaks from the spectrum alone and
    lands on exactly the frequencies the noise was added at — 0.0 px error — so
    it scores identically to the version handed the true peaks.
    """
    rows, extra = ff.evaluate_notch(images=ff.IMAGES[:6])
    blind = next(r["psnr_db"] for r in rows if "blind" in r["method"])
    oracle = next(r["psnr_db"] for r in rows if "oracle" in r["method"])

    assert extra["peak_localisation_error_px"] == 0.0
    assert blind == pytest.approx(oracle, abs=0.01)


def test_homomorphic_filtering_is_scored_on_brightness_unless_matched():
    """The same trap as the high-boost row in project 16.

    `gamma_low = 0.5` halves the low-frequency band of the *log* image, and the
    DC term is in that band — so the result comes back at about a third of the
    original brightness. Raw, it scores below the uneven input it was meant to
    fix. Matched, it is clearly ahead.
    """
    rows = {r["method"]: r for r in ff.evaluate_homomorphic(images=ff.IMAGES[:6])}
    homo, uneven = rows["Homomorphic"], rows["Uneven input"]

    assert homo["psnr_db"] < uneven["psnr_db"]                 # raw: looks broken
    assert homo["psnr_matched_db"] > uneven["psnr_db"] + 0.5   # matched: it works
    assert homo["psnr_matched_db"] - homo["psnr_db"] > 4.0     # the brightness gap


def test_gaussian_is_the_best_low_pass_and_ideal_the_worst():
    rows = {r["filter"]: r["psnr_db"]
            for r in ff.evaluate_lowpass(cutoff=40.0, images=ff.IMAGES[:6], runs=1)}
    assert max(rows, key=rows.get) == "Gaussian"
    assert rows["Ideal"] < rows["Gaussian"]


def test_a_lower_cutoff_always_costs_more():
    rows = sorted(ff.sweep_cutoff(images=ff.IMAGES[:4]), key=lambda r: r["cutoff"])
    for key in rows[0]:
        if key == "cutoff":
            continue
        scores = [r[key] for r in rows]
        assert scores == sorted(scores), key


def test_the_detected_peaks_are_not_the_dc_term():
    """The trap in blind peak finding: DC is the brightest point in every
    spectrum, so a naive argmax finds the image's own mean and notches it out."""
    clean = to_gray(ff.load_scene("tower_and_spire"))
    noisy, _ = ff.add_periodic_noise(clean, 40, 25, 0.25)
    h, w = clean.shape
    for y, x in ff.detect_noise_peaks(noisy):
        assert abs(y - h // 2) > 5 or abs(x - w // 2) > 5
