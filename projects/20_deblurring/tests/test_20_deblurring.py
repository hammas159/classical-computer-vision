"""Tests for project 20, deblurring.

Most of these pin a *finding* rather than a number, so that a later change which
quietly reverses one of the project's conclusions fails loudly instead of
rewriting the README by accident.
"""

from __future__ import annotations

import numpy as np
import pytest

import deblurring as db
from shared import synth
from shared.io import to_gray
from shared.metrics import psnr


# --------------------------------------------------------------------------- #
# the setup
# --------------------------------------------------------------------------- #


def test_the_blur_actually_damages_the_image():
    """If the degradation is mild there is nothing to recover and no experiment."""
    clean, blurred, _ = db.make_blurred("castle_gatehouse", "motion", noise_sigma=3.0)
    assert psnr(to_gray(blurred), clean) < 30.0


def test_every_method_returns_the_right_shape_and_no_nans():
    clean, blurred, psf = db.make_blurred("three_owlets", "motion", noise_sigma=3.0)
    for name, fn in db.METHODS.items():
        out = fn(blurred, psf)
        assert out.shape == clean.shape, name
        assert np.isfinite(out).all(), name


def test_the_table_says_which_methods_were_handed_the_kernel():
    """Four of the six get the true PSF and two do not.

    A comparison that mixes those without saying so is not a comparison. The
    flag is in the results table rather than in a footnote.
    """
    rows, _ = db.evaluate_methods(images=db.IMAGES[:2], runs=1)
    flagged = {r["method"]: r["knows_kernel"] for r in rows}
    assert flagged["Richardson-Lucy"] is True
    assert flagged["Wiener"] is True
    assert flagged["Unsharp (no kernel)"] is False
    assert flagged["Do nothing (control)"] is False
    assert sum(flagged.values()) == 4


# --------------------------------------------------------------------------- #
# the blind angle estimator -- it was wrong twice over
# --------------------------------------------------------------------------- #


def test_the_blind_angle_estimator_works_at_angles_that_are_not_multiples_of_45():
    """The bug that made a broken estimator look like a working one.

    It used `cv2.HoughLines` on a thresholded spectrum, which found the FFT's own
    axis-aligned and diagonal structure instead of the sinc stripes — so it
    snapped to multiples of 45 degrees. And it never took the perpendicular, so
    the number it returned was the stripe orientation rather than the motion
    direction.

    Scored against the truth those two errors cancelled at 90 degrees (3.25
    degrees of error, which reads as a working method) and produced 50 to 80
    degrees everywhere else — worse than guessing on a quantity with only 180
    degrees of range.

    The angles below are deliberately off the 45-degree grid.
    """
    rows = db.evaluate_blind_angle(images=db.IMAGES[:6], angles=(15.0, 30.0, 120.0, 160.0))
    for r in rows:
        assert r["mean_abs_error_deg"] < 20.0, r


def test_the_blind_angle_estimator_beats_guessing():
    """A uniform guess over 180 degrees averages 45 degrees of error."""
    rows = db.evaluate_blind_angle(images=db.IMAGES[:6],
                                   angles=(0.0, 30.0, 60.0, 90.0, 135.0))
    assert float(np.mean([r["mean_abs_error_deg"] for r in rows])) < 15.0


# --------------------------------------------------------------------------- #
# the findings
# --------------------------------------------------------------------------- #


def test_the_inverse_filter_is_catastrophically_worse_than_doing_nothing():
    """The textbook's first method, and it is unusable on any real image.

    Division by the transfer function amplifies every frequency the blur
    attenuated — including the noise sitting in those same frequencies, which
    the blur did nothing to. It scores 17 dB *below* leaving the image alone.
    """
    rows, _ = db.evaluate_methods(images=db.IMAGES[:6], runs=1)
    inverse = next(r for r in rows if r["method"].startswith("Inverse"))
    control = next(r for r in rows if r["method"].startswith("Do nothing"))
    assert control["psnr_db"] - inverse["psnr_db"] > 10.0


def test_richardson_lucy_diverges_past_its_optimum():
    """The signature finding: more iterations is not more deblurring.

    RL is a maximum-likelihood iteration with no regulariser, so it keeps going
    until it has explained the *noise* as well as the signal. PSNR rises, peaks
    and falls, and past the peak it heads back toward the do-nothing control.
    """
    rows = sorted(db.sweep_rl_iterations(images=db.IMAGES[:6]),
                  key=lambda r: r["iterations"])
    psnrs = [r["psnr_db"] for r in rows]
    peak = int(np.argmax(psnrs))

    assert 0 < peak < len(psnrs) - 1, "the optimum is at an end, so it is not an optimum"
    assert psnrs[-1] < psnrs[peak] - 1.0        # it really does come back down
    assert psnrs[0] < psnrs[peak]               # and it really did climb


def test_psnr_and_ssim_disagree_about_when_to_stop():
    """And at run time you have neither, which is the practical problem.

    The two metrics put Richardson-Lucy's optimum at different iteration counts.
    Choosing between them needs the clean image, which is the thing being
    reconstructed.
    """
    rows = db.sweep_rl_iterations(images=db.IMAGES[:6])
    best_psnr = max(rows, key=lambda r: r["psnr_db"])["iterations"]
    best_ssim = max(rows, key=lambda r: r["ssim"])["iterations"]
    assert best_psnr != best_ssim


def test_only_the_iterative_method_clearly_beats_doing_nothing():
    """The uncomfortable result, kept rather than tuned away.

    Handed the *true* kernel, three of the four kernel-aware methods score at or
    below the untouched blurred image. Only Richardson-Lucy is clearly ahead.
    """
    rows, _ = db.evaluate_methods(images=db.IMAGES[:6], runs=1)
    control = next(r for r in rows if r["method"].startswith("Do nothing"))
    ahead = [r["method"] for r in rows
             if not r["method"].startswith("Do nothing")
             and r["psnr_db"] > control["psnr_db"] + 1.0]
    assert ahead == ["Richardson-Lucy"]


def test_deblurring_stops_helping_once_the_noise_is_large():
    """The practically useful part: knowing when not to bother.

    Every method's advantage over the control shrinks as noise rises, because
    deblurring amplifies whatever is in the frequencies the blur suppressed and
    at high noise that is mostly noise.
    """
    rows = sorted(db.sweep_noise(images=db.IMAGES[:6]), key=lambda r: r["noise_sigma"])
    gains = [r["Richardson-Lucy"] - r["Do nothing (control)"] for r in rows]
    assert gains[0] > 1.0        # noiseless, deblurring clearly helps
    assert gains[-1] < gains[0]  # and the advantage erodes


def test_wiener_needs_its_parameter_tuned_to_the_noise():
    """One number, and it is the difference between useful and useless.

    Swept across four decades the PSNR curve has an interior optimum; the
    textbook 0.01 is not it, and the extremes are far worse than doing nothing.
    """
    rows = sorted(db.sweep_nsr(images=db.IMAGES[:6]), key=lambda r: r["nsr"])
    psnrs = [r["psnr_db"] for r in rows]
    peak = int(np.argmax(psnrs))
    assert 0 < peak < len(psnrs) - 1
    assert psnrs[0] < psnrs[peak] - 5.0    # too small a ratio is a near-inverse filter
    assert psnrs[-1] < psnrs[peak] - 3.0   # too large just smooths


@pytest.mark.parametrize("kind", ["motion", "defocus"])
def test_both_blur_kinds_are_recoverable_by_something(kind):
    rows, stats = db.evaluate_methods(kind=kind, images=db.IMAGES[:4], runs=1)
    best = max(rows, key=lambda r: r["psnr_db"])
    assert best["psnr_db"] > stats["psnr"]


def test_an_unknown_blur_kind_is_not_silently_treated_as_defocus():
    """`make_blurred` chooses motion for 'motion' and defocus for anything else.

    That is a real trap: a typo produces a defocus blur and a full table of
    plausible numbers about the wrong experiment. The two kernels are different
    enough that the difference is measurable, which is what this checks.
    """
    _, motion, motion_psf = db.make_blurred("owl_in_grass", "motion", noise_sigma=0.0)
    _, defocus, defocus_psf = db.make_blurred("owl_in_grass", "defocus", noise_sigma=0.0)
    assert motion_psf.shape != defocus_psf.shape or not np.allclose(motion_psf, defocus_psf)
    assert psnr(to_gray(motion), to_gray(defocus)) < 35.0
