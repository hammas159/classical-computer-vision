"""Tests for project 16, sharpening.

Most of these pin a *finding* rather than a number, so that a later change which
quietly reverses one of the project's conclusions fails loudly instead of
rewriting the README by accident.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import sharpening as sh
from shared.metrics import psnr


# --------------------------------------------------------------------------- #
# the operators
# --------------------------------------------------------------------------- #


def test_the_two_laplacian_signs_are_the_same_operator():
    """A 4-centre kernel added and a -4-centre kernel subtracted are identical.

    They are presented as two methods in every textbook and are one method. The
    only way to get this wrong is to mix the signs, which is what the third
    variant does deliberately.
    """
    img = sh.load_scene("lionesses")
    assert np.array_equal(sh.sharpen_laplacian_add(img), sh.sharpen_laplacian_sub(img))


def test_acutance_cannot_tell_a_sharpener_from_its_sign_error():
    """The mistake worth having a row for — and what it exposes about the metric.

    `sharpen_laplacian_wrong_sign` mixes the kernel sign with the combination
    sign, which is the classic way to get this wrong. It produces a visibly
    worse image: on the full set SSIM falls from **0.689 to 0.339** and PSNR
    from 23.29 to 22.48.

    **On a sharp input it still raises acutance.** The expectation going in was
    that the wrong sign simply blurs, and on a sharp photograph it does not — it
    leaves an inverted rim either side of every contour, whose gradient
    magnitude is *higher* than the original's. So the no-reference metric that
    everyone reaches for to say "this looks sharper" cannot distinguish a
    correct sharpener from a sign error; only a comparison against the truth
    can.

    On an already-blurred input it does soften further (0.090 against the
    blurred input's 0.114), which is where the folklore comes from. This test
    pins the sharp-input case, because that is the one that fools the metric.
    """
    img = sh.load_scene("deer_water")
    base = sh.acutance(img)
    right = sh.sharpen_laplacian_add(img)
    wrong = sh.sharpen_laplacian_wrong_sign(img)

    # both raise apparent sharpness above the original
    assert sh.acutance(wrong) > base
    assert sh.acutance(right) > sh.acutance(wrong)
    # but only a reference metric sees that one of them is damage
    assert psnr(wrong, img) < psnr(right, img)
    from shared.metrics import ssim

    assert ssim(wrong, img) < ssim(right, img) - 0.1


def test_sharpening_a_flat_image_changes_nothing():
    flat = np.full((64, 64, 3), 120, np.uint8)
    for name, fn in sh.METHODS.items():
        if name.startswith("High-boost"):
            continue  # documented to scale flat regions; see its own test
        out = fn(flat)
        assert np.abs(out.astype(int) - 120).max() <= 1, name


def test_high_boost_halves_a_flat_region():
    """Not a bug — the textbook formula, and the reason the table needs two columns.

    In a flat region `blur(I) == I`, so `A*I - blur(I)` is `(A-1)*I`. At the
    usual A = 1.5 that is half brightness, and raw PSNR then scores the
    brightness change rather than anything about sharpening.
    """
    flat = np.full((64, 64, 3), 200, np.uint8)
    out = sh.sharpen_high_boost(flat, boost=1.5)
    assert float(out.mean()) == pytest.approx(100, abs=3)


def test_matching_the_mean_recovers_what_brightness_cost():
    img = sh.load_scene("elk_water")
    out = sh.sharpen_high_boost(img)
    assert psnr(sh.match_mean(out, img), img) > psnr(out, img) + 10.0


# --------------------------------------------------------------------------- #
# the oracle
# --------------------------------------------------------------------------- #


def test_the_oracle_is_scored_on_the_same_thing_as_the_methods():
    """It used to deconvolve in grayscale and be scored against a grey reference.

    A grayscale reconstruction has one third as much to get wrong, so its PSNR
    was not comparable with any other row in the table. It now runs per channel.
    """
    img = sh.load_scene("lionesses")
    k = cv2.getGaussianKernel(9, 1.5)
    out = sh.deconvolve_oracle(img, (k @ k.T).astype(np.float32))
    assert out.shape == img.shape


def test_the_oracle_is_a_ceiling_over_every_method():
    """Per image, not on the average — an average hid a failure.

    The oracle is handed the true kernel and the clean image. Nothing that has
    neither should beat it. On `elk_water` one did: a bare FFT deconvolution
    wraps circularly, and that image's top and bottom edges differ enough that
    the seam rang across the whole frame. The oracle scored **30.68 dB against
    the 30.84 dB blurred input**, below its own floor, while a 3x3 Laplacian
    reached 32.33.

    Averaged over six images it showed up only as the oracle trailing the
    Laplacian by 0.04 dB, which looks like rounding. Checking each image
    separately is what made it visible. Reflect-padding before the transform
    fixes it: 34.41 dB on that image, and a clear ceiling on all six.
    """
    sigma = 1.5
    ksize = int(max(3, round(sigma * 6) | 1))
    k = cv2.getGaussianKernel(ksize, sigma)
    kernel = (k @ k.T).astype(np.float32)

    for name in sh.IMAGES:
        clean = sh.load_scene(name)
        blurred = cv2.GaussianBlur(clean, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
        oracle = psnr(sh.deconvolve_oracle_best(blurred, kernel, clean), clean)

        assert oracle > psnr(blurred, clean), f"{name}: oracle below its own floor"
        for method, fn in sh.METHODS.items():
            assert oracle > psnr(fn(blurred), clean), f"{name}: {method} beat the oracle"


def test_the_oracle_beats_doing_nothing_on_a_known_blur():
    """It is handed the kernel. If it cannot use it, it is not a ceiling."""
    rows, stats = sh.evaluate_on_blurred(sigma=1.5, images=sh.IMAGES[:3])
    oracle = next(r for r in rows if r["method"] == sh.ORACLE_NAME)
    assert oracle["psnr_matched_db"] > stats["psnr"]
    assert oracle["acutance"] > stats["acutance"]


# --------------------------------------------------------------------------- #
# the findings
# --------------------------------------------------------------------------- #


def test_sharpening_a_clean_image_always_loses():
    """The project's headline, pinned.

    There is nothing to restore in an image that was never blurred, so every
    sharpener can only move away from the truth — while every one of them makes
    the picture look crisper.
    """
    rows = sh.evaluate_on_clean(images=sh.IMAGES[:3], runs=1)
    control = next(r for r in rows if r["method"].startswith("Do nothing"))
    others = [r for r in rows if not r["method"].startswith("Do nothing")]

    assert control["psnr_db"] == float("inf")
    for r in others:
        assert r["psnr_db"] < 40.0, r["method"]
    # ... and yet apparent sharpness goes up
    best = max(others, key=lambda r: r["acutance"])
    assert best["acutance"] > control["acutance"] * 1.5


def test_apparent_sharpness_and_fidelity_move_in_opposite_directions():
    """Monotonically, across the whole amount sweep. That is the whole claim."""
    rows = sh.sweep_amount(images=sh.IMAGES[:3])
    rows = sorted(rows, key=lambda r: r["amount"])
    psnrs = [r["psnr_db"] for r in rows]
    acut = [r["acutance"] for r in rows]

    assert acut == sorted(acut)                      # sharpness rises
    assert psnrs[1:] == sorted(psnrs[1:], reverse=True)  # fidelity falls


def test_raw_and_matched_psnr_disagree_about_the_best_method():
    """Why the table has two PSNR columns.

    High-boost is last on raw PSNR and first on matched. Nothing about its
    output changed between those two rankings.
    """
    rows, _ = sh.evaluate_on_blurred(sigma=1.5, images=sh.IMAGES[:3])
    real = [r for r in rows if r["method"] != sh.ORACLE_NAME]
    worst_raw = min(real, key=lambda r: r["psnr_db"])["method"]
    best_matched = max(real, key=lambda r: r["psnr_matched_db"])["method"]
    assert worst_raw == best_matched == "High-boost"


def test_sharpening_only_helps_while_the_blur_is_mild():
    """The practically useful part: knowing when a sharpener is the wrong tool."""
    rows = sorted(sh.sweep_blur(images=sh.IMAGES[:3]), key=lambda r: r["blur_sigma"])
    gains = [r["Unsharp mask"] - r["blurred_input"] for r in rows if r["blur_sigma"] > 0]
    assert gains[0] > 0            # a mild blur is partly recoverable
    assert gains[-1] < gains[0]    # a heavy one is much less so
