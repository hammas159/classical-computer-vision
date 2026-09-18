"""Tests for project 40, multi-frame super-resolution.

Most of these pin a *finding* rather than a number, so that a later change which
quietly reverses one of the project's conclusions fails loudly instead of
rewriting the README by accident.

`test_ecc_returns_the_offset_and_not_its_negative` exists because a sign
inversion in `register_ecc` had the better of the two registration methods
reported as eleven times worse than the other, with no symptom except a number
that looked plausible.
"""

from __future__ import annotations

import numpy as np
import pytest

import multiframe_sr as sr

from shared.metrics import psnr

SCALE = 3


@pytest.fixture(scope="module")
def counts():
    return {r["frames"]: r for r in sr.sweep_frame_count(scale=SCALE)}


@pytest.fixture(scope="module")
def registration():
    return {r["method"]: r for r in sr.evaluate_registration(scale=SCALE, n_frames=8)}


# --------------------------------------------------------------------------- #
# the harness
# --------------------------------------------------------------------------- #


def test_ecc_returns_the_offset_and_not_its_negative():
    """Checked against known offsets, which is the only way this shows up.

    A negated translation has exactly the right magnitude and exactly the wrong
    direction. The only symptom is a registration error of roughly twice the true
    offset — a number that looks like a method simply being inaccurate.
    """
    hr = sr._prepare(sr.IMAGES[-1], SCALE)
    frames, offsets = sr.make_stack(hr, n_frames=6, scale=SCALE, seed=0)

    for frame, (dx, dy) in zip(frames[1:], offsets[1:]):
        ex, ey = sr.register_ecc(frames[0], frame, SCALE)
        assert np.sign(ex) == np.sign(dx), (ex, dx)
        assert np.sign(ey) == np.sign(dy), (ey, dy)
        assert abs(ex - dx) < 0.3 and abs(ey - dy) < 0.3


def test_the_stack_shifts_before_it_decimates():
    """Shifting after decimation could never produce new sample positions.

    A sub-pixel shift applied in the high-resolution world lands the sensor grid
    somewhere new; the same shift applied to an already-decimated frame just
    interpolates the samples it has. The first is super-resolution and the second
    is not.
    """
    hr = sr._prepare(sr.IMAGES[-1], SCALE)
    frames, offsets = sr.make_stack(hr, n_frames=4, scale=SCALE, noise_sigma=0.0, seed=0)

    assert offsets[0] == (0.0, 0.0)
    assert all(abs(o[0]) <= SCALE and abs(o[1]) <= SCALE for o in offsets)
    assert any(abs(o[0] % 1.0) > 0.05 for o in offsets[1:]), "offsets must be sub-pixel"

    for frame in frames:
        assert frame.shape[0] == hr.shape[0] // SCALE
        assert frame.shape[1] == hr.shape[1] // SCALE


def test_whole_pixel_offsets_really_are_whole_low_resolution_pixels():
    _, offsets = sr.make_stack(sr._prepare(sr.IMAGES[0], SCALE), n_frames=8,
                               scale=SCALE, subpixel=False, seed=0)
    for dx, dy in offsets:
        assert dx % SCALE == 0 and dy % SCALE == 0, (dx, dy)


def test_the_pool_spans_the_axis_it_was_selected_on():
    from shared import io

    values = []
    for name in sr.IMAGES:
        assert name in io.REAL_PHOTOS, name
        values.append(sr.detail(sr._prepare(name, SCALE)))

    assert len(set(sr.IMAGES)) == len(sr.IMAGES)
    assert values == sorted(values), "IMAGES should be ordered by detail"
    assert max(values) > 20 * min(values), "the pool must span the axis"


# --------------------------------------------------------------------------- #
# the findings
# --------------------------------------------------------------------------- #


def test_multiple_frames_break_the_single_image_plateau():
    """Project 21's whole nearest-to-Lanczos argument was worth 0.49 dB.

    Extra frames with sub-pixel offsets are worth 2.25 — four and a half times
    as much — because they carry samples no single frame has.
    """
    rows = {r["method"]: r for r in sr.evaluate_methods(n_frames=8, scale=SCALE)}
    best = max(rows.values(), key=lambda r: r["psnr_db"])

    assert best["method"] == "Iterative back-projection"
    assert best["gain_over_single_db"] > 2.0
    assert best["gain_over_single_db"] > 4 * 0.49


def test_part_of_the_gain_is_deblurring_and_not_fusion(counts):
    """The control the frame-count sweep exists to provide.

    Iterative back-projection inverts the known blur whether or not there is
    anything to fuse, so it beats bicubic by 0.68 dB **from a single frame**.
    Quoting the full 2.25 dB as a multi-frame gain would credit fusion with
    deblurring.
    """
    one = counts[1]
    assert one["Iterative back-projection"] > one["Single frame (bicubic)"] + 0.3
    assert one["Iterative back-projection"] < one["Single frame (bicubic)"] + 1.5

    # and the fusion is the larger half
    fusion = counts[32]["Iterative back-projection"] - one["Iterative back-projection"]
    deblur = one["Iterative back-projection"] - one["Single frame (bicubic)"]
    assert fusion > deblur


def test_more_frames_monotonically_help_and_with_diminishing_returns(counts):
    values = [counts[n]["Iterative back-projection"] for n in sorted(counts)]
    assert values == sorted(values)

    first_double = values[1] - values[0]     # 1 -> 2 frames
    last_double = values[-1] - values[-2]    # 16 -> 32 frames
    assert first_double > 3 * last_double


def test_sub_pixel_offsets_are_the_part_that_is_actually_resolution():
    """Whole-pixel offsets land on the same sensor grid and add no new samples."""
    rows = {r["offsets"]: r for r in sr.subpixel_vs_integer(scale=SCALE, n_frames=8)}
    sub = rows["Sub-pixel offsets"]
    whole = rows["Integer-pixel offsets"]

    for method in ("Shift-and-add", "Iterative back-projection"):
        assert sub[method] > whole[method] + 0.5, method
    assert sub["Single frame (bicubic)"] == whole["Single frame (bicubic)"]


def test_naive_averaging_is_worse_than_one_frame():
    """Averaging frames that are not aligned is a blur, not a reconstruction."""
    rows = {r["method"]: r for r in sr.evaluate_methods(n_frames=8, scale=SCALE)}
    assert rows["Naive average"]["gain_over_single_db"] < 0
    assert rows["Naive average"]["ssim"] < rows["Single frame (bicubic)"]["ssim"]


def test_ecc_is_the_more_accurate_registration(registration):
    """By 6x, once its sign is right."""
    ecc = registration["ECC"]["mean_offset_error_px"]
    phase = registration["Phase correlation"]["mean_offset_error_px"]

    assert ecc < 0.2
    assert phase > 3 * ecc


def test_fusion_stops_paying_once_the_frames_are_two_pixels_out():
    """And it degrades gracefully until then, which is the useful half."""
    rows = sorted(sr.sweep_registration_error(scale=SCALE, n_frames=8),
                  key=lambda r: r["registration_error_px"])
    baseline = rows[0]["Iterative back-projection"]
    single = rows[0]["Single frame (bicubic)"]

    half_pixel = next(r for r in rows if r["registration_error_px"] == 0.5)
    assert half_pixel["Iterative back-projection"] > baseline - 0.3

    worst = rows[-1]
    assert worst["registration_error_px"] >= 2.0
    assert worst["Iterative back-projection"] <= single + 0.1


def test_the_harder_the_upscale_the_less_a_fixed_number_of_frames_buys():
    """The opposite of the intuition, and it follows from counting samples.

    Sixteen frames deliver 3.46 dB at x2 and 2.16 dB at x4. Upscaling by s needs
    s^2 times as many samples per output pixel, so the same sixteen frames fill
    the x4 grid a quarter as densely. Extra frames do not buy an arbitrary scale
    factor; they buy a fixed amount of information that is spread thinner the
    more you ask of it.
    """
    rows = {r["scale"]: r for r in sr.sweep_scale()}
    gains = [rows[s]["gain_db"] for s in sorted(rows)]

    assert gains == sorted(gains, reverse=True), gains
    assert gains[0] > 1.5 * gains[-1]
    assert all(rows[s]["best_method"] == "Iterative back-projection" for s in rows)


# --------------------------------------------------------------------------- #
# the reconstructions themselves
# --------------------------------------------------------------------------- #


def test_every_method_returns_the_full_resolution_image():
    hr = sr._prepare(sr.IMAGES[0], SCALE)
    frames, offsets = sr.make_stack(hr, n_frames=4, scale=SCALE, seed=0)
    for name, fn in sr.METHODS.items():
        out = fn(frames, offsets, scale=SCALE)
        assert out.shape[:2] == hr.shape[:2], name
        assert out.dtype == np.uint8, name
        assert psnr(out, hr) > 12.0, name


def test_reconstruction_with_the_true_offsets_beats_reconstruction_with_none():
    """If it did not, the offsets would not be doing anything."""
    hr = sr._prepare(sr.IMAGES[-1], SCALE)
    frames, offsets = sr.make_stack(hr, n_frames=8, scale=SCALE, seed=0)

    with_truth = sr.sr_shift_and_add(frames, offsets, scale=SCALE)
    without = sr.sr_shift_and_add(frames, [(0.0, 0.0)] * len(frames), scale=SCALE)
    assert psnr(with_truth, hr) > psnr(without, hr) + 0.5
