"""Tests for project 11, HDR exposure fusion.

Most of these pin a *finding* rather than a number, so that a later change which
quietly reverses one of the project's conclusions fails loudly instead of
rewriting the README by accident.
"""

from __future__ import annotations

import numpy as np
import pytest

import hdr
from shared import io, synth


SCENES = ("windmills", "lake_shrine")


# --------------------------------------------------------------------------- #
# the generated scene
# --------------------------------------------------------------------------- #


def test_the_scene_is_wider_than_eight_bits():
    """Otherwise there is nothing for a bracket to do.

    This is the assumption the whole project rests on, and two earlier versions
    of the generator failed it silently — one by scaling an already-8-bit
    photograph, the other by narrowing the sensor instead of widening the scene.
    In both, the middle exposure clipped under 1% and every fusion method lost
    to the trivial control.
    """
    radiance = synth.hdr_scene(io.real_photo("windmills"))
    lum = synth.to_gray_float(radiance)
    lo, hi = np.percentile(lum, [1, 99])
    stops = np.log2(hi / max(lo, 1e-6))
    assert stops > 8.0


def test_no_single_exposure_holds_the_scene():
    frames, _, _ = synth.hdr_bracket(io.real_photo("windmills"))
    for f in frames:
        lost = float((f >= 254).mean() + (f <= 2).mean())
        assert lost > 0.001, "an exposure that loses nothing means the scene is too narrow"


def test_the_bracket_as_a_whole_records_almost_everything():
    """The ceiling. What is clipped in *every* frame is gone for good."""
    frames, _, _ = synth.hdr_bracket(io.real_photo("windmills"))
    loss = synth.bracket_loss(frames)
    assert loss["unrecoverable"] < 0.02


def test_exposure_times_are_relative_to_the_middle_frame():
    _, times, _ = synth.hdr_bracket(io.real_photo("windmills"))
    assert times[len(times) // 2] == pytest.approx(1.0, rel=1e-5)
    assert np.all(np.diff(times) > 0)


def test_srgb_round_trips():
    """A wrong transfer function silently changes what "one stop" means."""
    img = io.real_photo("windmills")
    back = synth.linear_to_srgb(synth.srgb_to_linear(img)) * 255.0
    assert np.abs(back - img.astype(np.float32)).mean() < 1.0


# --------------------------------------------------------------------------- #
# the oracle
# --------------------------------------------------------------------------- #


def test_the_oracle_is_a_ceiling_not_a_method():
    """It is handed the exact radiance the bracket only samples.

    Pinned because the reference used to be the flat photograph, which asked
    fusion to remove the illumination as well — an intrinsic-image problem — and
    every method then failed identically at around SSIM 0.5, which says nothing
    about any of them.
    """
    radiance = synth.hdr_scene(io.real_photo("windmills"))
    rendered = synth.tonemap_oracle(radiance)
    assert rendered.dtype == np.uint8
    assert rendered.shape == radiance.shape
    # it uses the whole range without clipping most of it away
    assert 60 < float(rendered.mean()) < 190
    assert float((rendered >= 254).mean()) < 0.05


# --------------------------------------------------------------------------- #
# methods
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", list(hdr.METHODS))
def test_every_method_returns_a_full_size_uint8_image(name):
    frames, times, reference = synth.hdr_bracket(io.real_photo("windmills"))
    out = hdr.METHODS[name](frames, times)
    assert out.dtype == np.uint8
    assert out.shape == reference.shape


def test_mertens_needs_uint8_frames():
    """The silent failure that made a working method look broken.

    Handed floats in [0,1], OpenCV's Mertens returns an image whose maximum is
    about 0.004 — black — and raises nothing. It scored 6.5 dB and SSIM 0.006
    that way, which reads as a broken algorithm rather than a wrong call.
    """
    import cv2

    frames, _, _ = synth.hdr_bracket(io.real_photo("windmills"))
    as_float = cv2.createMergeMertens().process([f.astype(np.float32) / 255.0 for f in frames])
    as_uint8 = cv2.createMergeMertens().process(list(frames))
    assert as_float.max() < 0.05
    assert as_uint8.max() > 0.5


def test_brightness_matching_only_removes_a_global_scale():
    """It must not be able to repair detail, only exposure."""
    reference = io.real_photo("windmills")
    darker = (reference.astype(np.float32) * 0.45).astype(np.uint8)
    matched = hdr.match_mean(darker, reference)
    from shared.metrics import psnr

    assert psnr(matched, reference) > psnr(darker, reference) + 5.0


# --------------------------------------------------------------------------- #
# the findings
# --------------------------------------------------------------------------- #


def test_the_naive_mean_beats_every_real_fusion_method():
    """The project's headline, and it is uncomfortable.

    Averaging the encoded frames — which weights a blown highlight exactly like
    a well-exposed pixel — scores higher than Mertens and than all three
    Debevec+tonemap pipelines.

    Part of that is real and part of it is the metric, and the README says so:
    the reference is rendered with a *global* Reinhard curve, so methods that
    behave globally match it more closely than methods that blend locally. The
    number is pinned here; the caveat is pinned in the README.
    """
    rows, _ = hdr.evaluate_methods(images=SCENES)
    by_name = {r["method"]: r for r in rows}
    mean_ctl = by_name["Mean of frames (control)"]
    fusion = [r for r in rows if r["method"] not in hdr.CONTROLS]
    assert mean_ctl["ssim"] > max(r["ssim"] for r in fusion)


def test_one_exposure_beats_the_sophisticated_pipelines():
    """The control that should not win, winning.

    Taking the middle frame and doing nothing beats all three Debevec pipelines
    on SSIM, at 0.002 ms against 1.8 s.
    """
    rows, _ = hdr.evaluate_methods(images=SCENES)
    by_name = {r["method"]: r for r in rows}
    single = by_name["Middle exposure only (control)"]
    debevec = [r for r in rows if r["method"].startswith("Debevec")]
    assert all(single["ssim"] > r["ssim"] for r in debevec)
    assert all(single["median_ms"] < r["median_ms"] for r in debevec)


def test_a_wider_bracket_makes_every_method_worse():
    """Adding frames removes information, which should not happen.

    Every method degrades from 2 frames to 5 on the full image set. The extra
    frames are the ±4-stop ones, which are roughly half clipped, and none of
    these methods discounts a frame for being mostly saturated.

    **The full set, not the two-scene subset the other tests use.** On two
    scenes Mertens *improves* with more frames (22.28 -> 23.77) and on six it
    degrades (24.10 -> 20.92). The effect is an average over scenes, not a
    property of every scene, and a subset that happens to contain the
    exceptions would assert the opposite. Recorded here because it is exactly
    the kind of claim a small sample can invert.
    """
    rows = hdr.sweep_bracket_size(images=hdr.IMAGES)
    fusion = [n for n in hdr.METHODS if n not in hdr.CONTROLS]
    for name in fusion:
        series = [r[name] for r in rows]
        assert series[0] > series[-1], f"{name} improved with more frames"


def test_a_wider_bracket_does_record_more_of_the_scene():
    """The counterpart to the above: the frames are not useless, the fusion is.

    Unrecoverable area falls as the bracket widens, so the extra exposures
    genuinely add information — the methods just make worse use of it.
    """
    rows = hdr.sweep_bracket_size(images=SCENES)
    lost = [r["unrecoverable"] for r in rows]
    assert lost[0] > lost[-1]


def test_raw_psnr_and_matched_psnr_disagree_about_the_winner():
    """Why the table needs both columns.

    A tone mapper picks its own overall brightness, and raw PSNR scores that
    choice far more heavily than the detail recovered.
    """
    rows, _ = hdr.evaluate_methods(images=SCENES)
    best_raw = max(rows, key=lambda r: r["psnr_db"])["method"]
    best_matched = max(rows, key=lambda r: r["psnr_matched_db"])["method"]
    assert best_raw != best_matched


def test_the_contribution_map_covers_more_than_one_frame():
    """If one frame supplied everything, the bracket was pointless."""
    frames, _, _ = synth.hdr_bracket(io.real_photo("windmills"))
    used = np.unique(hdr.contribution_map(frames))
    assert len(used) >= 3
