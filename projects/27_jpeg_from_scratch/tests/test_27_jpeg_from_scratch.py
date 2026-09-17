"""Tests for project 27, JPEG built from the DCT up.

Most of these pin a *finding* rather than a number, so that a later change which
quietly reverses one of the project's conclusions fails loudly instead of
rewriting the README by accident.
"""

from __future__ import annotations

import numpy as np
import pytest

import jpeg as jp
from shared.metrics import psnr


# --------------------------------------------------------------------------- #
# the transform
# --------------------------------------------------------------------------- #


def test_the_dct_is_its_own_inverse_to_floating_point():
    """The claim the whole ablation rests on: the transform loses nothing."""
    rng = np.random.default_rng(0)
    block = rng.uniform(-128, 127, (8, 8)).astype(np.float32)
    assert np.allclose(jp.idct_2d(jp.dct_2d(block)), block, atol=1e-3)


def test_the_zigzag_visits_every_coefficient_once():
    order = jp.ZIGZAG.ravel()
    assert sorted(order.tolist()) == list(range(64))


def test_the_quality_scale_follows_the_standard():
    """Quality 50 leaves the tables untouched; below it they grow, above they shrink."""
    assert jp.quality_scale(50) == pytest.approx(1.0)
    assert jp.quality_scale(10) > jp.quality_scale(50) > jp.quality_scale(90)
    assert jp.scaled_table(jp.Q_LUMA, 50).max() == pytest.approx(jp.Q_LUMA.max())
    assert jp.scaled_table(jp.Q_LUMA, 100).min() >= 1   # never divides by zero


# --------------------------------------------------------------------------- #
# the ablation -- this is where the bug was
# --------------------------------------------------------------------------- #


def test_turning_off_the_colour_transform_keeps_the_colour():
    """The bug that made a stage look like it costs bits rather than saves them.

    "No colour transform" means compress R, G and B directly. The branch used to
    call `to_gray`, so the row was actually measuring "throw the colour away": a
    grayscale reconstruction, three times smaller by construction, scored
    against a colour original.

    On the old grayscale image pool that produced a plausible wrong number. On a
    colour-only pool it raised a shape error, which is the only reason it was
    caught at all.
    """
    img = jp.load_scene("tulip_beds")
    recon, _ = jp.encode_decode(img, quality=50, use_colour_transform=False)

    assert recon.shape == img.shape
    # and it really is still in colour, not three copies of a grey channel
    spread = recon.astype(int).max(axis=2) - recon.astype(int).min(axis=2)
    assert spread.mean() > 10


def test_every_stage_off_is_near_lossless():
    """With the DCT and quantisation both off, only rounding remains."""
    rows = {r["configuration"]: r for r in jp.stage_ablation(images=jp.IMAGES[:4])}
    assert rows["No DCT, no quantisation"]["psnr_db"] > 40.0
    assert rows["No DCT, no quantisation"]["ssim"] > 0.99


def test_quantisation_is_where_all_the_loss_lives():
    """Turning quantisation off recovers most of the PSNR, at several times the bits."""
    rows = {r["configuration"]: r for r in jp.stage_ablation(images=jp.IMAGES[:4])}
    full, no_q = rows["Full codec"], rows["No quantisation"]

    assert no_q["psnr_db"] > full["psnr_db"] + 4.0
    assert no_q["estimated_bpp"] > 2.5 * full["estimated_bpp"]


def test_the_dct_earns_its_place_twice_over():
    """The project's headline, and it is a bigger effect than the textbooks imply.

    Quantising *pixels* instead of DCT coefficients — same quantisation tables,
    same everything else — scores **9.3 dB worse at more than twice the
    bitrate**. The transform is not a convenience; it is what makes the
    quantisation affordable, because it concentrates the energy into a few
    coefficients that survive and many that do not.
    """
    rows = {r["configuration"]: r for r in jp.stage_ablation(images=jp.IMAGES[:4])}
    full, no_dct = rows["Full codec"], rows["No DCT (quantise pixels)"]

    assert full["psnr_db"] > no_dct["psnr_db"] + 8.0        # far better
    assert full["estimated_bpp"] < 0.6 * no_dct["estimated_bpp"]   # and far smaller


def test_chroma_subsampling_is_nearly_free():
    """A quarter of the chroma data thrown away for hundredths of a decibel."""
    rows = {r["configuration"]: r for r in jp.stage_ablation(images=jp.IMAGES[:4])}
    full, no_sub = rows["Full codec"], rows["No chroma subsampling"]

    assert abs(full["psnr_db"] - no_sub["psnr_db"]) < 1.0
    assert full["estimated_bpp"] < 0.85 * no_sub["estimated_bpp"]


# --------------------------------------------------------------------------- #
# the findings
# --------------------------------------------------------------------------- #


def test_the_highest_frequency_coefficient_never_survives_below_quality_75():
    """What quantisation actually deletes, counted rather than described.

    The bottom-right coefficient of every 8x8 block is quantised by 99 in the
    luma table and more after scaling, so below quality 75 it rounds to zero in
    **every block of every image**. The DC term survives 91-99% of the time.
    """
    rows = {r["quality"]: r for r in jp.coefficient_statistics(images=jp.IMAGES[:4])}
    for q in (10, 30, 50):
        assert rows[q]["highest_freq_survival"] == 0.0, q
    assert rows[95]["highest_freq_survival"] > 0.0
    for q, r in rows.items():
        assert r["dc_survival"] > 0.85, q


def test_rate_and_distortion_both_rise_with_quality():
    rows = sorted(jp.rate_distortion(images=jp.IMAGES[:4], qualities=(10, 50, 95)),
                  key=lambda r: r["quality"])
    assert [r["bpp_ours"] for r in rows] == sorted(r["bpp_ours"] for r in rows)
    assert [r["psnr_ours"] for r in rows] == sorted(r["psnr_ours"] for r in rows)
    assert [r["ssim_ours"] for r in rows] == sorted(r["ssim_ours"] for r in rows)


def test_blockiness_rises_as_quality_falls():
    """The characteristic artefact, measured. Nothing but the 8x8 grid creates it."""
    rows = sorted(jp.rate_distortion(images=jp.IMAGES[:4], qualities=(10, 50, 95)),
                  key=lambda r: r["quality"])
    blockiness = [r["blockiness"] for r in rows]
    assert blockiness == sorted(blockiness, reverse=True)
    assert blockiness[0] > 1.5 * blockiness[-1]


def test_libjpeg_beats_this_codec_and_the_gap_is_reported():
    """A textbook codec against a production one, as a number rather than a hedge.

    libjpeg spends more bits at the same nominal quality and gets more PSNR for
    them. The entropy estimate here is an approximation of real Huffman coding,
    so the bitrates are not strictly comparable — which is stated rather than
    quietly relied on.
    """
    rows = {r["quality"]: r for r in jp.rate_distortion(images=jp.IMAGES[:4],
                                                        qualities=(50,))}
    at50 = rows[50]
    assert at50["psnr_opencv"] > at50["psnr_ours"]
    assert at50["bpp_opencv"] > at50["bpp_ours"]
