"""Tests for project 26, full-reference quality metrics.

Most of these pin a *finding* rather than a number, so that a later change which
quietly reverses one of the project's conclusions fails loudly instead of
rewriting the README by accident.
"""

from __future__ import annotations

import numpy as np
import pytest

import quality_metrics as qm


# --------------------------------------------------------------------------- #
# the metrics themselves
# --------------------------------------------------------------------------- #


def test_every_metric_is_perfect_on_an_identical_image():
    img = qm.load_scene("elephant_pair")
    for name, (metric, higher_better) in qm.METRICS.items():
        value = metric(img, img)
        if name == "PSNR":
            assert value > 60, name
        elif higher_better:
            assert value == pytest.approx(1.0, abs=0.02), name
        else:
            assert value == pytest.approx(0.0, abs=0.02), name


def test_every_metric_moves_the_right_way_as_damage_increases():
    """Monotone in the strength of the damage — with two documented exceptions.

    A metric that is not monotone in the thing it measures cannot be compared
    against another, and this is the check that comes before everything else.
    Two metrics fail it in small, specific, explicable ways, and both are
    reported rather than smoothed over:

    * **GMSD saturates and then reverses** under heavy noise. It is the standard
      deviation of a gradient-similarity map, and once the noise is strong
      enough that every pixel is equally dissimilar, that deviation *falls*. So
      a higher GMSD does not always mean worse.
    * **VIF rises slightly on the mildest JPEG** (0.999 to 1.024), because
      quantisation removes a little of the image's own noise along with the
      detail.

    Anything beyond `TOLERANCE` would be a new failure and should be
    investigated rather than added to this list.
    """
    TOLERANCE = 0.03
    known = {("GMSD", "Gaussian noise"), ("GMSD", "Salt & pepper"),
             ("VIF (approx)", "JPEG")}

    for degradation in qm.DEGRADATIONS:
        rows = qm.sweep_strength(degradation, images=qm.IMAGES[:4])
        for name, (_, higher_better) in qm.METRICS.items():
            values = [r[name] for r in rows if r[name] is not None]
            expected = sorted(values, reverse=higher_better)
            if values == expected:
                continue
            deviation = max(abs(a - b) for a, b in zip(values, expected))
            assert (name, degradation) in known, (
                f"{name} is newly non-monotone on {degradation} "
                f"by {deviation:.4f}: {values}"
            )
            assert deviation < TOLERANCE, (
                f"{name} on {degradation} deviates by {deviation:.4f}, "
                f"far beyond the documented wobble"
            )


def test_vif_is_bounded_and_does_not_reward_the_damage():
    """The bug that made a fidelity metric climb to 40 as the image got worse.

    VIF's denominator is the information in the **reference**. The arguments
    were the other way round, so the *distorted* image was treated as the
    reference — and on contrast loss, which shrinks the distorted variance
    without bound, the ratio ran away:

        0.98, 1.18, 1.40, 1.72, 2.23, 3.14, 5.03, 10.42, **40.66**

    rising monotonically with the damage. It did not look obviously wrong
    because VIF above 1 is meaningful: it means the distorted image carries more
    information than the reference, which enhancement genuinely can do.
    """
    rows = qm.sweep_strength("Contrast loss", images=qm.IMAGES[:4])
    values = [r["VIF (approx)"] for r in rows]

    assert max(values) <= 1.05, f"VIF is unbounded again: {values}"
    assert values == sorted(values, reverse=True)
    assert values[-1] < 0.2      # heavy contrast loss really is heavy damage


def test_the_bisection_really_hits_the_psnr_it_was_asked_for():
    """Every claim in this project rests on the degradations being equal-PSNR."""
    clean = qm.load_scene("stone_bridge_river")
    for degradation in qm.DEGRADATIONS:
        strength, got = qm.find_strength_for_psnr(clean, degradation, 28.0)
        if degradation == "Sub-pixel shift":
            continue  # cannot reach it; see its own test
        assert got == pytest.approx(28.0, abs=0.6), degradation


# --------------------------------------------------------------------------- #
# the findings
# --------------------------------------------------------------------------- #


def test_equal_psnr_images_are_not_equally_damaged():
    """The project's headline.

    Six different damages, every one tuned to the same 28 dB. If PSNR measured
    what people mean by quality they would be interchangeable. SSIM spans 0.692
    to 0.955 across them — a 0.26 spread at identical PSNR.
    """
    rows = qm.equal_psnr_comparison(28.0, images=qm.IMAGES[:6])
    ssims = {r["degradation"]: r["SSIM"] for r in rows}

    assert max(ssims.values()) - min(ssims.values()) > 0.15
    assert min(ssims, key=ssims.get) == "Gaussian noise"
    assert max(ssims, key=ssims.get) == "Contrast loss"


def test_psnr_and_ssim_disagree_about_the_worst_damage():
    """At every PSNR target tried, not just a convenient one."""
    for target in qm.TARGET_PSNRS:
        d = qm.ranking_disagreement(target, images=qm.IMAGES[:6])
        assert not d["psnr_and_ssim_agree"], target


def test_the_perceptual_metrics_genuinely_reorder_the_damages():
    """Kendall tau well below 1 against the PSNR ranking, for all four.

    An earlier version of this reported MSE at tau 0.733 — apparently
    disagreeing with PSNR, which is impossible per image — because the ranking
    was computed on values rounded to five decimals and the MSE column was full
    of ties. The rounding was also *masking* how far the perceptual metrics
    disagree: GMSD moved from 0.600 to 0.200 once the ties were gone.
    """
    taus = qm.ranking_disagreement(28.0, images=qm.IMAGES[:6])["kendall_tau_vs_psnr"]
    for metric in ("SSIM", "MS-SSIM", "GMSD"):
        assert taus[metric] < 0.8, f"{metric} tau {taus[metric]}"
    # MSE is PSNR by another name and must agree exactly; a tau below 1 here
    # would mean the ranking is being computed on rounded values again.
    assert taus["MSE"] == 1.0
    assert taus["GMSD"] <= taus["SSIM"]


def test_mse_and_psnr_rank_identically_on_a_single_image():
    """Because PSNR is a monotone function of MSE. Per image this is a theorem."""
    clean = qm.load_scene("gulls_on_ledge")
    pairs = []
    for degradation in qm.DEGRADATIONS:
        strength, _ = qm.find_strength_for_psnr(clean, degradation, 28.0)
        fn, _ = qm.DEGRADATIONS[degradation]
        bad = fn(clean, strength)
        pairs.append((qm.METRICS["MSE"][0](bad, clean),
                      qm.METRICS["PSNR"][0](bad, clean)))

    by_mse = [p for p in sorted(pairs, key=lambda p: p[0])]
    by_psnr = [p for p in sorted(pairs, key=lambda p: -p[1])]
    assert by_mse == by_psnr


def test_averaging_can_reorder_mse_and_psnr_across_a_set():
    """And across a set it is not a theorem, which is worth knowing.

    PSNR is a logarithm, so the mean of PSNRs is not a function of the mean of
    MSEs. At a 24 dB target this reverses JPEG and contrast loss. It is the
    reason "average PSNR over a dataset" is a quantity to be suspicious of.
    """
    reordered = [qm.ranking_disagreement(t, images=qm.IMAGES)["averaging_reorders_mse_and_psnr"]
                 for t in qm.TARGET_PSNRS]
    assert any(reordered), "the averaging effect has gone; the claim may be stale"


def test_a_one_pixel_shift_cannot_be_made_mild_enough_to_reach_28_db():
    """The clearest single illustration of what PSNR is not measuring.

    Shifting an image by one pixel is very nearly invisible and costs more PSNR
    than any of the other five damages can be turned up to. It tops out at about
    24.3 dB, so the equal-PSNR row cannot even be built for it.
    """
    rows = {r["degradation"]: r for r in qm.equal_psnr_comparison(28.0, images=qm.IMAGES[:6])}
    shift = rows["Sub-pixel shift"]

    assert shift["achieved_psnr"] < 26.0
    # ... while SSIM says it is milder than plain noise at a *better* PSNR
    assert shift["SSIM"] > rows["Gaussian noise"]["SSIM"]


def test_contrast_loss_is_the_damage_the_metrics_disagree_about_most():
    """PSNR sees a large squared error; SSIM sees the structure intact."""
    rows = {r["degradation"]: r for r in qm.equal_psnr_comparison(28.0, images=qm.IMAGES[:6])}
    contrast = rows["Contrast loss"]

    assert contrast["SSIM"] > 0.93            # structurally almost untouched
    assert contrast["GMSD"] < 0.05            # gradients almost untouched
    assert contrast["achieved_psnr"] == pytest.approx(28.0, abs=0.6)   # same PSNR
