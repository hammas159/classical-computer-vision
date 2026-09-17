"""Tests for project 17, histogram equalisation.

Most of these pin a *finding* rather than a number, so that a later change which
quietly reverses one of the project's conclusions fails loudly instead of
rewriting the README by accident.
"""

from __future__ import annotations

import numpy as np
import pytest

import histogram_eq as he
from shared.io import to_gray
from shared.metrics import entropy, psnr


# --------------------------------------------------------------------------- #
# the operators
# --------------------------------------------------------------------------- #


def test_every_method_leaves_colour_balance_alone():
    """The lurid-output bug, pinned.

    Equalising R, G and B independently shifts the colour balance and produces
    the garish result people associate with HE. Every method here works on
    luminance only, so a grey pixel must stay grey.
    """
    grey = np.full((64, 64, 3), 130, np.uint8)
    grey[:32] = 90  # two levels, so equalisation has something to do
    for name, fn in he.METHODS.items():
        out = fn(grey)
        spread = out.astype(int).max(axis=2) - out.astype(int).min(axis=2)
        assert spread.max() <= 2, f"{name} introduced a colour cast"


def test_a_flat_image_survives_every_method():
    flat = np.full((64, 64, 3), 120, np.uint8)
    for name, fn in he.METHODS.items():
        out = fn(flat)
        assert out.shape == flat.shape, name
        assert out.std() < 3.0, f"{name} invented structure in a flat image"


def test_degrading_really_does_compress_the_tone_range():
    """The scene's contract. If the degradation is mild, the experiment is empty."""
    clean = he.load_scene("covered_wagons")
    bad = he.degrade(clean, kind="low_contrast", noise_sigma=0.0)

    def spread(img):
        lo, hi = np.percentile(to_gray(img), (1, 99))
        return float(hi - lo)

    assert spread(bad) < 0.45 * spread(clean)


# --------------------------------------------------------------------------- #
# the oracle
# --------------------------------------------------------------------------- #


def test_the_oracle_beats_every_real_method_on_every_image():
    """Per image, not on the average — an average can hide a single failure.

    The oracle is handed the clean image's own histogram, so it is the best any
    *tone curve* can do on this degradation. Nothing that has to guess the
    target should beat it.
    """
    for name in he.SCORED_IMAGES:
        clean = he.load_scene(name)
        bad = he.degrade(clean, kind="low_contrast", noise_sigma=3.0, seed=0)
        oracle = psnr(he.eq_match_oracle(bad, clean), clean)
        for method, fn in he.METHODS.items():
            assert oracle > psnr(fn(bad), clean), f"{name}: {method} beat the oracle"


def test_the_oracle_is_not_a_no_op():
    """It has to actually recover something, or it bounds nothing."""
    clean = he.load_scene("skiers_woods")
    bad = he.degrade(clean, kind="low_contrast", noise_sigma=3.0, seed=0)
    assert psnr(he.eq_match_oracle(bad, clean), clean) > psnr(bad, clean) + 5.0


def test_histogram_matching_is_never_handed_its_own_target():
    """`Histogram matching` matches to a fixed reference photograph.

    If that reference were also one of the scored images, the method would score
    a perfect result on one row for a reason that has nothing to do with the
    method. It is excluded from `SCORED_IMAGES` instead.
    """
    assert he.MATCH_REFERENCE in he.IMAGES
    assert he.MATCH_REFERENCE not in he.SCORED_IMAGES


# --------------------------------------------------------------------------- #
# the findings
# --------------------------------------------------------------------------- #


def test_entropy_ranks_the_methods_almost_backwards():
    """The project's headline, pinned.

    Entropy is the metric global HE provably maximises, and it is the one
    reached for when there is no ground truth. It picks `AHE (unclipped)` — the
    *worst* method in the table by PSNR — and it ranks the oracle, which is the
    correct answer by construction, below doing nothing.
    """
    rows, _ = he.evaluate_methods(images=he.SCORED_IMAGES[:5], runs=1)
    by_psnr = sorted(rows, key=lambda r: -r["psnr_db"])
    by_entropy = sorted(rows, key=lambda r: -r["entropy_bits"])

    assert by_psnr[0]["method"] == he.ORACLE_NAME
    assert by_entropy[0]["method"] == "AHE (unclipped)"

    # the highest-entropy method is near the bottom on fidelity
    worst_psnr = [r["method"] for r in by_psnr[-2:]]
    assert "AHE (unclipped)" in worst_psnr

    # and the correct answer scores *below the untouched image* on entropy
    oracle_e = next(r["entropy_bits"] for r in rows if r["method"] == he.ORACLE_NAME)
    control_e = next(r["entropy_bits"] for r in rows
                     if r["method"].startswith("Do nothing"))
    assert oracle_e < control_e


def test_entropy_is_blind_to_a_nine_decibel_improvement():
    """The sharpest form of the finding, and the one that is robust per image.

    The oracle turns the degraded image into something **+9.18 dB** closer to
    the truth. Entropy notices a change of **-0.019 bits** on average, and never
    more than 0.13 bits on any image -- it moves *down* as often as up.

    So the no-reference metric that stands in for "did this get better" cannot
    see the single largest improvement available in the whole project. An
    earlier version of this test asserted the stronger claim that the oracle's
    entropy is below Global HE's; that is true on only 5 of the 11 images, so it
    was replaced with what actually holds rather than kept with a friendly image.
    """
    gains_psnr, gains_entropy = [], []
    for name in he.SCORED_IMAGES:
        clean = he.load_scene(name)
        bad = he.degrade(clean, kind="low_contrast", noise_sigma=3.0, seed=0)
        out = he.eq_match_oracle(bad, clean)
        gains_psnr.append(psnr(out, clean) - psnr(bad, clean))
        gains_entropy.append(entropy(out) - entropy(bad))

    assert float(np.mean(gains_psnr)) > 8.0      # a large, real improvement
    assert max(abs(e) for e in gains_entropy) < 0.25   # that entropy cannot see
    assert abs(float(np.mean(gains_entropy))) < 0.1


def test_the_clip_limit_has_an_optimum_that_entropy_points_away_from():
    """Why CLAHE's parameter cannot be tuned on a no-reference metric.

    PSNR peaks at a moderate clip and falls away either side. Entropy rises
    monotonically all the way to the unclipped end — so tuning on entropy picks
    the *worst* setting in the sweep.
    """
    rows = sorted(he.sweep_clip_limit(images=he.SCORED_IMAGES[:5]),
                  key=lambda r: r["clip_limit"])
    psnrs = [r["psnr_db"] for r in rows]
    ents = [r["entropy_bits"] for r in rows]

    best = int(np.argmax(psnrs))
    assert 0 < best < len(psnrs) - 1, "the optimum is at an end, so it is not an optimum"
    assert ents == sorted(ents), "entropy should rise monotonically with the clip limit"
    assert psnrs[int(np.argmax(ents))] == min(psnrs)


def test_on_most_images_nothing_beats_doing_nothing():
    """The uncomfortable result, kept rather than tuned away.

    On 5 of the 11 photographs **every** method in the family scores worse than
    leaving the image alone: they amplify the noise along with the contrast. The
    five are the darker ones, which is also the half that looks most like it
    needs equalising.
    """
    losers = []
    for name in he.SCORED_IMAGES:
        clean = he.load_scene(name)
        bad = he.degrade(clean, kind="low_contrast", noise_sigma=3.0, seed=0)
        control = psnr(he.eq_none(bad), clean)
        beat = [m for m, fn in he.METHODS.items()
                if not m.startswith("Do nothing") and psnr(fn(bad), clean) > control]
        if not beat:
            losers.append(name)
    assert len(losers) >= 5, f"only {losers} had no method beat the control"

    # and where something does win, it is always the contrast-limited one
    for name in he.SCORED_IMAGES:
        clean = he.load_scene(name)
        bad = he.degrade(clean, kind="low_contrast", noise_sigma=3.0, seed=0)
        control = psnr(he.eq_none(bad), clean)
        beat = [m for m, fn in he.METHODS.items()
                if not m.startswith("Do nothing") and psnr(fn(bad), clean) > control]
        if beat:
            assert "CLAHE (clip 2.0)" in beat, name


def test_clahe_amplifies_less_noise_than_unclipped_ahe():
    """What the clip limit actually buys, measured rather than asserted."""
    rows, _ = he.evaluate_methods(images=he.SCORED_IMAGES[:5], runs=1)
    clahe = next(r for r in rows if r["method"].startswith("CLAHE"))
    ahe = next(r for r in rows if r["method"].startswith("AHE"))
    assert clahe["noise_sigma"] < 0.5 * ahe["noise_sigma"]


def test_the_metrics_do_not_agree_on_a_winner():
    d = he.metric_disagreement(images=he.SCORED_IMAGES[:5])
    assert not d["winners_agree"]
    assert d["psnr_winner"] != d["entropy_winner"]


@pytest.mark.parametrize("kind", ["low_contrast", "gamma"])
def test_both_degradations_are_recoverable_by_a_tone_curve(kind):
    """If the oracle cannot fix it, the comparison is about the wrong thing."""
    clean = he.load_scene("beached_dinghy")
    bad = he.degrade(clean, kind=kind, noise_sigma=0.0)
    assert psnr(he.eq_match_oracle(bad, clean), clean) > psnr(bad, clean)


def test_an_unknown_degradation_raises():
    with pytest.raises(ValueError, match="unknown degradation"):
        he.degrade(he.load_scene("ostrich_head"), kind="solarise")
