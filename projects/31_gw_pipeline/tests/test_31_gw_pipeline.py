"""Tests for project 31, the Gonzalez & Woods eight-stage pipeline.

Most of these pin a *finding* rather than a number, so that a later change which
quietly reverses one of the project's conclusions fails loudly instead of
rewriting the README by accident.
"""

from __future__ import annotations

import numpy as np
import pytest

import gw_pipeline as gw
from shared.io import to_gray


# --------------------------------------------------------------------------- #
# the pipeline
# --------------------------------------------------------------------------- #


def test_the_pipeline_runs_and_returns_every_stage():
    img = gw.load_scene("diver_dark_reef")
    final, stages = gw.run_pipeline(img)
    assert final.shape == to_gray(img).shape
    assert len(stages) >= 8
    for name, stage in stages.items():
        assert stage.shape == final.shape, name
        assert np.isfinite(stage).all(), name


def test_skipping_an_unknown_stage_is_refused():
    """A typo used to run the full pipeline and report it as an ablation row.

    Six identical rows read as "no stage contributes anything", which is a
    conclusion rather than a bug report.
    """
    with pytest.raises(ValueError, match="unknown stage"):
        gw.run_pipeline(gw.load_scene("red_canoes"), skip="not_a_stage")


def test_an_unknown_variant_is_refused():
    with pytest.raises(ValueError, match="unknown method"):
        gw.apply_variant(gw.load_scene("red_canoes"), "magic")


# --------------------------------------------------------------------------- #
# the ablation
# --------------------------------------------------------------------------- #


def test_one_stage_contributes_essentially_nothing():
    """The project's headline, and the reason to ablate rather than admire.

    Stage (e) — smoothing the Sobel gradient — changes dark-region detail by
    about 0.01 and acutance by 0.002 when removed, and both changes are
    *improvements*. It is in the book and it is doing nothing here.
    """
    rows = gw.ablation(images=gw.IMAGES[:6])
    deltas = [r for r in rows if "dark_detail_delta" in r]
    smallest = min(deltas, key=lambda r: abs(r["dark_detail_delta"]))

    assert "e_smoothed_sobel" in smallest["configuration"]
    assert abs(smallest["dark_detail_delta"]) < 0.05
    assert abs(smallest["acutance_delta"]) < 0.02


def test_the_mask_stage_restrains_rather_than_adds():
    """Removing it *raises* every enhancement metric and wrecks the fidelity.

    Stage (f) multiplies the sharpened image by a smoothed gradient mask.
    Without it acutance rises by 0.28 and dark detail by 0.61 — while SSIM
    against the original falls from 0.77 to 0.44. The mask is not contributing
    enhancement, it is preventing the sharpening from running away, which is a
    different job and is invisible unless each stage is removed separately.
    """
    rows = {r["configuration"]: r for r in gw.ablation(images=gw.IMAGES[:6])}
    full, no_mask = rows["Full pipeline"], rows["Without f_mask"]

    assert no_mask["acutance"] > full["acutance"]
    assert no_mask["dark_detail"] > full["dark_detail"]
    assert no_mask["ssim_vs_original"] < full["ssim_vs_original"] - 0.2


def test_removing_the_power_law_leaves_the_image_far_closer_to_the_original():
    """It is a tone curve, so it moves brightness a long way and structure very little."""
    rows = {r["configuration"]: r for r in gw.ablation(images=gw.IMAGES[:6])}
    full, no_gamma = rows["Full pipeline"], rows["Without h_power_law"]
    assert no_gamma["psnr_vs_original"] > full["psnr_vs_original"] + 5.0
    assert no_gamma["ssim_vs_original"] > full["ssim_vs_original"]


# --------------------------------------------------------------------------- #
# the findings
# --------------------------------------------------------------------------- #


def test_a_single_unsharp_mask_beats_the_whole_pipeline():
    """Eight stages, and one line of code does better on every column.

    Unsharp mask alone reaches 0.788 dark detail against the pipeline's 0.509,
    higher acutance, and SSIM 0.907 against 0.772 — so it is *closer* to the
    original as well as more enhanced. CLAHE alone also beats it on both
    enhancement metrics.
    """
    rows = {r["method"]: r for r in gw.compare_to_simple_alternatives(images=gw.IMAGES[:6])}
    pipeline, unsharp = rows["G&W 8-stage pipeline"], rows["Unsharp mask only"]

    assert unsharp["dark_detail"] > pipeline["dark_detail"]
    assert unsharp["acutance"] > pipeline["acutance"]
    assert unsharp["ssim_vs_original"] > pipeline["ssim_vs_original"]
    assert rows["CLAHE only"]["dark_detail"] > pipeline["dark_detail"]


def test_the_pipeline_still_beats_doing_nothing():
    """Which is worth checking before concluding it is useless."""
    rows = {r["method"]: r for r in gw.compare_to_simple_alternatives(images=gw.IMAGES[:6])}
    pipeline, control = rows["G&W 8-stage pipeline"], rows["Original (control)"]
    assert pipeline["dark_detail"] > control["dark_detail"]
    assert pipeline["acutance"] > control["acutance"]


def test_acutance_cannot_tell_the_laplacian_sign_but_ssim_can():
    """The same trap project 16 found, on a different operator and a different scene.

    Adding a +8-centre Laplacian sharpens; adding a −8-centre one is the classic
    bug. **Both raise acutance far above the original** — 1.067 and 0.916
    against 0.301 — so the no-reference sharpness measure reports success on the
    bug. SSIM says 0.457 against **−0.055**: the wrong sign produces an image
    anti-correlated with the original.
    """
    rows = {r["configuration"]: r for r in gw.laplacian_sign_test(images=gw.IMAGES[:6])}
    original = next(v for k, v in rows.items() if k.startswith("Original"))
    correct = next(v for k, v in rows.items() if k.startswith("Correct"))
    wrong = next(v for k, v in rows.items() if k.startswith("Wrong"))

    assert correct["acutance"] > original["acutance"]
    assert wrong["acutance"] > original["acutance"]        # the bug looks sharper too
    assert wrong["ssim_vs_original"] < 0.1                 # and is structurally ruined
    assert correct["ssim_vs_original"] > wrong["ssim_vs_original"] + 0.3


def test_the_books_gamma_is_not_the_best_gamma():
    """The prescribed 0.5 is close to optimal and is not optimal.

    Dark-region detail peaks at **gamma 0.6** and falls away either side: too
    aggressive a curve crushes the highlights it has just lifted the shadows
    into. A monotone assumption would have missed this — the interior optimum is
    the finding.
    """
    rows = sorted(gw.sweep_gamma(images=gw.IMAGES[:4]), key=lambda r: r["gamma"])
    detail = [r["dark_detail"] for r in rows]
    peak = int(np.argmax(detail))

    assert 0 < peak < len(detail) - 1, "the optimum is at an end, so it is not one"
    assert rows[peak]["gamma"] != 0.5, "the book's value is now the measured best"
    # ... and it is a tone curve, so SSIM tracks gamma almost perfectly
    ssims = [r["ssim_vs_original"] for r in rows]
    assert ssims == sorted(ssims)


def test_stage_e_smoothing_size_barely_matters_either():
    """Its own parameter, on a stage that contributes nothing — as expected."""
    rows = sorted(gw.sweep_smoothing(images=gw.IMAGES[:4]), key=lambda r: r["smooth_ksize"])
    detail = [r["dark_detail"] for r in rows]
    assert max(detail) - min(detail) < 0.25
