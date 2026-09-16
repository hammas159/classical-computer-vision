"""Tests for project 05, old photo restoration.

Most of these pin a *finding* rather than a number, so that a later "improvement"
that quietly reverses one of the project's conclusions fails loudly instead of
rewriting the README by accident.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import restoration as rs  # noqa: E402
from shared import io, synth
from shared.metrics import iou, psnr

IMAGES = ("astronaut", "coffee")


# --------------------------------------------------------------------------- #
# the setup
# --------------------------------------------------------------------------- #


def test_the_mask_marks_exactly_the_changed_pixels():
    """Everything downstream is scored against this mask, so it has to be exact."""
    clean = io.sample("astronaut")
    damaged, mask = synth.add_scratches(clean, thickness=3, seed=0)
    changed = np.any(damaged != clean, axis=2)
    assert np.array_equal(damaged[mask == 0], clean[mask == 0])
    assert changed[mask > 0].mean() > 0.97  # a scratch may land on an identical value


def test_damage_is_not_detectable_by_brightness_alone():
    """The generator must not hand the detectors the answer.

    An earlier version painted every damaged pixel 255, and `pixel >= 250`
    recovered the mask at 0.68 IoU — which measured the generator, not the
    detectors.
    """
    clean = io.sample("astronaut")
    damaged, mask = synth.add_scratches(clean, thickness=3, seed=0)
    cheat = (rs.to_gray(damaged) >= 250).astype(np.uint8) * 255
    assert iou(cheat, mask) < 0.25


def test_fading_is_reversible_only_up_to_its_non_diagonal_part():
    """The model's central claim, asserted.

    fade_photo applies a desaturation (mixes channels) and then per-channel gains
    and lifts (does not). A per-channel stretch can undo the second group and
    provably cannot undo the first, so adding the saturation step must help.
    """
    clean = io.sample("coffee")
    faded = synth.fade_photo(clean, seed=0)
    stretched = rs.correct_channel_stretch(faded)
    full = rs.correct_full(faded)
    assert rs.saturation_of(stretched) < rs.saturation_of(clean)
    assert psnr(full, clean) > psnr(stretched, clean)


# --------------------------------------------------------------------------- #
# inpainting
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", list(rs.METHODS))
def test_every_method_returns_rgb_uint8_of_the_same_shape(name):
    clean = io.sample("chelsea")
    damaged, mask = synth.add_scratches(clean, thickness=3, seed=0)
    out = rs.METHODS[name](damaged, mask)
    assert out.dtype == np.uint8 and out.shape == damaged.shape


@pytest.mark.parametrize("name", list(rs.METHODS))
def test_every_method_leaves_undamaged_pixels_almost_alone(name):
    """Inpainting is supposed to touch the mask and nothing else.

    "Almost" because Telea and Navier-Stokes blend a narrow band at the boundary.
    A method that rewrote the whole image would still score well on whole-image
    PSNR, so this is asserted directly.
    """
    clean = io.sample("coffee")
    damaged, mask = synth.add_scratches(clean, thickness=3, seed=0)
    out = rs.METHODS[name](damaged, mask)
    far = cv2.dilate(mask, np.ones((9, 9), np.uint8)) == 0
    assert psnr(out[far], damaged[far]) > 38.0


@pytest.mark.parametrize("name", list(rs.METHODS))
def test_every_method_beats_doing_nothing(name):
    clean = io.sample("astronaut")
    damaged, mask = synth.add_scratches(clean, thickness=3, seed=0)
    out = rs.METHODS[name](damaged, mask)
    assert rs.psnr_on_damage(out, clean, mask) > rs.psnr_on_damage(damaged, clean, mask) + 10.0


def test_the_masked_mean_must_not_average_in_the_damage():
    """Guards the bug this baseline was written wrong twice for.

    Version 1 filled every damaged pixel at once from a plain median; version 2
    peeled rings but still used an unmasked filter. Both mixed the scratch back
    into its own replacement and scored ~6 dB. The fix is normalized convolution
    — divide by the number of KNOWN pixels in the window, not by the window size.
    """
    clean = io.sample("astronaut")
    damaged, mask = synth.add_scratches(clean, thickness=3, seed=0)
    good = rs.inpaint_masked_mean(damaged, mask)

    # the broken version, reconstructed: an unmasked blur applied to the whole hole
    naive = damaged.copy()
    naive[mask > 0] = cv2.medianBlur(damaged, 7)[mask > 0]

    assert rs.psnr_on_damage(good, clean, mask) > rs.psnr_on_damage(naive, clean, mask) + 10.0


def test_whole_image_psnr_hides_what_damage_only_psnr_shows():
    """The reason `psnr_on_damage` exists, asserted rather than explained.

    It is the *level* that differs, not the improvement. Doing nothing already
    scores 17 dB whole-image — a number that sounds like a working method —
    while scoring 6 dB on the pixels that actually changed.

    The improvement is deliberately **not** asserted to differ, because it
    cannot: an inpainter that leaves undamaged pixels alone changes the same
    squared error in both the whole-image and the damage-only sums, so the two
    dB gaps are the same number to seven decimal places. (Asserting otherwise is
    how this test failed first time round.)
    """
    clean = io.sample("astronaut")
    damaged, mask = synth.add_scratches(clean, thickness=3, seed=0)
    assert psnr(damaged, clean) > 15.0
    assert rs.psnr_on_damage(damaged, clean, mask) < 8.0


# --------------------------------------------------------------------------- #
# the findings
# --------------------------------------------------------------------------- #


def test_width_separates_the_methods_and_3px_does_not():
    """The project's first finding, pinned.

    At 3 px all four methods are within about a decibel. At 40 px harmonic
    diffusion has collapsed and the others have not. Anyone ranking inpainting
    methods on one damage width is measuring nothing.
    """
    rows = rs.sweep_thickness(images=IMAGES, levels=(3, 40))
    thin, wide = rows[0], rows[1]
    thin_vals = [thin[n] for n in rs.METHODS]
    assert max(thin_vals) - min(thin_vals) < 2.5

    assert wide["Harmonic diffusion"] < wide["Telea (fast marching)"] - 5.0
    # and the 20-line baseline is no longer behind at all
    assert wide["Iterative masked mean"] > wide["Harmonic diffusion"] + 5.0


def test_knowing_the_mask_matters_far_more_than_the_method():
    """The project's headline, pinned.

    The detection tax is an order of magnitude larger than the method spread.
    """
    method_rows, _ = rs.evaluate_methods(thickness=3, images=IMAGES, runs=1)
    det_rows = rs.evaluate_detectors(thickness=3, images=IMAGES)

    true_mask_best = max(r["damage_psnr_db"] for r in method_rows)
    method_spread = true_mask_best - min(r["damage_psnr_db"] for r in method_rows)
    detected_best = max(r["damage_psnr_db"] for r in det_rows)
    detection_tax = true_mask_best - detected_best

    assert detection_tax > 5.0 * method_spread


def test_the_best_iou_detector_is_not_the_best_restoration():
    """Pins the precision/recall asymmetry.

    Over-detection is nearly free — a wrongly flagged healthy pixel is replaced
    by an average of its healthy neighbours. Under-detection is not: a missed
    scratch stays in the picture. Ranking detectors by IoU therefore gets the
    answer wrong, and this asserts that it does.
    """
    rows = rs.evaluate_detectors(thickness=3, images=rs.IMAGES)
    by_iou = max(rows, key=lambda r: r["mask_iou"])
    by_psnr = max(rows, key=lambda r: r["damage_psnr_db"])
    assert by_iou["detector"] != by_psnr["detector"]
    assert by_psnr["recall"] > by_iou["recall"]


def test_the_median_residual_detector_goes_blind_below_its_window():
    """Pins the window cliff, and therefore the reason the default is 21 and not 5."""
    rows = rs.sweep_detector_window(images=IMAGES, thickness=3, windows=(5, 21))
    narrow, wide = rows[0], rows[1]
    assert narrow["recall"] < 0.35
    assert wide["recall"] > 3.0 * narrow["recall"]


def test_gray_world_looks_neutral_while_getting_the_colour_wrong():
    """The metric trap, pinned.

    Gray-world drives the *no-reference* cast measure toward zero by forcing the
    mean to grey — and in doing so moves further from the photograph's real
    colour balance than the faded print it started from.
    """
    rows, faded = rs.evaluate_fade(images=("astronaut", "coffee", "chelsea", "rocket"), runs=1)
    by_name = {r["method"]: r for r in rows}
    grayworld = by_name["Gray-world balance"]
    control = by_name["None (control)"]

    assert rs.colour_cast(rs.correct_gray_world(synth.fade_photo(io.sample("coffee")))) < 1.0
    assert grayworld["cast_error_deg"] > control["cast_error_deg"]
    assert grayworld["chroma"] < control["chroma"]  # it removed colour rather than correcting it


def test_the_saturation_factor_lands_on_the_originals_chroma():
    """The stopping point for the one free parameter, asserted rather than eyeballed."""
    rows, faded = rs.evaluate_fade(images=rs.COLOUR_IMAGES, runs=1)
    recommended = next(r for r in rows if r["method"] == "Stretch + saturate")
    assert recommended["chroma"] == pytest.approx(faded["original_chroma"], rel=0.05)


def test_inpainting_before_fade_correction_beats_the_other_order():
    """Pins the ordering result: 4-5 dB from sequence alone.

    Correcting the fade first stretches the damage along with everything else,
    handing the inpainter a higher-contrast scratch to remove.
    """
    rows = rs.evaluate_pipeline(thickness=3, images=("astronaut", "coffee"))
    by_stage = {r["stage"]: r for r in rows}
    assert by_stage["Inpaint then fade correct"]["psnr_db"] > (
        by_stage["Fade correct then inpaint"]["psnr_db"] + 2.0
    )


def test_neither_half_fixes_the_others_problem():
    """The reason the module has two halves that never share a method."""
    rows = rs.evaluate_pipeline(thickness=3, images=("astronaut", "coffee"))
    by_stage = {r["stage"]: r for r in rows}
    inpaint_only = by_stage["Inpaint only"]
    fade_only = by_stage["Fade correct only"]
    both = by_stage["Inpaint then fade correct"]

    # inpainting removes the damage and leaves the cast
    assert inpaint_only["damage_psnr_db"] > by_stage["Damaged + faded (input)"]["damage_psnr_db"] + 8
    assert inpaint_only["cast_error_deg"] > both["cast_error_deg"]
    # fade correction fixes the cast and leaves the damage
    assert fade_only["cast_error_deg"] < by_stage["Damaged + faded (input)"]["cast_error_deg"]
    assert fade_only["damage_psnr_db"] < inpaint_only["damage_psnr_db"] - 8


# --------------------------------------------------------------------------- #
# the end-to-end entry point
# --------------------------------------------------------------------------- #


def test_restore_detects_a_mask_when_none_is_given():
    """With no mask supplied, the detector must find the damage and the
    inpainter must repair it.

    This used to assert that saturation rises after restoration. That is not
    universally true and `chelsea` is the counterexample: its fur is genuinely
    warm brown, so the grey-world fade corrector reads the subject's own colour
    as a cast and pulls it out, leaving saturation a shade *lower* than the
    faded input (18.93 vs 19.09) and the cast error *worse* (5.77 vs 3.00).
    That is a property of grey-world on a tinted subject, not a bug, and it is
    reported in the README rather than asserted away here.

    What *is* universal across every image tried is the damage repair, so that
    is what this pins: the detector recovers the scratches and inpainting gains
    better than 10 dB over them.
    """
    clean = io.sample("chelsea")
    faded = synth.fade_photo(clean, seed=0)
    damaged, true_mask = synth.add_scratches(faded, thickness=3, seed=0)

    mask, inpainted, restored = rs.restore(damaged)
    assert mask.dtype == np.uint8 and mask.shape == damaged.shape[:2]
    assert inpainted.shape == restored.shape == damaged.shape

    # the detector found the damage rather than returning an empty mask
    recall = float(((mask > 0) & (true_mask > 0)).sum() / max((true_mask > 0).sum(), 1))
    assert recall > 0.6

    # and inpainting actually repaired it. The reference is the *faded* image,
    # not the clean one: fading happened before the scratches, so removing a
    # scratch can only ever recover the faded print underneath it.
    before = rs.psnr_on_damage(damaged, faded, true_mask)
    after = rs.psnr_on_damage(inpainted, faded, true_mask)
    assert after > before + 10.0


def test_restore_uses_a_supplied_mask_verbatim():
    clean = io.sample("chelsea")
    damaged, true_mask = rs.add_damage_and_fade(clean, thickness=3, seed=0)
    mask, _, _ = rs.restore(damaged, mask=true_mask)
    assert np.array_equal(mask, true_mask)


def test_an_empty_mask_changes_nothing_in_the_inpainting_stage():
    """A detector that finds nothing must not corrupt the image."""
    clean = io.sample("coffee")
    empty = np.zeros(clean.shape[:2], np.uint8)
    for name, fn in rs.METHODS.items():
        assert psnr(fn(clean, empty), clean) > 45.0, name


def test_unknown_names_raise_rather_than_silently_defaulting():
    damaged, mask = synth.add_scratches(io.sample("coffee"), thickness=3, seed=0)
    with pytest.raises(KeyError):
        rs.restore(damaged, mask=mask, method="not a method")
    with pytest.raises(KeyError):
        rs.restore(damaged, detector="not a detector")
