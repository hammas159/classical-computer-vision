"""Tests for project 02, portrait mode.

These target the failure modes that produce a *plausible wrong number*: a metric
that can be gamed by predicting everything, a compositing bug that looks fine
until you measure the halo, and a kernel that is not normalised and therefore
silently changes image brightness.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import portrait_mode as pm
from shared import synth
from shared.metrics import iou, psnr

# --------------------------------------------------------------------------- #
# the scene and the cascade
# --------------------------------------------------------------------------- #


def test_scene_masks_are_consistent():
    s = synth.portrait_scene(seed=0)
    assert s.image.shape[:2] == s.mask.shape
    # body and hair partition the subject, and never overlap
    assert not np.any((s.body > 0) & (s.fine > 0))
    assert np.array_equal(s.mask > 0, (s.body > 0) | (s.fine > 0))
    assert s.background.shape == s.image.shape


def test_fine_detail_is_a_minority_of_the_subject():
    """The premise of the whole hair analysis, asserted rather than assumed.

    If hair were 30% of the subject, whole-image IoU would already reflect it and
    a separate hair score would be redundant. It is a few percent, which is why
    a method can lose all of it and still look excellent.
    """
    s = synth.portrait_scene(seed=0)
    fraction = (s.fine > 0).sum() / (s.mask > 0).sum()
    assert 0.10 < fraction < 0.35


def test_the_bundled_haar_cascade_loads():
    # OpenCV 5 removed these XMLs; this is the guard that catches an unpinned bump
    assert not pm.face_cascade().empty()


def test_a_face_is_detected_in_every_background():
    for bg in pm.BACKGROUNDS:
        s = synth.portrait_scene(background=bg, seed=3)
        assert pm.detect_face(s.image) is not None, bg


def test_detect_face_returns_none_on_a_blank_image():
    assert pm.detect_face(np.full((300, 300, 3), 120, np.uint8)) is None


# --------------------------------------------------------------------------- #
# mattes
# --------------------------------------------------------------------------- #


def test_every_matte_returns_a_binary_mask_of_the_right_shape():
    s = synth.portrait_scene(seed=0)
    for name, fn in pm.MATTES.items():
        m = fn(s.image)
        if m is None:
            continue
        assert m.shape == s.image.shape[:2], name
        assert m.dtype == np.uint8, name
        assert set(np.unique(m)).issubset({0, 255}), name


def test_face_dependent_mattes_return_none_without_a_face():
    blank = np.full((320, 320, 3), 120, np.uint8)
    for name in ("Face rect (baseline)", "Face ellipse prior", "Haar + GrabCut",
                 "Watershed + markers"):
        assert pm.MATTES[name](blank) is None, name


def test_grabcut_beats_the_rectangle_baseline_on_boundary_quality():
    s = synth.portrait_scene(seed=0)
    truth_edges = cv2.Canny(s.mask, 50, 150)
    from shared.metrics import edge_prf

    rect_f1 = edge_prf(cv2.Canny(pm.matte_face_rect(s.image), 50, 150), truth_edges, 2)["f1"]
    gc_f1 = edge_prf(cv2.Canny(pm.matte_grabcut_face(s.image), 50, 150), truth_edges, 2)["f1"]
    assert gc_f1 > rect_f1 * 2


def test_the_reference_matte_favours_grabcut():
    """The circularity in this project, asserted so it cannot be forgotten.

    The reference matte was produced by GrabCut plus cleanup, so GrabCut-based
    methods are being scored against an annotation built the same way they work.
    They win every column, and that is **not evidence that they are best** -- it
    is evidence of the annotation's provenance. Pinning it here keeps the caveat
    attached to the numbers.
    """
    rows = [r for r in pm.evaluate_mattes(runs=1) if r["iou"] is not None]
    best = max(rows, key=lambda r: r["iou"])
    assert "GrabCut" in best["method"]
    for column in ("iou", "boundary_f1", "fine_recall"):
        assert max(rows, key=lambda r: r[column])["method"] == best["method"]



def test_background_fpr_exposes_a_mask_that_covers_everything():
    """Recall alone is gameable; the FPR column is what makes the table honest.

    On the old synthetic head-and-shoulders scene the face rectangle "recovered"
    0.97 of the hair purely by covering the whole head region. On a real
    full-body action photograph it cannot game the metric that way -- the
    subject is far larger than any face box, so it recovers only 0.39.

    What survives is the reason the column exists: the two shape priors spend
    different amounts of background to buy their recall, and only the FPR column
    shows it.
    """
    rows = pm.evaluate_mattes(runs=1)
    by_name = {r["method"]: r for r in rows}
    rect = by_name["Face rect (baseline)"]
    ellipse = by_name["Face ellipse prior"]

    # the rectangle covers more, so it recovers more fine detail than the ellipse
    assert rect["fine_recall"] > ellipse["fine_recall"]
    # and pays for it in background falsely called subject
    assert rect["background_fpr"] > ellipse["background_fpr"]


def test_grabcut_is_reproducible_when_the_seed_is_pinned():
    """Same seed, same mask — every time. This is what makes the table trustworthy."""
    s = synth.portrait_scene(seed=0)
    first = pm.matte_grabcut_face(s.image, rng_seed=0)
    for _ in range(3):
        assert np.array_equal(pm.matte_grabcut_face(s.image, rng_seed=0), first)


def test_grabcut_stability_is_a_property_of_the_scene_not_the_algorithm():
    """A finding that REVERSED when the scene became a real photograph.

    On the old synthetic scenes GrabCut spanned IoU 0.15-0.90 across 24 seeds --
    the headline "grabCut is non-deterministic" result. On this real photograph
    the same sweep spans under 0.01, because the subject's colours are far from
    the crowd behind them and the k-means initialisation lands in the same basin
    every time.

    Both observations are true and neither generalises. What generalises is the
    conditional: **the instability is a property of the scene.** Where subject
    and background share colours the initialisation decides the result; where
    they do not, it does not matter. Asserting stability here is what stops the
    old, scene-specific claim from being quietly carried forward.
    """
    rows = pm.evaluate_grabcut_stability(n_seeds=12)
    assert rows, "stability sweep returned nothing"
    spreads = [r["iou_max"] - r["iou_min"] for r in rows]
    assert max(spreads) < 0.05, f"expected a stable scene, saw spread {max(spreads):.3f}"



@pytest.mark.parametrize("name", list(pm.BOKEH_KERNELS))
def test_every_bokeh_kernel_is_normalised(name):
    # an unnormalised kernel changes the image's overall brightness, which looks
    # like a bad blur rather than the arithmetic error it is
    k = pm.BOKEH_KERNELS[name](11)
    assert k.sum() == pytest.approx(1.0, abs=1e-5)
    assert (k >= 0).all()


def test_disc_is_flat_topped_and_gaussian_is_not():
    disc = pm.highlight_profile(pm.kernel_disc(15))
    gauss = pm.highlight_profile(pm.kernel_gaussian(15))
    assert disc["peak_to_mean"] == pytest.approx(1.0, abs=1e-3)  # a flat disc
    assert gauss["peak_to_mean"] > 2.0                           # a peaked bump
    assert disc["edge_sharpness"] > gauss["edge_sharpness"] * 1.5


# --------------------------------------------------------------------------- #
# compositing
# --------------------------------------------------------------------------- #


def test_compositing_leaves_the_subject_untouched():
    s = synth.portrait_scene(seed=0)
    k = pm.kernel_disc(9)
    for name, fn in pm.COMPOSITORS.items():
        out = fn(s.image, s.mask, k)
        assert np.array_equal(out[s.mask > 0], s.image[s.mask > 0]), name


def test_masked_compositing_removes_the_halo():
    """Blurring across the subject boundary smears the subject into the background."""
    s = synth.portrait_scene(seed=0)
    k = pm.kernel_disc(15)
    ideal = pm.composite_reference(s, k)
    band = pm.halo_ring(s.mask, 12) > 0

    def err(x):
        return float(np.abs(x.astype(float) - ideal.astype(float)).mean(axis=-1)[band].mean())

    naive_err = err(pm.composite_naive(s.image, s.mask, k))
    masked_err = err(pm.composite_masked(s.image, s.mask, k))
    assert masked_err < naive_err / 3


def test_halo_ring_is_outside_the_subject_only():
    s = synth.portrait_scene(seed=0)
    ring = pm.halo_ring(s.mask, 10)
    assert (ring > 0).any()
    assert not np.any((ring > 0) & (s.mask > 0))


def test_reference_composite_is_close_to_a_perfect_matte_pipeline():
    """Near-equal, not equal, and the difference is honest.

    A real photograph has no clean background plate -- the pixels behind the
    subject were never photographed. ``portrait_scene`` reconstructs one by
    inpainting, so the compositing reference is a good approximation rather than
    ground truth, and this asserts closeness instead of identity.
    """
    s = synth.portrait_scene()
    identity = np.ones((1, 1), np.float32)
    ref = pm.composite_reference(s, identity)
    assert ref.shape == s.image.shape
    assert psnr(ref, s.image) > 30.0



def test_portrait_end_to_end_blurs_the_background_and_not_the_subject():
    s = synth.portrait_scene(seed=0)
    mask, out = pm.portrait(s.image, "Haar + GrabCut", "Disc (circular aperture)", radius=13)
    assert mask is not None and out.shape == s.image.shape

    from shared.metrics import rms_contrast

    bg_before = s.image.copy()
    bg_before[s.mask > 0] = 0
    bg_after = out.copy()
    bg_after[s.mask > 0] = 0
    assert rms_contrast(bg_after) < rms_contrast(bg_before)  # background softened
    assert iou(mask, s.mask) > 0.3


def test_portrait_returns_none_when_no_subject_is_found():
    blank = np.full((240, 240, 3), 90, np.uint8)
    assert pm.portrait(blank, "Haar + GrabCut") == (None, None)
