"""Tests for project 01, the document scanner.

The point of these is to catch the failures that produce a *plausible wrong
number* rather than a crash: corners returned in the wrong order, a homography
that silently mirrors the page, or a metric that rewards a detector for giving
up.
"""

from __future__ import annotations

import numpy as np
import pytest

import document_scanner as ds
from shared import synth
from shared.io import to_gray
from shared.metrics import corner_error, iou

# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #


def test_order_corners_is_canonical_regardless_of_input_order():
    canonical = np.float32([[10, 20], [200, 30], [190, 260], [20, 250]])
    rng = np.random.default_rng(0)
    for _ in range(10):
        shuffled = canonical[rng.permutation(4)]
        assert np.allclose(ds.order_corners(shuffled), canonical)


def test_order_corners_is_idempotent():
    c = np.float32([[10, 20], [200, 30], [190, 260], [20, 250]])
    assert np.allclose(ds.order_corners(ds.order_corners(c)), ds.order_corners(c))


def test_rectify_recovers_a_known_warp():
    # warp a flat page by a known homography, then rectify with the true corners:
    # the result must match the original page closely
    _photo, page, _truth = synth.document_scene(seed=0)
    h, w = page.shape[:2]
    corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    out = ds.rectify(page, corners, (w, h))
    assert out.shape == page.shape
    assert np.abs(out.astype(int) - page.astype(int)).mean() < 1.0


def test_quad_mask_area_matches_the_quadrilateral():
    corners = np.float32([[10, 10], [110, 10], [110, 60], [10, 60]])
    m = ds.quad_mask(corners, (200, 200))
    assert abs(int((m > 0).sum()) - 100 * 50) < 250  # rasterisation tolerance


# --------------------------------------------------------------------------- #
# detectors
# --------------------------------------------------------------------------- #


def test_every_detector_returns_four_ordered_corners_or_none():
    photo, _page, _truth = synth.document_scene(seed=0)
    for name, fn in ds.DETECTORS.items():
        out = fn(photo)
        if out is None:
            continue
        assert out.shape == (4, 2), name
        assert np.allclose(out, ds.order_corners(out)), f"{name} returned unordered corners"


def test_quad_detectors_land_within_ten_pixels_on_a_clean_scene():
    # minAreaRect is excluded on purpose: it fits a rectangle to a perspective
    # quadrilateral and cannot be accurate. That is what it is here to show.
    photo, _page, truth = synth.document_scene(seed=0)
    for name, fn in ds.DETECTORS.items():
        if name.startswith("minAreaRect"):
            continue
        corners = fn(photo)
        assert corners is not None, f"{name} found no page at all"
        assert corner_error(corners, truth) < 10.0, name


def test_minarea_rect_baseline_is_measurably_worse():
    photo, _page, truth = synth.document_scene(seed=0)
    rect_err = corner_error(ds.detect_minarea_rect(photo), truth)
    quad_err = corner_error(ds.detect_otsu_contour(photo), truth)
    assert rect_err > quad_err * 3


def test_detector_survives_a_blank_image_without_crashing():
    blank = np.full((240, 320, 3), 128, np.uint8)
    for name, fn in ds.DETECTORS.items():
        out = fn(blank)
        assert out is None or out.shape == (4, 2), name


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def test_usable_rate_does_not_reward_a_detector_for_failing():
    # a detector that never returns anything must score 0 usable, not 100%
    s = ds.DetectorScore("never-detects", n_scenes=5, n_success=0)
    assert s.success_rate == 0.0
    assert s.usable_rate == 0.0

    # one good detection out of five is 20% usable, even though the *errors*
    # list contains only that one good value
    s2 = ds.DetectorScore("one-good", n_scenes=5, n_success=1, corner_errors=[1.0])
    assert s2.usable_rate == pytest.approx(0.2)
    assert s2.success_rate == pytest.approx(0.2)


def test_evaluate_detectors_returns_a_row_per_method():
    rows = ds.evaluate_detectors(n_scenes=2, runs=1, warmup=0)
    assert len(rows) == len(ds.DETECTORS)
    assert {r["method"] for r in rows} == set(ds.DETECTORS)
    for r in rows:
        assert 0.0 <= r["success_rate"] <= 1.0
        assert r["usable_rate"] <= r["success_rate"] + 1e-9


# --------------------------------------------------------------------------- #
# binarisation
# --------------------------------------------------------------------------- #


def test_binarisers_return_a_two_valued_image():
    _photo, page, _truth = synth.document_scene(seed=0)
    gray = to_gray(page)
    for name, fn in ds.BINARISERS.items():
        out = fn(gray)
        assert out.dtype == np.uint8, name
        assert set(np.unique(out)).issubset({0, 255}), name


def _rectified_page(seed: int, illum_min: float):
    """Rectify a scene with its true corners and return (gray page, true text mask)."""
    photo, page, truth = synth.document_scene(seed=seed, illum_min=illum_min)
    h, w = page.shape[:2]
    gray = to_gray(ds.rectify(photo, truth, (w, h)))
    return gray, ds.text_mask(to_gray(page))


def _text_iou(seed: int, illum_min: float, fn):
    gray, truth_text = _rectified_page(seed, illum_min)
    return iou(255 - fn(gray), truth_text)


def test_page_illumination_ratio_is_one_under_flat_light():
    photo, _page, truth = synth.document_scene(seed=0, illum_min=1.0)
    H, W = photo.shape[:2]
    assert ds.page_illumination_ratio(truth, (W, H), 1.0) == pytest.approx(1.0)
    # and strictly less than one once a gradient exists
    photo2, _p2, truth2 = synth.document_scene(seed=0, illum_min=0.2)
    assert ds.page_illumination_ratio(truth2, (W, H), 0.2) < 0.9


def test_otsu_is_optimal_under_flat_light():
    """Under even lighting a global threshold is not merely adequate — it is best.

    The bound is 0.85 rather than 0.95 because the page now carries **real
    rendered glyphs** instead of 4 px bars. Thin anti-aliased strokes are
    genuinely harder to binarise, so every method's ceiling dropped — the
    *ordering*, which is what this test is about, did not.
    """
    flat = _text_iou(0, 1.0, ds.binarise_otsu)
    assert flat > 0.85
    assert flat > _text_iou(0, 1.0, ds.binarise_sauvola)


def test_otsu_collapses_in_deep_shadow_while_local_methods_hold():
    hard = ds.ILLUM_LEVELS[-1]
    otsu_flat = _text_iou(0, 1.0, ds.binarise_otsu)
    otsu_dark = _text_iou(0, hard, ds.binarise_otsu)
    sauvola_flat = _text_iou(0, 1.0, ds.binarise_sauvola)
    sauvola_dark = _text_iou(0, hard, ds.binarise_sauvola)

    # Bounds reflect real glyphs rather than solid bars: Otsu falls from 0.902
    # to 0.678 across the sweep instead of 0.999 to 0.430. Still a cliff, and
    # still in the same direction -- the drop is what matters, not its size.
    assert otsu_dark < 0.75
    assert otsu_flat - otsu_dark > 0.15            # a cliff, not a slope
    assert abs(sauvola_flat - sauvola_dark) < 0.15  # a local threshold barely moves
    assert sauvola_dark > otsu_dark


def test_the_oracle_shows_otsu_fails_while_a_global_cut_still_exists():
    """The finding that distinguishes this project from the textbook account.

    The usual explanation for an Otsu failure under uneven light is that ink and
    paper overlap, so no global threshold can separate them. The oracle — the
    best global threshold found by exhaustive search — tests that claim directly,
    and at this illumination it still scores well. So the separation is available
    and Otsu's criterion is what fails, not global thresholding itself.
    """
    hard = ds.ILLUM_LEVELS[-1]
    gray, truth_text = _rectified_page(0, hard)

    otsu_iou = iou(255 - ds.binarise_otsu(gray), truth_text)
    oracle, t = ds.binarise_best_global(gray, truth_text)
    oracle_iou = iou(255 - oracle, truth_text)

    assert 0 < t < 255
    assert oracle_iou > 0.85, "a global threshold should still work at this ratio"
    assert otsu_iou < 0.6, "Otsu should nonetheless have failed"
    assert oracle_iou - otsu_iou > 0.3


def test_a_perfect_global_cut_is_impossible_below_the_ink_reflectance():
    """The theoretical bound, checked rather than assumed.

    Paper reflects 245 and ink 45, so a shading ratio below 45/245 puts the
    darkest paper below the brightest ink and no global threshold can separate
    them. This asserts the arithmetic the sweep's reference line depends on.
    """
    paper, ink = 245.0, 45.0
    assert ds.INK_REFLECTANCE == pytest.approx(ink / paper)
    ratio = ds.INK_REFLECTANCE * 0.9  # just past the limit
    assert paper * ratio < ink        # darkest paper is now darker than brightest ink


def test_text_mask_marks_dark_pixels_only():
    _photo, page, _truth = synth.document_scene(seed=0)
    gray = to_gray(page)
    m = ds.text_mask(gray)
    assert m.max() == 255
    assert gray[m > 0].max() < 128
    assert gray[m == 0].min() >= 128


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #


def test_perspective_aspect_recovery_is_near_exact_on_true_corners():
    """A rectangle's true aspect ratio is recoverable from four corners alone.

    This works only because the scene is generated by a real pinhole camera. A
    homography stitched together from four arbitrary corners is not necessarily
    the image of any rectangle, and the closed form then has nothing to recover.
    """
    _photo, page, _ = synth.document_scene(seed=0)
    true_ratio = page.shape[1] / page.shape[0]

    for seed in range(5):
        photo, _page, truth = synth.document_scene(seed=seed)
        recovered = ds.aspect_from_perspective(truth, photo.shape)
        assert recovered is not None, f"seed {seed} was reported degenerate"
        assert abs(recovered - true_ratio) / true_ratio < 0.02


def test_edge_length_aspect_is_measurably_worse_than_perspective():
    _photo, page, _ = synth.document_scene(seed=0)
    true_ratio = page.shape[1] / page.shape[0]
    rows = ds.evaluate_aspect_recovery(n_scenes=8)
    edges, persp = rows[0], rows[1]
    assert persp["rel_error_pct"] < 1.0
    assert edges["rel_error_pct"] > persp["rel_error_pct"] * 3
    assert edges["true_ratio"] == pytest.approx(true_ratio, abs=1e-3)


def test_aspect_from_perspective_returns_none_for_an_affine_view():
    # a parallelogram has its vanishing points at infinity: focal length and
    # aspect ratio are not recoverable, and the function must say so rather
    # than return a confident wrong number
    parallelogram = np.float32([[100, 100], [300, 80], [340, 260], [140, 280]])
    assert ds.aspect_from_perspective(parallelogram, (400, 500)) is None


def test_scan_end_to_end_recovers_the_page_aspect_ratio():
    photo, page, _truth = synth.document_scene(seed=1)
    corners, rect, binary = ds.scan(photo, "Otsu + contour", "Sauvola")
    assert corners is not None and rect is not None and binary is not None
    assert binary.ndim == 2
    true_ar = page.shape[1] / page.shape[0]
    assert 0.85 * true_ar < rect.shape[1] / rect.shape[0] < 1.15 * true_ar


def test_scan_returns_none_when_no_page_is_present():
    blank = np.full((200, 200, 3), 30, np.uint8)
    corners, rect, binary = ds.scan(blank, "Canny + contour")
    assert (corners, rect, binary) == (None, None, None)
