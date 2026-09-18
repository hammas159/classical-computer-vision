"""Tests for project 56: hand gesture recognition.

They pin the result (the background moves the answer further than the hand does;
skin-likeness predicts it; a drawn ellipse beats two real segmenters; the finger
rule fails even on a perfect mask), the mechanism (the matte really is exact, and
the composite is built from it), the construction (labels come from the dataset
and are not invented, the cleanup is shared, no method sees the truth except the
oracle) and the degenerate inputs that broke the palm circle.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import gestures as gs  # noqa: E402

pytestmark = pytest.mark.skipif(
    not gs.available(),
    reason="run `python tools/fetch_assets.py --set hands` to fetch the photographs")


# --------------------------------------------------------------------------- #
# the data and its labels
# --------------------------------------------------------------------------- #


def test_there_are_enough_hands_and_people():
    hands = gs.hand_names()
    assert len(hands) >= 20
    assert len({gs.person_of(h) for h in hands}) >= 5


def test_the_gesture_label_comes_from_the_dataset():
    """Parsed from the filename the dataset shipped, never inferred."""
    for h in gs.hand_names():
        g = gs.gesture_of(h)
        assert g and h.startswith(g)
        assert len(g) <= 2


def test_only_the_numbered_gestures_get_a_finger_count():
    """The letters are deliberately not given one.

    Whether the thumb counts as extended in `A` or `S` is a judgement, and
    inventing an answer would turn the dataset's label into this project's.
    """
    for h in gs.hand_names():
        want = gs.fingers_of(h)
        if gs.gesture_of(h).isdigit():
            assert want == int(gs.gesture_of(h))
        else:
            assert want is None


def test_every_background_exists_and_is_distinct():
    seen = set()
    for b in gs.BACKGROUNDS:
        img = gs.load_background(b)
        assert img.ndim == 3
        seen.add(img.tobytes()[:4096])
    assert len(seen) == len(gs.BACKGROUNDS)


def test_the_backgrounds_span_the_difficulty_axis():
    """The axis has to have a range or the project measures nothing."""
    likeness = [gs.skin_likeness(b) for b in gs.BACKGROUNDS]
    assert min(likeness) < 0.05
    assert max(likeness) > 0.80


# --------------------------------------------------------------------------- #
# the matte is exact, and the composite is built from it
# --------------------------------------------------------------------------- #


def test_the_matte_is_binary_and_not_empty():
    for h in gs.hand_names()[:6]:
        m = gs.matte(gs.load_hand(h))
        assert set(np.unique(m)).issubset({0, 255})
        assert 0.02 < (m > 0).mean() < 0.8, h


def test_the_matte_is_one_connected_blob():
    """A speck of sleeve left in a corner would make the exact truth inexact."""
    for h in gs.hand_names()[:6]:
        m = gs.matte(gs.load_hand(h))
        n, _, _, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
        assert n == 2, (h, n)


def test_the_composite_pixels_come_from_the_hand_exactly_where_the_mask_says():
    """The construction, asserted directly: inside the mask the frame is the
    hand, outside it the frame is the background. If that ever stopped being
    true, every IoU in this project would be measuring something else."""
    hand, bg = gs.hand_names()[0], list(gs.BACKGROUNDS)[0]
    frame, mask = gs.composite(hand, bg)
    plain = gs.load_background(bg)
    outside = mask == 0
    assert np.array_equal(frame[outside], plain[outside])
    assert (mask > 0).sum() > 100


def test_the_composite_is_deterministic():
    a, _ = gs.composite(gs.hand_names()[0], list(gs.BACKGROUNDS)[0])
    b, _ = gs.composite(gs.hand_names()[0], list(gs.BACKGROUNDS)[0])
    assert np.array_equal(a, b)


def test_the_hand_lands_inside_the_frame():
    for h in gs.hand_names()[:4]:
        for b in list(gs.BACKGROUNDS)[:3]:
            frame, mask = gs.composite(h, b)
            assert mask.shape == frame.shape[:2]
            ys, xs = np.nonzero(mask)
            assert ys.min() >= 0 and xs.min() >= 0
            assert ys.max() < mask.shape[0] and xs.max() < mask.shape[1]


# --------------------------------------------------------------------------- #
# the controls
# --------------------------------------------------------------------------- #


def test_the_oracle_is_the_only_method_given_the_truth():
    """Every other entry must ignore the `truth` keyword entirely.

    Asserted by handing them a deliberately wrong mask and checking nothing
    changes -- a leak here would quietly turn a segmenter into an oracle.
    """
    frame, truth = gs.composite(gs.hand_names()[0], list(gs.BACKGROUNDS)[0])
    lie = np.zeros_like(truth)
    for name in gs.REAL_SEGMENTERS:
        a = gs.SEGMENTERS[name](frame, truth=truth)
        b = gs.SEGMENTERS[name](frame, truth=lie)
        assert np.array_equal(a, b), name


def test_the_oracle_scores_a_perfect_iou():
    r = gs.evaluate("Oracle mask (control)")
    assert r["mean_iou"] == pytest.approx(1.0)
    assert r["median_feature_gap"] == pytest.approx(0.0, abs=1e-9)


def test_the_nothing_control_scores_zero():
    r = gs.evaluate("Nothing (control)")
    assert r["mean_iou"] == 0.0
    assert r["found"] == 0


def test_a_drawn_ellipse_beats_some_real_segmenters():
    """The control that earns its place.

    An ellipse placed where a hand usually is, with the image never read, beats
    at least one method that does read it.
    """
    results = {r["segmenter"]: r for r in gs.evaluate_all()}
    ellipse = results["Centre ellipse (control)"]["mean_iou"]
    beaten = [k for k in gs.REAL_SEGMENTERS if results[k]["mean_iou"] < ellipse]
    assert beaten, "the ellipse control beat nothing, so it is not fencing anything in"


def test_the_whole_frame_control_does_not_crash_the_palm_circle():
    """A regression test for a real bug.

    `distanceTransform` on a mask with no background pixels returns FLT_MAX for
    every pixel. That is finite, so it passed the first guard, and then
    overflowed the integer cast inside the finger counter. The whole-frame
    control is exactly that input.
    """
    frame, _ = gs.composite(gs.hand_names()[0], list(gs.BACKGROUNDS)[0])
    full = gs.control_whole_frame(frame)
    centre, radius = gs.palm_circle(full)
    assert np.isfinite(radius)
    assert radius <= 0.5 * min(full.shape[:2])
    assert isinstance(gs.count_fingers(full), int)


def test_counting_fingers_on_an_empty_mask_is_zero():
    assert gs.count_fingers(np.zeros((100, 100), np.uint8)) == 0


# --------------------------------------------------------------------------- #
# the result
# --------------------------------------------------------------------------- #


def test_the_background_moves_the_answer_further_than_the_hand():
    """The headline, measured on the full grid of every hand on every
    background so that only one thing varies at a time."""
    m = gs.which_matters_more(gs.REAL_SEGMENTERS[2])
    assert m["background_spread"] > 1.5 * m["hand_spread"], m


def test_skin_likeness_predicts_the_score():
    """And it is computed from the background alone, before any composite is
    made, so it predicts rather than explains."""
    m = gs.which_matters_more("YCrCb skin")
    assert m["skin_likeness_r"] < -0.5, m


def test_the_hardest_background_is_the_most_skin_coloured_one():
    rows = gs.background_effect("YCrCb skin")
    worst = min(rows, key=lambda r: r["mean_iou"])
    assert worst["skin_likeness"] > 0.8, worst


def test_at_least_one_segmenter_recovers_most_of_the_hands():
    """If nothing worked, the comparison would have no top end."""
    results = gs.evaluate_all()
    best = max((r for r in results if r["segmenter"] in gs.REAL_SEGMENTERS),
               key=lambda r: r["mean_iou"])
    assert best["mean_iou"] > 0.6
    assert best["found"] >= 0.7 * best["frames"]


def test_a_worse_mask_costs_the_shape_description():
    """The two stages are connected, and this is the number that connects them.

    Better masks must give shape features closer to the ones the exact matte
    produces, or the segmentation result would have no downstream meaning.
    """
    rows = [r for r in gs.evaluate_all() if r["segmenter"] in gs.REAL_SEGMENTERS]
    ious = [r["mean_iou"] for r in rows]
    gaps = [r["median_feature_gap"] for r in rows]
    assert float(np.corrcoef(ious, gaps)[0, 1]) < -0.7


def test_the_hardest_background_fails_as_a_size_contest_not_a_degradation():
    """The mechanism behind the worst background, and a correction.

    The first explanation written for `a black dog on grass` was that the hand
    *merges* into the dog. It does not. Every segmenter keeps the largest
    connected component, and that photograph contains a varnished orange-red
    cart which reads as skin -- so when the hand is smaller than the cart, the
    rule selects the cart and the score is not degraded but **zero**.

    The prediction that follows is specific: on such a background the score
    should track how much of the frame the hand covers, and the failures should
    be genuinely bimodal rather than merely low.
    """
    c = gs.size_contest("247012", "Lab skin")
    assert c["area_vs_iou_r"] > 0.6, c
    assert c["total_failures"] >= 5, c
    assert c["mean_area_when_failed"] < c["mean_area_when_not"], c
    # bimodal: a mean well above zero with a median at zero
    assert c["mean_iou"] > 0.2 and c["median_iou"] < 0.05, c


def test_an_easy_background_has_no_total_failures():
    """The same measurement where the mechanism cannot fire, so the test above
    is not just asserting that segmentation is hard."""
    rows = {r["background"]: r for r in gs.background_effect("Lab skin")}
    assert rows["314016"]["total_failures"] == 0, rows["314016"]


def test_the_mean_and_median_are_both_reported():
    """A mean of 0.363 with a median of 0.000 describes neither half of a
    bimodal result, so both are carried through to the tables."""
    for r in gs.background_effect("Lab skin"):
        assert "mean_iou" in r and "median_iou" in r and "total_failures" in r


# --------------------------------------------------------------------------- #
# the second stage does not work even with a perfect mask
# --------------------------------------------------------------------------- #


def test_finger_counting_fails_on_a_perfect_mask():
    """Settled before any segmenter is blamed for anything.

    Two rules and ten settings between them, all run on the exact matte. The
    best manages three of five. If this ever passes comfortably the project's
    framing has changed and the README is wrong.
    """
    ceiling = gs.finger_rule_ceiling()
    assert ceiling["best_correct"] < ceiling["of"], ceiling
    assert len(ceiling["sweeps"]) >= 8


def test_the_palm_circle_rule_beats_raw_defect_counting():
    """The repair that was worth making, kept as a test so it stays made.

    Counting convexity defects directly returns three or four whatever the hand
    is doing, because the forearm is in the silhouette.
    """
    ceiling = gs.finger_rule_ceiling()
    best_defects = max(r["correct"] for r in ceiling["sweeps"]
                       if r["rule"].startswith("convexity"))
    best_circle = max(r["correct"] for r in ceiling["sweeps"]
                      if r["rule"].startswith("palm"))
    assert best_circle > best_defects, (best_circle, best_defects)


def test_the_palm_radius_is_scale_free():
    """The count must not change when the same hand is photographed larger.

    Scaling the mask scales the palm radius with it, so the circle lands in the
    same place relative to the fingers -- the same fractions-not-pixels point
    project 06 makes about a region of interest.
    """
    _, truth = gs.composite("5_P__5_P_hgr1_id04_2", list(gs.BACKGROUNDS)[0])
    big = cv2.resize(truth, None, fx=1.7, fy=1.7, interpolation=cv2.INTER_NEAREST)
    assert gs.count_fingers(truth) == gs.count_fingers(big)


# --------------------------------------------------------------------------- #
# the matte's own error is reported rather than assumed away
# --------------------------------------------------------------------------- #


def test_the_matte_quality_is_measured_and_mostly_clean():
    rows = gs.matte_quality()
    halos = [r["halo"] for r in rows]
    assert float(np.median(halos)) < 0.03
    assert rows[0]["halo"] >= rows[-1]["halo"]


def test_the_shared_cleanup_is_shared():
    """Every segmenter gets the identical morphology, so the comparison is of
    the colour rule rather than of who cleaned up harder."""
    source = Path(gs.__file__).read_text(encoding="utf-8")
    for name in ("segment_ycrcb", "segment_hsv", "segment_lab", "segment_otsu",
                 "segment_adaptive"):
        body = source.split(f"def {name}(")[1].split("\ndef ")[0]
        assert "_largest(" in body, name
