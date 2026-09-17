"""Tests for project 19, keypoint detectors.

Most of these pin a *finding* rather than a number, so that a later change which
quietly reverses one of the project's conclusions fails loudly instead of
rewriting the README by accident.
"""

from __future__ import annotations

import numpy as np
import pytest

import keypoints as kp
from shared.io import to_gray
from shared.metrics import repeatability


# --------------------------------------------------------------------------- #
# the measurement -- this was wrong, and it was wrong in a flattering direction
# --------------------------------------------------------------------------- #


def test_repeatability_ignores_keypoints_the_rotation_pushed_out_of_frame():
    """A keypoint that is no longer in the picture cannot be redetected.

    `apply_homography` keeps the canvas size, so rotating a rectangle carries
    its corners outside the view — at 45 degrees roughly a third of the image is
    gone. Counting those as misses measures the crop, not the detector, and made
    every detector's rotation curve sag for a reason that has nothing to do with
    rotation invariance.
    """
    img = kp.load_scene("glass_pyramid")
    gray = to_gray(img)
    H = kp.homography_rotation(gray.shape, 45.0)
    wgray = to_gray(kp.apply_homography(img, H))
    a, b = kp.kp_harris(gray), kp.kp_harris(wgray)

    cropped = repeatability(a, b, H, kp.MATCH_THRESHOLD)
    fair = repeatability(a, b, H, kp.MATCH_THRESHOLD, shape=wgray.shape)
    assert fair > cropped + 0.1


def test_an_identity_transform_is_perfectly_repeatable():
    """The sanity check the whole table rests on. If this is not 1.0, nothing else
    in the project means anything."""
    rows = kp.evaluate_detectors("rotation", 0.0, images=kp.IMAGES[:4], runs=1)
    for r in rows:
        assert r["repeatability"] == pytest.approx(1.0), r["detector"]


def test_an_unknown_transform_raises():
    with pytest.raises(ValueError, match="unknown transform"):
        kp.evaluate_detectors("shear", 1.0, images=kp.IMAGES[:1], runs=1)


# --------------------------------------------------------------------------- #
# the detectors
# --------------------------------------------------------------------------- #


def test_every_detector_returns_xy_pairs():
    gray = to_gray(kp.load_scene("bobcat_rock"))
    for name, fn in kp.DETECTORS.items():
        pts = fn(gray)
        assert pts.ndim == 2 and pts.shape[1] == 2, name
        assert np.isfinite(pts).all(), name


def test_a_blank_image_yields_almost_no_keypoints():
    """There is nothing to detect in a flat field, and a detector that finds
    hundreds of corners there is responding to nothing."""
    flat = np.full((240, 320), 128, np.uint8)
    for name, fn in kp.DETECTORS.items():
        assert len(fn(flat)) < 20, f"{name} invented keypoints in a flat image"


# --------------------------------------------------------------------------- #
# the findings
# --------------------------------------------------------------------------- #


def test_harris_is_the_most_repeatable_detector():
    """The project's headline, and it is not what the folklore predicts.

    Harris is from 1988, has no scale selection, no orientation and no
    descriptor. It is nonetheless the most repeatable *detector* in the table —
    more repeatable than SIFT, ORB, AKAZE and BRISK.

    The reason is that the modern detectors' invariance lives in their
    **descriptors**, not in where they put their keypoints. This project only
    measures the latter. See the README's limitation on exactly that point, and
    project 25 for the matching side.
    """
    rows = kp.evaluate_detectors("rotation", 30.0, images=kp.IMAGES[:6], runs=1)
    best = max(rows, key=lambda r: r["repeatability"])
    assert best["detector"] == "Harris"

    scored = {r["detector"]: r["repeatability"] for r in rows}
    for other in ("SIFT", "ORB", "AKAZE", "BRISK", "FAST"):
        assert scored["Harris"] > scored[other], other


def test_harris_does_not_win_by_finding_more_keypoints():
    """The obvious objection to the table above, answered with data.

    Harris returns ~1000 keypoints and AKAZE ~350, and more keypoints means more
    chances for one to land inside the 3 px match radius. If that were driving
    the result, randomly discarding Harris keypoints would lower its score. It
    does not: the line is flat from 100 to 1000.
    """
    rows = kp.sweep_keypoint_budget(images=kp.IMAGES[:6])
    scores = [r["repeatability"] for r in rows]
    assert max(scores) - min(scores) < 0.05


def test_harris_trades_frame_coverage_for_repeatability():
    """The qualification that keeps the headline honest.

    Harris is the most repeatable detector and covers the *least* of the frame:
    33.5% of an 8x8 grid against 58-81% for everything else. It concentrates on
    the strongest corners, which are also the most stable ones — so the property
    that wins it the repeatability column is the same property that makes it a
    poor choice for estimating a global transform.
    """
    rows = kp.evaluate_detectors("rotation", 30.0, images=kp.IMAGES[:6], runs=1)
    scored = {r["detector"]: r for r in rows}

    assert scored["Harris"]["repeatability"] == max(r["repeatability"] for r in rows)
    assert scored["Harris"]["coverage_pct"] == min(r["coverage_pct"] for r in rows)
    assert scored["Harris"]["coverage_pct"] < 0.6 * scored["SIFT"]["coverage_pct"]


def test_orb_is_both_faster_and_more_repeatable_than_sift():
    """"ORB is faster than SIFT but less accurate" — half of that is wrong here.

    ORB is ~6x faster *and* scores higher on detector repeatability. The
    accuracy half of the folklore is about descriptor matching, which this
    project does not measure.
    """
    rows = kp.evaluate_detectors("rotation", 30.0, images=kp.IMAGES[:6], runs=1)
    sift = next(r for r in rows if r["detector"] == "SIFT")
    orb = next(r for r in rows if r["detector"] == "ORB")

    assert orb["median_ms"] * 3 < sift["median_ms"]
    assert orb["repeatability"] > sift["repeatability"]


def test_right_angle_rotations_test_nothing():
    """Why the headline runs at 30 degrees and not 90.

    A 90 or 180 degree rotation is a transpose and a flip of the pixel grid: it
    is exact, with no resampling at all. Every detector scores near-perfectly,
    including the ones with no rotation invariance whatsoever, so the number is
    about arithmetic rather than about the method.
    """
    rows = kp.sweep_rotation(images=kp.IMAGES[:4], angles=(30.0, 180.0))
    at_30 = next(r for r in rows if r["rotation_deg"] == 30.0)
    at_180 = next(r for r in rows if r["rotation_deg"] == 180.0)

    for det in ("Harris", "FAST", "Shi-Tomasi"):
        assert at_180[det] > at_30[det], det
    assert at_180["Harris"] > 0.95


def test_scale_invariance_is_asymmetric():
    """Shrinking and enlarging are different problems, and the ranking flips.

    Harris holds up far better when the image is *enlarged* (detail is
    interpolated, corners stay corners) than when it is shrunk (detail is
    destroyed). ORB is the other way round. A single "scale invariance" number
    would average these into nonsense.
    """
    rows = kp.sweep_scale(images=kp.IMAGES[:6], scales=(0.6, 1.6))
    down = next(r for r in rows if r["scale"] == 0.6)
    up = next(r for r in rows if r["scale"] == 1.6)

    assert up["Harris"] > down["Harris"] + 0.1      # Harris prefers enlarging
    assert down["ORB"] > up["ORB"] + 0.1            # ORB prefers shrinking


def test_sift_is_the_least_stable_detector_under_noise():
    """Counterintuitive, consistent across the sweep, and worth stating plainly.

    SIFT locates keypoints as extrema of a difference-of-Gaussian scale space.
    An extremum is a comparison between neighbouring samples, so noise moves it;
    FAST's segment test on a 16-pixel ring is a vote, and votes are stabler.
    """
    rows = sorted(kp.sweep_noise(images=kp.IMAGES[:6]), key=lambda r: r["noise_sigma"])
    worst = rows[-1]
    assert worst["SIFT"] == min(worst[d] for d in kp.DETECTORS)
    assert worst["FAST"] > worst["SIFT"] + 0.3


def test_brisk_is_the_worst_trade_in_the_table():
    """Slowest by a wide margin *and* near the bottom on repeatability.

    Reported because a comparison that only ever finds trade-offs is not being
    honest — sometimes one option is simply dominated.
    """
    rows = kp.evaluate_detectors("rotation", 30.0, images=kp.IMAGES[:6], runs=1)
    scored = {r["detector"]: r for r in rows}
    brisk = scored["BRISK"]

    assert brisk["median_ms"] == max(r["median_ms"] for r in rows)
    assert brisk["repeatability"] < scored["Harris"]["repeatability"]
    assert brisk["repeatability"] < scored["ORB"]["repeatability"]
    assert brisk["median_ms"] > 5 * scored["ORB"]["median_ms"]
