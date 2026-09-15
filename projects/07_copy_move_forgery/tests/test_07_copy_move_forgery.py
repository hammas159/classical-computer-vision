"""Tests for project 07, copy-move forgery detection.

Most of these pin a *finding* rather than a number, so that a later "improvement"
which quietly reverses one of the project's conclusions fails loudly instead of
rewriting the README by accident.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import forgery as fg
from shared import io, synth
from shared.metrics import iou

IMAGES = ("astronaut", "coffee")


# --------------------------------------------------------------------------- #
# the generator
# --------------------------------------------------------------------------- #


def test_the_pasted_region_is_an_exact_duplicate():
    """Everything downstream assumes this. If it is false, nothing is measurable."""
    clean = io.sample("astronaut")
    f = synth.copy_move_forgery(clean, size=96, seed=0)
    sx, sy = f.src_xy
    dx, dy = f.dst_xy
    src = f.image[sy : sy + 96, sx : sx + 96]
    dst = f.image[dy : dy + 96, dx : dx + 96]
    assert np.array_equal(src, dst)


def test_mask_marks_the_paste_and_mask_both_marks_the_pair():
    clean = io.sample("coffee")
    f = synth.copy_move_forgery(clean, size=96, seed=1)
    assert f.mask.sum() / 255 == 96 * 96
    assert f.mask_both.sum() > f.mask.sum()
    # mask_both must contain mask
    assert np.all(f.mask_both[f.mask > 0] == 255)


def test_a_rotated_paste_is_filled_with_real_content_not_reflection():
    """Guards a generator bug that silently capped recall for every method.

    Rotating a size x size patch *in place* leaves the corners to BORDER_REFLECT,
    so roughly 20% of a 15-degree "forgery" was not a duplicate of anything. No
    detector could find it, and the rotation sweep measured that instead of the
    detectors. The generator now samples a window wide enough to crop from.
    """
    clean = io.sample("astronaut")
    f = synth.copy_move_forgery(clean, size=96, angle_deg=15.0, seed=0)
    dx, dy = f.dst_xy
    pasted = f.image[dy : dy + 96, dx : dx + 96]
    # every corner of the paste must be real image content, so the transform
    # RANSAC recovers from real keypoints must be the one that was applied
    fit = fg.describe_match(f.image)
    assert fit["inliers"] >= 4
    assert abs(abs(fit["angle_deg"]) - 15.0) < 2.5
    assert pasted.std() > 5.0  # not a flat reflection-filled square


@pytest.mark.parametrize("scale", [0.8, 1.0, 1.5])
def test_the_generator_survives_every_scale(scale):
    """Pins a broadcast bug: scale > 1 asked for a source window smaller than the crop."""
    f = synth.copy_move_forgery(io.sample("coffee"), size=96, scale=scale, seed=0)
    assert f.image.shape == io.sample("coffee").shape
    assert f.mask.sum() > 0


def test_textured_source_beats_a_uniform_random_one():
    """The source patch has to contain something, or the scores measure the RNG."""
    clean = io.sample("astronaut")
    textured = synth.copy_move_forgery(clean, size=96, seed=0, textured_source=True)
    sx, sy = textured.src_xy
    patch_std = float(io.to_gray(clean)[sy : sy + 96, sx : sx + 96].std())
    assert patch_std > float(io.to_gray(clean).std()) * 0.7


# --------------------------------------------------------------------------- #
# the detectors
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", list(fg.METHODS))
def test_every_detector_returns_a_binary_mask_of_the_right_shape(name):
    f = synth.copy_move_forgery(io.sample("chelsea"), size=96, seed=0)
    out = fg.METHODS[name](f.image)
    assert out.dtype == np.uint8 and out.shape == f.image.shape[:2]
    assert set(np.unique(out)) <= {0, 255}


def test_block_matching_is_near_perfect_on_an_exact_copy():
    rows = fg.evaluate_methods(images=IMAGES)
    block = next(r for r in rows if r["method"] == "Block matching")
    assert block["iou"] > 0.95


def test_block_matching_needs_stride_one():
    """Pins the bug that made this method look useless.

    A block at (x, y) in the source has its duplicate at (x+dx, y+dy). If the
    stride does not divide both dx and dy, the copy is never sampled and the two
    blocks describe different pixels. At stride 8 the method scored 0.068 on an
    exact copy; at stride 1 it scores 0.99. This is not a speed/quality dial.
    """
    f = synth.copy_move_forgery(io.sample("astronaut"), size=96, seed=0)
    fine = fg.detect_block_matching(f.image, stride=1)
    coarse = fg.detect_block_matching(f.image, stride=8)
    assert iou(fine, f.mask_both) > 0.9
    assert iou(coarse, f.mask_both) < 0.5


def test_sift_needs_a_lowered_contrast_threshold():
    """Pins the silent failure that returned 'no forgery' on a real forgery.

    At SIFT's default contrastThreshold of 0.04 the whole 96x96 duplicated region
    produced 3 usable match pairs — below the 4 the verifier needs — so the
    detector returned an empty mask and raised nothing.
    """
    f = synth.copy_move_forgery(io.sample("astronaut"), size=96, seed=0)
    default = cv2.SIFT.create(nfeatures=3000)
    tuned = fg.make_sift()
    assert len(fg.self_matches(f.image, tuned, cv2.NORM_L2, 0.6, 30)) > 3 * len(
        fg.self_matches(f.image, default, cv2.NORM_L2, 0.6, 30)
    )


def test_dense_verification_beats_painting_blobs():
    """Same keypoints, same matches — only the way they become a region differs."""
    rows = fg.evaluate_methods(images=fg.IMAGES)
    verified = next(r for r in rows if r["method"] == "SIFT + similarity verify")
    blobs = next(r for r in rows if r["method"] == "SIFT blobs (no verify)")
    assert verified["iou"] > blobs["iou"] + 0.15


# --------------------------------------------------------------------------- #
# the findings
# --------------------------------------------------------------------------- #


def test_the_best_method_on_an_exact_copy_is_the_worst_under_rotation():
    """The project's headline, pinned.

    Block matching does not degrade under rotation, it stops. Two degrees is
    enough, and there is no partial credit.
    """
    rows = fg.sweep_rotation(images=IMAGES, angles=(0.0, 2.0))
    exact, turned = rows[0], rows[1]
    assert exact["Block matching"] > 0.9
    assert turned["Block matching"] == 0.0
    # while the similarity-verified detector barely notices
    assert turned["SIFT + similarity verify"] > 0.9 * exact["SIFT + similarity verify"]


def test_the_verifier_hypothesis_matters_more_than_the_descriptor():
    """Two rows, identical SIFT keypoints and matches, opposite behaviour.

    `SIFT + translation verify` beats `SIFT + similarity verify` on an exact copy
    and scores zero at 90 degrees, where the other still works. The descriptor
    was rotation invariant in both; the hypothesis was not.
    """
    # the full image set, not the two-image subset the other tests use: the
    # 0-degree half of this claim is a 0.12 IoU effect on six images and a
    # 0.005 coin-flip on two, which is how this test failed first time round
    rows = fg.sweep_rotation(images=fg.IMAGES, angles=(0.0, 90.0))
    exact, turned = rows[0], rows[1]
    assert exact["SIFT + translation verify"] > exact["SIFT + similarity verify"] + 0.05
    assert turned["SIFT + translation verify"] < 0.05
    assert turned["SIFT + similarity verify"] > 0.3


def test_rotation_robustness_is_paid_for_in_false_alarms():
    """The trade that decides which method a forensic tool should use.

    The method that survives any rotation is the method that accuses untampered
    photographs; the one that never raises a false alarm is the one that fails at
    2 degrees.
    """
    rows = fg.evaluate_false_alarms(images=fg.IMAGES)
    by_name = {r["method"]: r for r in rows}
    assert by_name["Block matching"]["mean_flagged"] == 0.0
    assert by_name["SIFT + similarity verify"]["mean_flagged"] > 0.01


def test_pixel_accuracy_is_worthless_here():
    """The control earns over 90% on the metric, and zero on the useful one."""
    rows = fg.evaluate_methods(images=IMAGES)
    control = next(r for r in rows if r["method"].startswith("Predict nothing"))
    assert control["pixel_accuracy"] > 0.9
    assert control["iou"] == 0.0


def test_every_method_fails_below_its_size_floor():
    """Pins the floor, and that it is built into the algorithms rather than tuned."""
    rows = fg.sweep_size(images=IMAGES, sizes=(32, 96))
    small, large = rows[0], rows[1]
    for name in fg.METHODS:
        if name.startswith("Predict nothing") or name.startswith("SIFT blobs"):
            continue
        assert small[name] < 0.1, name
    assert large["Block matching"] > 0.9


def test_scoring_against_the_pasted_half_alone_caps_precision():
    """Why `mask_both` exists, asserted rather than explained.

    A detector finds a duplicated *pair* and cannot say which half is the copy,
    so roughly half of a perfect prediction lands on the source region — which
    `mask` does not contain.
    """
    f = synth.copy_move_forgery(io.sample("astronaut"), size=96, seed=0)
    pred = fg.detect_block_matching(f.image) > 0
    on_paste = float((pred & (f.mask > 0)).sum()) / max(float(pred.sum()), 1.0)
    on_pair = float((pred & (f.mask_both > 0)).sum()) / max(float(pred.sum()), 1.0)
    assert on_paste < 0.6
    assert on_pair > 0.9


# --------------------------------------------------------------------------- #
# the entry points
# --------------------------------------------------------------------------- #


def test_describe_match_recovers_the_applied_transform():
    f = synth.copy_move_forgery(io.sample("astronaut"), size=128, angle_deg=30.0, seed=0)
    fit = fg.describe_match(f.image)
    assert fit["inliers"] >= 4
    assert abs(abs(fit["angle_deg"]) - 30.0) < 3.0
    assert abs(fit["scale"] - 1.0) < 0.06


def test_detect_dispatches_by_name():
    f = synth.copy_move_forgery(io.sample("coffee"), size=96, seed=0)
    assert np.array_equal(
        fg.detect(f.image, "Block matching"), fg.detect_block_matching(f.image)
    )
    with pytest.raises(KeyError):
        fg.detect(f.image, "not a method")


def test_an_untampered_image_does_not_crash_any_detector():
    clean = io.sample("chelsea")
    for name, fn in fg.METHODS.items():
        out = fn(clean)
        assert out.shape == clean.shape[:2], name
