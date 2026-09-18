"""Tests for project 38, panorama stitching.

The one worth reading first is `test_psnr_ranks_the_blenders_backwards`. Scored by
PSNR against the original photograph, the winner is the control that does not
stitch at all — and it comes **last** on the seam. A panorama cannot be scored by
fidelity to one of its own inputs, because the first frame *is* the original in
its own region and any blending that mixes the second frame in can only move away
from it.

`test_blending_is_worth_nothing_at_matched_exposure` is the other one: with the
two frames at the same exposure, all five methods land within half a grey level
of each other. A blender comparison run on matched exposures measures nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

import panorama as pa


@pytest.fixture(scope="module")
def overall():
    return {r["blender"]: r for r in pa.evaluate()}


@pytest.fixture(scope="module")
def disagree():
    return pa.metrics_disagree()


# --------------------------------------------------------------------------- #
# the result
# --------------------------------------------------------------------------- #


def test_psnr_ranks_the_blenders_backwards(disagree):
    """The PSNR winner is the non-stitching control, and it is the seam loser."""
    assert disagree["psnr_winner_is_the_non_stitching_control"], disagree
    assert disagree["psnr_winner_seam_rank"] == len(pa.BLENDERS), disagree
    assert disagree["by_psnr"] != disagree["by_seam_step"]


def test_the_non_stitching_control_has_the_worst_seam(overall):
    keep = overall["Keep the first frame (control)"]
    for name, r in overall.items():
        if name == "Keep the first frame (control)":
            continue
        assert r["seam_step"] <= keep["seam_step"], name


def test_blending_beats_pasting_at_the_seam(overall):
    paste = overall["Overwrite (control)"]["seam_step"]
    for name in ("Feather (41 px)", "Multi-band (5 levels)"):
        assert overall[name]["seam_step"] < paste, name


def test_blending_is_worth_nothing_at_matched_exposure():
    """With nothing to hide, every method scores the same."""
    rows = {r["exposure"]: r for r in
            pa.exposure_is_the_whole_story(images=pa.IMAGES[:6],
                                           exposures=(1.0, 1.4))}
    flat = rows[1.0]
    values = [flat[b] for b in pa.BLENDERS]
    assert max(values) - min(values) < 2.0, flat

    lit = rows[1.4]
    values = [lit[b] for b in pa.BLENDERS]
    assert max(values) - min(values) > 8.0, lit


def test_the_seam_step_grows_with_the_exposure_difference():
    rows = pa.exposure_is_the_whole_story(images=pa.IMAGES[:4],
                                          exposures=(1.0, 1.1, 1.4))
    steps = [r["Overwrite (control)"] for r in rows]
    assert steps == sorted(steps), rows


def test_a_repeating_pattern_cannot_be_registered():
    """Every brick course looks like every other one."""
    rows = {r["image"]: r for r in pa.registrability_table()}
    brick = rows["brick_wall_courses"]
    best = max(rows.values(), key=lambda r: r["per_megapixel"])
    assert brick["per_megapixel"] < 100, brick
    assert best["per_megapixel"] > 100 * brick["per_megapixel"], (brick, best)
    assert brick["matches"] < 50, brick


def test_a_textured_scene_hides_its_own_seam():
    """Where the scene's own gradients are large, no blender changes the number.

    Worth pinning because it bounds the result: blending is worth a lot on smooth
    scenes and nothing on busy ones, and a mean over both understates the first
    and overstates the second.
    """
    rows = {r["image"]: r for r in pa.per_image(images=pa.IMAGES)}
    busy = rows["hillside_town_from_the_air"]
    smooth = rows["tugboat_under_the_bridge"]
    busy_spread = (max(busy[b] for b in pa.BLENDERS)
                   - min(busy[b] for b in pa.BLENDERS))
    smooth_spread = (max(smooth[b] for b in pa.BLENDERS)
                     - min(smooth[b] for b in pa.BLENDERS))
    assert smooth_spread > busy_spread, (smooth_spread, busy_spread)


# --------------------------------------------------------------------------- #
# the construction
# --------------------------------------------------------------------------- #


def test_neither_frame_alone_covers_the_canvas():
    """An earlier version made the first frame the whole photograph, so the
    do-nothing control scored an infinite PSNR and nothing was measured."""
    image = pa.load(pa.IMAGES[6])
    left, lc, right, rc, H, truth = pa.make_pair(image)
    assert 0.3 < lc.mean() < 0.9, lc.mean()
    assert 0.2 < rc.mean() < 0.9, rc.mean()
    assert not lc.all()
    aligned, ac = pa.align(right, rc, H, image.shape)
    overlap, right_only, covered = pa.regions(lc, ac)
    assert overlap.any() and right_only.any()
    assert covered.mean() > 0.95, covered.mean()


def test_the_overlap_is_a_realistic_fraction():
    image = pa.load(pa.IMAGES[6])
    left, lc, right, rc, H, _ = pa.make_pair(image)
    aligned, ac = pa.align(right, rc, H, image.shape)
    overlap, _, _ = pa.regions(lc, ac)
    assert 0.15 < overlap.mean() < 0.45, overlap.mean()


def test_the_second_frame_really_is_brighter():
    image = pa.load(pa.IMAGES[6])
    _, _, right, rc, _, _ = pa.make_pair(image, exposure=1.2)
    _, _, plain, pc, _, _ = pa.make_pair(image, exposure=1.0)
    assert float(right[rc].mean()) > float(plain[pc].mean()) * 1.05


def test_every_blender_leaves_the_uncovered_region_alone():
    image = pa.load(pa.IMAGES[4])
    left, lc, right, rc, H, _ = pa.make_pair(image)
    aligned, ac = pa.align(right, rc, H, image.shape)
    overlap, right_only, covered = pa.regions(lc, ac)
    left_only = lc & ~ac
    for name, fn in pa.BLENDERS.items():
        out = fn(left, aligned, overlap, right_only=right_only)
        assert np.array_equal(out[left_only], left[left_only]), name


def test_the_joining_control_fills_the_second_frames_region():
    """It does the joining and none of the blending, which is what isolates
    what blending is worth."""
    image = pa.load(pa.IMAGES[4])
    left, lc, right, rc, H, _ = pa.make_pair(image)
    aligned, ac = pa.align(right, rc, H, image.shape)
    overlap, right_only, _ = pa.regions(lc, ac)
    out = pa.blend_keep_first(left, aligned, overlap, right_only=right_only)
    assert np.array_equal(out[right_only], aligned[right_only])
    assert np.array_equal(out[overlap], left[overlap])


def test_the_seam_step_is_zero_on_an_unbroken_image():
    """A flat gradient has no step anywhere, so the metric must report none."""
    ramp = np.tile(np.linspace(0, 255, 400, dtype=np.uint8), (200, 1))
    rgb = np.dstack([ramp] * 3)
    line = np.zeros((200, 400), bool)
    line[:, 199:201] = True
    step = pa.seam_step(rgb, line, reach=6)
    # a linear ramp of 255 over 400 px steps about 7.6 levels over 12 px
    assert 5.0 < step < 10.0, step


def test_more_pyramid_levels_help_up_to_a_point():
    rows = {r["bands"]: r["seam_step"] for r in
            pa.sweep_bands(images=pa.IMAGES[:4], bands=(2, 5))}
    assert rows[5] <= rows[2], rows


# --------------------------------------------------------------------------- #
# the real pair
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not pa.graf_available(),
                    reason="graf1/graf3 not cached; run "
                           "`python tools/fetch_assets.py --set graf`")
def test_the_real_pair_registers():
    r = pa.real_pair()
    assert r["matches"] > 200
    assert r["inlier_rate"] > 0.5
    assert r["cycle_error_px"] < 10.0


@pytest.mark.skipif(not pa.graf_available(), reason="graf1/graf3 not cached")
def test_the_real_pair_has_no_external_truth_and_none_is_claimed():
    """The Oxford set's homography files 404 upstream.

    `cycle_error` cannot detect a consistently wrong homography, because H and
    its own inverse always compose to the identity. The docstring says so; this
    test is what stops the number being upgraded to an accuracy later.
    """
    import inspect

    source = inspect.getsource(pa.cycle_error)
    assert "not as an accuracy" in source
    assert not (pa.GRAF / "H1to3p").exists()
