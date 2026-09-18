"""Tests for project 44, Poisson blending.

Most of these pin a *finding* rather than a number, so that a later change which
quietly reverses one of the project's conclusions fails loudly instead of
rewriting the README by accident.

`test_the_boundary_ring_keeps_the_targets_values` exists because the from-scratch
solver was a complete no-op: it seeded its own boundary from the source, which
makes `f = source` an exact fixed point, and it reported numbers identical to
copy-paste rather than failing.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import poisson_blending as pb


@pytest.fixture(scope="module")
def scored():
    return {r["method"]: r for r in pb.evaluate_methods(runs=1)}


@pytest.fixture(scope="module")
def case():
    return pb.make_case(*pb.PAIRS[0], size=120, brightness_offset=0.25)


# --------------------------------------------------------------------------- #
# the solver
# --------------------------------------------------------------------------- #


def test_the_boundary_ring_keeps_the_targets_values(case):
    """The one line that makes Poisson blending Poisson blending.

    Gradients from the source inside, values from the destination on the edge.
    Seeding the boundary from the source instead makes `f = source` a fixed
    point of the iteration — the solver then "converges" in zero steps to the
    copy-paste answer, and every metric reports exactly the control's numbers.
    """
    target, source, mask, centre = case
    out = pb.blend_poisson_jacobi(target, source, mask, centre, iterations=200)
    control = pb.blend_copy_paste(target, source, mask, centre)

    assert not np.array_equal(out, control), "the solver must actually do something"
    assert pb.pixel_fidelity(out, source, mask, centre, target) > 5.0


def test_the_from_scratch_solver_agrees_with_opencv(scored):
    """Two independent implementations of the same equation, as a cross-check."""
    jacobi = scored["Poisson (Jacobi, from scratch)"]["seam_visibility"]
    opencv = scored["Poisson (OpenCV)"]["seam_visibility"]
    assert abs(jacobi - opencv) < 0.05

    assert scored["Poisson (Jacobi, from scratch)"]["median_ms"] > \
        10 * scored["Poisson (OpenCV)"]["median_ms"]


def test_the_jacobi_iteration_converges():
    rows = sorted(pb.convergence(), key=lambda r: r["iterations"])
    assert rows[0]["seam_visibility"] > rows[-1]["seam_visibility"] - 0.2

    settled = [r for r in rows if r["iterations"] >= 100]
    spread = max(r["seam_visibility"] for r in settled) - \
        min(r["seam_visibility"] for r in settled)
    assert spread < 0.02, "it should be flat past 100 iterations"


def test_the_placement_uses_the_patch_size_not_the_frame_size(case):
    """`seam_visibility` passed the whole composite where the patch belonged."""
    target, source, mask, centre = case
    x, y, w, h = pb._placement(target, mask, mask, centre)
    assert (h, w) == mask.shape[:2]
    assert 0 <= x and x + w <= target.shape[1]
    assert 0 <= y and y + h <= target.shape[0]


# --------------------------------------------------------------------------- #
# the findings
# --------------------------------------------------------------------------- #


def test_the_method_that_works_moves_the_pixels_most(scored):
    """The project's headline, and it is why the obvious metric is wrong.

    Poisson changes the pasted region by 51 grey levels; alpha feather changes it
    by 4.5 and copy-paste by 0. Scored on fidelity to the source, the ranking is
    exactly inverted — the best score belongs to the method with the worst seam.
    """
    control = scored["Copy-paste (control)"]
    feather = scored["Alpha feather"]
    poisson = scored["Poisson (OpenCV)"]

    assert control["pixel_difference"] == 0.0
    assert poisson["pixel_difference"] > 8 * feather["pixel_difference"]

    # and on the seam, that ranking reverses
    assert control["seam_visibility"] > 2 * poisson["seam_visibility"]


def test_copy_paste_is_the_worst_seam_by_a_wide_margin(scored):
    control = scored["Copy-paste (control)"]["seam_visibility"]
    for name, row in scored.items():
        if name != "Copy-paste (control)":
            assert row["seam_visibility"] < 0.6 * control, name


def test_the_seam_metric_cannot_separate_feathering_from_blending(scored):
    """Which is why the figure is the result and the table is the caveat.

    Alpha feather edges the mean (1.054 against 1.140) by attenuating the
    boundary gradient rather than reconciling it — it only moved the pasted
    pixels 4.5 levels, so the brightness step is still there and the patch is
    still plainly a patch. Per pair the two split two-all.
    """
    feather = scored["Alpha feather"]["seam_visibility"]
    poisson = scored["Poisson (OpenCV)"]["seam_visibility"]
    assert abs(feather - poisson) < 0.2, "the metric does not separate them"

    per_pair = {"feather": 0, "poisson": 0}
    for target_name, source_name in pb.PAIRS[:4]:
        target, source, mask, centre = pb.make_case(target_name, source_name,
                                                    size=120, brightness_offset=0.25)
        f = pb.seam_visibility(pb.blend_alpha_feather(target, source, mask, centre),
                               mask, centre, target)
        p = pb.seam_visibility(pb.blend_poisson_opencv(target, source, mask, centre),
                               mask, centre, target)
        per_pair["feather" if f < p else "poisson"] += 1
    assert min(per_pair.values()) >= 1, per_pair


def test_mixed_gradients_wins_the_seam_by_making_the_paste_transparent(scored):
    """The best seam score in the table belongs to the method you can see through.

    Mixed gradients keeps whichever of the two gradients is stronger, so the
    target's own texture bleeds into the pasted region. That gives the lowest
    boundary contrast of anything here (0.850) and by far the worst gradient
    fidelity to the source (0.469) — the skier in row 4 of the figure becomes a
    ghost.
    """
    mixed = scored["Poisson (mixed gradients)"]
    plain = scored["Poisson (OpenCV)"]

    assert mixed["seam_visibility"] == min(r["seam_visibility"] for r in scored.values())
    assert mixed["gradient_fidelity"] < 0.7 * plain["gradient_fidelity"]


def test_the_harder_the_brightness_mismatch_the_more_the_gradient_domain_is_worth():
    rows = sorted(pb.sweep_brightness_offset(), key=lambda r: r["brightness_offset"])
    easy, hard = rows[0], rows[-1]

    control_cost = hard["Copy-paste (control)"] - easy["Copy-paste (control)"]
    poisson_cost = hard["Poisson (OpenCV)"] - easy["Poisson (OpenCV)"]
    assert control_cost > 4 * poisson_cost
    assert hard["Copy-paste (control)"] > 4 * hard["Poisson (OpenCV)"]


def test_every_method_beats_the_control_at_every_region_size():
    rows = pb.sweep_region_size()
    for row in rows:
        control = row["Copy-paste (control) seam"]
        for name in pb.METHODS:
            if name != "Copy-paste (control)":
                assert row[f"{name} seam"] < control, (row["region_size"], name)


# --------------------------------------------------------------------------- #
# the harness
# --------------------------------------------------------------------------- #


def test_the_pairs_use_each_photograph_exactly_once():
    used = [n for pair in pb.PAIRS for n in pair]
    assert sorted(used) == sorted(pb.IMAGES)
    assert len(set(used)) == len(pb.IMAGES)


def test_the_pool_spans_the_axis_it_was_selected_on():
    from shared import io

    values = []
    for name in pb.IMAGES:
        assert name in io.REAL_PHOTOS, name
        values.append(pb.tonal_range(pb.load_scene(name)))

    assert values == sorted(values), "IMAGES should be ordered by tonal range"
    assert max(values) > 2.5 * min(values)


def test_the_brightness_offset_is_applied_to_the_source_only(case):
    target, source, mask, centre = case
    plain = pb.make_case(*pb.PAIRS[0], size=120, brightness_offset=0.0)

    assert np.array_equal(target, plain[0]), "the target must not be touched"
    assert source.mean() > plain[1].mean() + 20


def test_every_blend_returns_the_target_untouched_outside_the_mask(case):
    target, source, mask, centre = case
    x, y, w, h = pb._placement(target, mask, mask, centre)

    for name, fn in pb.METHODS.items():
        out = fn(target, source, mask, centre)
        assert out.shape == target.shape, name
        # a strip far from the pasted region must be identical
        assert np.array_equal(out[:20, :20], target[:20, :20]), name
