"""Tests for project 52, focus stacking.

The one worth reading first is `test_the_pooling_window_matters_more_than_the_measure`.
The five focus measures land within 1.0 dB of each other and the best is 0.016 dB
from an oracle that is handed the answer; changing the window the response is
pooled over — a parameter almost nobody reports — moves the result 2.3× further.

`test_no_measure_can_be_right_in_a_flat_region` is the other one: where every
frame is identically smooth there is no evidence, agreement with the truth falls
from 0.98 to 0.55, and the five measures disagree with each other on 58% of those
pixels against 12% elsewhere.
"""

from __future__ import annotations

import numpy as np
import pytest

import focus as fo


@pytest.fixture(scope="module")
def overall():
    return {r["method"]: r for r in fo.evaluate()}


@pytest.fixture(scope="module")
def spread():
    return fo.measure_spread()


# --------------------------------------------------------------------------- #
# the result
# --------------------------------------------------------------------------- #


def test_the_pooling_window_matters_more_than_the_measure(spread):
    assert spread["pooling_matters_more"], spread
    assert spread["pooling_spread_db"] > 2 * spread["measure_spread_db"], spread


def test_the_best_measure_is_within_a_hair_of_the_oracle(overall):
    """There is almost nothing left for a better measure to win."""
    oracle = overall["Oracle (truth)"]["psnr_db"]
    best = max(r["psnr_db"] for r in overall.values() if not r["is_control"])
    assert oracle - best < 0.1, (oracle, best)


def test_every_measure_captures_most_of_what_the_task_is_worth(overall):
    oracle = overall["Oracle (truth)"]["psnr_db"]
    random_pick = overall["Random pick (control)"]["psnr_db"]
    worth = oracle - random_pick
    assert worth > 10.0, worth
    for name, r in overall.items():
        if r["is_control"]:
            continue
        assert (r["psnr_db"] - random_pick) > 0.9 * worth, name


def test_pooling_is_what_makes_the_raw_response_usable():
    rows = {r["window"]: r for r in fo.sweep_pooling(images=fo.IMAGES[:4])}
    assert rows[1]["agreement_with_truth"] < 0.8
    assert rows[9]["agreement_with_truth"] > 0.9
    # and too much pooling is worse again: there is an optimum
    assert rows[41]["agreement_with_truth"] < rows[9]["agreement_with_truth"]


def test_no_measure_can_be_right_in_a_flat_region():
    result = fo.flat_regions_are_unwinnable(images=fo.IMAGES[:6])
    assert result["agreement_in_detailed_regions"] > 0.9
    assert result["agreement_in_flat_regions"] < 0.8
    assert result["images_with_any_flat_region"] >= 1


def test_the_measures_disagree_where_there_is_nothing_to_agree_about():
    rows = [r for r in fo.disagreement(images=fo.IMAGES[:6])
            if r["flat_share"] > 0.01]
    assert rows, "no photograph in the subset has a flat region"
    for r in rows:
        assert r["disagreement_in_flat"] > 2 * r["disagreement_in_detailed"], r


def test_detail_predicts_the_ceiling_and_the_sign_is_negative():
    """More detail means a *lower* ceiling: a blurred detailed region loses more."""
    fit = fo.detail_predicts_the_ceiling()
    assert fit["pearson_r"] < -0.8, fit
    lo, hi = fit["detail_range"]
    assert hi > 4 * lo, fit


def test_the_gap_to_the_oracle_stays_tiny_at_every_stack_depth():
    rows = fo.sweep_frames(images=fo.IMAGES[:4], counts=(3, 9))
    for r in rows:
        assert r["gap_db"] < 0.3, r


def test_there_has_to_be_enough_defocus_to_see():
    """At 1 sigma across the whole depth range, adjacent frames differ by 0.25
    sigma and every measure falls to about half right.

    The sweep was non-monotone in an earlier version, with a dip at sigma 2. That
    was an artefact of the stack being approximated with nine blur levels
    whatever `max_sigma` was, so a small-sigma stack was approximated eight times
    more finely. The levels now use a fixed sigma step.
    """
    rows = {r["max_sigma"]: r for r in fo.sweep_blur(images=fo.IMAGES[:4],
                                                     sigmas=(1.0, 2.0, 4.0))}
    for measure in fo.MEASURES:
        assert rows[1.0][measure] < 0.75, measure
        assert rows[2.0][measure] > rows[1.0][measure], measure
        assert rows[4.0][measure] > rows[2.0][measure], measure


# --------------------------------------------------------------------------- #
# the construction
# --------------------------------------------------------------------------- #


def test_the_truth_is_recorded_not_estimated():
    import inspect

    source = inspect.getsource(fo.build_stack)
    assert "argmin" in source
    for name in fo.MEASURES:
        assert name.lower().split()[0] not in source.lower()


def test_each_frame_is_sharpest_in_its_own_band():
    """The property the whole construction rests on."""
    image = fo.load(fo.IMAGES[4])
    frames, truth, depth = fo.build_stack(image)
    responses = np.stack([fo.measure_variance_of_laplacian(f) for f in frames])
    picked = np.argmax(responses, axis=0)
    # not every pixel, but the great majority: this is the same statement as
    # the measures scoring 0.97 agreement
    assert float((picked == truth).mean()) > 0.9


def test_the_oracle_merge_is_close_to_the_original():
    image = fo.load(fo.IMAGES[5])
    frames, truth, _ = fo.build_stack(image)
    assert fo.psnr(fo.merge(frames, truth), image) > 30.0


def test_every_frame_of_the_stack_is_worse_than_the_original():
    """If one were not, that frame would be the answer and there is no problem."""
    image = fo.load(fo.IMAGES[6])
    frames, _, _ = fo.build_stack(image)
    for i, frame in enumerate(frames):
        assert fo.psnr(frame, image) < 40.0, i
        assert fo.detail(frame) < fo.detail(image), i


def test_the_depth_map_is_smooth():
    """A salt-and-pepper depth would make the truth unwinnable for everyone."""
    depth = fo.depth_map((321, 481))
    gradient = np.abs(np.diff(depth, axis=1))
    assert float(gradient.max()) < 0.05, float(gradient.max())
    assert depth.min() == pytest.approx(0.0, abs=1e-6)
    assert depth.max() == pytest.approx(1.0, abs=1e-6)


def test_the_controls_do_what_they_say():
    image = fo.load(fo.IMAGES[0])
    frames, _, _ = fo.build_stack(image)
    first = fo.select_first(frames)
    assert first.max() == 0
    random_idx = fo.select_random(frames, seed=0)
    assert len(np.unique(random_idx)) == len(frames)
    assert np.array_equal(random_idx, fo.select_random(frames, seed=0))


def test_merging_by_index_takes_each_pixel_from_that_frame():
    image = fo.load(fo.IMAGES[0])
    frames, _, _ = fo.build_stack(image)
    indices = np.zeros(frames[0].shape[:2], np.int32)
    indices[:, 100:] = 2
    merged = fo.merge(frames, indices)
    assert np.array_equal(merged[:, :100], frames[0][:, :100])
    assert np.array_equal(merged[:, 100:], frames[2][:, 100:])


def test_feathering_softens_rather_than_sharpens():
    rows = {r["feather"]: r["psnr_db"] for r in
            fo.feathering_effect(images=fo.IMAGES[:4], feathers=(0, 21))}
    assert rows[21] < rows[0], rows


def test_the_twelve_are_ordered_by_detail():
    values = [fo.detail(fo.load(n)) for n in fo.IMAGES]
    assert values == sorted(values), list(zip(fo.IMAGES, values))
