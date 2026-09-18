"""Tests for project 58, red-eye removal.

The two worth reading first are
`test_the_generated_scene_reports_the_problem_solved_and_the_photographs_do_not`
and `test_pupil_psnr_cannot_see_what_a_detector_does_to_the_rest_of_the_frame`.
Between them they pin why the project has a real arm at all: on the generated
scene the face-constrained detector scores 0.984 and has nothing left to
improve, and the metric that would normally be quoted rates the naive detector
within 0.13 dB of it while it marks two hundred times more of the frame.

The rest pin the ground truth (the planted mask is exactly the recorded pupils),
the controls (six photographs with no red-eye, and doing nothing), and the
finding that zeroing the red channel is not always an improvement.
"""

from __future__ import annotations

import numpy as np
import pytest

import red_eye as re58


@pytest.fixture(scope="module")
def real():
    return {r["detector"]: r for r in re58.evaluate_on_portraits()}


@pytest.fixture(scope="module")
def both():
    return {r["detector"]: r for r in re58.generated_versus_real_background(scenes=4)}


@pytest.fixture(scope="module")
def clean():
    return re58.false_positives_on_clean_photographs()


@pytest.fixture(scope="module")
def corrections():
    return re58.corrections_on_portraits()


# --------------------------------------------------------------------------- #
# the findings
# --------------------------------------------------------------------------- #


def test_the_generated_scene_reports_the_problem_solved_and_the_photographs_do_not(both):
    """0.984 on drawn distractors, 0.60 on photographs, same detector.

    The generated distractors are discs and rectangles the shape filter is
    entitled to reject. Lipstick, brake lights and sugar-shelled sweets are
    round, small and red, and it is not.
    """
    face = both["Face-constrained"]
    assert face["generated_iou"] > 0.95
    assert face["real_iou"] < 0.75
    assert face["generated_iou"] - face["real_iou"] > 0.3


def test_pupil_psnr_cannot_see_what_a_detector_does_to_the_rest_of_the_frame(real):
    """The metric everyone quotes rates the worst detector next to the best.

    Pupil PSNR is computed on pupil pixels. The naive detector's false positives
    are all somewhere else, so the score never notices them — which is exactly
    why this project measures a clean photograph separately.
    """
    pipeline = {r["detector"]: r["pupil_psnr_db"]
                for r in re58.end_to_end_on_portraits()}
    gap_db = abs(pipeline["Face-constrained"] - pipeline["Colour only (control)"])
    fp_ratio = (real["Colour only (control)"]["false_positive_area_ratio"]
                / max(real["Face-constrained"]["false_positive_area_ratio"], 1e-9))
    assert gap_db < 0.5, pipeline
    assert fp_ratio > 50, fp_ratio


def test_a_clean_photograph_is_the_control_no_threshold_can_hide(clean):
    """Truth is empty on these six, so every detected pixel is a mistake."""
    total = {d: sum(r[d] for r in clean) for d in re58.DETECTORS}
    assert total["Colour only (control)"] > 50_000
    assert total["Colour + shape"] < total["Colour only (control)"] / 4
    assert total["Face-constrained"] < 200
    assert total["Face-constrained"] < total["Colour + shape"] / 50


def test_the_eye_cascade_finds_eyes_in_a_pile_of_sweets(clean):
    """The tightest constraint is not the safest one.

    The face constraint marks nothing in `scattered_sweets`; the eye constraint
    marks 47 pixels, because the eye cascade fires on round dark-rimmed discs and
    that is what the photograph is full of.
    """
    sweets = next(r for r in clean if r["photograph"] == "scattered_sweets")
    assert sweets["Eye-constrained"] > 0
    assert sweets["Face-constrained"] == 0


def test_zeroing_the_red_channel_is_not_always_an_improvement(corrections):
    """On half the portraits it scores below leaving the red-eye alone.

    It removes the red and leaves a dark blue hole, which is further from the
    original pupil than the red-eye was. Only a do-nothing control shows this;
    against the other corrections it merely looks like the weakest of three.
    """
    worse = [r for r in corrections if r["Zero red channel"] < r["did nothing"]]
    assert len(worse) >= 3, corrections
    for r in corrections:
        assert r["Mean of G and B"] > r["Zero red channel"] or \
               r["Desaturate (feathered)"] > r["Zero red channel"], r


def test_both_luminance_preserving_corrections_beat_doing_nothing(corrections):
    for r in corrections:
        assert r["Mean of G and B"] > r["did nothing"], r
        assert r["Desaturate (feathered)"] > r["did nothing"], r


def test_the_face_constraint_is_what_buys_the_precision(real):
    assert real["Face-constrained"]["iou"] > 3 * real["Colour + shape"]["iou"]
    assert (real["Face-constrained"]["false_positive_area_ratio"]
            < real["Colour + shape"]["false_positive_area_ratio"] / 20)


def test_the_eye_constraint_trades_recall_for_precision(real):
    """Tighter than the face box, and the eye cascade misses more often."""
    assert (real["Eye-constrained"]["false_positive_area_ratio"]
            <= real["Face-constrained"]["false_positive_area_ratio"])
    assert real["Eye-constrained"]["pupil_recall"] < real["Face-constrained"]["pupil_recall"]


# --------------------------------------------------------------------------- #
# the ground truth
# --------------------------------------------------------------------------- #


def test_the_planted_mask_is_exactly_the_recorded_pupils():
    for name, entry in re58.PORTRAITS.items():
        _, _, mask = re58.portrait_scene(name)
        n, _, _, _ = __import__("cv2").connectedComponentsWithStats(
            (mask > 0).astype(np.uint8), 8)
        assert n - 1 == len(entry["pupils"]), name
        for (px, py, _r) in entry["pupils"]:
            assert mask[py, px] == 255, (name, px, py)


def test_the_pre_flash_image_is_the_untouched_photograph():
    """Nothing but the pupils may differ, or the correction is scored against
    a target that has been quietly improved."""
    from shared import io

    for name in re58.PORTRAITS:
        flash, pre_flash, mask = re58.portrait_scene(name)
        assert np.array_equal(pre_flash, io.real_photo(name)), name
        untouched = mask == 0
        assert np.array_equal(flash[untouched], pre_flash[untouched]), name


def test_red_eye_is_added_to_the_pupil_rather_than_painted_over_it():
    """A flat disc would give every pupil the same value; a real one glows."""
    flash, pre_flash, mask = re58.portrait_scene("woman_in_red_scarf")
    sel = mask > 0
    assert np.all(flash[..., 0][sel] >= pre_flash[..., 0][sel])
    assert flash[..., 0][sel].std() > 5.0


def test_which_pupils_the_cascade_found_is_recorded():
    """Half the point of the table is that it says which half is hand-placed."""
    found = total = 0
    for entry in re58.PORTRAITS.values():
        assert len(entry["source"]) == len(entry["pupils"])
        assert set(entry["source"]) <= {"cascade", "hand"}
        found += entry["source"].count("cascade")
        total += len(entry["pupils"])
    assert total == 16
    assert found == 12


def test_no_photograph_serves_two_roles():
    assert not set(re58.PORTRAITS) & set(re58.CLEAN_PHOTOGRAPHS)
    assert len(re58.PHOTOGRAPHS) == 12
    assert len(set(re58.PHOTOGRAPHS)) == 12


def test_the_selection_axis_is_the_methods_own_response():
    """Pupil-like blobs is `detect_colour_shape` up to the accept step."""
    from shared import io

    counts = {n: re58.pupil_like_blobs(io.real_photo(n)) for n in re58.PHOTOGRAPHS}
    assert max(counts.values()) > 30
    assert min(counts.values()) < 5
    # and it predicts what the shape filter will wrongly mark
    clean = re58.false_positives_on_clean_photographs()
    blobs = [r["pupil_like_blobs"] for r in clean]
    marked = [r["Colour + shape"] for r in clean]
    assert np.corrcoef(blobs, marked)[0, 1] > 0.3, list(zip(blobs, marked))


# --------------------------------------------------------------------------- #
# the pieces
# --------------------------------------------------------------------------- #


def test_redness_is_relative_not_the_red_channel():
    """White has a full red channel and is not red."""
    white = np.full((4, 4, 3), 255, np.uint8)
    red = np.zeros((4, 4, 3), np.uint8)
    red[..., 0] = 255
    assert re58.redness(white).max() == pytest.approx(0.0)
    assert re58.redness(red).min() == pytest.approx(1.0)


def test_the_threshold_has_an_optimum_rather_than_a_direction():
    curve = {r["threshold"]: r["Face-constrained IoU"]
             for r in re58.sweep_threshold_on_portraits()}
    best = max(curve, key=curve.get)
    assert 0.18 <= best <= 0.35, curve
    assert curve[min(curve)] < curve[best]
    assert curve[max(curve)] < curve[best]


def test_the_shape_filter_has_a_working_range_with_both_ends_visible():
    """Any fixed geometric constraint fails outside its bounds, in both directions.

    At radius 2 the blob is below `min_area` and at 45 it is above `max_area`;
    both report 0. The original sweep only spanned 4 to 20 and therefore showed
    a filter that always works.
    """
    sweep = {r["pupil_radius"]: r["Colour + shape"] for r in re58.sweep_pupil_size(scenes=3)}
    assert sweep[min(sweep)] == 0.0, sweep
    assert sweep[max(sweep)] == 0.0, sweep
    assert max(sweep.values()) > 0.2, sweep


def test_a_pupil_too_large_for_the_face_cascade_loses_the_face_constraint():
    sweep = {r["pupil_radius"]: r["Face-constrained"]
             for r in re58.sweep_pupil_size(scenes=3)}
    assert sweep[9] > 0.9
    assert sweep[32] == 0.0, sweep
