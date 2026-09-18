"""Tests for project 41, point transforms.

The central claim here is exact rather than statistical — a point transform on
8-bit data *is* a 256-entry table — so most of these are equalities rather than
tolerances.

`test_a_table_and_the_arithmetic_it_replaces_are_bit_identical` is the one that
found a real defect: tables built with a plain `.astype(np.uint8)` truncate where
the arithmetic path rounds, and the two gamma curves disagreed on 136 of 256
input levels with no visible symptom.
"""

from __future__ import annotations

import numpy as np
import pytest

import point_transforms as pt

from shared.io import to_float, to_gray, to_uint8


@pytest.fixture(scope="module")
def properties():
    return {r["transform"]: r for r in pt.table_properties()}


# --------------------------------------------------------------------------- #
# the exact claims
# --------------------------------------------------------------------------- #


def test_a_table_and_the_arithmetic_it_replaces_are_bit_identical():
    """Not close — identical, on every pixel of every photograph.

    This is the project's correctness claim and it is checkable to the bit. It
    was false until `_to_level` replaced `.astype(np.uint8)`: truncation against
    round-half-up differs by exactly one level on more than half the inputs,
    which is invisible in an image and completely changes what the table is.
    """
    for row in pt.lut_versus_arithmetic(runs=1):
        assert row["identical"] is True, row["transform"]


def test_truncating_instead_of_rounding_breaks_it_on_half_the_levels():
    """The defect itself, pinned so it cannot come back quietly."""
    v = np.arange(256, dtype=np.float64) / 255.0
    curve = np.power(v, 0.5) * 255.0

    truncating = np.clip(curve, 0, 255).astype(np.uint8)
    rounding = pt._to_level(curve)
    arithmetic = to_uint8(np.power(to_float(np.arange(256, dtype=np.uint8)), 0.5))

    assert np.array_equal(rounding, arithmetic)
    assert not np.array_equal(truncating, arithmetic)
    assert int(np.count_nonzero(truncating != arithmetic)) > 128
    assert int(np.abs(truncating.astype(int) - arithmetic.astype(int)).max()) == 1


def test_the_lookup_is_faster_than_the_maths_it_replaces():
    rows = {r["transform"]: r for r in pt.lut_versus_arithmetic(runs=7)}
    for name in ("Gamma 0.5 (brighten)", "Gamma 2.2 (darken)"):
        assert rows[name]["speedup"] > 10, name
    # the identity is the control: its "arithmetic" is a memory copy, so the
    # table cannot win by much and does not
    assert rows["Identity (control)"]["speedup"] < 5


def test_only_the_bijections_are_invertible(properties):
    for name, row in properties.items():
        expected = name in ("Identity (control)", "Negative")
        assert row["invertible"] is expected, name
        assert (row["levels_surviving"] == 256) is expected, name


def test_the_table_predicts_the_loss_without_looking_at_an_image():
    """And on this pool it is exact, because every input level occurs somewhere."""
    predicted = {r["transform"]: r["levels_surviving"] for r in pt.table_properties()}
    measured = {r["transform"]: r["levels_surviving"] for r in pt.evaluate_on_images(runs=1)}

    for name in predicted:
        assert measured[name] <= predicted[name], name
        assert predicted[name] - measured[name] <= 2, name


# --------------------------------------------------------------------------- #
# the findings
# --------------------------------------------------------------------------- #


def test_brightening_bands_and_darkening_smears(properties):
    """The two failures are opposite and a single loss number hides both.

    A flat part of the curve maps many inputs to one output: detail is lost and
    the result is smooth. A steep part maps adjacent inputs far apart: no detail
    is lost, but the output histogram has holes and a gradient becomes steps.

    Gamma 0.5 and 2.2 lose about the same number of levels (64 and 72) by
    opposite mechanisms.
    """
    bright = properties["Gamma 0.5 (brighten)"]
    dark = properties["Gamma 2.2 (darken)"]

    assert abs(bright["levels_lost"] - dark["levels_lost"]) < 20

    assert bright["max_output_gap"] > 4 * bright["max_collapse_run"]
    assert dark["max_collapse_run"] > 4 * dark["max_output_gap"]


def test_log_and_inverse_log_lose_exactly_the_same_levels_in_opposite_ways(properties):
    """The cleanest demonstration that 'levels lost' is not a sufficient measure."""
    log, inverse = properties["Log"], properties["Inverse log"]

    assert log["levels_lost"] == inverse["levels_lost"]
    assert log["max_output_gap"] > 3 * log["max_collapse_run"]
    assert inverse["max_collapse_run"] > 3 * inverse["max_output_gap"]


def test_contrast_and_information_are_not_the_same_quantity():
    """Thresholding maximises one and nearly minimises the other."""
    rows = {r["transform"]: r for r in pt.evaluate_on_images(runs=1)}
    threshold = rows["Threshold 127"]

    assert threshold["rms_contrast"] == max(r["rms_contrast"] for r in rows.values())
    assert threshold["entropy_bits"] < 1.0
    assert threshold["entropy_bits"] < rows["Identity (control)"]["entropy_bits"] / 5


def test_the_gamma_sweep_is_symmetric_about_one():
    """Both directions lose, the identity loses nothing, and the shape flips."""
    rows = {r["gamma"]: r for r in pt.sweep_gamma()}
    assert rows[1.0]["levels_surviving"] == 256
    assert rows[1.0]["max_collapse_run"] == 1

    for g in rows:
        if g != 1.0:
            assert rows[g]["levels_surviving"] < 256, g

    assert rows[0.3]["max_output_gap"] > rows[0.3]["max_collapse_run"]
    assert rows[3.0]["max_collapse_run"] > rows[3.0]["max_output_gap"]


def test_posterising_loses_exactly_what_it_says_it_will():
    for row in pt.sweep_posterise():
        assert row["actual_levels"] == row["requested_levels"], row
        assert row["entropy_bits"] <= np.log2(row["requested_levels"]) + 0.01, row


def test_half_the_bits_carry_most_of_the_picture():
    rows = {r["top_planes_kept"]: r for r in pt.bit_plane_contribution()}
    assert rows[4]["psnr_db"] > 25.0
    assert rows[8]["psnr_db"] == float("inf")

    values = [rows[n]["psnr_db"] for n in range(1, 8)]
    assert values == sorted(values)


# --------------------------------------------------------------------------- #
# the tables themselves
# --------------------------------------------------------------------------- #


def test_every_table_is_256_entries_of_uint8():
    for name, build in pt.LUTS.items():
        lut = build()
        assert lut.shape == (256,), name
        assert lut.dtype == np.uint8, name


def test_bit_plane_reconstruction_masks_the_right_bits():
    img = pt.load_scene(pt.IMAGES[0])
    gray = to_gray(img)
    for planes in range(1, 9):
        out = pt.reconstruct_from_planes(img, planes)
        assert out.shape == gray.shape
        assert (out <= gray).all(), planes
        assert int((gray.astype(int) - out.astype(int)).max()) < 2 ** (8 - planes)


def test_the_pool_spans_the_axis_it_was_selected_on():
    from shared import io

    values = []
    for name in pt.IMAGES:
        assert name in io.REAL_PHOTOS, name
        values.append(pt.mean_brightness(pt.load_scene(name)))

    assert len(set(pt.IMAGES)) == len(pt.IMAGES)
    assert values == sorted(values), "IMAGES should be ordered by brightness"
    assert min(values) < 60 and max(values) > 180
