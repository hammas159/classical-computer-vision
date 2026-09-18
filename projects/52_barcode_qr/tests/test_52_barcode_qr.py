"""Tests for project 52, barcode and QR detection.

Most of these pin a *finding* rather than a number, so that a later change which
quietly reverses one of the project's conclusions fails loudly instead of
rewriting the README by accident.

`test_the_generated_background_flatters_the_gradient_localiser` is the one worth
reading first: it pins a result that only appeared once real photographs were
used as backgrounds, and that the synthetic scene had been hiding completely.
"""

from __future__ import annotations

import numpy as np
import pytest

import barcode as bc


@pytest.fixture(scope="module")
def backgrounds():
    return {r["background"]: r for r in bc.photo_versus_generated_background()}


LOCATORS = ("Gradient + morphology", "Local variance", "QRCodeDetector")


# --------------------------------------------------------------------------- #
# the ground truth
# --------------------------------------------------------------------------- #


def test_a_decoded_payload_is_the_only_objective_answer_here():
    """No metric choice at all: the string comes back or it does not."""
    for payload in bc.PAYLOADS:
        code = bc.make_qr(payload)
        assert bc.decode_qr(code) == payload, payload


def test_the_true_corners_are_where_the_code_actually_is():
    code = bc.make_qr("CLASSICAL")
    img, corners = bc.place_on_background(code, seed=0)

    assert corners.shape == (4, 2)
    assert bc.localisation_iou(corners, corners, img.shape) == pytest.approx(1.0)
    assert bc.decode_qr(img) == "CLASSICAL"


def test_the_barcode_decoder_arity_is_handled():
    """OpenCV changed detectAndDecode from 4 return values to 3 between versions.

    Unpacking a fixed number raised ValueError on the other one and took the
    whole experiment down instead of reporting an empty decode.
    """
    code = bc.make_code39("HELLO123")
    result = bc.decode_barcode(code)
    assert isinstance(result, str)
    if bc.barcode_decoder_available():
        assert bc.decode_barcode(np.zeros((64, 64), np.uint8)) == ""


def test_the_pool_spans_the_axis_it_was_ordered_by():
    from shared import io

    values = []
    for name in bc.IMAGES:
        assert name in io.REAL_PHOTOS, name
        values.append(bc.barcode_like_share(bc.load_scene(name)))

    assert values == sorted(values), "IMAGES should be ordered by barcode-like share"
    assert min(values) < 12.0 and max(values) > 70.0


# --------------------------------------------------------------------------- #
# the findings
# --------------------------------------------------------------------------- #


def test_the_generated_background_flatters_the_gradient_localiser(backgrounds):
    """The finding the synthetic scene was hiding.

    Random rectangles have no repeating vertical structure, and a 1-D barcode
    localiser looks for exactly that. On generated clutter it scores 1.000; on
    twelve photographs, 0.021. The other two localisers are unaffected, which is
    what makes it a property of the cue rather than of the harder images.
    """
    generated = backgrounds["Generated clutter"]
    photographs = backgrounds["Photographs"]

    assert generated["Gradient + morphology"] > 0.9
    assert photographs["Gradient + morphology"] < 0.1

    for name in ("Local variance", "QRCodeDetector"):
        assert generated[name] == pytest.approx(photographs[name], abs=0.05), name


def test_decoding_does_not_care_about_the_background(backgrounds):
    """Because it reads the code itself rather than hunting for it."""
    for row in backgrounds.values():
        assert row["decoded"] > 0.95, row["background"]


def test_localisation_survives_long_after_decoding_stops():
    """The project's central claim, on three separate degradations.

    A detection rate overstates a pipeline by exactly this gap: the code is
    still found every time and read none of the time.
    """
    for rows, key in ((bc.sweep_blur(), "blur_sigma"),
                      (bc.sweep_scale(), "scale"),
                      (bc.sweep_noise(), "noise_sigma")):
        collapsed = [r for r in rows if r["decode_rate"] <= 0.25]
        assert collapsed, key
        for row in collapsed:
            assert row["best_found_rate"] >= 0.75, (key, row[key])


def test_resolution_is_the_sharpest_version_of_the_gap():
    """At 6.7 pixels per module the code is located 100% and decoded 0%."""
    rows = sorted(bc.sweep_scale(), key=lambda r: -r["approx_px_per_module"])
    readable = [r for r in rows if r["decode_rate"] >= 0.99]
    unreadable = [r for r in rows if r["decode_rate"] < 0.5]

    assert readable and unreadable
    assert min(r["approx_px_per_module"] for r in readable) > \
        max(r["approx_px_per_module"] for r in unreadable)
    assert unreadable[0]["best_found_rate"] >= 0.99


def test_perspective_is_the_one_that_degrades_gradually():
    """Every other degradation is a cliff; this one is a slope."""
    rows = sorted(bc.sweep_perspective(), key=lambda r: r["perspective"])
    rates = [r["decode_rate"] for r in rows]

    assert rates[0] >= 0.99
    assert rates[-1] < 0.5
    assert len({round(r, 2) for r in rates}) >= 3, rates


def test_rotation_is_the_degradation_qr_shrugs_off():
    """A QR code carries its own orientation markers, so turning it costs nothing."""
    for row in bc.sweep_rotation():
        assert row["decode_rate"] >= 0.99, row["rotation_deg"]


def test_the_gradient_localiser_is_the_one_rotation_breaks():
    """It looks for horizontal gradient, and 45 degrees is where that vanishes."""
    rows = {r["rotation_deg"]: r for r in bc.sweep_rotation()}
    assert rows[45.0]["Gradient + morphology"] < 0.5
    assert rows[0.0]["Gradient + morphology"] >= 0.99
    assert rows[45.0]["Local variance"] >= 0.99


# --------------------------------------------------------------------------- #
# the generators
# --------------------------------------------------------------------------- #


def test_the_generated_codes_are_binary_and_quiet_zoned():
    qr = bc.make_qr("VISION42")
    assert set(np.unique(qr)) <= {0, 255}

    code = bc.make_code39("ABC789")
    assert set(np.unique(code)) <= {0, 255}
    assert code[:, :10].min() == 255, "a barcode needs a quiet zone to be readable"


def test_placing_a_code_on_a_photograph_uses_that_photograph():
    code = bc.make_qr("HELLO123")
    plain, _ = bc.place_on_background(code, seed=0)
    on_photo, _ = bc.place_on_background(code, seed=0, background=bc.IMAGES[0])

    assert plain.shape == on_photo.shape
    assert not np.array_equal(plain, on_photo)
    assert bc.decode_qr(on_photo) == "HELLO123"
