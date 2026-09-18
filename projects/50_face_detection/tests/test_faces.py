"""Tests for project 50: face detection.

They pin the result (the 45x45 cascade is blind below its window and upscaling
is the only fix; rotation collapses; the cascades do not share an operating
point), the mechanism (`minSize` below the training window changes nothing), the
construction (the no-face photographs really contain no face the project scores,
the carved faces are kept out of both arms, and the controls behave as controls),
and the box arithmetic everything else rests on.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import faces as fc  # noqa: E402

pytestmark = pytest.mark.skipif(
    not fc.available(),
    reason="run `python tools/fetch_assets.py --set faces --set cascades`")


# --------------------------------------------------------------------------- #
# box arithmetic
# --------------------------------------------------------------------------- #


def test_iou_of_a_box_with_itself_is_one():
    assert fc.iou((10, 10, 30, 30), (10, 10, 30, 30)) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    assert fc.iou((0, 0, 10, 10), (50, 50, 10, 10)) == 0.0


def test_iou_of_a_half_overlap():
    """Two 10x10 boxes offset by 5 in x share 50 of 150 -- checked against
    arithmetic, not against another run."""
    assert fc.iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)


def test_each_box_matches_at_most_one_target():
    """One huge box must not be credited with finding three faces."""
    big = np.array([[0, 0, 100, 100]])
    three = [(0, 0, 100, 100), (0, 0, 100, 100), (0, 0, 100, 100)]
    assert fc.matched(big, three) == 1


def test_matching_nothing_against_nothing_is_zero():
    assert fc.matched(np.zeros((0, 4)), []) == 0


# --------------------------------------------------------------------------- #
# the controls
# --------------------------------------------------------------------------- #


def test_the_nothing_control_returns_nothing():
    img = fc.load(fc.image_names()[0])
    assert len(fc.run(img, "Nothing (control)")) == 0


def test_the_nothing_control_is_perfect_on_the_empty_arm():
    """The headline reason the empty arm cannot be the only score.

    A detector that never fires has zero false alarms, which is the best possible
    result on that arm and a useless detector.
    """
    assert fc.false_alarms("Nothing (control)")["false_alarms"] == 0


def test_the_every_box_control_floods_the_frame():
    """Compared against the cascades rather than against a round number.

    A first version asserted `> 500` boxes, which is true on ten of the eleven
    photographs and false on the smallest -- the grid is defined in fractions of
    the frame, so its box count is roughly constant and its *threshold* cannot
    be. What matters is that it returns orders of magnitude more than any real
    detector, and that is scale-free.
    """
    img = fc.load(fc.image_names()[0])
    flood = len(fc.run(img, "Every box (control)"))
    most = max(len(fc.detect(img, m)) for m in fc.DETECTORS)
    assert flood > 20 * max(most, 1), (flood, most)
    assert fc.false_alarms("Every box (control)")["false_alarms"] > 500


def test_the_centre_box_control_ignores_the_image():
    """Two different photographs of the same size must give the same box."""
    a = np.zeros((400, 600, 3), np.uint8)
    b = np.full((400, 600, 3), 255, np.uint8)
    assert np.allclose(fc.run(a, "One centre box (control)"),
                       fc.run(b, "One centre box (control)"))


# --------------------------------------------------------------------------- #
# the training window
# --------------------------------------------------------------------------- #


def test_the_training_windows_are_what_the_xml_says():
    """Read from the file rather than remembered, and asserted so a change in
    the shipped cascades is caught rather than silently changing every number."""
    assert fc.training_window("LBP improved") == 45
    assert fc.training_window("Haar default") == 24
    for method in fc.DETECTORS:
        assert 16 <= fc.training_window(method) <= 64


def test_min_size_below_the_training_window_changes_nothing():
    """The mechanism, asserted directly.

    `minSize` is a floor on the search, not a resampling. Below the cascade's own
    window there is no classifier to evaluate, so lowering it cannot add a
    detection -- and the counts come out byte-identical.
    """
    names = fc.image_names()
    for method in fc.DETECTORS:
        at12 = sum(len(fc.detect(fc.load(n), method, min_size=12)) for n in names)
        at24 = sum(len(fc.detect(fc.load(n), method, min_size=24)) for n in names)
        assert at12 == at24, (method, at12, at24)


def test_the_45x45_cascade_is_blind_at_native_scale():
    names = fc.image_names()
    improved = sum(len(fc.detect(fc.load(n), "LBP improved")) for n in names)
    plain = sum(len(fc.detect(fc.load(n), "LBP frontal")) for n in names)
    assert improved < 0.15 * plain, (improved, plain)


def test_upscaling_recovers_it_and_only_it():
    """The prediction the mechanism makes, which is what makes it a mechanism.

    If the 45x45 window is the cause, resampling the image up must recover that
    cascade and must barely move the ones already below their window.
    """
    rows = {r["method"]: r for r in fc.window_blindness()}
    assert rows["LBP improved"]["recovery"] > 10
    for method, r in rows.items():
        if method != "LBP improved":
            assert r["recovery"] < 2.0, (method, r["recovery"])


# --------------------------------------------------------------------------- #
# rotation
# --------------------------------------------------------------------------- #


def test_a_zero_degree_rotation_keeps_everything():
    """A regression test for the whole recorded-transform apparatus: with no
    rotation applied, every box must map to itself and be recovered."""
    rows = fc.rotation_survival("Haar alt2", degrees=(0,))
    assert rows[0]["survival"] == pytest.approx(1.0)


def test_mapping_a_box_through_a_zero_rotation_is_the_identity():
    M = fc.rotation_matrix((400, 600, 3), 0.0)
    assert fc.map_box(M, (10.0, 20.0, 30.0, 40.0)) == pytest.approx(
        (10.0, 20.0, 30.0, 40.0), abs=1e-6)


def test_mapping_a_box_through_180_degrees_lands_on_the_other_side():
    M = fc.rotation_matrix((400, 600, 3), 180.0)
    out = fc.map_box(M, (0.0, 0.0, 100.0, 100.0))
    assert out[0] == pytest.approx(500.0, abs=1e-6)
    assert out[1] == pytest.approx(300.0, abs=1e-6)


def test_every_cascade_loses_most_of_its_boxes_by_thirty_degrees():
    """The headline, asserted for all six rather than for the worst one."""
    for method in fc.DETECTORS:
        if method == "LBP improved":
            continue  # it has almost nothing to lose at native scale
        rows = {r["degrees"]: r["survival"]
                for r in fc.rotation_survival(method, degrees=(0, 30))}
        assert rows[30] < 0.35, (method, rows)


def test_the_every_box_control_out_survives_every_cascade():
    """And that is exactly why survival cannot be the only score either.

    A grid of boxes at every position and scale is nearly invariant to rotation
    by construction -- it only loses the boxes the rotation carries out of the
    frame -- so it beats every real cascade at 30 degrees while finding nothing
    in particular.
    """
    control = fc.rotation_survival("Every box (control)", degrees=(30,))[0]
    for method in fc.DETECTORS:
        real = fc.rotation_survival(method, degrees=(30,))[0]
        assert control["survival"] > real["survival"], (method, real["survival"])


# --------------------------------------------------------------------------- #
# the empty truth
# --------------------------------------------------------------------------- #


def test_the_no_face_photographs_all_exist_and_are_distinct():
    seen = set()
    for name in fc.NO_FACE:
        img = fc.load_no_face(name)
        assert img.ndim == 3
        seen.add(img.tobytes()[:2048])
    assert len(seen) == len(fc.NO_FACE)


def test_the_carved_faces_are_in_neither_arm():
    """They are reported on their own and scored in neither direction.

    Deciding that a wooden totem is or is not a face would put an opinion inside
    a false-alarm rate.
    """
    for name in fc.CARVED:
        assert name not in fc.NO_FACE
    assert name not in fc.image_names()


def test_false_alarms_are_reported_per_megapixel():
    """A raw count would rank the cascades by the sizes of the images they were
    given, because a detector is slid over every position."""
    r = fc.false_alarms("Haar default")
    assert r["per_megapixel"] == pytest.approx(
        r["false_alarms"] / r["megapixels"], rel=1e-9)


def test_the_cascades_do_not_share_an_operating_point():
    """At the loosest setting of the one knob everyone turns, two cascades that
    return a similar number of boxes differ by an order of magnitude in false
    alarms."""
    loose = {m: fc.operating_points(m, neighbors=(1,))[0] for m in fc.DETECTORS}
    alarms = [r["false_alarms"] for r in loose.values()]
    assert max(alarms) >= 10
    assert min(alarms) == 0


def test_looser_settings_never_return_fewer_boxes():
    """A monotonicity check on the sweep itself. `minNeighbors` only removes
    candidates, so the curve cannot go up with the threshold -- and if it did,
    the sweep would be measuring noise."""
    for method in fc.DETECTORS:
        rows = fc.operating_points(method, neighbors=(1, 3, 5, 8))
        boxes = [r["boxes"] for r in rows]
        assert boxes == sorted(boxes, reverse=True), (method, boxes)


# --------------------------------------------------------------------------- #
# agreement is agreement
# --------------------------------------------------------------------------- #


def test_consensus_is_monotone_in_the_vote_threshold():
    img = fc.load("addams-family")
    counts = [len(fc.consensus(img, votes=v)[0]) for v in (1, 2, 3, 4, 5, 6)]
    assert counts == sorted(counts, reverse=True), counts


def test_the_six_cascades_do_not_all_agree():
    """If they did, the comparison would have nothing in it."""
    rows = fc.agreement()
    distinct = sum(r["distinct_boxes"] for r in rows)
    unanimous = sum(r["6_or_more"] for r in rows)
    assert unanimous < 0.8 * distinct, (unanimous, distinct)
