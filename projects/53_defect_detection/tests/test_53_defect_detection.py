"""Tests for project 53, industrial defect detection.

The one worth reading first is `test_the_detectors_are_complementary_not_competing`.
A smear — a local loss of texture at unchanged brightness — is found by the local
standard deviation and by four of the other five *exactly never*, because no
intensity residual can see it. Picking "the best defect detector" is picking which
defects to miss.

`test_a_clean_surface_is_half_the_problem` is the other one: on twelve surfaces
with nothing wrong with them, where the true answer is an empty mask, these
detectors mark between 3.7% and 14.3% of the surface.

`test_the_flag_rule_closes_and_never_opens` is a regression test for the bug that
made every number in this project zero: a 3×3 morphological opening erases a
two-pixel-wide scratch, so 2011 correctly thresholded pixels became 82.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import defects as df


@pytest.fixture(scope="module")
def overall():
    return {r["detector"]: r for r in df.evaluate()}


@pytest.fixture(scope="module")
def by_defect():
    return {r["defect"]: r for r in df.per_defect()}


# --------------------------------------------------------------------------- #
# the result
# --------------------------------------------------------------------------- #


def test_the_detectors_are_complementary_not_competing(by_defect):
    """No detector is best at everything, and one defect has only one answer."""
    winners = set()
    for kind, row in by_defect.items():
        winners.add(max(df.REAL_DETECTORS, key=lambda n: row[n]))
    assert len(winners) > 1, by_defect

    smear = by_defect["smear"]
    blind = [n for n in df.REAL_DETECTORS if smear[n] == 0.0]
    assert len(blind) >= 3, smear
    assert smear["Local standard deviation"] > 0.4, smear


def test_only_a_texture_measure_sees_a_smear():
    """A smear has the same mean as its surround, so no residual can see it.

    Checked directly rather than inferred from the table: the mean inside the
    smear is unchanged and the local standard deviation is not.
    """
    image = df.load("surface_coarse_cloth")
    bad, mask = df.plant(image, "smear", seed=0)
    g0 = df.to_gray(image).astype(np.float32)
    g1 = df.to_gray(bad).astype(np.float32)
    sel = mask > 0
    assert abs(float(g1[sel].mean() - g0[sel].mean())) < 6.0
    assert float(g1[sel].std()) < 0.8 * float(g0[sel].std())


def test_a_clean_surface_is_half_the_problem(overall):
    """The true answer is an empty mask and nothing returns one."""
    real = [r for r in overall.values() if not r["is_control"]]
    for r in real:
        assert r["false_alarm_share"] > 0.0, r
    assert max(r["false_alarm_share"] for r in real) > 0.02
    assert overall["Flag nothing (control)"]["false_alarm_share"] == 0.0


def test_the_two_arms_rank_the_detectors_differently():
    result = df.arms_disagree()
    assert not result["rankings_agree"], result
    assert result["detection_winner"] != result["alarm_winner"], result


def test_the_surface_decides_more_than_the_detector():
    rows = df.per_surface()
    best = [r["best_detection_rate"] for r in rows]
    assert max(best) - min(best) > 0.5, rows


def test_pixel_accuracy_is_unusable_here():
    """A defect covers about one per cent, so flagging nothing is 99% right."""
    result = df.pixel_accuracy_is_broken()
    assert result["mean_defect_share"] < 0.05
    assert result["flag_nothing_pixel_accuracy"] > 0.95


def test_a_bigger_defect_is_easier_to_find():
    rows = {r["severity"]: r for r in df.sweep_severity(images=df.IMAGES[:4],
                                                        severities=(1, 16))}
    better = sum(1 for n in df.REAL_DETECTORS if rows[16][n] >= rows[1][n])
    assert better >= len(df.REAL_DETECTORS) - 1, rows


def test_filling_does_not_help_and_that_is_reported():
    """The obvious repair for an edge response, measured and found worthless.

    Kept as a test because deleting the finding would leave the next reader to
    try it again.
    """
    rows = df.filling_helps(images=df.IMAGES[:4])
    gains = [r["gain"] for r in rows]
    assert max(abs(g) for g in gains) < 0.05, rows
    import inspect

    assert "It does not" in inspect.getdoc(df.fill)


# --------------------------------------------------------------------------- #
# the bug that made every number zero
# --------------------------------------------------------------------------- #


def test_the_flag_rule_closes_and_never_opens():
    import inspect

    source = inspect.getsource(df._flag)
    assert "MORPH_CLOSE" in source
    assert "MORPH_OPEN" not in source


def test_a_thin_scratch_survives_the_flag_rule():
    """The invariant the opening broke: a two-pixel line must not vanish."""
    residual = np.zeros((200, 200), np.float32)
    residual[100:102, 40:160] = 60.0
    flagged = df._flag(residual, k=4.0, min_area=40)
    assert (flagged > 0).sum() > 100, int((flagged > 0).sum())


def test_every_detector_finds_a_scratch_it_should(by_defect):
    """If a scratch were missed by most of them, the flag rule would be suspect."""
    scratch = by_defect["scratch"]
    finding = [n for n in df.REAL_DETECTORS if scratch[n] > 0.3]
    assert len(finding) >= 4, scratch


# --------------------------------------------------------------------------- #
# the construction
# --------------------------------------------------------------------------- #


def test_defect_strength_is_scaled_to_the_noise_ceiling_not_the_contrast():
    """An earlier version scaled by `surface_contrast`, which made every defect
    the same multiple of the texture it was hiding in — so nothing worked
    anywhere and the comparison measured only the construction."""
    import inspect

    source = inspect.getsource(df.plant)
    assert "noise_ceiling(image)" in source
    assert "surface_contrast" not in source


def test_the_noise_ceiling_is_what_a_defect_has_to_clear():
    for name in df.IMAGES[:4]:
        image = df.load(name)
        ceiling = df.noise_ceiling(image)
        assert ceiling > 0.5, (name, ceiling)
        bad, mask = df.plant(image, "scratch", severity=8.0)
        g0 = df.to_gray(image).astype(np.float32)
        g1 = df.to_gray(bad).astype(np.float32)
        applied = float(np.abs(g1 - g0)[mask > 0].mean())
        assert applied > 2.0 * ceiling, (name, applied, ceiling)


def test_a_clean_surface_has_an_empty_truth_mask():
    image = df.load(df.IMAGES[3])
    ok, mask = df.clean(image)
    assert np.array_equal(ok, image)
    assert mask.max() == 0


def test_the_twelve_are_ordered_by_uniformity():
    values = [df.uniformity(df.load(n)) for n in df.IMAGES]
    assert values == sorted(values), list(zip(df.IMAGES, values))


def test_every_defect_kind_changes_the_image_inside_its_mask():
    image = df.load(df.IMAGES[6])
    for kind in df.DEFECTS:
        bad, mask = df.plant(image, kind, seed=1)
        assert mask.max() == 255, kind
        assert 0.0005 < float((mask > 0).mean()) < 0.10, (kind, (mask > 0).mean())
        changed = np.abs(bad.astype(float) - image.astype(float)).sum(axis=2)
        assert float(changed[mask > 0].mean()) > 1.0, kind


def test_the_controls_do_what_they_say():
    image = df.load(df.IMAGES[0])
    assert df.detect_nothing(image).max() == 0
    assert df.detect_everything(image).min() == 255
    assert df.iou(df.detect_nothing(image), np.zeros(image.shape[:2], np.uint8)) == 1.0


def test_found_needs_real_overlap():
    truth = np.zeros((100, 100), np.uint8)
    truth[40:60, 40:60] = 255
    empty = np.zeros((100, 100), np.uint8)
    assert not df.found(empty, truth)
    assert df.found(truth, truth)
    partial = np.zeros((100, 100), np.uint8)
    partial[40:42, 40:60] = 255          # 10% of the defect
    assert df.found(partial, truth, fraction=0.10)
    assert not df.found(partial, truth, fraction=0.50)
