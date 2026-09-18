"""Tests for project 43, grayscale to colour.

The one worth reading first is `test_the_selection_axis_predicts_the_ceiling`.
The statistic the twelve photographs were *chosen* on — how much chroma survives
inside one luminance level — predicts, at r = 0.995, what an oracle handed that
image's own luminance-to-colour mapping still cannot recover. The hardest
photograph in the set was identified before a single method ran.

`test_two_methods_are_worse_than_returning_the_grey_image` is the other one:
pseudo-colour and Welsh transfer both score worse on colour error than emitting
no colour at all.

Note what is *not* claimed. PSNR ranks these six methods identically to a
chroma-only error, so the usual complaint about it does not apply here, and
`test_psnr_ranks_the_methods_the_same_way` pins that rather than the opposite.
"""

from __future__ import annotations

import numpy as np
import pytest

import colourise as co


@pytest.fixture(scope="module")
def overall():
    return {r["method"]: r for r in co.evaluate()}


@pytest.fixture(scope="module")
def references():
    return co.reference_sensitivity()


# --------------------------------------------------------------------------- #
# the result
# --------------------------------------------------------------------------- #


def test_the_selection_axis_predicts_the_ceiling():
    """A statistic measured before anything ran predicts how well anything can do."""
    fit = co.axis_predicts_the_ceiling()
    assert fit["pearson_r"] > 0.95, fit
    assert 0.5 < fit["slope"] < 1.2, fit
    # and the range is wide enough for the correlation to mean something
    lo, hi = fit["ambiguity_range"]
    assert hi > 5 * lo, fit


def test_the_hardest_and_easiest_photographs_are_the_predicted_ones():
    rows = co.oracle_residual()
    by_ambiguity = sorted(rows, key=lambda r: r["ambiguity"])
    by_residual = sorted(rows, key=lambda r: r["oracle_chroma_error"])
    assert by_ambiguity[0]["image"] == by_residual[0]["image"]
    assert by_ambiguity[-1]["image"] == by_residual[-1]["image"]


def test_two_methods_are_worse_than_returning_the_grey_image(overall):
    """Inventing colour made the colour error worse."""
    control = overall["Do nothing (grey, control)"]["chroma_error"]
    worse = [name for name, r in overall.items()
             if name != "Do nothing (grey, control)" and r["chroma_error"] > control]
    assert "Pseudo-colour (viridis)" in worse
    assert "Welsh transfer (reference)" in worse
    assert len(worse) == 2, worse


def test_the_worse_methods_are_the_more_colourful_ones(overall):
    """They are not failing by being timid."""
    control = overall["Do nothing (grey, control)"]
    for name in ("Pseudo-colour (viridis)", "Welsh transfer (reference)"):
        assert overall[name]["colourfulness"] > 30 * control["colourfulness"], name


def test_psnr_ranks_the_methods_the_same_way():
    """The usual accusation against PSNR here is simply not true on this set."""
    comparison = co.metric_comparison()
    assert comparison["rankings_agree"], comparison


def test_psnr_is_nonetheless_mostly_luminance():
    """The same amount of information is worth several dB more as luminance."""
    rows = co.psnr_is_luminance()
    grey = float(np.mean([r["grey_only_psnr"] for r in rows]))
    chroma = float(np.mean([r["chroma_only_psnr"] for r in rows]))
    assert grey > chroma + 5.0, (grey, chroma)
    # and the two are opposite on the metric that measures colour. The
    # chroma-only residual is not exactly zero because flattening L pushes
    # saturated colours out of the sRGB gamut and the round trip clips them;
    # the two largest are the tropical sky and the red top, which is the
    # explanation rather than a coincidence.
    assert all(r["chroma_only_chroma_error"] < 1.5 for r in rows), rows
    assert all(r["grey_only_chroma_error"] > 2.0 for r in rows), rows
    for r in rows:
        assert r["chroma_only_chroma_error"] < 0.1 * r["grey_only_chroma_error"], r


def test_forty_scribbles_beat_the_whole_luminance_mapping(overall):
    """Where the colour is beats what the colour is."""
    rows = {r["scribbles"]: r["chroma_error"] for r in co.sweep_scribbles(
        counts=(16, 40))}
    assert rows[40] < overall["Luminance lookup (oracle)"]["chroma_error"]


def test_more_scribbles_always_help_and_with_diminishing_returns():
    rows = co.sweep_scribbles(counts=(1, 4, 16, 40, 100))
    errors = [r["chroma_error"] for r in rows]
    assert errors == sorted(errors, reverse=True), rows
    first_gain = errors[0] - errors[1]
    last_gain = errors[-2] - errors[-1]
    assert first_gain > last_gain, errors


def test_the_reference_matters_far_more_than_the_method(references, overall):
    """Welsh transfer's score is the photograph it was handed."""
    spread = float(np.mean([r["spread"] for r in references]))
    gap = abs(overall["Welsh transfer (reference)"]["chroma_error"]
              - overall["Do nothing (grey, control)"]["chroma_error"])
    assert spread > 4 * gap, (spread, gap)


def test_on_most_images_the_transfer_is_worse_than_doing_nothing(references):
    rule_worse = sum(1 for r in references
                     if r["with_rule_reference"] > r["grey_control"])
    best_worse = sum(1 for r in references if r["best"] > r["grey_control"])
    assert rule_worse >= 6, references
    assert best_worse >= 1, references


def test_no_image_is_its_own_reference():
    """An earlier version shuffled the references and three kept their own."""
    for name in co.IMAGES:
        assert co.reference_for(name) != name
    rows = co.reference_sensitivity(images=co.IMAGES[:4])
    for r in rows:
        assert r["best_reference"] != r["image"]
        assert r["worst_reference"] != r["image"]


# --------------------------------------------------------------------------- #
# the setup
# --------------------------------------------------------------------------- #


def test_every_method_preserves_luminance_exactly():
    """Which is why a chroma-only metric is the right one, and why PSNR is high."""
    truth = co.load(co.IMAGES[3])
    grey = co.greyscale(truth)
    target = co.to_lab(grey)[..., 0]
    for method in co.METHODS:
        _, prediction, _ = co.colourise(co.IMAGES[3], method)
        got = co.to_lab(prediction)[..., 0]
        assert float(np.mean(np.abs(got - target))) < 2.0, method


def test_the_grey_input_really_has_no_colour():
    for name in co.IMAGES[:4]:
        grey = co.greyscale(co.load(name))
        assert co.colourfulness(grey) < 1.0, name
        lab = co.to_lab(grey)
        assert float(np.abs(lab[..., 1]).max()) < 3.0
        assert float(np.abs(lab[..., 2]).max()) < 3.0


def test_the_twelve_are_ordered_by_ambiguity():
    values = [co.ambiguity(co.load(n)) for n in co.IMAGES]
    assert values == sorted(values), list(zip(co.IMAGES, values))


def test_ambiguity_is_zero_for_a_grey_image():
    grey = co.greyscale(co.load(co.IMAGES[0]))
    assert co.ambiguity(grey) < 0.5


def test_ambiguity_is_large_when_one_grey_means_two_colours():
    """A constructed case, so the statistic is checked against something known.

    The two halves must land in the *same* luminance bin for the statistic to see
    them, so the green is searched for rather than guessed: an earlier version
    used a green 25 Lab units away, which fell in the next bin and scored 0.
    """
    red = np.array([200, 40, 40], np.uint8)
    target_L = float(co.to_lab(red.reshape(1, 1, 3))[0, 0, 0])
    best, best_gap = None, 1e9
    for g in range(60, 200):
        candidate = np.array([40, g, 40], np.uint8)
        L = float(co.to_lab(candidate.reshape(1, 1, 3))[0, 0, 0])
        if abs(L - target_L) < best_gap:
            best, best_gap = candidate, abs(L - target_L)
    assert best_gap < 2.0, (best, best_gap)

    img = np.zeros((64, 64, 3), np.uint8)
    img[:, :32] = red
    img[:, 32:] = best
    assert co.ambiguity(img) > 20, co.ambiguity(img)

    # and a single flat colour, which is one grey meaning one colour, scores zero
    flat = np.zeros((64, 64, 3), np.uint8)
    flat[:] = red
    assert co.ambiguity(flat) < 0.5


def test_scribbles_are_placed_on_a_grid_not_where_they_help():
    truth = co.load(co.IMAGES[0])
    a = co.sample_scribbles(truth, 16, seed=0)
    b = co.sample_scribbles(truth, 16, seed=1)
    assert a.sum() > 0 and b.sum() > 0
    # different seeds jitter within cells but keep the same coverage
    assert abs(int(a.sum()) - int(b.sum())) < 0.3 * int(a.sum())
    h, w = truth.shape[:2]
    ys, xs = np.nonzero(a)
    assert ys.max() - ys.min() > 0.5 * h
    assert xs.max() - xs.min() > 0.5 * w


def test_the_oracles_are_labelled_as_oracles(overall):
    """A table that ranked these against the real methods without saying so
    would be reporting that being given the answer helps."""
    for name in co.ORACLES:
        assert overall[name]["is_oracle"], name
    assert not overall["Welsh transfer (reference)"]["is_oracle"]
    assert not overall["Pseudo-colour (viridis)"]["is_oracle"]


def test_lab_round_trips():
    truth = co.load(co.IMAGES[5])
    back = co.from_lab(co.to_lab(truth))
    assert float(np.mean(np.abs(back.astype(float) - truth.astype(float)))) < 2.0
