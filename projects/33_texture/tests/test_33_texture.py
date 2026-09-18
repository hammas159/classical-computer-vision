"""Tests for project 33, the texture descriptors.

Most of these pin a *finding* rather than a number, so that a later change which
quietly reverses one of the project's conclusions fails loudly instead of
rewriting the README by accident.

Three of them exist because the first version of this project was wrong in ways
that produced a perfectly plausible-looking table:
`test_degrading_both_sides_of_the_comparison_measures_nothing`,
`test_the_benchmark_saturates_and_stops_ranking_anything` and
`test_a_rotated_probe_shows_the_same_surface_region`.
"""

from __future__ import annotations

import numpy as np
import pytest

import texture as tx

CHANCE = 1.0 / len(tx.TEXTURES)

#: Two crop seeds rather than the five `run.py` reports. Every finding asserted
#: below is a gap of 0.1 or more against a seed-to-seed spread of about 0.04, so
#: two seeds resolve them and five would only make the suite slower. The README's
#: numbers come from `run.py`; these tests pin the *findings*.
SEEDS = (0, 1)


@pytest.fixture(scope="module")
def summary():
    """The invariance table, computed once for the whole module.

    Built per descriptor per column per seed, so recomputing it in each of the
    five tests below would cost about two minutes for nothing.
    """
    return {r["descriptor"]: r for r in tx.invariance_summary(seeds=SEEDS)}


# --------------------------------------------------------------------------- #
# the dataset
# --------------------------------------------------------------------------- #


def test_the_gallery_and_the_probes_are_cut_from_disjoint_crops():
    """A probe that can match its own pixels is not being classified."""
    gallery, probes, labels = tx.build_split()
    assert len(gallery) == len(probes) == len(labels)
    assert len(labels) == len(tx.TEXTURES) * tx.PER_CLASS

    for g, p in zip(gallery, probes):
        assert not np.array_equal(g, p)


def test_a_rotated_probe_shows_the_same_surface_region():
    """Rotation has to be rotation, not relocation.

    The crop centre is carried through the same matrix that warps the plate, so
    a rotated probe is the same piece of surface turned. Before this, the probe
    landed at the same *coordinates* in a rotated plate — a different piece of
    surface — and a 90 degree rotation appeared to raise LBP's accuracy from
    0.674 to 0.847. It was measuring the crop, not the angle.
    """
    plate = tx.load_scene(tx.TEXTURES[0])
    centre = tx.crop_centres(plate.shape, tx.PATCH, 1, seed=0)[0]
    warped, matrix = tx._warp_plate(plate, 180.0, 1.0)

    upright = tx._cut(plate, centre, tx.PATCH)
    turned = tx._cut(warped, tx._move(matrix, centre), tx.PATCH)

    # 180 degrees is exact, so the same region must come back reversed in both
    # axes to within interpolation error.
    assert np.abs(turned.astype(float) - upright[::-1, ::-1].astype(float)).mean() < 3.0


def test_crop_centres_stay_where_a_rotation_cannot_push_them_out():
    plate = tx.load_scene(tx.TEXTURES[0])
    h, w = plate.shape
    radius = min(h, w) / 2.0 - tx.PATCH
    for y, x in tx.crop_centres(plate.shape, tx.PATCH, 2 * tx.PER_CLASS, seed=0):
        assert (y - (h - 1) / 2.0) ** 2 + (x - (w - 1) / 2.0) ** 2 <= radius ** 2 + 1e-6


def test_every_texture_is_a_distinct_surface():
    assert len(set(tx.TEXTURES)) == len(tx.TEXTURES)
    from shared import io

    for name in tx.TEXTURES:
        assert name in io.REAL_PHOTOS, name


# --------------------------------------------------------------------------- #
# the protocol, and the two ways of getting it wrong
# --------------------------------------------------------------------------- #


def test_degrading_both_sides_of_the_comparison_measures_nothing():
    """The harness bug that made every descriptor look illumination-invariant.

    Nearest-neighbour matching is invariant to anything that moves every sample
    the same way. Degrade the gallery as well as the probes and a brightness
    change costs *nothing* — not even to a plain intensity histogram, which has
    no such invariance at all. The number is real, the experiment is empty.
    """
    fn = tx.DESCRIPTORS["Raw histogram (control)"]
    gallery, probes, labels = tx.build_split(degradation="illumination", strength=0.6)

    both_degraded = tx.leave_one_out_accuracy(tx._features(probes, fn), labels)
    honest = tx.cross_condition_accuracy(tx._features(gallery, fn),
                                         tx._features(probes, fn), labels)

    assert both_degraded > 0.7, "degrading both sides hides the damage"
    assert honest < 0.3, "the honest protocol shows it"


def test_the_benchmark_saturates_and_stops_ranking_anything():
    """Why the operating point is 32 px and not the obvious 96.

    At 96 px three of the four real descriptors score a perfect 1.000 and the
    fourth is at 0.997 — a total spread of 0.003, which is an eighth of the
    seed-to-seed noise. The benchmark cannot say which is better. At 32 px the
    spread is 0.25 and it can.
    """
    rows = {r["patch"]: r for r in tx.sweep_patch_size(seeds=SEEDS)}
    real = [d for d in tx.DESCRIPTORS if "control" not in d]

    saturated = [rows[96][d] for d in real]
    assert min(saturated) > 0.99
    assert max(saturated) - min(saturated) < 0.02

    working = [rows[tx.PATCH][d] for d in real]
    assert max(working) < 1.0
    assert max(working) > 0.9
    assert max(working) - min(working) > 0.15


def test_the_clean_row_uses_the_same_protocol_as_the_degraded_rows():
    """Or the drop attributed to a degradation is partly a change of method."""
    clean = {r["descriptor"]: r["accuracy"] for r in tx.evaluate_descriptors()}
    zero = {r["descriptor"]: r["accuracy"]
            for r in tx.evaluate_descriptors(degradation="noise", strength=0.0)}
    assert clean == zero


# --------------------------------------------------------------------------- #
# the findings
# --------------------------------------------------------------------------- #


def test_lbp_is_the_worst_descriptor_clean_and_the_only_survivor_of_a_relight(summary):
    """The project's headline, and it is the opposite of a ranking.

    LBP keeps only the *sign* of local differences, so a brightness change
    leaves its code untouched by construction. Clean it is the weakest real
    descriptor here (0.701, below a plain intensity histogram); pull the light
    down and it is the only thing in the table above chance at all — 0.640 while
    GLCM, Gabor, Laws and the control are all at 1/12.
    """
    real = [d for d in summary if "control" not in d]

    assert min(real, key=lambda d: summary[d]["clean_accuracy"]) == "LBP (uniform)"
    assert max(summary, key=lambda d: summary[d]["illumination_accuracy"]) == "LBP (uniform)"
    assert summary["LBP (uniform)"]["illumination_drop"] < 0.15

    for other in summary:
        if other != "LBP (uniform)":
            assert summary[other]["illumination_accuracy"] == pytest.approx(
                CHANCE, abs=0.03), other


def test_no_descriptor_is_best_everywhere(summary):
    """A descriptor is an invariance, not a quality.

    Three different descriptors and the do-nothing control each win at least one
    column, and the descriptor that wins the clean column is at chance in
    another. There is no row of this table that is good everywhere, which is why
    the project reports a matrix and not a ranking.
    """
    columns = ["clean_accuracy"] + [f"{c}_accuracy" for c in tx.INVARIANCE_COLUMNS]
    winners = {c: max(summary, key=lambda d: summary[d][c]) for c in columns}
    assert len(set(winners.values())) >= 3, winners

    best_clean = winners["clean_accuracy"]
    at_chance = [c for c in columns if summary[best_clean][c] < CHANCE + 0.03]
    assert at_chance, f"{best_clean} never falls to chance: {summary[best_clean]}"


def test_gabor_is_the_noise_robust_descriptor_and_laws_is_not(summary):
    """Bandwidth is the whole explanation.

    A Gabor filter is a sinusoid under a 21x21 Gaussian envelope: it averages
    over hundreds of pixels and rejects everything outside its passband. Laws
    uses 5x5 high-pass masks, which is where broadband noise lives — so a
    descriptor tied for best on clean patches lands at chance while Gabor is
    still at 0.611. The runner-up in the noise column is the do-nothing control.
    """
    assert max(summary, key=lambda d: summary[d]["noise_accuracy"]) == "Gabor bank"
    assert summary["Laws energy"]["noise_accuracy"] < 0.2
    assert summary["Gabor bank"]["noise_accuracy"] > 3 * summary["Laws energy"]["noise_accuracy"]

    runner_up = sorted(summary, key=lambda d: -summary[d]["noise_accuracy"])[1]
    assert runner_up == "Raw histogram (control)"


def test_laws_is_the_one_descriptor_that_survives_a_right_angle(summary):
    """And it is invariant by construction rather than by luck.

    `feat_laws` averages each mask with its transpose (L5E5 with E5L5), and a 90
    degree rotation is exactly what swaps them — so it loses nothing at all.
    GLCM computes four angles and is *not* invariant, because it keeps them in a
    fixed order instead of canonicalising them. Four angles are not the same
    thing as rotation invariance.
    """
    assert summary["Laws energy"]["rotation_drop"] == pytest.approx(0.0, abs=0.02)
    assert summary["GLCM (Haralick)"]["rotation_drop"] > 0.25


def test_at_45_degrees_the_do_nothing_control_beats_every_real_descriptor(summary):
    """Because it is the only descriptor with no geometry in it at all.

    An intensity histogram throws away every spatial relationship, which makes
    it weak on clean patches and exactly rotation-invariant. At 45 degrees — the
    worst angle for a square-sampled operator — that trade pays off, and the
    control wins the column outright.
    """
    control = summary["Raw histogram (control)"]["rotation45_accuracy"]
    for name in summary:
        if "control" not in name:
            assert summary[name]["rotation45_accuracy"] < control, name


def test_lbp_needs_far_more_surface_than_anything_else():
    """Its weakness is sample size, not illumination.

    LBP is a 26-bin histogram. A 24x24 patch gives it a few hundred codes to
    fill those bins with, and it collapses to 0.45 while Laws is still above
    0.9 — a gap that closes entirely by 64 px.
    """
    rows = {r["patch"]: r for r in tx.sweep_patch_size(seeds=SEEDS)}
    small = min(rows)
    assert rows[small]["LBP (uniform)"] < 0.6
    assert rows[small]["Laws energy"] > 0.9
    assert rows[64]["LBP (uniform)"] > 0.95


# --------------------------------------------------------------------------- #
# the descriptors themselves
# --------------------------------------------------------------------------- #


def test_lbp_is_invariant_to_a_linear_intensity_change_at_the_feature_level():
    """Not measured through a classifier — measured on the vector itself."""
    patch = tx.build_patches()[0][0]
    brighter = np.clip(patch.astype(np.float32) * 0.55 + 40, 0, 255).astype(np.uint8)

    a, b = tx.feat_lbp(patch), tx.feat_lbp(brighter)
    assert np.abs(a - b).max() < 0.05

    c, d = tx.feat_glcm(patch), tx.feat_glcm(brighter)
    assert np.abs(c - d).max() > 0.1, "GLCM should not survive this"


def test_every_descriptor_returns_a_finite_fixed_length_vector():
    patches, _ = tx.build_patches()
    for name, fn in tx.DESCRIPTORS.items():
        vectors = [fn(p) for p in patches[:6]]
        assert len({v.shape for v in vectors}) == 1, name
        assert all(np.isfinite(v).all() for v in vectors), name


def test_separability_keeps_discriminating_where_accuracy_does_not():
    """The reason a second metric is reported at all.

    At 96 px every real descriptor is at 1.000 and accuracy has nothing left to
    say. The Fisher-style scatter ratio still separates them by a factor of
    three, which is the difference between classifying correctly and classifying
    correctly with room to spare.
    """
    rows = {r["descriptor"]: r for r in tx.evaluate_descriptors(patch=96)}
    real = [d for d in rows if "control" not in d]

    assert {rows[d]["accuracy"] for d in real} == {1.0}
    scores = [rows[d]["separability"] for d in real]
    assert max(scores) > 2 * min(scores)


def test_the_confusion_matrix_accounts_for_every_probe():
    matrix = tx.confusion("Laws energy")
    assert matrix.shape == (len(tx.TEXTURES), len(tx.TEXTURES))
    assert matrix.sum() == len(tx.TEXTURES) * tx.PER_CLASS
    assert (matrix.sum(axis=1) == tx.PER_CLASS).all()
    # the diagonal is the accuracy, by another route
    accuracy = next(r["accuracy"] for r in tx.evaluate_descriptors()
                    if r["descriptor"] == "Laws energy")
    assert np.trace(matrix) / matrix.sum() == pytest.approx(accuracy, abs=1e-4)
