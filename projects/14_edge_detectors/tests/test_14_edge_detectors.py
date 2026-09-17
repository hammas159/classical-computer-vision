"""Tests for project 14, edge detectors.

Most of these pin a *finding* rather than a number, so that a later change which
quietly reverses one of the project's conclusions fails loudly instead of
rewriting the README by accident.
"""

from __future__ import annotations

import numpy as np
import pytest

import edges as ed
from shared.io import to_gray
from shared.metrics import edge_prf


# --------------------------------------------------------------------------- #
# the operators
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", list(ed.GRADIENTS))
def test_every_gradient_returns_a_normalised_magnitude(name):
    img, _ = ed.scene(seed=0)
    mag = ed.GRADIENTS[name](to_gray(img))
    assert mag.dtype == np.float32
    assert mag.min() >= 0.0
    assert mag.max() == pytest.approx(1.0, abs=1e-5)


def test_a_flat_image_has_no_edges():
    """The check that would catch a gradient reading its own padding."""
    flat = np.full((64, 64), 128, np.uint8)
    for name, fn in ed.GRADIENTS.items():
        mag = fn(flat)
        assert float(mag.max()) < 1e-6, f"{name} found an edge in a flat image"


def test_every_operator_finds_a_step_edge():
    step = np.zeros((64, 64), np.uint8)
    step[:, 32:] = 255
    for name, fn in ed.GRADIENTS.items():
        mag = fn(step)
        column = mag[:, 30:35].max(axis=1)
        assert float(column.min()) > 0.3, f"{name} missed a clean step edge"


# --------------------------------------------------------------------------- #
# the harness
# --------------------------------------------------------------------------- #


def test_every_operator_is_given_its_own_best_threshold():
    """Otherwise the comparison measures tuning, not operators.

    A fixed threshold across operators whose magnitudes are normalised
    differently is a comparison of normalisation.
    """
    img, truth = ed.scene(noise_sigma=15.0, seed=0)
    gray = to_gray(img)
    for name, fn in ed.GRADIENTS.items():
        f1, t, e = ed.best_threshold(gray, truth, fn)
        at_fixed = edge_prf(ed.threshold_magnitude(fn(gray), 0.1), truth, ed.TOLERANCE)["f1"]
        assert f1 >= at_fixed - 1e-9, f"{name}: the swept best is worse than a fixed 0.1"


def test_the_scene_has_exact_edges():
    _, truth = ed.scene(seed=0)
    assert truth.dtype == np.uint8
    assert set(np.unique(truth)) <= {0, 255}
    # edges are a thin minority of the frame; anything near half is a filled mask
    assert 0.001 < float((truth > 0).mean()) < 0.05


# --------------------------------------------------------------------------- #
# the findings
# --------------------------------------------------------------------------- #


def test_a_clean_scene_cannot_separate_the_operators():
    """Why the comparison is run under noise at all.

    On a noise-free scene almost everything scores 1.000. A project that
    compared edge detectors on clean synthetic shapes would conclude they are
    identical, which is the wrong conclusion reached honestly.
    """
    rows = ed.evaluate_operators(noise_sigma=0.0, seeds=(0,), runs=1)
    perfect = [r for r in rows if r["f1"] >= 0.999]
    assert len(perfect) >= 4


def test_the_auto_canny_recipe_collapses_under_noise():
    """The project's headline, pinned.

    "Automatic Canny" — thresholds at 0.66 and 1.33 times the image median — is
    a widely copied recipe. The median of a noisy image is not a statistic about
    its edges, and the recipe fails catastrophically while fixed thresholds do
    not.
    """
    rows = {r["noise_sigma"]: r for r in ed.sweep_noise(seeds=(0,))}
    auto_clean = rows[0.0]["Canny (auto median)"]
    auto_noisy = rows[30.0]["Canny (auto median)"]
    fixed_noisy = rows[30.0]["Canny (fixed 50/150)"]

    assert auto_clean > 0.99
    assert auto_noisy < 0.30
    assert fixed_noisy > 0.95
    assert fixed_noisy > auto_noisy + 0.6


def test_the_matching_tolerance_decides_the_ranking():
    """The second finding: an F1 without a tolerance is not a number.

    The *same* detections score 0.246 and 1.000 depending only on how far a
    detected pixel may be from the true edge and still count.
    """
    rows = {r["tolerance_px"]: r for r in ed.sweep_tolerance(seeds=(0,))}
    strict, loose = rows[0], rows[3]
    ops = [k for k in strict if k != "tolerance_px"]

    assert min(strict[o] for o in ops) < 0.30
    assert min(loose[o] for o in ops) > 0.99
    # and the ORDER changes, not just the values
    assert sorted(ops, key=lambda o: -strict[o]) != sorted(ops, key=lambda o: -loose[o])


def test_roberts_degrades_worst_because_it_has_no_smoothing():
    """A 2x2 operator has no averaging in it, so noise goes straight through.

    It is competitive up to sigma 15 and collapses by sigma 50, while the 3x3
    operators degrade gracefully. That is the whole reason the larger kernels
    exist, stated as a number.
    """
    rows = {r["noise_sigma"]: r for r in ed.sweep_noise(seeds=(0,))}
    assert rows[15.0]["Roberts"] > 0.99
    assert rows[50.0]["Roberts"] < 0.60
    assert rows[50.0]["Sobel"] > rows[50.0]["Roberts"] + 0.3


def test_the_best_operator_changes_with_the_noise_level():
    """No operator wins everywhere, which is the point of the sweep."""
    rows = ed.sweep_noise(seeds=(0,))
    ops = [k for k in rows[0] if k != "noise_sigma"]
    winners = {max(ops, key=lambda o: r[o]) for r in rows}
    assert len(winners) > 1
