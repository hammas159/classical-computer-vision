"""Tests for the shared layer.

These assert *numerical* behaviour, not just that the functions run. Every
project in the repo reports numbers produced by this code, so a silent error
here would corrupt every result in the repo at once.
"""

from __future__ import annotations

import numpy as np
import pytest

from shared import bench, figures, io, metrics, synth

# --------------------------------------------------------------------------- #
# io
# --------------------------------------------------------------------------- #


def test_sample_is_rgb_uint8():
    img = io.sample("astronaut")
    assert img.dtype == np.uint8
    assert img.ndim == 3 and img.shape[2] == 3


def test_sample_gray_is_2d():
    assert io.sample("camera", gray=True).ndim == 2


def test_every_declared_sample_loads():
    for name in io.sample_names():
        img = io.sample(name)
        assert img.dtype == np.uint8 and img.ndim == 3, name


def test_float_uint8_roundtrip_is_lossless():
    img = io.sample("chelsea")
    assert np.array_equal(io.to_uint8(io.to_float(img)), img)


def test_to_uint8_clips_instead_of_wrapping():
    # the uint8 wraparound bug: 1.2 must become 255, never 51
    assert io.to_uint8(np.array([[1.2]], np.float32))[0, 0] == 255
    assert io.to_uint8(np.array([[-0.4]], np.float32))[0, 0] == 0


def test_imwrite_imread_roundtrip_preserves_channel_order(tmp_path):
    # a pure-red RGB image must still be pure red after a disk round trip;
    # a BGR mix-up anywhere in the chain turns it blue
    red = np.zeros((16, 16, 3), np.uint8)
    red[..., 0] = 255
    io.imwrite(tmp_path / "red.png", red)
    assert np.array_equal(io.imread(tmp_path / "red.png"), red)


def test_imread_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        io.imread("does-not-exist-anywhere.png")


# --------------------------------------------------------------------------- #
# synth
# --------------------------------------------------------------------------- #


def test_gaussian_noise_has_requested_sigma():
    # a mid-gray field avoids clipping at either end, so the measured
    # std-dev should match the requested sigma closely
    flat = np.full((256, 256, 3), 128, np.uint8)
    noisy = synth.gaussian_noise(flat, sigma=20.0, seed=0)
    measured = float(np.std(noisy.astype(np.float64) - 128.0))
    assert 18.0 < measured < 22.0


def test_salt_pepper_density_is_close_to_requested():
    flat = np.full((400, 400, 3), 128, np.uint8)
    out = synth.salt_pepper_noise(flat, density=0.10, seed=0)
    corrupted = np.any(out != 128, axis=2).mean()
    assert 0.08 < corrupted < 0.12


def test_noise_is_reproducible_from_seed():
    img = io.sample("camera")
    assert np.array_equal(
        synth.gaussian_noise(img, 25.0, seed=7), synth.gaussian_noise(img, 25.0, seed=7)
    )


def test_blur_kernels_are_normalised():
    assert synth.motion_blur_kernel(15, 30.0).sum() == pytest.approx(1.0, abs=1e-5)
    assert synth.defocus_kernel(7).sum() == pytest.approx(1.0, abs=1e-6)


def test_blur_reduces_high_frequency_energy():
    img = io.sample("camera", gray=True)
    blurred = synth.apply_kernel(img, synth.defocus_kernel(5))
    assert np.std(blurred.astype(float)) < np.std(img.astype(float))


def test_random_homography_is_invertible():
    H = synth.random_homography((300, 400), seed=3)
    assert H.shape == (3, 3)
    assert abs(np.linalg.det(H)) > 1e-8  # a collinear config would fail silently


def test_warp_by_known_homography_is_recoverable():
    img = io.sample("astronaut")
    H = synth.random_homography(img.shape, jitter=0.04, seed=1)
    warped = synth.warp_by_homography(img, H)
    back = synth.warp_by_homography(warped, np.linalg.inv(H))
    # centre crop avoids the border region that warped off-canvas and back
    c = lambda a: a[80:-80, 80:-80]  # noqa: E731
    assert metrics.psnr(c(back), c(img)) > 20.0


def test_uniform_haze_scales_contrast_by_exactly_the_transmission():
    # with a constant t, I = J*t + A*(1-t) is affine, so std(I) == t * std(J).
    # This is the one exact statement available about haze, so it is the one
    # worth asserting. "Haze reduces global contrast" is NOT generally true:
    # if the far plane is already bright sky, whitening it can raise the
    # global std-dev.
    img = io.sample("rocket")
    beta = 1.0
    hazy, t = synth.add_haze(img, beta=beta, depth=np.ones(img.shape[:2], np.float32))
    expected_t = float(np.exp(-beta))
    assert t.min() == pytest.approx(expected_t, abs=1e-5)
    ratio = metrics.rms_contrast(hazy) / metrics.rms_contrast(img)
    assert ratio == pytest.approx(expected_t, rel=0.02)


def test_haze_gradient_puts_the_far_plane_at_the_top():
    img = io.sample("rocket")
    hazy, t = synth.add_haze(img, beta=1.5)
    assert t.shape == img.shape[:2]
    assert 0.0 < t.min() <= t.max() <= 1.0
    # the horizon is distant, the foreground is near: transmission must be
    # lowest at the top of the frame, not the bottom
    assert t[0].mean() < t[-1].mean()
    # and contrast must fall where the haze actually is
    far = t < 0.4
    assert far.any()
    g_hazy = io.to_float(io.to_gray(hazy))
    g_orig = io.to_float(io.to_gray(img))
    assert np.std(g_hazy[far]) < np.std(g_orig[far])


def test_low_light_darkens():
    img = io.sample("coffee")
    assert metrics.mean_brightness(synth.low_light(img, gamma=3.0)) < metrics.mean_brightness(img)


def test_downsample_for_sr_shape_and_not_naive_resize():
    import cv2

    img = io.sample("astronaut")  # 512x512
    lr = synth.downsample_for_sr(img, scale=4)
    assert lr.shape[0] == img.shape[0] // 4 and lr.shape[1] == img.shape[1] // 4
    # blur-then-decimate must differ measurably from cv2.resize, or the whole
    # super-resolution comparison is measuring the wrong degradation
    naive = cv2.resize(img, (lr.shape[1], lr.shape[0]), interpolation=cv2.INTER_AREA)
    assert metrics.psnr(lr, naive) < 45.0


def test_copy_move_mask_matches_pasted_pixels():
    img = io.sample("chelsea")
    f = synth.copy_move_forgery(img, size=64, seed=2)
    dx, dy = f.dst_xy
    assert f.mask[dy + 32, dx + 32] == 255
    assert f.mask.sum() / 255 == 64 * 64
    assert np.array_equal(f.image[dy : dy + 64, dx : dx + 64], f.image[dy : dy + 64, dx : dx + 64])


def test_scratches_mask_marks_only_damaged_pixels():
    img = io.sample("astronaut")
    damaged, mask = synth.add_scratches(img, seed=4)
    assert mask.sum() > 0
    assert np.all(damaged[mask > 0] == 255)
    assert np.array_equal(damaged[mask == 0], img[mask == 0])


def test_shapes_scene_has_edges_and_matching_size():
    img, edges = synth.shapes(size=256, seed=0)
    assert img.shape == (256, 256) and edges.shape == (256, 256)
    assert edges.max() == 255 and 0 < (edges > 0).mean() < 0.1


def test_document_scene_corners_are_inside_the_photo():
    photo, page, corners = synth.document_scene(seed=0)
    H, W = photo.shape[:2]
    assert corners.shape == (4, 2)
    assert corners[:, 0].min() >= 0 and corners[:, 0].max() < W
    assert corners[:, 1].min() >= 0 and corners[:, 1].max() < H


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def test_psnr_of_identical_images_is_infinite():
    img = io.sample("camera")
    assert metrics.psnr(img, img) == float("inf")
    assert metrics.ssim(img, img) == pytest.approx(1.0, abs=1e-6)


def test_psnr_matches_the_closed_form_for_known_noise():
    # PSNR = 10*log10(1/MSE); for a constant offset of 10/255 the MSE is exact
    a = np.full((64, 64), 100, np.uint8)
    b = np.full((64, 64), 110, np.uint8)
    expected = 10 * np.log10(1.0 / (10.0 / 255.0) ** 2)
    assert metrics.psnr(a, b) == pytest.approx(expected, abs=0.01)


def test_psnr_decreases_as_noise_increases():
    img = io.sample("astronaut")
    scores = [metrics.psnr(synth.gaussian_noise(img, s, seed=0), img) for s in (5, 15, 40)]
    assert scores[0] > scores[1] > scores[2]


def test_iou_and_dice_bounds():
    a = np.zeros((64, 64), np.uint8)
    a[:32] = 255
    b = np.zeros((64, 64), np.uint8)
    b[16:48] = 255
    assert metrics.iou(a, a) == 1.0
    assert metrics.dice(a, a) == 1.0
    assert metrics.iou(a, b) == pytest.approx(1 / 3, abs=1e-6)  # 16 overlap / 48 union
    assert metrics.dice(a, b) >= metrics.iou(a, b)


def test_iou_of_two_empty_masks_is_one():
    z = np.zeros((16, 16), np.uint8)
    assert metrics.iou(z, z) == 1.0


def test_edge_prf_perfect_and_tolerance_matters():
    _, edges = synth.shapes(size=256, seed=0)
    perfect = metrics.edge_prf(edges, edges, tolerance=0)
    assert perfect["f1"] == pytest.approx(1.0)

    shifted = np.roll(edges, 1, axis=1)  # a one-pixel offset is not an error
    assert metrics.edge_prf(shifted, edges, tolerance=0)["f1"] < 0.9
    assert metrics.edge_prf(shifted, edges, tolerance=2)["f1"] > 0.95


def test_pratt_fom_is_one_for_a_perfect_match():
    _, edges = synth.shapes(size=256, seed=0)
    assert metrics.pratt_fom(edges, edges) == pytest.approx(1.0, abs=1e-6)


def test_endpoint_error_of_known_flow():
    true = synth.synthetic_flow((64, 64), magnitude=5.0, kind="translation")
    assert metrics.endpoint_error(true, true) == pytest.approx(0.0)
    # a flow that predicts no motion at all has EPE equal to the true magnitude
    zero = np.zeros_like(true)
    assert metrics.endpoint_error(zero, true) == pytest.approx(np.hypot(5.0, 2.5), abs=1e-4)


def test_reprojection_error_is_zero_for_the_true_homography():
    import cv2

    H = synth.random_homography((300, 400), seed=5)
    src = np.float32([[10, 10], [200, 30], [180, 240], [30, 260]])
    dst = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    assert metrics.reprojection_error(H, src, dst) == pytest.approx(0.0, abs=1e-3)


def test_corner_error_of_identical_corners_is_zero():
    c = np.float32([[0, 0], [10, 0], [10, 10], [0, 10]])
    assert metrics.corner_error(c, c) == 0.0
    assert metrics.corner_error(c + 3.0, c) == pytest.approx(np.hypot(3, 3), abs=1e-6)


# --------------------------------------------------------------------------- #
# bench
# --------------------------------------------------------------------------- #


def test_timeit_returns_result_and_plausible_timing():
    calls = {"n": 0}

    def work():
        calls["n"] += 1
        return 42

    result, t = bench.timeit(work, runs=5, warmup=2)
    assert result == 42
    assert calls["n"] == 7  # warmup runs really happened
    assert t.runs == 5 and t.min_ms <= t.median_ms <= t.max_ms
    assert t.median_ms >= 0.0


def test_run_methods_times_every_entry():
    out = bench.run_methods({"a": lambda: 1, "b": lambda: 2}, runs=3, warmup=1)
    assert set(out) == {"a", "b"}
    assert out["a"][0] == 1 and out["b"][1].runs == 3


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #


def test_figure_helpers_write_png_files(tmp_path):
    img = io.sample("chelsea")
    noisy = synth.gaussian_noise(img, 25.0, seed=0)

    g = figures.grid([("a", img), ("b", noisy), ("c", img)], tmp_path / "g.png", ncols=2)
    p = figures.pair(noisy, img, tmp_path / "p.png")
    h = figures.error_heatmap(noisy, img, tmp_path / "h.png")
    b = figures.metric_bars(["x", "y"], [1.0, 2.0], tmp_path / "b.png", ylabel="v")

    for path in (g, p, h, b):
        assert path.exists() and path.stat().st_size > 1000
