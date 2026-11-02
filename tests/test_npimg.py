import numpy as np
import pytest

from idphoto import color, npimg


@pytest.mark.parametrize("shape,r", [((7, 9), 2), ((5, 4), 10), ((12, 11, 2), 3), ((1, 6), 1)])
def test_box_sum_matches_brute_force(rng, shape, r):
    a = rng.random(shape).astype(np.float32)
    ref = np.zeros_like(a)
    for i in range(shape[0]):
        for j in range(shape[1]):
            ref[i, j] = a[max(i - r, 0) : i + r + 1, max(j - r, 0) : j + r + 1].sum(axis=(0, 1))
    np.testing.assert_allclose(npimg.box_sum(a, r), ref, atol=1e-4)


def test_box_mean_of_constant_is_constant():
    np.testing.assert_allclose(npimg.box_mean(np.full((20, 30), 0.3, np.float32), 7), 0.3, atol=1e-6)


def test_guided_filter_keeps_an_aligned_edge(rng):
    guide = np.zeros((40, 40), np.float32)
    guide[:, 20:] = 0.8
    guide += rng.normal(0, 0.005, guide.shape).astype(np.float32)
    mask = (guide > 0.4).astype(np.float32)
    out = npimg.guided_filter(guide, mask, 6, 1e-3)
    assert out[:, :17].max() < 0.05 and out[:, 23:].min() > 0.95


def test_color_guided_filter_separates_equal_brightness_colors():
    # Skin-like and wall-like colors with the same luminance: only color tells them apart.
    img = np.zeros((40, 40, 3), np.float32)
    img[:, :20] = (0.62, 0.42, 0.33)
    img[:, 20:] = (0.45, 0.47, 0.47)
    mask = np.zeros((40, 40), np.float32)
    mask[:, :20] = 1
    out = npimg.guided_filter_color(np.sqrt(img), mask, 6, 1e-3)
    assert out[:, :18].min() > 0.95 and out[:, 22:].max() < 0.05


def test_otsu_splits_two_clusters(rng):
    v = np.concatenate([rng.normal(0.2, 0.03, 500), rng.normal(0.8, 0.03, 500)])
    assert 0.35 < npimg.otsu_threshold(v) < 0.65


def test_opening_removes_specks_and_closing_fills_holes():
    m = np.zeros((40, 40), bool)
    m[10:30, 10:30] = True
    m[2, 2] = True
    m[20, 20] = False
    assert not npimg.opening(m, 1)[2, 2]
    assert npimg.closing(m, 1)[20, 20]


def test_resize_identity_and_ramp():
    img = np.random.default_rng(1).random((30, 40, 3)).astype(np.float32)
    np.testing.assert_allclose(npimg.resize(img, 30, 40, "cubic"), img, atol=1e-6)
    ramp = np.tile(np.arange(64, dtype=np.float32), (8, 1))
    down = npimg.resize(ramp, 8, 32, "linear")
    expected = np.broadcast_to(np.arange(32, dtype=np.float32)[2:-2] * 2 + 0.5, (8, 28))
    np.testing.assert_allclose(down[:, 2:-2], expected, atol=1e-5)


def test_resize_box_outside_frame_clamps_to_edge():
    out = npimg.resize(np.full((10, 10), 0.4, np.float32), 5, 5, "cubic", box=(-5, -5, 15, 15))
    np.testing.assert_allclose(out, 0.4, atol=1e-6)


def test_downscale_u8_averages_blocks():
    img = np.arange(16, dtype=np.uint8).reshape(4, 4, 1).repeat(3, axis=2)
    out = npimg.downscale_u8(img, 2)
    assert out.shape == (2, 2, 3)
    assert out[0, 0, 0] == (0 + 1 + 4 + 5 + 2) // 4  # rounded half up


def test_fill_polygon_square_area():
    m = npimg.fill_polygon((20, 20), np.array([[5, 5], [15, 5], [15, 15], [5, 15]]))
    assert m.sum() == 100


def test_rasterize_two_triangles_cover_a_quad_without_gaps():
    v = np.array([[2, 2], [18, 2], [18, 14], [2, 14]], np.float64)
    tris = np.array([[0, 1, 2], [0, 2, 3]])
    tri_id, bary = npimg.rasterize_triangles(v, np.zeros(4), tris, (16, 20))
    covered = tri_id >= 0
    assert covered[2:14, 2:18].all() and covered.sum() == 12 * 16
    np.testing.assert_allclose(bary[covered].sum(axis=1), 1.0, atol=1e-5)


def test_rasterize_keeps_the_nearest_triangle():
    v = np.array([[0, 0], [10, 0], [0, 10], [0, 0], [10, 0], [0, 10]], np.float64)
    depth = np.array([5, 5, 5, 1, 1, 1], np.float64)  # second triangle is closer
    tri_id, _ = npimg.rasterize_triangles(v, depth, np.array([[0, 1, 2], [3, 4, 5]]), (10, 10))
    assert set(np.unique(tri_id[tri_id >= 0])) == {1}


def test_srgb_round_trip():
    x = np.linspace(0, 1, 101, dtype=np.float32)
    np.testing.assert_allclose(color.linear_to_srgb(color.srgb_to_linear(x)), x, atol=1e-5)
    u8 = np.arange(256, dtype=np.uint8)
    assert np.abs(color.encode_u8(color.decode_u8(u8)).astype(int) - u8).max() <= 1
