"""mesh.py against known values: topology, box and grid mapping, normals, regions, R_LR."""

import numpy as np
import pytest

from idphoto import mesh, synthetic

from .conftest import FRONT


@pytest.fixture(scope="module")
def lm():
    return synthetic.landmarks_for(synthetic.Scene(**FRONT))


# --- canonical mesh -------------------------------------------------------------------


def test_canonical_mesh_shape_and_indices():
    vertices, tris = mesh.canonical_mesh()
    assert vertices.shape == (468, 3) and tris.shape == (898, 3)
    assert tris.min() == 0 and tris.max() <= 467
    assert np.all((tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2]))
    assert len(np.unique(tris)) > 450  # nearly every vertex belongs to a triangle


def test_canonical_pitch_ratio_is_the_value_the_readme_quotes():
    assert mesh.canonical_pitch_ratio() == pytest.approx(1.134, abs=0.005)


def test_landmark_groups_use_valid_indices():
    groups = [mesh.FACE_OVAL, mesh.EYE_R, mesh.EYE_L, mesh.BROW_R, mesh.BROW_L, mesh.LIPS,
              mesh.CHEEK_R, mesh.CHEEK_L, mesh.EAR_R, mesh.EAR_L]
    for g in groups:
        assert all(0 <= i < 468 for i in g) and len(set(g)) == len(g)
    assert mesh.IRIS_R >= 468 and mesh.IRIS_L >= 468


# --- face_box and to_grid -------------------------------------------------------------


def test_face_box_is_square_and_contains_the_face(lm):
    x0, y0, x1, y1 = mesh.face_box(lm)
    assert (x1 - x0) == pytest.approx(y1 - y0)
    p = lm[:468, :2]
    assert x0 < p[:, 0].min() and x1 > p[:, 0].max() and y0 < p[:, 1].min() and y1 > p[:, 1].max()
    extent = max(np.ptp(p[:, 0]), np.ptp(p[:, 1]))
    assert (x1 - x0) == pytest.approx(1.08 * extent, rel=1e-5)


def test_face_box_is_centered_on_the_face(lm):
    x0, y0, x1, y1 = mesh.face_box(lm)
    p = lm[:468, :2]
    assert 0.5 * (x0 + x1) == pytest.approx(0.5 * (p[:, 0].min() + p[:, 0].max()), abs=1e-3)
    assert 0.5 * (y0 + y1) == pytest.approx(0.5 * (p[:, 1].min() + p[:, 1].max()), abs=1e-3)


def test_face_box_ignores_iris_points(lm):
    far = lm.copy()
    far[468:, :2] = 5000.0
    assert mesh.face_box(far) == mesh.face_box(lm)


def test_face_box_follows_shift_and_scale(lm):
    moved = lm.copy()
    moved[:, :2] = lm[:, :2] * 2.0 + 10.0
    a, b = np.array(mesh.face_box(lm)), np.array(mesh.face_box(moved))
    np.testing.assert_allclose(b, a * 2.0 + 10.0, rtol=1e-5)


def test_to_grid_maps_the_box_onto_the_grid_and_leaves_the_input_alone(lm):
    box, size = mesh.face_box(lm), 128
    before = lm.copy()
    g = mesh.to_grid(lm, box, size)
    np.testing.assert_array_equal(lm, before)
    s = size / (box[2] - box[0])
    np.testing.assert_allclose(g[:, 0], (lm[:, 0] - box[0]) * s, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(g[:, 1], (lm[:, 1] - box[1]) * s, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(g[:, 2], lm[:, 2] * s, rtol=1e-5, atol=1e-4)
    assert g[:468, :2].min() > 0 and g[:468, :2].max() < size  # the padded box holds the face


# --- vertex and pixel normals ---------------------------------------------------------


def test_vertex_normals_are_unit_and_face_the_camera(lm):
    vn = mesh.vertex_normals(lm)
    assert vn.shape == (468, 3) and vn.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vn, axis=1), 1.0, atol=1e-5)
    assert vn[:, 2].mean() > 0.6
    assert vn[mesh.NOSE_TIP, 2] > 0.6
    assert vn[mesh.FACE_EDGE_R, 0] < 0 < vn[mesh.FACE_EDGE_L, 0]  # the subject's right edge points left in the image
    assert vn[mesh.FOREHEAD, 1] > 0 > vn[mesh.CHIN, 1]  # the forehead looks up, the chin down


def test_vertex_normals_ignore_translation_and_uniform_scale(lm):
    moved = lm.copy()
    moved[:, :2] = lm[:, :2] * 1.7 + np.array([40.0, -25.0], np.float32)
    moved[:, 2] = lm[:, 2] * 1.7
    np.testing.assert_allclose(mesh.vertex_normals(moved), mesh.vertex_normals(lm), atol=1e-4)


def test_vertex_normals_of_a_mirrored_face_are_mirrored(lm):
    mirrored = lm.copy()
    mirrored[:, 0] = 640.0 - lm[:, 0]
    expected = mesh.vertex_normals(lm) * np.array([-1.0, 1.0, 1.0], np.float32)
    np.testing.assert_allclose(mesh.vertex_normals(mirrored), expected, atol=1e-4)


def test_vertex_normals_turn_with_the_head():
    for yaw in (-15.0, 15.0):
        vn = mesh.vertex_normals(synthetic.landmarks_for(synthetic.Scene(yaw=yaw, **FRONT)))
        assert np.sign(vn[:, 0].mean()) == np.sign(yaw)
        assert abs(np.degrees(np.arctan2(vn[:, 0].mean(), vn[:, 2].mean())) - yaw) < 2.0


def test_mesh_normals_cover_the_face_with_unit_normals_and_nothing_else(lm):
    box = mesh.face_box(lm)
    normals, cov = mesh.mesh_normals(lm, box, 96)
    assert normals.shape == (96, 96, 3) and cov.shape == (96, 96)
    assert 0.4 < cov.mean() < 0.9
    np.testing.assert_allclose(np.linalg.norm(normals[cov], axis=1), 1.0, atol=1e-4)
    assert not normals[~cov].any()
    assert cov[48, 48]  # the grid center is on the face
    assert not cov[0, 0] and not cov[-1, -1]  # the corners of the padded box are not


def test_mesh_normals_accept_precomputed_vertex_normals(lm):
    box = mesh.face_box(lm)
    a, cov_a = mesh.mesh_normals(lm, box, 64)
    b, cov_b = mesh.mesh_normals(lm, box, 64, mesh.vertex_normals(lm))
    np.testing.assert_array_equal(cov_a, cov_b)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_mesh_normals_follow_the_left_right_slope_of_the_face(lm):
    normals, cov = mesh.mesh_normals(lm, mesh.face_box(lm), 96)
    left_half, right_half = cov.copy(), cov.copy()
    left_half[:, 48:] = False
    right_half[:, :48] = False
    assert normals[left_half][:, 0].mean() < -0.05 < 0.05 < normals[right_half][:, 0].mean()


# --- regions --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def regions(lm):
    box = mesh.face_box(lm)
    return mesh.region_masks(mesh.to_grid(lm, box, 128), (128, 128))


def _centroid_x(mask):
    return np.nonzero(mask)[1].mean()


def test_regions_have_sensible_sizes_and_nesting(regions):
    oval, eyes, features = regions["oval"], regions["eyes"], regions["features"]
    assert oval.sum() > 0.35 * oval.size
    assert 0 < eyes.sum() < 0.05 * oval.sum()
    assert not (eyes & ~features).any()  # features include the eyes, grown by one pixel
    assert (features & ~oval).sum() < 0.02 * oval.sum()  # eyes, brows and lips lie inside the face


def test_cheeks_are_on_the_expected_sides_and_clear_of_features(regions):
    r, l = regions["cheek_r"], regions["cheek_l"]
    assert _centroid_x(r) < _centroid_x(l)  # the subject's right cheek is on the image left
    assert 0.8 < r.sum() / l.sum() < 1.25
    assert not (r & l).any()
    assert not ((r | l) & regions["features"]).any()
    assert not ((r | l) & ~regions["oval"]).any()


def test_cheek_centroids_are_mirror_symmetric_for_a_frontal_face(regions):
    cx = 0.5 * (_centroid_x(regions["cheek_r"]) + _centroid_x(regions["cheek_l"]))
    assert cx == pytest.approx(_centroid_x(regions["oval"]), abs=2.0)


# --- R_LR and shadow share ------------------------------------------------------------


def _regions_for_balance():
    shape = (10, 10)
    left, right = np.zeros(shape, bool), np.zeros(shape, bool)
    left[2:6, 6:9] = True
    right[2:6, 1:4] = True
    return {"cheek_l": left, "cheek_r": right}


def test_cheek_balance_known_values():
    reg = _regions_for_balance()
    Y = np.full((10, 10), 0.4, np.float32)
    assert mesh.cheek_balance(Y, reg) == pytest.approx(1.0, abs=1e-5)
    Y[reg["cheek_l"]] = 0.8  # the subject's left cheek twice as bright
    assert mesh.cheek_balance(Y, reg) == pytest.approx(2.0, abs=1e-4)
    Y[reg["cheek_l"]] = 0.2
    assert mesh.cheek_balance(Y, reg) == pytest.approx(0.5, abs=1e-4)


def test_cheek_balance_uses_only_valid_pixels():
    reg = _regions_for_balance()
    Y = np.full((10, 10), 0.4, np.float32)
    Y[reg["cheek_l"]] = 0.8
    valid = np.ones((10, 10), bool)
    valid[2:4, 6:9] = False  # drop the first two rows of the left cheek
    Y[2:4, 6:9] = 100.0  # garbage there must not matter
    assert mesh.cheek_balance(Y, reg, valid) == pytest.approx(2.0, abs=1e-4)


def test_cheek_balance_is_nan_without_enough_pixels():
    reg = _regions_for_balance()
    Y = np.full((10, 10), 0.4, np.float32)
    assert np.isnan(mesh.cheek_balance(Y, reg, np.zeros((10, 10), bool)))
    few = np.zeros((10, 10), bool)
    few[2, 6] = few[2, 1] = True
    assert np.isnan(mesh.cheek_balance(Y, reg, few))


def test_shadow_share_known_values():
    Y = np.array([[1.0, 1.0, 1.0, 1.0], [0.1, 0.1, 0.1, 0.1]], np.float32)
    skin = np.ones((2, 4), bool)
    # mean 0.55, shadow below 0.5 x mean = 0.275: the four dark pixels
    assert mesh.shadow_share(Y, skin, 0.5) == pytest.approx(0.5)
    assert mesh.shadow_share(Y, skin, 0.1) == pytest.approx(0.0)
    skin[1] = False  # only the bright row counts
    assert mesh.shadow_share(Y, skin, 0.5) == pytest.approx(0.0)


def test_shadow_share_of_an_even_face_is_zero_and_empty_skin_is_nan():
    Y = np.full((8, 8), 0.5, np.float32)
    assert mesh.shadow_share(Y, np.ones((8, 8), bool), 0.7) == 0.0
    assert np.isnan(mesh.shadow_share(Y, np.zeros((8, 8), bool), 0.7))


# --- usable ---------------------------------------------------------------------------


def test_usable_accepts_a_real_detection_and_rejects_broken_ones(lm):
    assert mesh.usable(lm)
    assert not mesh.usable(None)
    assert not mesh.usable(lm[:100])  # too few points
    assert not mesh.usable(np.full_like(lm, np.nan))
    assert not mesh.usable(np.zeros_like(lm))  # no size
    assert not mesh.usable(np.full_like(lm, 123.0))
    one_nan = lm.copy()
    one_nan[200, 0] = np.nan
    assert not mesh.usable(one_nan)
    inf = lm.copy()
    inf[3, 1] = np.inf
    assert not mesh.usable(inf)


def test_usable_looks_only_at_the_face_points_not_the_iris(lm):
    iris_broken = lm.copy()
    iris_broken[468:] = np.nan
    assert mesh.usable(iris_broken)


def test_usable_has_a_minimum_size(lm):
    tiny = lm.copy()
    tiny[:, :2] = (lm[:, :2] - lm[:, :2].mean(axis=0)) * 0.01 + 100.0  # a few pixels wide
    assert not mesh.usable(tiny)
    assert mesh.usable(tiny, min_size=1.0)
