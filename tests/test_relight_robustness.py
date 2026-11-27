"""relight.py: its mathematics against known values, and its behaviour on bad input.

The live loop must never crash on a strange frame, so a face that is half outside the
frame, far away, missing or broken has to give back the frame unchanged (or a sane ratio).
"""

import numpy as np
import pytest

from idphoto import config, synthetic
from idphoto.relight import LEFT_RIGHT_TERMS, Relighter, face_yaw, fit_light, sh_basis, target_light

from .conftest import FRONT, SIDE, make_ctx

# --- the SH basis and the target light ------------------------------------------------


def _sphere(n=20000, seed=0):
    v = np.random.default_rng(seed).normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_sh_basis_is_orthonormal_on_the_sphere():
    n = _sphere()
    H = sh_basis(n)
    gram = 4 * np.pi * (H.T @ H) / len(n)  # integral of H_i H_j over the sphere
    np.testing.assert_allclose(gram, np.eye(9), atol=0.08)


def test_sh_basis_known_values():
    H = sh_basis(np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    c0, c1, c2, c3, c4 = 0.282095, 0.488603, 1.092548, 0.315392, 0.546274
    np.testing.assert_allclose(H[0], [c0, 0, c1, 0, 0, 0, 2 * c3, 0, 0], atol=1e-6)
    np.testing.assert_allclose(H[1], [c0, 0, 0, c1, 0, 0, -c3, 0, c4], atol=1e-6)
    np.testing.assert_allclose(H[2], [c0, c1, 0, 0, 0, 0, -c3, 0, -c4], atol=1e-6)


def test_target_light_is_mirror_antisymmetric_in_yaw():
    left, right = target_light(yaw_deg=-15.0), target_light(yaw_deg=15.0)
    lr = list(LEFT_RIGHT_TERMS)
    other = [i for i in range(9) if i not in lr]
    # the hemisphere is sampled with 4000 points, so the symmetry holds to about 0.5%
    np.testing.assert_allclose(left[lr], -right[lr], rtol=5e-3, atol=2e-3)
    np.testing.assert_allclose(left[other], right[other], rtol=5e-3, atol=2e-3)


def test_target_light_follows_the_head_turn_and_the_elevation():
    n = np.array([[0.5, 0.0, 0.87], [-0.5, 0.0, 0.87]])  # a cheek on each side
    right_cheek, left_cheek = sh_basis(n) @ target_light(yaw_deg=25.0)
    assert right_cheek > left_cheek  # turned right: the right-side surface faces the light
    level, raised = target_light(elevation_deg=0.0)[1], target_light(elevation_deg=40.0)[1]
    assert raised > level  # more elevation: more light on up-facing surfaces


def test_target_light_is_positive_over_the_visible_face():
    n = _sphere(4000, seed=1)
    n[:, 2] = np.abs(n[:, 2])
    for yaw in (-20.0, 0.0, 20.0):
        assert (sh_basis(n) @ target_light(yaw_deg=yaw)).min() > -0.05


def test_face_yaw_matches_the_head_turn():
    from idphoto import mesh

    for yaw in (-12.0, 0.0, 9.0):
        lm = synthetic.landmarks_for(synthetic.Scene(yaw=yaw, **FRONT))
        assert face_yaw(mesh.vertex_normals(lm)) == pytest.approx(yaw, abs=1.5)


# --- fit_light ------------------------------------------------------------------------


def _hemisphere_normals(n=3000, seed=0):
    v = _sphere(n, seed)
    v[:, 2] = np.abs(v[:, 2])
    return v.astype(np.float32)


def test_fit_rejects_dark_outliers_like_hair_and_beard():
    n = _hemisphere_normals()
    truth = np.array([0.8, 0.1, 0.4, -0.3, 0.02, 0.0, 0.05, -0.04, 0.01], np.float32)
    Y = (sh_basis(n) @ truth).astype(np.float32)
    dirty = Y.copy()
    dirty[:300] *= 0.1  # a tenth of the "skin" is really dark hair
    skin = np.ones((len(n), 1), bool)
    coef = fit_light(dirty[:, None], n[:, None, :], skin)
    err = np.abs(sh_basis(n[300:]) @ coef - Y[300:]).mean()
    plain = fit_light(dirty[:, None], n[:, None, :], skin, iterations=0)
    plain_err = np.abs(sh_basis(n[300:]) @ plain - Y[300:]).mean()
    assert err < 0.02 and err < 0.5 * plain_err


def test_fit_survives_normals_that_barely_vary():
    n = np.tile(np.array([0.0, 0.0, 1.0], np.float32), (500, 1)) + np.random.default_rng(0).normal(0, 1e-4, (500, 3))
    n = (n / np.linalg.norm(n, axis=1, keepdims=True)).astype(np.float32)
    Y = np.full(500, 0.4, np.float32)
    coef = fit_light(Y[:, None], n[:, None, :], np.ones((500, 1), bool))
    assert np.isfinite(coef).all()
    # whatever the split between terms, the light it describes at the face is 0.4
    assert float((sh_basis(n) @ coef).mean()) == pytest.approx(0.4, abs=0.02)


def test_fit_with_exactly_constant_normals_does_not_crash():
    n = np.tile(np.array([0.0, 0.0, 1.0], np.float32), (300, 1))
    coef = fit_light(np.full(300, 0.4, np.float32)[:, None], n[:, None, :], np.ones((300, 1), bool))
    assert np.isfinite(coef).all()
    assert float((sh_basis(n[:1]) @ coef)[0]) == pytest.approx(0.4, abs=0.02)


# --- Relighter: properties of the result ----------------------------------------------


def _run(rgb, lm, frames=5, **kw):
    ctx, relighter = make_ctx(rgb, lm), Relighter()
    for _ in range(frames):
        relighter.process(ctx, **kw)
    return ctx, relighter


def test_relit_frame_is_a_new_array_and_the_input_frame_is_untouched(side_lit):
    rgb, lm, _ = side_lit
    ctx = make_ctx(rgb, lm)
    before = ctx.linear_rgb.copy()
    Relighter().process(ctx)
    assert ctx.relit_frame is not ctx.linear_rgb
    np.testing.assert_array_equal(ctx.linear_rgb, before)
    gain = ctx.ratio_map[..., None] * (1.0 if ctx.color_gain is None else ctx.color_gain)
    np.testing.assert_allclose(ctx.relit_frame, np.minimum(before * gain, 1.0), rtol=1e-6)


def test_ratio_stays_in_the_clip_range_and_the_frame_stays_valid(side_lit):
    rgb, lm, _ = side_lit
    ctx, _ = _run(rgb, lm)
    assert np.isfinite(ctx.ratio_map).all() and np.isfinite(ctx.relit_frame).all()
    lo, hi = config.RATIO_CLIP
    assert ctx.ratio_map.min() >= lo - 1e-4 and ctx.ratio_map.max() <= hi + 1e-4
    assert ctx.relit_frame.min() >= 0.0 and ctx.relit_frame.max() <= 1.0
    assert ctx.relit_frame.shape == ctx.linear_rgb.shape and ctx.relit_frame.dtype == np.float32


def test_relighting_does_not_depend_on_the_exposure(side_lit):
    """Halving the linear exposure must give the same ratio: the target keeps the mean face
    brightness, so a camera that gets darker or brighter does not change the correction."""
    rgb, lm, _ = side_lit
    ctx, _ = _run(rgb, lm)
    from idphoto import color

    dark = make_ctx(rgb, lm)
    dark.linear_rgb = (dark.linear_rgb * 0.5).astype(np.float32)
    dark.Y = color.luminance(dark.linear_rgb)
    relighter = Relighter()
    for _ in range(5):
        relighter.process(dark)
    m = ctx.face_mask
    assert np.abs(dark.ratio_map[m] - ctx.ratio_map[m]).mean() < 0.03
    assert dark.stats["r_lr_after"] == pytest.approx(ctx.stats["r_lr_after"], abs=0.03)


def test_strength_scales_the_correction_linearly(side_lit):
    rgb, lm, _ = side_lit
    full, _ = _run(rgb, lm, strength=1.0)
    half, _ = _run(rgb, lm, strength=0.5)
    none, _ = _run(rgb, lm, strength=0.0)
    m = full.face_mask
    assert np.abs(none.ratio_map[m] - 1.0).max() < 1e-4
    dev_full, dev_half = np.abs(full.ratio_map[m] - 1.0).mean(), np.abs(half.ratio_map[m] - 1.0).mean()
    assert dev_full > 0.1
    assert dev_half == pytest.approx(0.5 * dev_full, rel=0.1)


def test_the_shutter_frame_reuses_the_live_light_without_changing_it(side_lit):
    rgb, lm, _ = side_lit
    live, relighter = _run(rgb, lm)
    light, yaw = relighter.light.copy(), float(np.mean(relighter.yaws))
    history = len(relighter.history)
    shutter = make_ctx(rgb, lm)
    relighter.process(shutter, light=light, target_yaw=yaw, update=False)
    assert len(relighter.history) == history
    np.testing.assert_allclose(relighter.light, light)
    np.testing.assert_allclose(shutter.light_coeffs, light)
    m = live.face_mask
    assert np.abs(shutter.ratio_map[m] - live.ratio_map[m]).mean() < 0.02


def test_averaging_steadies_the_fitted_light():
    """Sensor noise moves the per-frame fit; the 5-frame average moves less."""
    coefs, averaged = [], []
    relighter = Relighter()
    for seed in range(12):
        rgb, lm, _ = synthetic.render(synthetic.Scene(noise=6.0, seed=seed, **SIDE))
        relighter.process(make_ctx(rgb, lm))
        coefs.append(relighter.history[-1])
        averaged.append(relighter.light)
    raw, smooth = np.std(coefs[4:], axis=0), np.std(averaged[4:], axis=0)
    assert smooth.sum() < raw.sum()


def test_smoothing_length_can_change_live_and_keeps_the_history(side_lit):
    rgb, lm, _ = side_lit
    _, relighter = _run(rgb, lm, frames=5)
    relighter.set_smoothing(10)
    assert relighter.history.maxlen == 10 and len(relighter.history) == 5
    relighter.set_smoothing(2)
    assert relighter.history.maxlen == 2 and len(relighter.history) == 2


# --- Relighter: bad input -------------------------------------------------------------


def _assert_untouched(ctx):
    np.testing.assert_array_equal(ctx.relit_frame, ctx.linear_rgb)
    assert (ctx.ratio_map == 1.0).all() and not ctx.face_mask.any()


def test_no_face_returns_the_frame_unchanged(side_lit):
    rgb, _, _ = side_lit
    ctx = make_ctx(rgb, None)
    Relighter().process(ctx)
    _assert_untouched(ctx)


def test_a_face_far_outside_the_frame_returns_the_frame_unchanged(side_lit):
    rgb, lm, _ = side_lit
    moved = lm.copy()
    moved[:, 0] += 5000.0
    ctx = make_ctx(rgb, moved)
    Relighter().process(ctx)
    _assert_untouched(ctx)


@pytest.mark.parametrize("dx, dy", [(-200, 0), (200, 0), (0, -150), (0, 150), (-200, -150)])
def test_a_face_cut_by_the_frame_edge_does_not_crash(side_lit, dx, dy):
    rgb, lm, _ = side_lit
    moved = lm.copy()
    moved[:, 0] += dx
    moved[:, 1] += dy
    ctx = make_ctx(rgb, moved)
    Relighter().process(ctx)
    assert ctx.relit_frame.shape == ctx.linear_rgb.shape
    assert np.isfinite(ctx.relit_frame).all() and np.isfinite(ctx.ratio_map).all()
    assert ctx.ratio_map.min() >= 0.5 - 1e-4 and ctx.ratio_map.max() <= 2.0 + 1e-4


def test_a_tiny_distant_face_does_not_crash():
    rgb, lm, _ = synthetic.render(synthetic.Scene(face_height=0.05, **SIDE))
    ctx = make_ctx(rgb, lm)
    Relighter().process(ctx)
    assert np.isfinite(ctx.relit_frame).all()


def test_a_huge_face_filling_the_frame_does_not_crash():
    rgb, lm, _ = synthetic.render(synthetic.Scene(face_height=1.1, center=(0.5, 0.5), **SIDE))
    ctx = make_ctx(rgb, lm)
    Relighter().process(ctx)
    assert np.isfinite(ctx.relit_frame).all() and np.isfinite(ctx.ratio_map).all()


def test_broken_landmarks_are_treated_like_a_lost_face(side_lit):
    rgb, lm, _ = side_lit
    for broken in (np.full_like(lm, np.nan), np.zeros_like(lm), np.full_like(lm, 123.0)):
        ctx = make_ctx(rgb, broken)
        Relighter().process(ctx)
        assert np.isfinite(ctx.relit_frame).all()
        assert ctx.relit_frame.shape == ctx.linear_rgb.shape


def test_a_few_nan_landmarks_do_not_poison_the_light(side_lit):
    rgb, lm, _ = side_lit
    ctx, relighter = _run(rgb, lm, frames=3)
    bad = lm.copy()
    bad[::50, :] = np.nan
    ctx2 = make_ctx(rgb, bad)
    relighter.process(ctx2)
    assert np.isfinite(relighter.light).all()
    assert np.isfinite(ctx2.relit_frame).all()


def test_a_dark_frame_and_a_blown_out_frame_do_not_crash(side_lit):
    rgb, lm, _ = side_lit
    for value in (0, 255):
        ctx = make_ctx(np.full_like(rgb, value), lm)
        Relighter().process(ctx)
        assert np.isfinite(ctx.relit_frame).all() and np.isfinite(ctx.ratio_map).all()
        assert ctx.relit_frame.min() >= 0.0 and ctx.relit_frame.max() <= 1.0


def test_a_face_cut_by_the_frame_edge_is_fitted_on_real_pixels_only():
    """Beyond the frame, resizing repeats the edge pixels. Fitting those as skin made the
    light 75% too bright for a face a quarter outside the frame; they are left out now."""
    def fit(cx):
        rgb, lm, _ = synthetic.render(synthetic.Scene(center=(cx, 0.42), **SIDE))
        ctx, relighter = _run(rgb, lm, frames=3)
        return relighter.light

    whole, cut = fit(0.5), fit(0.94)  # at 0.94 about a quarter of the face is outside
    assert abs(cut[0] - whole[0]) / whole[0] < 0.40
    assert np.abs(cut[1:4] - whole[1:4]).max() < 0.25
