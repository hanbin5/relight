import numpy as np

from idphoto import mesh, synthetic
from idphoto.relight import LEFT_RIGHT_TERMS, Relighter, fit_light, sh_basis, target_light

from .conftest import FRONT, make_ctx


def test_target_has_no_left_right_terms_and_is_brighter_in_front():
    coef = target_light()
    assert np.all(coef[list(LEFT_RIGHT_TERMS)] == 0)
    front, side = sh_basis(np.array([[0, 0, 1.0], [1.0, 0, 0]])) @ coef
    assert front > side


def test_fit_recovers_a_known_light(rng):
    n = rng.normal(size=(4000, 3))
    n[:, 2] = np.abs(n[:, 2])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    truth = np.array([0.8, 0.1, 0.4, -0.3, 0.02, 0.0, 0.05, -0.04, 0.01], np.float32)
    Y = (sh_basis(n) @ truth).astype(np.float32)
    coef = fit_light(Y[:, None], n[:, None, :].astype(np.float32), np.ones((len(n), 1), bool))
    np.testing.assert_allclose(sh_basis(n) @ coef, Y, atol=0.01)


def test_mesh_normals_face_the_camera(side_lit):
    _, lm, _ = side_lit
    normals, cov = mesh.mesh_normals(lm, mesh.face_box(lm), 128)
    assert cov.mean() > 0.4
    assert normals[cov][:, 2].mean() > 0.6
    # nose tip region points forward; the face edges point sideways, left and right
    vn = mesh.vertex_normals(lm)
    assert vn[mesh.FACE_EDGE_R, 0] < -0.5 and vn[mesh.FACE_EDGE_L, 0] > 0.5


def test_relighting_evens_a_side_light(side_lit):
    rgb, lm, _ = side_lit
    ctx = make_ctx(rgb, lm)
    relighter = Relighter()
    for _ in range(5):
        relighter.process(ctx)
    s = ctx.stats
    assert s["r_lr_before"] < 0.6
    assert 0.90 <= s["r_lr_after"] <= 1.10  # the plan's balance band
    assert s["shadow_after"] < 0.05
    assert ctx.relit_frame.shape == rgb.shape and ctx.light_coeffs.shape == (9,)


def test_relighting_barely_changes_an_even_light(front_lit):
    rgb, lm, _ = front_lit
    ctx = make_ctx(rgb, lm)
    Relighter().process(ctx)
    r = ctx.ratio_map[ctx.face_mask]
    assert abs(float(np.median(r)) - 1.0) < 0.1


def test_strength_zero_leaves_the_frame_unchanged(side_lit):
    rgb, lm, _ = side_lit
    ctx = make_ctx(rgb, lm)
    Relighter().process(ctx, strength=0.0)
    np.testing.assert_allclose(ctx.relit_frame, ctx.linear_rgb, atol=1e-6)


def test_relighting_keeps_skin_texture(side_lit):
    """Only a smooth ratio changes: fine detail (high-pass) survives up to the ratio's scale."""
    rgb, lm, _ = side_lit
    ctx = make_ctx(rgb, lm)
    Relighter().process(ctx)
    m = ctx.face_mask
    before = ctx.Y - _blur(ctx.Y)
    after_y = (ctx.relit_frame * [0.2126, 0.7152, 0.0722]).sum(-1)
    after = after_y / np.maximum(ctx.ratio_map, 1e-6) - _blur(after_y / np.maximum(ctx.ratio_map, 1e-6))
    assert np.corrcoef(before[m], after[m])[0, 1] > 0.95


def test_lost_face_keeps_then_drops_the_fit(side_lit):
    rgb, lm, _ = side_lit
    relighter = Relighter()
    relighter.process(make_ctx(rgb, lm))
    assert relighter.light is not None
    for _ in range(20):
        relighter.process(make_ctx(rgb, None))
    assert relighter.light is None


def _blur(a):
    from idphoto import npimg

    return npimg.box_mean(a.astype(np.float32), 2)


def test_canonical_scene_is_level():
    lm = synthetic.landmarks_for(synthetic.Scene(**FRONT))
    from idphoto.geometry import head_pose

    pose = head_pose(lm)
    assert abs(pose["roll_deg"]) < 1e-3 and abs(pose["yaw_ratio"]) < 1e-3
    assert abs(pose["pitch_ratio"] - 1.0) < 1e-3
