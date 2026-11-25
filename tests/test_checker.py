"""Rule 1's shadow share on its own (the side-shadow test in test_matting_checker.py fails
rule 1 on R_LR, so it doesn't show that the shadow share works)."""

from idphoto import config, mesh, synthetic
from idphoto.checker import Checker

from .conftest import FRONT, make_ctx


def even_light_unrelit(rgb, lm):
    ctx = make_ctx(rgb, lm)
    ctx.relit_frame = ctx.linear_rgb  # check the frame as it is, without relighting
    return Checker().even_light(ctx)


def test_side_shadow_counts_as_shadow():
    rgb, lm, _ = synthetic.render(synthetic.Scene(light=(-0.55, 0.2, 0.8), ambient=0.25))
    v = even_light_unrelit(rgb, lm)
    assert v.detail["shadow_share"] > config.SHADOW_MAX
    assert not v.passed


def test_hair_over_the_forehead_is_not_shadow():
    rgb, lm, _ = synthetic.render(synthetic.Scene(**FRONT))
    rgb = rgb.copy()
    # paint dark "bangs" from the top of the face down to the top of the eyes
    eyes = lm[list(mesh.EYE_R) + list(mesh.EYE_L)]
    top, eye_top = int(lm[:468, 1].min()), int(eyes[:, 1].min())
    left, right = int(lm[:468, 0].min()), int(lm[:468, 0].max())
    rgb[top:eye_top, left:right] = 20
    v = even_light_unrelit(rgb, lm)
    assert v.detail["shadow_share"] < config.SHADOW_MAX


def head_pose(pitch_ratio, pitch_deg=None):
    import numpy as np

    from idphoto.context import FrameContext

    ctx = FrameContext(raw_frame=np.zeros((4, 4, 3), np.uint8))
    ctx.geometry = {"roll_deg": 0.0, "yaw_ratio": 0.0, "pitch_ratio": pitch_ratio, "pitch_deg": pitch_deg}
    return Checker().head_pose(ctx)


def test_pitch_uses_degrees_when_the_detector_gives_them():
    # a real face 10 degrees down reads ~1.4 as a ratio; the angle should decide
    assert head_pose(1.4, pitch_deg=10.0).passed
    assert not head_pose(1.0, pitch_deg=15.5).passed  # leaning in, like shot5
    assert not head_pose(1.0, pitch_deg=-8.0).passed  # chin up


def test_pitch_falls_back_to_the_ratio_without_degrees():
    assert head_pose(1.0).passed
    assert not head_pose(1.4).passed
