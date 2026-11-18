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
