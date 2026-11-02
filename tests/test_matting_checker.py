import numpy as np
import pytest

from idphoto import synthetic
from idphoto.checker import AutoShutter

from .conftest import FRONT, SIDE, run_pipeline


@pytest.mark.parametrize("scene", [
    synthetic.Scene(**FRONT),
    synthetic.Scene(**SIDE),
    synthetic.Scene(wall="#9fb6cf", top="#222222", **FRONT),
    synthetic.Scene(wall="#e8dcc8", top="#9c2f2f", **FRONT),
], ids=["light-wall", "side-light", "blue-wall", "beige-wall"])
def test_mask_matches_the_true_person(scene):
    ctx, person, _, _ = run_pipeline(scene, frames=3)
    a = ctx.alpha > 0.5
    assert (a & person).sum() / (a | person).sum() >= 0.95  # the plan's IoU target
    assert ctx.rule_verdicts[7].passed


def test_background_is_replaced_and_face_is_kept():
    ctx, person, _, _ = run_pipeline(synthetic.Scene(**FRONT), frames=2)
    out = ctx.output_frame
    assert np.allclose(out[5, 5], 1.0, atol=1e-3)  # wall corner -> white
    assert np.allclose(out[ctx.face_mask], ctx.relit_frame[ctx.face_mask])


def test_original_wall_option_keeps_the_relit_frame():
    from idphoto.pipeline import Settings

    ctx, _, _, _ = run_pipeline(synthetic.Scene(**FRONT), frames=2, settings=Settings(background="wall"))
    assert ctx.output_frame is ctx.relit_frame


@pytest.mark.parametrize("scene,rule,message", [
    (synthetic.Scene(light=(-0.8, 0.2, 0.55), ambient=0.12), 1, "turn_to_light"),
    (synthetic.Scene(glare=True, **FRONT), 2, "glare_glasses"),
    (synthetic.Scene(blur=3, **FRONT), 3, "hold_still"),
    (synthetic.Scene(roll=6.0, **FRONT), 4, "straighten_head"),
    (synthetic.Scene(yaw=12.0, **FRONT), 4, "face_camera"),
    (synthetic.Scene(pitch=15.0, **FRONT), 4, "face_camera"),
    (synthetic.Scene(eyes_open=False, **FRONT), 5, "open_eyes"),
    (synthetic.Scene(mouth_open=0.12, **FRONT), 6, "close_mouth"),
    (synthetic.Scene(top="#d8d6d2", **SIDE), 7, "top_vs_wall"),
], ids=["side-shadow", "glare", "blur", "tilt", "turn", "pitch", "closed-eyes", "open-mouth", "top-like-wall"])
def test_each_staged_fault_fails_its_rule(scene, rule, message):
    ctx, _, _, _ = run_pipeline(scene)
    v = ctx.rule_verdicts[rule]
    assert not v.passed and v.message == message


def test_good_frame_passes_every_rule_and_fires_the_shutter():
    ctx, _, pipeline, fired = run_pipeline(synthetic.Scene(**SIDE), frames=6)
    assert all(v.passed for v in ctx.rule_verdicts.values()), {n: v.value for n, v in ctx.rule_verdicts.items()}
    assert fired and not pipeline.shutter.armed


def test_sharpness_bound_is_calibrated_per_camera():
    from idphoto import config
    from idphoto.checker import Checker

    c = Checker()
    assert c.sharpness_bound() == config.SHARPNESS_MIN  # until the camera has been seen
    c.sharpness_seen.extend([50.0] * 40)  # a soft webcam
    assert c.sharpness_bound() == pytest.approx(0.7 * 50.0)
    c.sharpness_seen.extend([400.0] * 90)  # a sharp camera is never held above B.4's 120
    assert c.sharpness_bound() == config.SHARPNESS_MIN
    c.sharpness_seen.extend([5.0] * 90)  # nor a hopeless one below the floor
    assert c.sharpness_bound() == config.SHARPNESS_FLOOR


def test_auto_shutter_needs_five_passing_frames_in_a_row():
    s = AutoShutter(frames=5)
    assert not any(s.update(ok) for ok in [True, True, True, True, False, True, True, True, True])
    assert s.update(True)  # the 5th in a row
    assert not s.update(True)  # fires once until rearmed
    s.rearm()
    assert not s.update(True)


def test_auto_shutter_off_counts_but_never_fires():
    s = AutoShutter(frames=2)
    assert not any(s.update(True, enabled=False) for _ in range(5))
    assert s.count == 5 and s.armed


def test_no_face_fails_every_rule():
    from idphoto.pipeline import Pipeline, Settings

    blank = np.full((360, 640, 3), 200, np.uint8)
    ctx, fired = Pipeline().process(blank, Settings(), landmarks=None)
    assert not fired
    assert all(not v.passed and v.message == "no_face" for v in ctx.rule_verdicts.values())
    assert ctx.output_frame.shape == (360, 640, 3)
