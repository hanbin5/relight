from types import SimpleNamespace

import numpy as np
import pytest

from idphoto.context import FrameContext
from idphoto.pipeline import Pipeline, Settings
from idphoto.presentation_modnet import ModnetPrediction, PresentationMODNet
from tests.test_center_subject import face


def scene():
    rgb = np.full((100, 160, 3), 140, np.uint8)
    alpha = np.zeros((100, 160), np.float32)
    alpha[20:80, 10:60] = .9
    alpha[20:80, 100:150] = .8
    faces = (face(35, 40).landmarks, face(125, 40).landmarks)
    return rgb, alpha, faces


def test_preview_uses_all_people_for_wall_but_selected_person_for_output(monkeypatch):
    from idphoto import presentation_modnet

    rgb, alpha, faces = scene()
    ctx = FrameContext(raw_frame=rgb, linear_rgb=rgb.astype(np.float32) / 255,
                       relit_frame=rgb.astype(np.float32) / 255,
                       landmarks=faces[1], subject_faces=faces, subject_index=1)
    pipeline = Pipeline()
    pipeline.modnet = SimpleNamespace(apply_prediction=PresentationMODNet.apply_prediction)
    observed = []

    def walls(linear, wall_alpha):
        observed.append(wall_alpha.copy())
        return np.ones(3, np.float32), np.ones_like(linear)

    monkeypatch.setattr(presentation_modnet, "estimate_walls_live", walls)
    pipeline.apply_modnet_prediction(ctx, ModnetPrediction(alpha, {}), np.ones(3, np.float32))
    assert ctx.subject_status == "ready"
    np.testing.assert_array_equal(observed[0], alpha)
    assert ctx.alpha[40, 35] == 0
    assert ctx.alpha[40, 125] > .7
    assert ctx.all_person_alpha[40, 35] > .8


def test_ambiguous_matte_has_placeholder_and_cannot_fire():
    rgb, alpha, faces = scene()
    alpha[:, 60:100] = .8
    background = np.array([.2, .4, .6], np.float32)
    ctx = FrameContext(raw_frame=rgb, linear_rgb=rgb.astype(np.float32) / 255,
                       relit_frame=rgb.astype(np.float32) / 255, landmarks=faces[1],
                       subject_faces=faces, subject_index=1)
    pipeline = Pipeline()
    pipeline.apply_modnet_prediction(ctx, ModnetPrediction(alpha, {}), background)
    result, fired = pipeline.finalize_deferred_model(ctx, Settings(modnet_demo=True))
    assert result.subject_status == "ambiguous"
    assert result.geometry is None and result.rule_verdicts is None and not fired
    np.testing.assert_allclose(result.output_frame, np.broadcast_to(background, rgb.shape))


@pytest.mark.parametrize("capture_model", ["modnet", "birefnet-portrait"])
@pytest.mark.parametrize("ambiguous", [False, True])
def test_full_capture_rechecks_ownership_from_same_frozen_frame(
    monkeypatch, capture_model, ambiguous
):
    from idphoto import geometry, pipeline as pipeline_module, presentation_modnet

    rgb, alpha, faces = scene()
    full_rgb = np.repeat(np.repeat(rgb, 2, axis=0), 2, axis=1)
    full_alpha = np.repeat(np.repeat(alpha, 2, axis=0), 2, axis=1)
    if ambiguous:
        full_alpha[:, 120:200] = .8
    calls = []

    class Model:
        def predict_alpha(self, raw_rgb, *, capture):
            calls.append((raw_rgb.copy(), capture))
            return ModnetPrediction(full_alpha, {})

        apply_prediction = staticmethod(PresentationMODNet.apply_prediction)

    def relight(_self, ctx, *_args, **_kwargs):
        ctx.relit_frame = ctx.linear_rgb
        ctx.ratio_map = np.ones(ctx.Y.shape, np.float32)

    monkeypatch.setattr(pipeline_module.Relighter, "process", relight)
    monkeypatch.setattr(geometry, "measure", lambda *_: {
        "crown": [125., 20.], "chin": [125., 60.], "head_length": 40.,
        "crown_source": "test", "roll_deg": 0., "yaw_ratio": 0., "pitch_ratio": 0.,
    })
    monkeypatch.setattr(geometry, "crop_box", lambda *_: (0, 0, 20, 20))
    monkeypatch.setattr(geometry, "crop_fits", lambda *_: True)
    monkeypatch.setattr(geometry, "render_photo", lambda *_: np.zeros((20, 20, 3), np.uint8))
    monkeypatch.setattr(geometry, "print_sheet", lambda x, *_: x)
    verdict = SimpleNamespace(rule=7, key="clean_cutout", passed=True, value=0,
                              bound=0, detail={})
    monkeypatch.setattr(pipeline_module.Checker, "evaluate", lambda *_: {7: verdict})
    observed = []
    original = presentation_modnet.estimate_wall_color

    def wall_color(linear, wall_alpha):
        observed.append(wall_alpha.copy())
        return original(linear, wall_alpha)

    monkeypatch.setattr(presentation_modnet, "estimate_wall_color", wall_color)
    live = FrameContext(raw_frame=rgb, landmarks=faces[1], light_coeffs=np.zeros(9),
                        subject_faces=faces, subject_index=1, subject_status="ready")
    pipeline = Pipeline(modnet_factory=Model, birefnet_factory=Model)
    result = pipeline.render_final(full_rgb, live, Settings(
        modnet_demo=True, capture_matte=capture_model))
    np.testing.assert_array_equal(calls[0][0], full_rgb)
    assert calls[0][1] is True
    if ambiguous:
        assert result == {"subject_blocked": "ambiguous"}
    else:
        ctx = result["ctx"]
        assert ctx.subject_status == "ready"
        assert ctx.alpha[80, 70] == 0 and ctx.alpha[80, 250] > .7
        np.testing.assert_array_equal(ctx.all_person_alpha, full_alpha)
        assert observed
        for wall_alpha in observed:
            np.testing.assert_array_equal(wall_alpha, full_alpha)
