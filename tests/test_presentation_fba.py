import numpy as np
import pytest

from idphoto import color
from idphoto.pipeline import Pipeline, Settings
from tests.test_matting_checker import clothing_scene
from tools.presentation_fba import PresentationFBA


def test_presentation_fba_is_lazy_and_requires_opt_in():
    rgb, lm, _, _ = clothing_scene("jacket")
    calls = []

    def factory():
        calls.append("load")
        return lambda ctx, bg: calls.append((ctx.raw_frame.shape, bg))

    pipeline = Pipeline(fba_factory=factory)
    settings = Settings(auto_shutter=False)
    pipeline.process(rgb, settings, landmarks=lm)
    assert calls == []
    settings.fba_demo = True
    pipeline.process(rgb, settings, landmarks=lm)
    assert calls[0] == "load" and calls[1][0] == rgb.shape
    pipeline.process(rgb, settings, landmarks=lm)
    assert calls.count("load") == 1
    with pytest.raises(RuntimeError, match="opt-in"):
        Pipeline().process(rgb, settings, landmarks=lm)


def test_deferred_fba_keeps_clothing_preview_and_capture_fba_opt_in():
    rgb, lm, _, _ = clothing_scene("jacket")
    calls = []

    def factory():
        return lambda ctx, bg: calls.append((ctx.raw_frame.shape, bg))

    pipeline = Pipeline(fba_factory=factory)
    settings = Settings(auto_shutter=False, fba_demo=True)
    ctx, _ = pipeline.process(rgb, settings, landmarks=lm, defer_fba=True)
    assert calls == []
    pipeline.render_final(rgb, ctx, settings)
    assert len(calls) == 1 and calls[0][0] == rgb.shape


def test_deferred_modnet_checks_the_selected_alpha():
    from idphoto.checker import Verdict

    rgb, lm, _, _ = clothing_scene("jacket")
    seen = []

    class Model:
        def predict_alpha(self, raw_rgb, *, capture=False):
            from tools.presentation_modnet import ModnetPrediction

            alpha = np.full(raw_rgb.shape[:2], 0.2, np.float32)
            alpha[0, :] = 0
            return ModnetPrediction(
                alpha,
                {"input_width": raw_rgb.shape[1], "input_height": raw_rgb.shape[0],
                 "capture": capture},
            )

        def apply_prediction(self, ctx, prediction, background):
            ctx.alpha = prediction.alpha
            ctx.output_frame = ctx.relit_frame

    class Check:
        def evaluate(self, ctx):
            seen.append(ctx.alpha.copy())
            passed = float(ctx.alpha.mean()) > 0.5
            return {7: Verdict(7, "clean_cutout", passed, float(ctx.alpha.mean()), "test")}

    settings = Settings(auto_shutter=False, modnet_demo=True)
    pipeline = Pipeline(modnet_factory=Model)
    pipeline.checker = Check()
    ctx, fired = pipeline.process(rgb, settings, landmarks=lm, defer_modnet=True)
    assert not fired and ctx.rule_verdicts is None

    pipeline._apply_modnet(ctx, settings, settings.background_linear())
    ctx, _ = pipeline.finalize_deferred_model(ctx, settings)
    assert not seen[-1].mean() > 0.5
    np.testing.assert_array_equal(seen[-1], ctx.alpha)
    assert not ctx.rule_verdicts[7].passed


def test_deferred_modnet_skips_unused_clothing_matting(monkeypatch):
    rgb, lm, _, _ = clothing_scene("jacket")
    pipeline = Pipeline()

    def unexpected_clothing(*_args, **_kwargs):
        raise AssertionError("MODNet must not run the unused Clothing matting pass")

    monkeypatch.setattr(pipeline.matter, "process", unexpected_clothing)
    ctx, fired = pipeline.process(
        rgb, Settings(auto_shutter=False, modnet_demo=True), landmarks=lm,
        defer_modnet=True,
    )

    assert not fired
    assert ctx.alpha.shape == rgb.shape[:2]
    np.testing.assert_array_equal(ctx.alpha, 0)
    np.testing.assert_array_equal(ctx.output_frame, np.ones_like(ctx.output_frame))


def test_deferred_modnet_shutter_resets_and_blocks_uncertain_tracks():
    from idphoto.checker import AutoShutter, Verdict

    rgb, lm, _, _ = clothing_scene("jacket")

    class Check:
        def evaluate(self, ctx):
            return {
                number: Verdict(number, f"rule_{number}", True, 1.0, "test")
                for number in range(1, 8)
            }

    settings = Settings(auto_shutter=True, modnet_demo=True)
    pipeline = Pipeline()
    pipeline.shutter = AutoShutter(frames=2)
    pipeline.checker = Check()

    def finalize(source, track_break=False, ambiguous=False):
        ctx, _ = pipeline.process(rgb, settings, landmarks=lm, defer_modnet=True)
        ctx.all_person_alpha = np.full(rgb.shape[:2], 0.9, np.float32)
        if not ambiguous:
            ctx.all_person_alpha[0, :] = 0
        pipeline._select_subject(ctx)
        assert ctx.subject_status == ("ambiguous" if ambiguous else "ready")
        ctx.stats["landmark_source"] = source
        ctx.stats["track_break"] = track_break
        return pipeline.finalize_deferred_model(ctx, settings)[1]

    assert not finalize("hint")
    assert pipeline.shutter.count == 0
    assert not finalize("hint")
    assert not finalize("native", track_break=True)
    assert pipeline.shutter.count == 0
    assert not finalize("native")
    assert finalize("native")
    assert not finalize("native", ambiguous=True)
    assert not finalize("native")
    assert finalize("native")


def test_modnet_track_break_clears_previous_light_and_capture_streak():
    rgb, lm, _, _ = clothing_scene("jacket")

    class Detector:
        source = "native"
        track_break = True
        calls = 1
        pitch_deg = yaw_deg = None

        def detect(self, _rgb, _timestamp_ms=None):
            return lm

    pipeline = Pipeline(detector=Detector())
    pipeline.relighter.history.append(np.ones(9, np.float32))
    pipeline.tracker.state = {"stale": True}
    pipeline.shutter.count = 4
    pipeline.shutter.armed = False
    ctx, _ = pipeline.process(
        rgb, Settings(auto_shutter=False, modnet_demo=True), defer_modnet=True,
    )
    assert len(pipeline.relighter.history) == 1
    assert pipeline.tracker.state is None
    assert pipeline.shutter.count == 0
    assert pipeline.shutter.armed
    assert ctx.stats["track_break"]


def test_modnet_full_resolution_capture_uses_its_own_cutout_verdict(monkeypatch):
    from idphoto.checker import Verdict
    import idphoto.pipeline as pipeline_module

    rgb, lm, _, _ = clothing_scene("jacket")
    captures = []

    class Model:
        def predict_alpha(self, raw_rgb, *, capture=False):
            from tools.presentation_modnet import ModnetPrediction

            captures.append((raw_rgb.shape, capture))
            alpha = np.zeros(raw_rgb.shape[:2], np.float32)
            alpha[raw_rgb.shape[0] // 3:] = 0.9
            return ModnetPrediction(
                alpha, {"input_width": raw_rgb.shape[1], "input_height": raw_rgb.shape[0],
                        "capture": capture},
            )

        def apply_prediction(self, ctx, prediction, background):
            ctx.alpha = prediction.alpha
            ctx.output_frame = ctx.relit_frame

    class Check:
        def evaluate(self, ctx):
            assert ctx.geometry is not None
            assert "crop_box" in ctx.geometry
            passed = float(ctx.alpha.mean()) > 0.5
            return {7: Verdict(7, "clean_cutout", passed, float(ctx.alpha.mean()), "test")}

    monkeypatch.setattr(pipeline_module, "Checker", Check)
    settings = Settings(strength=0, auto_shutter=False, modnet_demo=True)
    pipeline = Pipeline(modnet_factory=Model)
    live, _ = pipeline.process(rgb, settings, landmarks=lm, defer_modnet=True)
    result = pipeline.render_final(rgb, live, settings)

    assert captures[-1] == (rgb.shape, True)
    assert result["report"]["settings"]["modnet_demo"]
    assert result["report"]["original_wall_kept"] is False
    assert result["report"]["rules"][7]["passed"] is True
    assert np.all(result["ctx"].output_frame[:rgb.shape[0] // 3] == 1)


def test_presentation_models_are_mutually_exclusive():
    rgb, lm, _, _ = clothing_scene("jacket")
    pipeline = Pipeline()
    with pytest.raises(ValueError, match="mutually exclusive"):
        pipeline.process(rgb, Settings(fba_demo=True, modnet_demo=True), landmarks=lm)


def test_scheduled_capture_matches_synchronous_render():
    import threading

    import numpy as np

    from idphoto.fba_jobs import FbaJobs

    rgb, lm, _, _ = clothing_scene("jacket")

    def factory():
        def apply(ctx, background):
            ctx.alpha = ctx.alpha * .8
        return apply

    pipeline = Pipeline(fba_factory=factory)
    settings = Settings(auto_shutter=False, fba_demo=True)
    ctx, _ = pipeline.process(rgb, settings, landmarks=lm, defer_fba=True)
    expected = pipeline.render_final(rgb, ctx, settings)
    results, errors = [], []
    done = threading.Event()
    jobs = FbaJobs(lambda: None, pipeline.render_final, lambda _: None,
                   lambda result: (results.append(result), done.set()),
                   lambda *err: (errors.append(err), done.set()))
    try:
        assert jobs.submit_capture(rgb, ctx, settings)
        assert done.wait(10)
    finally:
        jobs.stop()
    assert not errors
    np.testing.assert_array_equal(results[0]["ctx"].alpha, expected["ctx"].alpha)
    for key in expected["photos"]:
        np.testing.assert_array_equal(results[0]["photos"][key], expected["photos"][key])
    assert results[0]["report"] == expected["report"]


def test_presentation_adapter_uses_fba_alpha_and_relighting(monkeypatch):
    from idphoto.context import FrameContext
    from tools import presentation_fba

    rgb = np.full((2, 2, 3), 128, np.uint8)
    ctx = FrameContext(raw_frame=rgb, landmarks=np.zeros((478, 3), np.float32),
                       face_mask=np.ones((2, 2), bool))
    ctx.relit_frame = np.full((2, 2, 3), .25, np.float32)
    ctx.ratio_map = np.full((2, 2), 1.5, np.float32)
    prediction = np.zeros((2, 2, 7), np.float32)
    prediction[..., 0] = .4
    prediction[..., 1:4] = .5

    class Model:
        def process(self, image, trimap, **kwargs):
            np.testing.assert_array_equal(image, rgb)
            return prediction, 1

    monkeypatch.setattr(presentation_fba, "fba_trimap", lambda *_: np.zeros((2, 2), np.uint8))
    adapter = PresentationFBA.__new__(PresentationFBA)
    adapter.model = Model()
    adapter(ctx, np.ones(3, np.float32))
    np.testing.assert_allclose(ctx.alpha, .4)
    np.testing.assert_allclose(ctx.output_frame, .4 * color.srgb_to_linear(.5) * 1.5 + .6)
    adapter(ctx, None)
    np.testing.assert_array_equal(ctx.output_frame, ctx.relit_frame)


@pytest.mark.parametrize("size", [1, 2, 3, 6])
@pytest.mark.parametrize("shape", [(3, 5), (9, 13), (12, 18)])
def test_mps_pool_workaround_matches_adaptive_pool(size, shape):
    torch = pytest.importorskip("torch")
    from tools.fba_reference import _exact_adaptive_pool

    x = torch.rand((2, 3, *shape), generator=torch.Generator().manual_seed(7))
    expected = torch.nn.functional.adaptive_avg_pool2d(x, size)
    pool = _exact_adaptive_pool(size)
    torch.testing.assert_close(pool(x), expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(pool(x), expected, atol=1e-6, rtol=1e-6)


def test_unavailable_mps_is_reported_before_loading_weights(monkeypatch):
    torch = pytest.importorskip("torch")
    from tools.presentation_fba import check_availability

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(ValueError, match="MPS acceleration is unavailable"):
        check_availability("mps")


def test_real_mps_prediction_matches_cpu_when_available():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    from tools.eval_clothing import fba_trimap
    from tools.fba_reference import FBAReference, verify_assets

    try:
        verify_assets()
    except FileNotFoundError:
        pytest.skip("local FBA research weights unavailable")
    torch.set_num_threads(4)
    rgb, lm, _, _ = clothing_scene("jacket")
    ctx, _ = Pipeline().process(rgb, Settings(strength=0, auto_shutter=False), landmarks=lm)
    trimap = fba_trimap(rgb, lm, ctx)
    cpu, _ = FBAReference(device="cpu").process(rgb, trimap)
    mps, _ = FBAReference(device="mps").process(rgb, trimap)
    np.testing.assert_allclose(mps, cpu, rtol=0, atol=3e-4)
