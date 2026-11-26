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
