import numpy as np
import pytest

from idphoto import color, synthetic
from idphoto.context import FrameContext
from idphoto.landmarks import MODEL_PATH
from idphoto.pipeline import Pipeline, Settings

FRONT = dict(light=(0.0, 0.25, 1.0), ambient=0.3)
SIDE = dict(light=(-0.55, 0.2, 0.8), ambient=0.25)

needs_model = pytest.mark.skipif(not MODEL_PATH.exists(), reason="run tools/fetch_model.py first")


def make_ctx(rgb, lm) -> FrameContext:
    ctx = FrameContext(raw_frame=rgb, landmarks=lm)
    ctx.linear_rgb = color.decode_u8(rgb)
    ctx.Y = color.luminance(ctx.linear_rgb)
    return ctx


def run_pipeline(scene: synthetic.Scene, frames: int = 6, settings: Settings | None = None):
    rgb, lm, person = synthetic.render(scene)
    pipeline = Pipeline()
    settings = settings or Settings()
    fired = False
    for i in range(frames):
        ctx, f = pipeline.process(rgb, settings, landmarks=lm, timestamp_ms=33 * i)
        fired |= f
    return ctx, person, pipeline, fired


@pytest.fixture(scope="session")
def side_lit():
    return synthetic.render(synthetic.Scene(**SIDE))


@pytest.fixture(scope="session")
def front_lit():
    return synthetic.render(synthetic.Scene(**FRONT))


@pytest.fixture
def rng():
    return np.random.default_rng(0)
