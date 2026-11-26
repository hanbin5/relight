"""Local, noncommercial presentation-only adapter; not part of the app package."""

import importlib.util

import numpy as np

from idphoto import color
from tools.eval_clothing import fba_trimap
from tools.fba_reference import FBAReference, verify_assets


def check_availability(device="cpu"):
    if importlib.util.find_spec("torch") is None:
        raise ImportError("FBA requires the optional extra: uv sync --extra fba --group dev")
    if device == "mps":
        import torch

        if not torch.backends.mps.is_available():
            raise ValueError("FBA MPS acceleration is unavailable on this system; select CPU.")
    verify_assets()


class PresentationFBA:
    def __init__(self, device="cpu"):
        import torch

        torch.set_num_threads(4)
        self.model = FBAReference(device=device)

    def __call__(self, ctx, background):
        if ctx.landmarks is None or ctx.face_mask is None or not ctx.face_mask.any():
            return
        trimap = fba_trimap(ctx.raw_frame, ctx.landmarks, ctx)
        prediction, _ = self.model.process(ctx.raw_frame, trimap, tile_size=768, overlap=96)
        ctx.alpha = prediction[..., 0]
        if background is not None:
            foreground = color.srgb_to_linear(prediction[..., 1:4])
            if ctx.ratio_map is not None:
                foreground = np.minimum(foreground * ctx.ratio_map[..., None], 1)
            a = ctx.alpha[..., None]
            ctx.output_frame = a * foreground + (1 - a) * background
        else:
            ctx.output_frame = ctx.relit_frame
