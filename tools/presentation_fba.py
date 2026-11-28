"""Local, noncommercial presentation-only adapter; not part of the app package."""

import importlib.util

import numpy as np

from idphoto import color
from tools.eval_clothing import fba_trimap
from tools.fba_reference import FBAReference, verify_assets

NO_KNOWN_REGION = "Trimap needs both known foreground and known background."


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
        try:
            trimap = fba_trimap(ctx.raw_frame, ctx.landmarks, ctx)
        except ValueError as exc:
            if not str(exc).startswith(NO_KNOWN_REGION):
                raise
            # The subject fills the frame (close to the camera), so no pixel is a sure
            # background. FBA cannot run on this frame; keep the main matte instead of failing.
            ctx.stats["fba"] = {"skipped": "no_known_background"}
            return
        prediction, _ = self.model.process(ctx.raw_frame, trimap, tile_size=768, overlap=96)
        ctx.alpha = prediction[..., 0]
        if background is not None:
            foreground = color.srgb_to_linear(prediction[..., 1:4])
            if ctx.ratio_map is not None:
                gain = ctx.ratio_map[..., None]
                if getattr(ctx, "color_gain", None) is not None:
                    gain = gain * ctx.color_gain
                foreground = np.minimum(foreground * gain, 1)
            a = ctx.alpha[..., None]
            ctx.output_frame = a * foreground + (1 - a) * background
        else:
            ctx.output_frame = ctx.relit_frame
