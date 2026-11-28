"""Local, noncommercial presentation-only adapter; not part of the app package."""

import importlib.util

import numpy as np

from idphoto import color, config, npimg
from tools.eval_clothing import fba_trimap
from tools.fba_reference import FBAReference, verify_assets

NO_KNOWN_REGION = "Trimap needs both known foreground and known background."

# FBA runs at the live preview width. At capture resolution the unknown band of the trimap
# is several times wider in pixels than the network can resolve, so it keeps the wall.
FBA_WORK_WIDTH = config.LIVE_WIDTH


def check_availability(device="cpu"):
    if importlib.util.find_spec("torch") is None:
        raise ImportError("FBA requires the optional extra: uv sync --extra fba --group dev")
    if device == "mps":
        import torch

        if not torch.backends.mps.is_available():
            raise ValueError("FBA MPS acceleration is unavailable on this system; select CPU.")
    verify_assets()


class WorkWidthFBA:
    """Run FBA on a copy downscaled to `width` and bring the result back to full size.

    Alpha is upsampled. The foreground keeps the full-resolution pixels and adds only the
    upsampled FBA color correction (foreground minus input), so skin and hair stay sharp.
    """

    def __init__(self, model, width=FBA_WORK_WIDTH):
        self.model = model
        self.width = width

    def process(self, rgb, trimap, tile_size=768, overlap=96):
        h, w = trimap.shape
        if w <= self.width:
            return self.model.process(rgb, trimap, tile_size=tile_size, overlap=overlap)
        sh, sw = max(1, round(h * self.width / w)), self.width
        small_rgb = np.clip(npimg.resize(rgb.astype(np.float32), sh, sw, "linear") + 0.5,
                            0, 255).astype(np.uint8)
        # Nearest sampling keeps the trimap's three labels.
        ys = np.minimum(((np.arange(sh) + 0.5) * h / sh).astype(np.int64), h - 1)
        xs = np.minimum(((np.arange(sw) + 0.5) * w / sw).astype(np.int64), w - 1)
        small_trimap = np.ascontiguousarray(trimap[ys][:, xs])
        small, tiles = self.model.process(small_rgb, small_trimap, tile_size=tile_size, overlap=overlap)
        correction = small[..., 1:4] - small_rgb.astype(np.float32) / 255
        up = npimg.resize(np.concatenate([small[..., :1], correction], axis=2), h, w, "linear")
        prediction = np.empty((h, w, 4), np.float32)
        prediction[..., 0] = np.clip(up[..., 0], 0, 1)
        prediction[..., 0][trimap == 255] = 1
        prediction[..., 0][trimap == 0] = 0
        prediction[..., 1:4] = np.clip(rgb.astype(np.float32) / 255 + up[..., 1:4], 0, 1)
        return prediction, tiles


class PresentationFBA:
    def __init__(self, device="cpu"):
        import torch

        torch.set_num_threads(4)
        self.model = WorkWidthFBA(FBAReference(device=device))

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
