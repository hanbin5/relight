"""The per-frame pipeline (plan 2.2, Figure 2) and the shutter render.

webcam frame -> face landmarks -> light fit and relight (A) -> person mask and new
background (B) -> head geometry (C) -> checker and auto shutter (D). The relit,
background-replaced frame is both the live view and, re-rendered at the camera's full
resolution, the saved photo.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from . import color, config, geometry, npimg
from .checker import AutoShutter, Checker
from .context import FrameContext
from .matting import Matter
from .relight import Relighter

# Background swatches of the wireframe (Figure 4); None keeps the original wall.
BACKGROUNDS = {
    "white": "#ffffff",
    "light_gray": "#e6e7e9",
    "light_blue": "#dbe5f0",
    "mint": "#e1eee8",
    "wall": None,
}


@dataclass
class Settings:
    """User choices from the GUI; read by the pipeline every frame."""

    format: str = "resume"
    head_share: dict = field(default_factory=lambda: {k: f.head_share for k, f in geometry.FORMATS.items()})
    background: str = "white"
    strength: float = config.DEFAULT_STRENGTH
    auto_shutter: bool = True

    def background_linear(self):
        hex_color = BACKGROUNDS.get(self.background)
        return None if hex_color is None else color.hex_to_linear(hex_color)


class Pipeline:
    def __init__(self, detector=None):
        self.detector = detector  # FaceLandmarkDetector; None when the source gives landmarks
        self.relighter = Relighter()
        self.matter = Matter()
        self.checker = Checker()
        self.tracker = geometry.GeometryTracker()
        self.shutter = AutoShutter()

    def set_light_smoothing(self, frames: int):
        self.relighter.set_smoothing(frames)

    def process(self, raw: np.ndarray, settings: Settings, landmarks=None, timestamp_ms: int = 0,
                full_frame: np.ndarray | None = None):
        """Run one live frame. full_frame is the camera frame raw was downscaled from, if any.
        Returns (ctx, shutter_fired)."""
        t = [time.perf_counter()]
        ctx = FrameContext(raw_frame=raw, timestamp_ms=timestamp_ms)
        if full_frame is not None and full_frame.shape[1] > raw.shape[1]:
            ctx.full_frame = full_frame
        ctx.linear_rgb = color.decode_u8(raw)
        ctx.Y = color.luminance(ctx.linear_rgb)
        if landmarks is None and self.detector is not None:
            landmarks = self.detector.detect(raw, timestamp_ms)
        ctx.landmarks = landmarks
        t.append(time.perf_counter())

        self.relighter.process(ctx, settings.strength)
        t.append(time.perf_counter())
        self.matter.process(ctx, settings.background_linear())
        t.append(time.perf_counter())
        if ctx.landmarks is not None:
            geo = self.tracker.update(geometry.measure(ctx.landmarks, ctx.alpha))
            fmt = geometry.FORMATS[settings.format]
            geo["crop_box"] = geometry.crop_box(geo, fmt, settings.head_share[settings.format])
            geo["crop_fits"] = geometry.crop_fits(geo["crop_box"], ctx.size)
            ctx.geometry = geo
        else:
            self.tracker.reset()
            ctx.geometry = None
        t.append(time.perf_counter())
        verdicts = self.checker.evaluate(ctx)
        all_pass = all(v.passed for v in verdicts.values())
        fired = self.shutter.update(all_pass, enabled=settings.auto_shutter)
        t.append(time.perf_counter())
        names = ("landmarks", "relight", "matting", "geometry", "checker")
        ctx.stats["timings_ms"] = {n: 1000 * (t[i + 1] - t[i]) for i, n in enumerate(names)}
        ctx.stats["all_pass"] = all_pass
        return ctx, fired

    def render_final(self, raw_full: np.ndarray, live: FrameContext, settings: Settings) -> dict:
        """Re-render the shutter frame at full resolution and make every output.

        Reuses the live frame's landmarks (scaled) and its 5-frame light average, so the
        saved photo matches what the user saw. Returns photos, print sheets and a report.
        """
        scale = raw_full.shape[1] / live.raw_frame.shape[1]
        ctx = FrameContext(raw_frame=raw_full, timestamp_ms=live.timestamp_ms)
        ctx.linear_rgb = color.decode_u8(raw_full)
        ctx.Y = color.luminance(ctx.linear_rgb)
        ctx.landmarks = live.landmarks * np.float32(scale)
        Relighter().process(ctx, settings.strength, light=live.light_coeffs, update=False)
        background = settings.background_linear()
        # C.4 fallback: when the cutout was not clean (rule 7), keep the original wall.
        cutout = (live.rule_verdicts or {}).get(7)
        kept_wall = background is not None and not (cutout and cutout.passed)
        if kept_wall:
            background = None
        Matter().process(ctx, background, update=False, fast=False)
        geo = geometry.measure(ctx.landmarks, ctx.alpha)
        ctx.geometry = geo
        pad = background if background is not None else ctx.wall_color
        photos, sheets, boxes = {}, {}, {}
        for key, fmt in geometry.FORMATS.items():
            boxes[key] = geometry.crop_box(geo, fmt, settings.head_share[key])
            photos[key] = geometry.render_photo(ctx.output_frame, boxes[key], fmt, pad)
            if fmt.sheet_grid:
                sheets[key] = geometry.print_sheet(photos[key], fmt)
        report = {
            "notice": geometry.NOTICE,
            "settings": {"background": settings.background, "strength": settings.strength,
                         "head_share": dict(settings.head_share)},
            "original_wall_kept": kept_wall,
            "relighting": {k: _num(live.stats.get(k)) for k in ("r_lr_before", "r_lr_after", "shadow_before", "shadow_after")},
            "light_coeffs": [float(x) for x in live.light_coeffs] if live.light_coeffs is not None else None,
            "rules": {
                v.rule: {"name": v.key, "passed": v.passed, "value": _num(v.value), "bound": v.bound,
                         "detail": {k: _num(x) for k, x in v.detail.items()}}
                for v in (live.rule_verdicts or {}).values()
            },
            "geometry": {"crown": [float(x) for x in geo["crown"]], "chin": [float(x) for x in geo["chin"]],
                         "head_length_px": geo["head_length"], "crown_source": geo["crown_source"],
                         "roll_deg": geo["roll_deg"], "yaw_ratio": geo["yaw_ratio"], "pitch_ratio": geo["pitch_ratio"],
                         "crop_boxes": {k: [float(x) for x in b] for k, b in boxes.items()},
                         "crop_fits": {k: geometry.crop_fits(b, ctx.size) for k, b in boxes.items()}},
            "frame_size": list(ctx.size),
        }
        return {"photos": photos, "sheets": sheets, "report": report, "ctx": ctx, "kept_wall": kept_wall}


def to_live(frame: np.ndarray, landmarks=None, width: int = config.LIVE_WIDTH):
    """Downscale a camera frame (and its landmarks, if given) to the live width."""
    h, w = frame.shape[:2]
    if w <= width:
        return frame, landmarks
    f = w / width
    if abs(f - round(f)) < 1e-6:
        small = npimg.downscale_u8(frame, int(round(f)))
    else:
        small = npimg.resize(frame, int(round(h / f)), width, "linear")
        small = np.clip(small + 0.5, 0, 255).astype(np.uint8)
    if landmarks is not None:
        landmarks = landmarks * np.float32(small.shape[1] / w)
    return small, landmarks


def _num(x):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(x) else x
