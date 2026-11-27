"""The per-frame pipeline (plan 2.2, Figure 2) and the shutter render.

webcam frame -> face landmarks -> light fit and relight (A) -> person mask and new
background (B) -> head geometry (C) -> checker and auto shutter (D). The relit,
background-replaced frame is both the live view and, re-rendered at the camera's full
resolution, the saved photo.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from . import color, config, geometry, matting, npimg
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
    clothing_enabled: bool = True
    fba_demo: bool = False
    modnet_demo: bool = False
    capture_matte: str = "modnet"

    def background_linear(self):
        hex_color = BACKGROUNDS.get(self.background)
        return None if hex_color is None else color.hex_to_linear(hex_color)


class Pipeline:
    def __init__(self, detector=None, fba_factory=None, modnet_factory=None,
                 birefnet_factory=None):
        self.detector = detector  # FaceLandmarkDetector; None when the source gives landmarks
        self.relighter = Relighter()
        self.matter = Matter()
        self.checker = Checker()
        self.tracker = geometry.GeometryTracker()
        self.shutter = AutoShutter()
        self.matting_choice = None
        self.fba_factory = fba_factory
        self.fba = None
        self.modnet_factory = modnet_factory
        self.modnet = None
        self.birefnet_factory = birefnet_factory
        self.birefnet = None

    def set_light_smoothing(self, frames: int):
        self.relighter.set_smoothing(frames)

    def process(self, raw: np.ndarray, settings: Settings, landmarks=None, timestamp_ms: int = 0,
                full_frame: np.ndarray | None = None, defer_fba: bool = False,
                defer_modnet: bool = False):
        """Run one live frame. full_frame is the camera frame raw was downscaled from, if any.
        Returns (ctx, shutter_fired)."""
        if settings.fba_demo and settings.modnet_demo:
            raise ValueError("FBA and MODNet presentation modes are mutually exclusive.")
        t = [time.perf_counter()]
        ctx = FrameContext(raw_frame=raw, timestamp_ms=timestamp_ms)
        if full_frame is not None and full_frame.shape[1] > raw.shape[1]:
            ctx.full_frame = full_frame
        ctx.linear_rgb = color.decode_u8(raw)
        ctx.Y = color.luminance(ctx.linear_rgb)
        angles = {"pitch_deg": None, "yaw_deg": None}  # only MediaPipe's own detection gives them
        detector_used = landmarks is None and self.detector is not None
        if detector_used:
            landmarks = self.detector.detect(raw, timestamp_ms)
            angles = {k: getattr(self.detector, k, None) for k in angles}
            if getattr(self.detector, "track_break", False):
                self.relighter.reset()
                self.tracker.reset()
                self.shutter.rearm()
                self.checker.sharpness_seen.clear()
        ctx.landmarks = landmarks
        if settings.modnet_demo:
            if detector_used and hasattr(self.detector, "candidates"):
                ctx.subject_faces = tuple(c.landmarks.copy() for c in self.detector.candidates)
                ctx.subject_index = self.detector.selected_index
            elif landmarks is not None:
                ctx.subject_faces = (landmarks.copy(),)
                ctx.subject_index = 0
            ctx.subject_status = "pending" if landmarks is not None else "no_face"
        if detector_used:
            ctx.stats["landmark_source"] = getattr(self.detector, "source", "native")
            ctx.stats["track_break"] = bool(getattr(self.detector, "track_break", False))
            ctx.stats["landmark_detector_calls"] = getattr(self.detector, "calls", 1)
        else:
            ctx.stats["landmark_source"] = "provided" if landmarks is not None else "none"
            ctx.stats["track_break"] = False
        t.append(time.perf_counter())

        self.relighter.process(ctx, settings.strength)
        t.append(time.perf_counter())
        choice = (settings.clothing_enabled, settings.fba_demo, settings.modnet_demo)
        if self.matting_choice != choice:
            self.matter.reset()
            self.shutter.count = 0
            self.matting_choice = choice
        if settings.modnet_demo:
            ctx.alpha = np.zeros(ctx.Y.shape, np.float32)
            self._subject_placeholder(ctx, settings.background_linear())
        else:
            self.matter.process(ctx, settings.background_linear(),
                                clothing_enabled=settings.clothing_enabled)
        if not defer_fba:
            self._apply_fba(ctx, settings, settings.background_linear())
        if not defer_modnet:
            self._apply_modnet(ctx, settings, settings.background_linear())
        t.append(time.perf_counter())
        subject_blocked = settings.modnet_demo and ctx.subject_status != "ready"
        if defer_modnet or subject_blocked:
            ctx.geometry = None
            ctx.rule_verdicts = None
            ctx.stats["deferred_model_angles"] = angles
            fired = False
            t.extend((time.perf_counter(), time.perf_counter()))
        elif ctx.landmarks is not None:
            geo = self.tracker.update(geometry.measure(ctx.landmarks, ctx.alpha))
            fmt = geometry.FORMATS[settings.format]
            geo["crop_box"] = geometry.crop_box(geo, fmt, settings.head_share[settings.format])
            geo["crop_fits"] = geometry.crop_fits(geo["crop_box"], ctx.size)
            geo.update(angles)
            ctx.geometry = geo
        else:
            self.tracker.reset()
            ctx.geometry = None
        t.append(time.perf_counter())
        if defer_modnet or subject_blocked:
            all_pass = False
            fired = False
            if subject_blocked and (not defer_modnet or ctx.subject_status != "pending"):
                self.shutter.rearm()
            t.append(time.perf_counter())
        else:
            verdicts = self.checker.evaluate(ctx)
            all_pass = all(v.passed for v in verdicts.values())
            safe_tracking = (
                ctx.stats.get("landmark_source") not in ("hint", "search")
                and not ctx.stats.get("track_break", False)
            )
            fired = self.shutter.update(all_pass and safe_tracking, enabled=settings.auto_shutter)
            t.append(time.perf_counter())
        names = ("landmarks", "relight", "matting", "geometry", "checker")
        ctx.stats["timings_ms"] = {n: 1000 * (t[i + 1] - t[i]) for i, n in enumerate(names)}
        if not defer_modnet and not subject_blocked:
            ctx.rule_verdicts = verdicts
        ctx.stats["all_pass"] = all_pass
        return ctx, fired

    def finalize_deferred_model(self, ctx: FrameContext, settings: Settings):
        """Run geometry, checks, and shutter only after the selected matte is ready."""
        start = time.perf_counter()
        angles = ctx.stats.pop("deferred_model_angles", {})
        if ctx.subject_status != "ready":
            self.tracker.reset()
            self.shutter.rearm()
            ctx.geometry = None
            ctx.rule_verdicts = None
            ctx.stats["all_pass"] = False
            return ctx, False
        if ctx.landmarks is not None:
            geo = self.tracker.update(geometry.measure(ctx.landmarks, ctx.alpha))
            fmt = geometry.FORMATS[settings.format]
            geo["crop_box"] = geometry.crop_box(geo, fmt, settings.head_share[settings.format])
            geo["crop_fits"] = geometry.crop_fits(geo["crop_box"], ctx.size)
            geo.update(angles)
            ctx.geometry = geo
        else:
            self.tracker.reset()
            ctx.geometry = None
        geometry_ms = 1000 * (time.perf_counter() - start)
        verdicts = self.checker.evaluate(ctx)
        all_pass = all(v.passed for v in verdicts.values())
        safe_tracking = (
            ctx.stats.get("landmark_source") not in ("hint", "search")
            and not ctx.stats.get("track_break", False)
        )
        fired = self.shutter.update(all_pass and safe_tracking, enabled=settings.auto_shutter)
        ctx.rule_verdicts = verdicts
        ctx.stats["all_pass"] = all_pass
        timings = ctx.stats.setdefault("timings_ms", {})
        timings["geometry"] = geometry_ms
        timings["checker"] = 1000 * (time.perf_counter() - start) - geometry_ms
        return ctx, fired

    def render_final(self, raw_full: np.ndarray, live: FrameContext, settings: Settings) -> dict:
        """Re-render the shutter frame at full resolution and make every output.

        Reuses the live frame's landmarks (scaled) and its 5-frame light average.
        An FBA demo capture may differ from its Clothing-only live preview.
        Returns photos, print sheets and a report.
        """
        if settings.fba_demo and settings.modnet_demo:
            raise ValueError("FBA and MODNet presentation modes are mutually exclusive.")
        if settings.capture_matte != "modnet" and not settings.modnet_demo:
            raise ValueError("BiRefNet portrait capture requires the MODNet experimental mode.")
        render_started = time.perf_counter()
        capture_timings = {
            "render_final_started_monotonic_ms": time.monotonic() * 1000,
            "output_wall_estimate_ms": 0.0,
            "composite_ms": 0.0,
        }
        scale = raw_full.shape[1] / live.raw_frame.shape[1]
        ctx = FrameContext(raw_frame=raw_full, timestamp_ms=live.timestamp_ms)
        ctx.full_frame = raw_full
        prepare_started = time.perf_counter()
        ctx.linear_rgb = color.decode_u8(raw_full)
        ctx.Y = color.luminance(ctx.linear_rgb)
        ctx.landmarks = live.landmarks * np.float32(scale)
        if settings.modnet_demo:
            ctx.subject_faces = tuple(face * np.float32(scale) for face in live.subject_faces)
            ctx.subject_index = live.subject_index
            ctx.subject_status = "pending"
        Relighter().process(ctx, settings.strength, light=live.light_coeffs, update=False,
                            target_yaw=live.stats.get("target_yaw"))
        capture_timings["frame_decode_relight_ms"] = \
            (time.perf_counter() - prepare_started) * 1000
        background = settings.background_linear()
        if settings.modnet_demo:
            matte_started = time.perf_counter()
            if settings.capture_matte == "birefnet-portrait":
                self._apply_birefnet(ctx)
            elif settings.capture_matte == "modnet":
                self._apply_modnet(ctx, settings, None, capture=True)
            else:
                raise ValueError(f"Unknown capture matte: {settings.capture_matte}")
            capture_timings["matte_apply_call_ms"] = \
                (time.perf_counter() - matte_started) * 1000
            if ctx.subject_status != "ready":
                return {"subject_blocked": ctx.subject_status}
            from .presentation_modnet import (
                composite_dense,
                estimate_wall_color,
                estimate_walls,
            )

            wall_estimate_started = time.perf_counter()
            ctx.wall_color = estimate_wall_color(ctx.linear_rgb, ctx.all_person_alpha)
            capture_timings["checker_wall_estimate_ms"] = \
                (time.perf_counter() - wall_estimate_started) * 1000
            if ctx.landmarks is not None:
                geo = geometry.measure(ctx.landmarks, ctx.alpha)
                fmt = geometry.FORMATS[settings.format]
                geo["crop_box"] = geometry.crop_box(geo, fmt, settings.head_share[settings.format])
                geo["crop_fits"] = geometry.crop_fits(geo["crop_box"], ctx.size)
                for key in ("pitch_deg", "yaw_deg"):
                    geo[key] = (live.geometry or {}).get(key)
                ctx.geometry = geo
            checker_started = time.perf_counter()
            capture_verdicts = Checker().evaluate(ctx)
            capture_timings["checker_ms"] = \
                (time.perf_counter() - checker_started) * 1000
            cutout = capture_verdicts.get(7)
            kept_wall = background is not None and not (cutout and cutout.passed) \
                and len(ctx.subject_faces) <= 1
            if kept_wall:
                ctx.output_frame = ctx.relit_frame
            elif background is not None or len(ctx.subject_faces) > 1:
                wall_estimate_started = time.perf_counter()
                ctx.wall_color, wall = estimate_walls(ctx.linear_rgb, ctx.all_person_alpha)
                capture_timings["output_wall_estimate_ms"] = \
                    (time.perf_counter() - wall_estimate_started) * 1000
                composite_started = time.perf_counter()
                ctx.output_frame = composite_dense(
                    ctx, wall, background if background is not None else ctx.wall_color)
                capture_timings["composite_ms"] = \
                    (time.perf_counter() - composite_started) * 1000
            else:
                ctx.output_frame = ctx.relit_frame
        else:
            # C.4 fallback: when the cutout was not clean (rule 7), keep the original wall.
            cutout = (live.rule_verdicts or {}).get(7)
            kept_wall = background is not None and not (cutout and cutout.passed)
            if kept_wall:
                background = None
            Matter().process(ctx, background, update=False, fast=False,
                             clothing_enabled=settings.clothing_enabled)
            self._apply_fba(ctx, settings, background)
            capture_verdicts = live.rule_verdicts
        geo = ctx.geometry or geometry.measure(ctx.landmarks, ctx.alpha)
        for k in ("pitch_deg", "yaw_deg"):
            geo[k] = (live.geometry or {}).get(k)
        ctx.geometry = geo
        pad = background if background is not None and not kept_wall else ctx.wall_color
        photos, sheets, boxes = {}, {}, {}
        output_render_started = time.perf_counter()
        for key, fmt in geometry.FORMATS.items():
            boxes[key] = geometry.crop_box(geo, fmt, settings.head_share[key])
            photos[key] = geometry.render_photo(ctx.output_frame, boxes[key], fmt, pad)
            if fmt.sheet_grid:
                sheets[key] = geometry.print_sheet(photos[key], fmt)
        capture_timings["crop_photo_sheet_render_ms"] = \
            (time.perf_counter() - output_render_started) * 1000
        report = {
            "notice": geometry.NOTICE,
            "settings": {"background": settings.background, "strength": settings.strength,
                         "clothing_enabled": settings.clothing_enabled,
                         "fba_demo": settings.fba_demo,
                         "modnet_demo": settings.modnet_demo,
                         "capture_matte": settings.capture_matte,
                         "head_share": dict(settings.head_share)},
            "original_wall_kept": kept_wall,
            "relighting": {k: _num(live.stats.get(k)) for k in ("r_lr_before", "r_lr_after", "shadow_before", "shadow_after")},
            "light_coeffs": [float(x) for x in live.light_coeffs] if live.light_coeffs is not None else None,
            "rules": {
                v.rule: {"name": v.key, "passed": v.passed, "value": _num(v.value), "bound": v.bound,
                         "detail": {k: _num(x) for k, x in v.detail.items()}}
                for v in (capture_verdicts or {}).values()
            },
            "geometry": {"crown": [float(x) for x in geo["crown"]],
                         "chin": [float(x) for x in geo["chin"]],
                         "head_length_px": geo["head_length"], "crown_source": geo["crown_source"],
                         "roll_deg": geo["roll_deg"], "yaw_ratio": geo["yaw_ratio"], "pitch_ratio": geo["pitch_ratio"],
                         "pitch_deg": _num(geo["pitch_deg"]), "yaw_deg": _num(geo["yaw_deg"]),
                         "crop_boxes": {k: [float(x) for x in b] for k, b in boxes.items()},
                         "crop_fits": {k: geometry.crop_fits(b, ctx.size) for k, b in boxes.items()}},
            "frame_size": list(ctx.size),
        }
        capture_timings["render_final_finished_monotonic_ms"] = time.monotonic() * 1000
        capture_timings["render_final_total_ms"] = \
            (time.perf_counter() - render_started) * 1000
        ctx.stats["capture_timing_ms"] = capture_timings
        return {"photos": photos, "sheets": sheets, "report": report, "ctx": ctx, "kept_wall": kept_wall}

    def _apply_fba(self, ctx, settings, background):
        if not settings.fba_demo or ctx.landmarks is None or ctx.face_mask is None or not ctx.face_mask.any():
            return
        if self.fba_factory is None:
            raise RuntimeError("FBA is only available in the opt-in presentation demo.")
        if self.fba is None:
            self.fba = self.fba_factory()
        self.fba(ctx, background)

    def _apply_modnet(self, ctx, settings, background, capture=False):
        if not settings.modnet_demo:
            return
        prediction = self.predict_modnet_alpha(ctx.raw_frame, capture=capture)
        self.apply_modnet_prediction(ctx, prediction, background)

    def predict_modnet_alpha(self, raw_rgb, *, capture=False):
        self._ensure_modnet()
        predictor = getattr(self.modnet, "predict_alpha", None)
        if predictor is None:
            raise TypeError("MODNet adapter must implement predict_alpha(raw_rgb, capture=...).")
        prediction = predictor(raw_rgb, capture=capture)
        alpha = getattr(prediction, "alpha", None)
        if not isinstance(alpha, np.ndarray) or alpha.shape != raw_rgb.shape[:2]:
            raise ValueError("MODNet prediction alpha must match the input frame dimensions.")
        if not np.isfinite(alpha).all():
            raise ValueError("MODNet prediction alpha contains non-finite values.")
        if np.any((alpha < 0) | (alpha > 1)):
            raise ValueError("MODNet prediction alpha must be in [0, 1].")
        return prediction

    def prepare_modnet_input(self, raw_rgb):
        self._ensure_modnet()
        prepare = getattr(self.modnet, "prepare_input", None)
        if prepare is None:
            raise TypeError("MODNet adapter must implement prepare_input(raw_rgb).")
        return prepare(raw_rgb)

    def _ensure_modnet(self):
        if self.modnet_factory is None:
            raise RuntimeError("MODNet is only available in the opt-in presentation demo.")
        if self.modnet is None:
            self.modnet = self.modnet_factory()

    def _apply_birefnet(self, ctx):
        if self.birefnet_factory is None:
            raise RuntimeError(
                "BiRefNet portrait capture is unavailable; configure a verified local model."
            )
        if self.birefnet is None:
            self.birefnet = self.birefnet_factory()
        prediction = self.birefnet.predict_alpha(ctx.raw_frame, capture=True)
        alpha = getattr(prediction, "alpha", None)
        if not isinstance(alpha, np.ndarray) or alpha.shape != ctx.raw_frame.shape[:2]:
            raise ValueError("BiRefNet portrait alpha must match the raw capture dimensions.")
        if not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha > 1)):
            raise ValueError("BiRefNet portrait alpha must be finite and in [0, 1].")
        ctx.all_person_alpha = prediction.alpha
        self._select_subject(ctx)
        ctx.stats["birefnet_portrait"] = dict(prediction.stats)

    def apply_modnet_prediction(self, ctx, prediction, background):
        from .presentation_modnet import ModnetPrediction

        ctx.all_person_alpha = prediction.alpha
        self._select_subject(ctx)
        if ctx.subject_status != "ready":
            self._subject_placeholder(ctx, background)
            return
        apply_prediction = getattr(self.modnet, "apply_prediction", None)
        if apply_prediction is None:
            raise TypeError("MODNet adapter must implement apply_prediction(ctx, prediction, background).")
        apply_prediction(ctx, ModnetPrediction(ctx.alpha, prediction.stats), background)

    @staticmethod
    def _select_subject(ctx):
        from .subject_matte import select_subject_alpha

        faces = ctx.subject_faces
        selected = ctx.subject_index
        if not faces and ctx.landmarks is not None:
            faces, selected = (ctx.landmarks,), 0
        ctx.alpha, ctx.subject_status = select_subject_alpha(ctx.all_person_alpha, faces, selected)

    @staticmethod
    def _subject_placeholder(ctx, background):
        ctx.output_frame = np.empty_like(ctx.linear_rgb)
        ctx.output_frame[:] = background if background is not None else 1.0


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
