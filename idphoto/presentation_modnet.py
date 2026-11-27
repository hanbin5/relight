"""Opt-in local MODNet adapter; the model file is never bundled with the app."""

import hashlib
import importlib.util
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from idphoto import config, matting

MODEL_SHA256 = "07c308cf0fc7e6e8b2065a12ed7fc07e1de8febb7dc7839d7b7f15dd66584df9"
LIVE_WIDTH = 512
CAPTURE_WIDTH = 512
CPU_THREADS = 8


@dataclass(frozen=True)
class ModnetPrediction:
    alpha: np.ndarray
    stats: dict


def check_availability(model_path, provider="cpu"):
    if provider == "coreml":
        raise ValueError(
            "Dynamic CoreML is disabled because its alpha parity is unreliable; "
            "use coreml-static or cpu."
        )
    if provider not in ("cpu", "coreml-static"):
        raise ValueError(f"Unsupported MODNet execution provider: {provider}")
    if importlib.util.find_spec("onnxruntime") is None:
        raise ImportError("MODNet requires the optional extra: uv sync --extra parsing --group dev")
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"MODNet model not found: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        raise ValueError(f"MODNet model SHA-256 mismatch: {digest}")
    if provider == "coreml-static":
        import onnxruntime

        if "CoreMLExecutionProvider" not in onnxruntime.get_available_providers():
            raise ValueError("ONNX Runtime CoreML provider is unavailable on this system.")
    return path


def model_input_size(height, width, width_limit):
    width_limit = min(width, width_limit)
    model_width = max(32, width_limit // 32 * 32)
    model_height = max(32, round(height * model_width / width / 32) * 32)
    return model_width, model_height


class PresentationMODNet:
    def __init__(self, model_path, provider="cpu", live_width=LIVE_WIDTH,
                 capture_width=CAPTURE_WIDTH):
        path = check_availability(model_path, provider)
        self.provider = provider
        self.model_path = path
        self.session = None
        self.static_sessions = {}
        self.input_names = {}
        self.prepared_grids = set()
        self.live_width = live_width
        self.capture_width = capture_width

    def _session_for_grid(self, model_width, model_height):
        import onnxruntime

        if self.provider != "coreml-static":
            if self.session is None:
                options = onnxruntime.SessionOptions()
                options.intra_op_num_threads = CPU_THREADS
                options.inter_op_num_threads = 1
                self.session = onnxruntime.InferenceSession(
                    str(self.model_path), sess_options=options,
                    providers=["CPUExecutionProvider"],
                )
                self.input_names["dynamic"] = self.session.get_inputs()[0].name
            return self.session, self.input_names["dynamic"], self.provider

        key = (model_width, model_height)
        if key not in self.static_sessions:
            self.static_sessions.clear()
            self.input_names.clear()
            self.prepared_grids.clear()
            probe = onnxruntime.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
            dims = probe.get_inputs()[0].shape
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = CPU_THREADS
            options.inter_op_num_threads = 1
            values = {0: 1, 2: model_height, 3: model_width}
            for axis, name in enumerate(dims):
                if isinstance(name, str):
                    options.add_free_dimension_override_by_name(name, values[axis])
                elif name is None:
                    raise ValueError("Unnamed dynamic MODNet input axis; static CoreML grid unavailable")
            cache_dir = (
                Path.home() / "Library" / "Caches" / "relight" / "modnet" / MODEL_SHA256
                / f"{model_width}x{model_height}"
                / f"coreml-MLProgram-CPUAndGPU-ort-{onnxruntime.__version__}"
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            provider_options = {
                "ModelFormat": "MLProgram",
                "MLComputeUnits": "CPUAndGPU",
                "RequireStaticInputShapes": "1",
                "EnableOnSubgraphs": "0",
                "ModelCacheDirectory": str(cache_dir),
            }
            session = onnxruntime.InferenceSession(
                str(self.model_path), sess_options=options,
                providers=[("CoreMLExecutionProvider", provider_options), "CPUExecutionProvider"],
            )
            if "CoreMLExecutionProvider" not in session.get_providers():
                raise RuntimeError(
                    "CoreML was requested for MODNet, but ONNX Runtime did not activate "
                    "CoreMLExecutionProvider."
                )
            self.static_sessions[key] = session
            self.input_names[key] = session.get_inputs()[0].name
        return self.static_sessions[key], self.input_names[key], "coreml-static"

    def prepare_input(self, raw_rgb, *, capture=False):
        height, width = raw_rgb.shape[:2]
        model_width, model_height = model_input_size(
            height, width, self.capture_width if capture else self.live_width
        )
        key = (model_width, model_height)
        if key in self.prepared_grids:
            return True
        session, input_name, _ = self._session_for_grid(*key)
        dummy = np.zeros((1, 3, model_height, model_width), np.float32)
        output = session.run(None, {input_name: dummy})[0]
        if output.ndim != 4 or output.shape[:2] != (1, 1) \
                or not np.isfinite(output).all():
            raise RuntimeError("MODNet warm-up returned an invalid matte")
        self.prepared_grids.add(key)
        return False

    def predict_alpha(self, raw_rgb, *, capture=False):
        height, width = raw_rgb.shape[:2]
        model_width, model_height = model_input_size(
            height, width, self.capture_width if capture else self.live_width
        )
        resized = cv2.resize(raw_rgb, (model_width, model_height),
                              interpolation=cv2.INTER_AREA)
        tensor = np.transpose(
            (resized.astype(np.float32) / 255.0 - 0.5) / 0.5, (2, 0, 1)
        )[None]
        self.prepare_input(raw_rgb, capture=capture)
        inference_started_ms = time.monotonic() * 1000
        started = time.perf_counter()
        session, input_name, used_provider = self._session_for_grid(model_width, model_height)
        prediction = session.run(None, {input_name: tensor})[0][0, 0]
        inference_ms = (time.perf_counter() - started) * 1000
        inference_finished_ms = time.monotonic() * 1000
        alpha = cv2.resize(prediction, (width, height), interpolation=cv2.INTER_LINEAR)
        alpha = np.clip(alpha, 0, 1).astype(np.float32)
        alpha.setflags(write=False)
        return ModnetPrediction(alpha, {
            "inference_ms": inference_ms,
            "inference_started_ms": inference_started_ms,
            "inference_finished_ms": inference_finished_ms,
            "input_width": model_width,
            "input_height": model_height,
            "capture": capture,
            "provider": used_provider,
            "provider_backends": session.get_providers(),
        })

    @staticmethod
    def apply_prediction(ctx, prediction, background):
        ctx.alpha = prediction.alpha
        ctx.stats["modnet"] = dict(prediction.stats)
        wall_alpha = ctx.all_person_alpha if ctx.all_person_alpha is not None else ctx.alpha
        if background is None:
            ctx.wall_color = estimate_wall_color(ctx.linear_rgb, wall_alpha)
            if len(ctx.subject_faces) <= 1:
                ctx.output_frame = ctx.relit_frame
                return
            background = ctx.wall_color
        wall_estimate_started = time.perf_counter()
        ctx.wall_color, wall = estimate_walls_live(ctx.linear_rgb, wall_alpha)
        ctx.stats["modnet"]["wall_estimate_ms"] = \
            (time.perf_counter() - wall_estimate_started) * 1000
        composite_started = time.perf_counter()
        ctx.output_frame = composite_dense(ctx, wall, background)
        ctx.stats["modnet"]["composite_ms"] = (time.perf_counter() - composite_started) * 1000

    def __call__(self, ctx, background, *, capture=False):
        self.apply_prediction(ctx, self.predict_alpha(ctx.raw_frame, capture=capture), background)


def estimate_wall_color(linear_rgb, alpha):
    """Return a stable scalar fallback from pixels confidently classified as background."""
    background = linear_rgb[alpha <= 0.02]
    if not len(background):
        background = linear_rgb.reshape(-1, 3)
    return background.mean(axis=0).astype(np.float32)


def estimate_local_wall(linear_rgb, alpha, fallback=None):
    """Estimate a spatial wall field from nearby confident background pixels."""
    confident_background = (alpha <= 0.02).astype(np.float32)
    kernel = (19, 19)
    count = cv2.boxFilter(confident_background, -1, kernel, normalize=False)
    total = cv2.boxFilter(
        linear_rgb * confident_background[..., None], -1, kernel, normalize=False
    )
    if fallback is None:
        fallback = estimate_wall_color(linear_rgb, alpha)
    return np.where(
        count[..., None] > 0,
        total / np.maximum(count[..., None], 1e-6),
        np.asarray(fallback, np.float32),
    ).astype(np.float32)


def estimate_walls(linear_rgb, alpha):
    """Estimate global and local wall colors with one masked multi-channel box filter."""
    confident_background = (alpha <= 0.02).astype(np.float32)
    stacked = np.empty(alpha.shape + (4,), np.float32)
    np.multiply(linear_rgb, confident_background[..., None], out=stacked[..., :3])
    stacked[..., 3] = confident_background
    totals = cv2.sumElems(stacked)
    if totals[3] > 0:
        global_wall = np.asarray(totals[:3], np.float64) / totals[3]
    else:
        global_wall = np.asarray(
            cv2.sumElems(np.ascontiguousarray(linear_rgb))[:3], np.float64
        ) / alpha.size
    global_wall = global_wall.astype(np.float32)
    sums = cv2.boxFilter(stacked, -1, (19, 19), normalize=False)
    counts = sums[..., 3]
    local_wall = np.divide(sums[..., :3], np.maximum(counts[..., None], 1e-6))
    local_wall[counts <= 0] = global_wall
    return global_wall, local_wall.astype(np.float32)


def estimate_walls_live(linear_rgb, alpha):
    """Estimate preview walls with the exact-output packed OpenCV path."""
    clear = alpha <= 0.02
    if not clear.any():
        mean = np.asarray(
            cv2.sumElems(np.ascontiguousarray(linear_rgb))[:3], np.float64
        ) / alpha.size
        global_wall = mean.astype(np.float32)
        local = np.empty(linear_rgb.shape, np.float32)
        local[...] = global_wall
        return global_wall, local

    scratch = cv2.cvtColor(linear_rgb, cv2.COLOR_RGB2RGBA)
    packed = cv2.copyTo(scratch, clear.view(np.uint8))
    totals = cv2.sumElems(packed)
    global_wall = (np.asarray(totals[:3], np.float64) / totals[3]).astype(np.float32)
    cv2.boxFilter(packed, -1, (19, 19), dst=scratch, normalize=False)
    del packed
    r, g, b, count = cv2.split(scratch)
    del scratch
    missing = count == 0
    np.maximum(count, 1, out=count)
    for channel, fallback in zip((r, g, b), global_wall):
        np.divide(channel, count, out=channel)
        np.copyto(channel, fallback, where=missing)
    return global_wall, cv2.merge((r, g, b))


def composite_dense(ctx, wall, background):
    """Apply the standard spill-removal equation without sparse gathers for dense alpha."""
    alpha = ctx.alpha
    inverse = 1.0 - alpha
    ratio = ctx.ratio_map if ctx.ratio_map is not None else 1.0
    subtraction = inverse * ratio
    denominator = np.maximum(alpha, config.SPILL_ALPHA_FLOOR)
    scale = np.minimum(alpha / config.SPILL_ALPHA_FLOOR, 1.0)
    bg = np.asarray(background, np.float32)
    relit_channels = cv2.split(ctx.relit_frame)
    wall_array = np.asarray(wall, np.float32)
    wall_channels = cv2.split(wall_array) if wall_array.ndim == 3 else None
    output = []
    for index in range(3):
        if wall_channels is None:
            spill = cv2.scaleAdd(
                subtraction, -float(wall_array[index]), relit_channels[index]
            )
        else:
            spill = cv2.subtract(
                relit_channels[index],
                cv2.multiply(subtraction, wall_channels[index]),
            )
        foreground = cv2.multiply(cv2.min(cv2.max(spill, 0.0), denominator), scale)
        output.append(cv2.scaleAdd(inverse, float(bg[index]), foreground))
    return cv2.merge(output)
