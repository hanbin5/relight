"""Opt-in local BiRefNet portrait adapter for shutter captures only."""

import hashlib
import importlib.util
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .presentation_modnet import (
    composite_dense,
    estimate_wall_color,
    estimate_walls,
)

MODEL_SHA256 = "1ba1c8ff5a7bbfadc8d8d13fb11d7be793f91f23d9d466549e37a854f6668f99"
MODEL_SIZE = (1024, 1024)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class BirefNetPrediction:
    alpha: np.ndarray
    stats: dict
    mask_u8: np.ndarray


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def birefnet_portrait_factory():
    model_path = os.environ.get("IDPHOTO_BIREFNET_MODEL")
    if not model_path:
        raise RuntimeError("Set IDPHOTO_BIREFNET_MODEL to the verified local ONNX path.")
    return BirefNetPortrait(model_path)


def check_availability(model_path, *, verify_hash=True):
    if not model_path:
        raise ValueError("Provide a local BiRefNet portrait ONNX model path.")
    if "OMP_NUM_THREADS" in os.environ:
        raise ValueError(
            "BiRefNet portrait requires OMP_NUM_THREADS to be unset to match the "
            "frozen rembg CPU contract."
        )
    if importlib.util.find_spec("onnxruntime") is None:
        raise ImportError(
            "BiRefNet portrait requires onnxruntime; install the optional extra with "
            "uv sync --extra parsing"
        )
    if importlib.util.find_spec("PIL") is None:
        raise ImportError(
            "BiRefNet portrait requires Pillow; install the optional extra with "
            "uv sync --extra parsing"
        )
    path = Path(model_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"BiRefNet portrait model not found: {path}")
    if verify_hash:
        digest = _sha256(path)
        if digest != MODEL_SHA256:
            raise ValueError(f"BiRefNet portrait model SHA-256 mismatch: {digest}")
    return path


class BirefNetPortrait:
    """Reproduce rembg 2.0.88 only_mask preprocessing and uint8 mask output."""

    def __init__(self, model_path):
        self.model_path = check_availability(model_path)
        self.session = None
        self.input_name = None
        self.versions = None
        self.session_create_ms = None
        self.inference_count = 0

    def _ensure_session(self):
        if self.session is not None:
            return
        import PIL
        import onnxruntime

        started = time.perf_counter()
        options = onnxruntime.SessionOptions()
        session = onnxruntime.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        providers = session.get_providers()
        if providers != ["CPUExecutionProvider"]:
            raise RuntimeError(
                f"BiRefNet portrait requires CPUExecutionProvider only; got {providers}."
            )
        inputs = session.get_inputs()
        if len(inputs) != 1:
            raise RuntimeError(f"Unexpected BiRefNet portrait input count: {len(inputs)}")
        self.session = session
        self.input_name = inputs[0].name
        self.versions = {
            "onnxruntime": onnxruntime.__version__,
            "pillow": PIL.__version__,
            "numpy": np.__version__,
        }
        self.session_create_ms = (time.perf_counter() - started) * 1000

    @staticmethod
    def _normalize(raw_rgb):
        from PIL import Image

        height, width = raw_rgb.shape[:2]
        resized = Image.fromarray(raw_rgb).convert("RGB").resize(
            MODEL_SIZE, Image.Resampling.LANCZOS
        )
        image = np.asarray(resized)
        image = image / max(np.max(image), 1e-6)
        normalized = np.zeros((image.shape[0], image.shape[1], 3))
        for channel in range(3):
            normalized[:, :, channel] = (
                image[:, :, channel] - MEAN[channel]
            ) / STD[channel]
        tensor = np.expand_dims(normalized.transpose((2, 0, 1)), 0).astype(np.float32)
        return tensor, (height, width)

    def predict_alpha(self, raw_rgb, *, capture=True):
        if raw_rgb.dtype != np.uint8 or raw_rgb.ndim != 3 or raw_rgb.shape[2] != 3:
            raise ValueError("BiRefNet portrait input must be an H×W×3 uint8 RGB frame.")
        if not capture:
            raise ValueError("BiRefNet portrait is available only for shutter captures.")
        predict_started = time.perf_counter()
        self._ensure_session()
        preprocess_started = time.perf_counter()
        tensor, (height, width) = self._normalize(raw_rgb)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000
        started = time.perf_counter()
        output = self.session.run(None, {self.input_name: tensor})
        inference_ms = (time.perf_counter() - started) * 1000
        self.inference_count = getattr(self, "inference_count", 0) + 1
        postprocess_started = time.perf_counter()
        if not output or output[0].ndim != 4 or output[0].shape[:2] != (1, 1):
            raise RuntimeError("BiRefNet portrait returned an invalid output tensor.")
        logits = output[0][:, 0, :, :]
        if not np.isfinite(logits).all():
            raise RuntimeError("BiRefNet portrait returned non-finite logits.")
        prediction = 1 / (1 + np.exp(-logits))
        maximum, minimum = np.max(prediction), np.min(prediction)
        if not maximum > minimum:
            raise RuntimeError("BiRefNet portrait returned a constant mask.")
        prediction = np.squeeze((prediction - minimum) / (maximum - minimum))
        mask = (prediction * 255).astype(np.uint8)
        from PIL import Image

        mask = Image.fromarray(mask, mode="L").resize(
            (width, height), Image.Resampling.LANCZOS
        )
        mask_u8 = np.asarray(mask, dtype=np.uint8)
        mask_u8.setflags(write=False)
        alpha = mask_u8.astype(np.float32) / np.float32(255.0)
        alpha.setflags(write=False)
        alpha_postprocess_ms = (time.perf_counter() - postprocess_started) * 1000
        return BirefNetPrediction(alpha, {
            "inference_ms": inference_ms,
            "session_create_ms": getattr(self, "session_create_ms", None),
            "preprocess_ms": preprocess_ms,
            "alpha_postprocess_ms": alpha_postprocess_ms,
            "total_predict_ms": (time.perf_counter() - predict_started) * 1000,
            "inference_count": self.inference_count,
            "input_width": MODEL_SIZE[0],
            "input_height": MODEL_SIZE[1],
            "provider": "CPUExecutionProvider",
            "mask_dtype": str(mask_u8.dtype),
            **self.versions,
        }, mask_u8)

    @staticmethod
    def apply_prediction(ctx, prediction, background):
        ctx.alpha = prediction.alpha
        ctx.stats["birefnet_portrait"] = dict(prediction.stats)
        if background is None:
            ctx.wall_color = estimate_wall_color(ctx.linear_rgb, ctx.alpha)
            ctx.output_frame = ctx.relit_frame
            return
        ctx.wall_color, wall = estimate_walls(ctx.linear_rgb, ctx.alpha)
        ctx.output_frame = composite_dense(ctx, wall, background)
