"""Compatibility re-exports for the app's packaged MODNet adapter."""

from idphoto.presentation_modnet import (
    CAPTURE_WIDTH,
    CPU_THREADS,
    LIVE_WIDTH,
    MODEL_SHA256,
    ModnetPrediction,
    PresentationMODNet,
    check_availability,
    composite_dense,
    estimate_local_wall,
    estimate_wall_color,
    estimate_walls,
    model_input_size,
)

__all__ = [
    "CAPTURE_WIDTH",
    "CPU_THREADS",
    "LIVE_WIDTH",
    "MODEL_SHA256",
    "ModnetPrediction",
    "PresentationMODNet",
    "check_availability",
    "composite_dense",
    "estimate_local_wall",
    "estimate_wall_color",
    "estimate_walls",
    "model_input_size",
]
