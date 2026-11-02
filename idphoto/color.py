"""Color conversions in NumPy: sRGB <-> linear, luminance, saturation (B.2)."""

import numpy as np

_DECODE_LUT = None
_ENCODE_LUT = None
_ENCODE_STEPS = 16384


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """Undo the sRGB gamma on values in [0, 1]."""
    x = np.asarray(x, np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Apply the sRGB gamma to linear values; clips to [0, 1]."""
    x = np.clip(np.asarray(x, np.float32), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055).astype(np.float32)


def decode_u8(img: np.ndarray) -> np.ndarray:
    """uint8 sRGB image -> float32 linear RGB in [0, 1] (table lookup)."""
    global _DECODE_LUT
    if _DECODE_LUT is None:
        _DECODE_LUT = srgb_to_linear(np.arange(256, dtype=np.float32) / 255.0)
    return _DECODE_LUT[img]


def encode_u8(lin: np.ndarray) -> np.ndarray:
    """float32 linear RGB -> uint8 sRGB (table lookup on a fine linear grid)."""
    global _ENCODE_LUT
    if _ENCODE_LUT is None:
        grid = np.linspace(0.0, 1.0, _ENCODE_STEPS, dtype=np.float32)
        _ENCODE_LUT = np.round(linear_to_srgb(grid) * 255.0).astype(np.uint8)
    idx = np.clip(lin, 0.0, 1.0) * (_ENCODE_STEPS - 1) + 0.5
    return _ENCODE_LUT[idx.astype(np.int32)]


def luminance(lin: np.ndarray) -> np.ndarray:
    """Y = 0.2126 R + 0.7152 G + 0.0722 B on linear RGB."""
    return (lin[..., 0] * 0.2126 + lin[..., 1] * 0.7152 + lin[..., 2] * 0.0722).astype(np.float32)


def gray_u8(rgb_u8: np.ndarray) -> np.ndarray:
    """8-bit-scale gray of an sRGB image (Rec. 601 weights, as in common blur checks)."""
    rgb = rgb_u8.astype(np.float32)
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def hsv_saturation(rgb: np.ndarray) -> np.ndarray:
    """HSV saturation (max - min) / max of RGB values (any scale)."""
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    return np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0).astype(np.float32)


def hex_to_linear(hex_color: str) -> np.ndarray:
    """'#rrggbb' -> linear RGB float32 (3,)."""
    h = hex_color.lstrip("#")
    srgb = np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0
    return srgb_to_linear(srgb)
