"""The shared data format: one FrameContext passed along every frame (plan B.6).

Fields follow B.6; the letter after each field is its producing module. Fields added
beyond B.6 are marked "added" and documented here so all four modules agree on them.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FrameContext:
    raw_frame: np.ndarray  # A: uint8 (H, W, 3) RGB, exposure-locked camera frame
    linear_rgb: np.ndarray | None = None  # A: float32 (H, W, 3) linear color in [0, 1]
    Y: np.ndarray | None = None  # A: float32 (H, W) linear luminance
    landmarks: np.ndarray | None = None  # C: float32 (478, 3) face and iris points in pixels
    # A: float32 (S, S, 3) mesh normals. B.6 lists (H, W, 3); normals are only needed on
    # the FIT_SIZE face crop the light is fitted on, so they live on that grid (face_box).
    normals: np.ndarray | None = None
    face_mask: np.ndarray | None = None  # A: bool (H, W) face pixels, sure foreground for B
    light_coeffs: np.ndarray | None = None  # A: float32 (9,) fitted SH light, 5-frame average
    relit_frame: np.ndarray | None = None  # A: float32 (H, W, 3) linear frame, relit
    alpha: np.ndarray | None = None  # B: float32 (H, W) foreground mask in [0, 1]
    output_frame: np.ndarray | None = None  # B: float32 (H, W, 3) relit frame on the new background
    rule_verdicts: dict | None = None  # D: rule number -> Verdict
    geometry: dict | None = None  # C: crown, chin, eye center, pose angles, crop box

    timestamp_ms: int = 0  # added: capture time
    # added: the camera frame at full resolution when raw_frame is a downscaled live copy;
    # rule 3 measures sharpness on it so the 256 x 256 face crop is not upsampled
    full_frame: np.ndarray | None = None
    face_box: tuple | None = None  # added (A): (x0, y0, x1, y1) square crop the normals live on
    ratio_map: np.ndarray | None = None  # added (A): float32 (H, W) relighting ratio, used by B's spill removal
    wall_color: np.ndarray | None = None  # added (B): float32 (3,) linear wall color mu_B
    stats: dict = field(default_factory=dict)  # added: per-module measurements and timings

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) of the frame."""
        return self.raw_frame.shape[1], self.raw_frame.shape[0]
