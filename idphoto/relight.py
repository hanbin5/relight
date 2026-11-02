"""Module A: real-time relighting from the face landmark mesh (plan B.2, Figure 3).

Every frame:
1. rasterize the landmark mesh on a FIT_SIZE x FIT_SIZE face crop to get a normal n per pixel;
2. fit the room light with nine spherical-harmonic terms H(n) by least squares over skin
   pixels (eyes, brows, lips and clipped pixels left out):  l_hat = argmin sum (Y - l^T H(n))^2;
3. average l_hat over the last 5 frames;
4. scale each pixel by clip_[0.5, 2](l_t^T H(n) / (l_hat^T H(n) + eps)), where the target l_t is
   even, frontal and slightly from above, has no left-right term and keeps the mean face
   brightness. Outside the mesh the ratio fades to its face mean; the ratio is upsampled to
   the frame with the guided filter, and a strength s in [0, 1] blends it toward 1.

Only a smooth shading ratio changes, so skin texture and identity stay. Cast shadows (the
nose's) are not modeled by nine SH terms; the checker flags them (rule 1).
"""

from collections import deque

import numpy as np

from . import config, mesh, npimg
from .context import FrameContext

# Real SH basis constants (Ramamoorthi & Hanrahan 2001).
_C0, _C1, _C2, _C3, _C4 = 0.282095, 0.488603, 1.092548, 0.315392, 0.546274
# Basis order: 1, y, z, x, xy, yz, 3z^2 - 1, xz, x^2 - y^2 (x right, y up, z to the camera).
LEFT_RIGHT_TERMS = (3, 4, 7)  # odd in x
SECOND_ORDER = slice(4, 9)


def sh_basis(n: np.ndarray) -> np.ndarray:
    """H(n): (..., 3) unit normals -> (..., 9) basis values."""
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    return np.stack(
        [np.full_like(x, _C0), _C1 * y, _C1 * z, _C1 * x, _C2 * x * y, _C2 * y * z,
         _C3 * (3.0 * z * z - 1.0), _C2 * x * z, _C4 * (x * x - y * y)],
        axis=-1,
    )


_HEMISPHERE = None


def _hemisphere():
    """Normals spread evenly over the visible hemisphere, and the pseudo-inverse of their basis."""
    global _HEMISPHERE
    if _HEMISPHERE is None:
        k = np.arange(4000) + 0.5
        z = k / 4000.0  # uniform area on the hemisphere z >= 0
        phi = np.pi * (1.0 + 5.0**0.5) * k
        rho = np.sqrt(1.0 - z * z)
        n = np.stack([rho * np.cos(phi), rho * np.sin(phi), z], axis=-1)
        _HEMISPHERE = (n, np.linalg.pinv(sh_basis(n)))
    return _HEMISPHERE


def target_light(elevation_deg: float = config.TARGET_ELEVATION_DEG,
                 ambient: float = config.TARGET_AMBIENT, yaw_deg: float = 0.0) -> np.ndarray:
    """SH coefficients of the target: a key light in front of the face, raised by
    elevation_deg, plus a soft fill.

    The key light turns with the head's yaw, so both cheeks get the same light even when the
    head is not square to the camera (with a camera-fixed light, the cheek facing the camera
    gets more and R_LR drifts with the head). The shading ambient + max(0, n . d) is projected
    on the nine terms over the visible hemisphere; at yaw 0 the left-right terms are exactly zero.
    """
    n, pinv = _hemisphere()
    e, y = np.radians(elevation_deg), np.radians(yaw_deg)
    d = np.array([np.sin(y) * np.cos(e), np.sin(e), np.cos(y) * np.cos(e)])
    coef = pinv @ (ambient + np.maximum(n @ d, 0.0))
    if yaw_deg == 0.0:
        coef[list(LEFT_RIGHT_TERMS)] = 0.0
    return coef.astype(np.float32)


def face_yaw(vertex_normals: np.ndarray) -> float:
    """Yaw of the face in degrees (+ turned toward the image right), from its mean normal."""
    m = vertex_normals.mean(axis=0)
    return float(np.degrees(np.arctan2(m[0], m[2])))


def fit_light(Y: np.ndarray, normals: np.ndarray, skin: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Least-squares SH light over skin pixels, with outlier rejection (hair, beard, glasses)."""
    H = sh_basis(normals[skin]).astype(np.float64)
    y = Y[skin].astype(np.float64)
    ridge = np.zeros(9)
    ridge[SECOND_ORDER] = config.FIT_RIDGE * len(y)
    w = np.ones_like(y)
    for it in range(iterations + 1):
        Hw = H * w[:, None]
        coef = np.linalg.solve(H.T @ Hw + np.diag(ridge), Hw.T @ y)
        if it == iterations:
            break
        res = y - H @ coef
        spread = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-6
        w = (np.abs(res) < 2.5 * spread).astype(np.float64)
    return coef.astype(np.float32)


class Relighter:
    """Stateful per-frame relighting: keeps the 5-frame light average."""

    def __init__(self, fit_size: int = config.FIT_SIZE, smooth_frames: int = config.SMOOTH_FRAMES):
        self.fit_size = fit_size
        self.history: deque = deque(maxlen=smooth_frames)
        self.yaws: deque = deque(maxlen=smooth_frames)  # the target follows the averaged yaw
        self.frames_lost = 0

    def set_smoothing(self, frames: int):
        self.history = deque(self.history, maxlen=frames)
        self.yaws = deque(self.yaws, maxlen=frames)

    def reset(self):
        self.history.clear()
        self.yaws.clear()

    @property
    def light(self):
        return np.mean(self.history, axis=0).astype(np.float32) if self.history else None

    def process(self, ctx: FrameContext, strength: float = config.DEFAULT_STRENGTH,
                light: np.ndarray | None = None, update: bool = True,
                target_yaw: float | None = None) -> FrameContext:
        """Fill normals, face_mask, light_coeffs, relit_frame and ratio_map of ctx.

        light and target_yaw: use these instead of fitting and measuring (the shutter frame
        reuses the live averages); update=False leaves the running averages untouched.
        """
        h, w = ctx.Y.shape
        ctx.face_mask = np.zeros((h, w), bool)
        if ctx.landmarks is None:
            self._lost()
            ctx.relit_frame, ctx.ratio_map = ctx.linear_rgb, np.ones((h, w), np.float32)
            return ctx

        S = self.fit_size
        lm = ctx.landmarks
        box = mesh.face_box(lm)
        vertex_normals = mesh.vertex_normals(lm)
        normals, cov = mesh.mesh_normals(lm, box, S, vertex_normals)
        regions = mesh.region_masks(mesh.to_grid(lm, box, S), (S, S))
        Yc = npimg.resize(ctx.Y, S, S, kernel="linear", box=box)
        core = npimg.erode(cov, 1)
        lo, hi = config.FIT_Y_RANGE
        skin = core & ~regions["features"] & (Yc > lo) & (Yc < hi)
        if skin.sum() < 200:
            self._lost()
            ctx.relit_frame, ctx.ratio_map = ctx.linear_rgb, np.ones((h, w), np.float32)
            return ctx

        if light is None:
            coef = fit_light(Yc, normals, skin)
            if update:
                self.history.append(coef)
                self.frames_lost = 0
            light = self.light if update else coef
        light = np.asarray(light, np.float32)
        if target_yaw is None:
            yaw = face_yaw(vertex_normals)
            if update:
                self.yaws.append(yaw)
            target_yaw = float(np.mean(self.yaws)) if update else yaw

        # Shading ratio on the face grid.
        Hn = sh_basis(normals[cov])
        fitted = Hn @ light
        target = Hn @ target_light(yaw_deg=target_yaw)
        on_skin = skin[cov]
        mean_fit = max(float(fitted[on_skin].mean()), 1e-4)
        target *= mean_fit / max(float(target[on_skin].mean()), 1e-6)  # keep mean brightness
        r = np.clip(target / np.maximum(fitted, 0.02 * mean_fit), *config.RATIO_CLIP)
        r = 1.0 + strength * (r - 1.0)
        # Face mean of the ratio, weighted by brightness so hair, neck and shoulders keep the
        # exposure the face keeps (an unweighted mean is pulled up by the brightened shadows).
        y_skin = Yc[cov][on_skin]
        face_mean = float((r[on_skin] * y_skin).sum() / max(float(y_skin.sum()), 1e-6))
        grid = np.full((S, S), face_mean, np.float32)
        grid[cov] = r
        fade = npimg.box_mean(core.astype(np.float32), 2)
        grid = face_mean + fade * (grid - face_mean)

        # Guided upsampling: coefficients on the grid, bilinear to the frame, a * guide + b.
        a, b = npimg.guided_filter_coeffs(np.sqrt(Yc), grid, config.RATIO_GF_RADIUS, config.RATIO_GF_EPS)
        fx0, fy0 = max(int(np.floor(box[0])), 0), max(int(np.floor(box[1])), 0)
        fx1, fy1 = min(int(np.ceil(box[2])), w), min(int(np.ceil(box[3])), h)
        ratio_map = np.full((h, w), face_mean, np.float32)
        if fx1 > fx0 and fy1 > fy0:
            g = S / (box[2] - box[0])  # grid pixels per frame pixel
            on_grid = ((fx0 - box[0]) * g, (fy0 - box[1]) * g, (fx1 - box[0]) * g, (fy1 - box[1]) * g)
            ab = npimg.resize(np.stack([a, b], axis=-1), fy1 - fy0, fx1 - fx0, "linear", box=on_grid)
            guide = np.sqrt(ctx.Y[fy0:fy1, fx0:fx1])
            lo_r, hi_r = 1.0 + strength * (np.array(config.RATIO_CLIP) - 1.0)
            ratio_map[fy0:fy1, fx0:fx1] = np.clip(ab[..., 0] * guide + ab[..., 1], lo_r, hi_r)
            gy = np.clip(((np.arange(fy0, fy1) + 0.5 - box[1]) * g).astype(np.int64), 0, S - 1)
            gx = np.clip(((np.arange(fx0, fx1) + 0.5 - box[0]) * g).astype(np.int64), 0, S - 1)
            ctx.face_mask[fy0:fy1, fx0:fx1] = cov[np.ix_(gy, gx)]

        ctx.normals, ctx.face_box, ctx.light_coeffs = normals, box, light
        ctx.ratio_map = ratio_map
        ctx.relit_frame = np.minimum(ctx.linear_rgb * ratio_map[..., None], 1.0)

        relit_grid = Yc * grid
        fitted_grid = np.zeros((S, S), np.float32)
        target_grid = np.zeros((S, S), np.float32)
        fitted_grid[cov], target_grid[cov] = fitted, target
        ctx.stats.update(
            r_lr_before=mesh.cheek_balance(Yc, regions, skin),
            r_lr_after=mesh.cheek_balance(relit_grid, regions, skin),
            shadow_before=mesh.shadow_share(Yc, skin, config.SHADOW_FACTOR),
            shadow_after=mesh.shadow_share(relit_grid, skin, config.SHADOW_FACTOR),
            target_yaw=target_yaw,
            fit_grid=dict(Y=Yc, skin=skin, coverage=cov, fitted=fitted_grid, target=target_grid, ratio=grid),
        )
        return ctx

    def _lost(self):
        self.frames_lost += 1
        if self.frames_lost > config.KEEP_FIT_FRAMES:
            self.history.clear()
