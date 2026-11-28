"""Module A: real-time relighting from the face landmark mesh (plan B.2, Figure 3).

Every frame:
1. rasterize the landmark mesh on a FIT_SIZE x FIT_SIZE face crop to get a normal n per pixel;
2. fit the room light with nine spherical-harmonic terms H(n) by least squares over skin
   pixels (eyes, brows, lips and clipped pixels left out):  l_hat = argmin sum (Y - l^T H(n))^2;
3. average l_hat over the last 5 frames;
4. scale each pixel by a soft bound of two factors: sqrt(Ys(p') / Ys(p)), which evens each
   point p with its mirror point p' on the other side of the face (Ys: widely averaged skin
   brightness), and target(n) / even(l_hat^T H(n)), which turns the fitted light, made even
   across the midline, into a soft studio light (frontal, slightly from above, wide fill)
   with the same mean face brightness. The left-right part is measured again on the relit
   grid and refined (MIRROR_PASSES). Per-channel gains likewise move each point's skin color
   halfway to its mirror point's (ctx.color_gain). Outside the mesh the ratio fades to its face
   mean; it is smoothed on the grid (light varies slowly), upsampled to the frame with the
   guided filter, and a strength s in [0, 1] blends it toward 1.

Only a smooth shading ratio changes, so skin texture and identity stay. Cast shadows (the
nose's) are not modeled by nine SH terms; the checker flags them (rule 1).
"""

from collections import deque

import numpy as np

from . import color, config, mesh, npimg
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


def mirror_normals(n: np.ndarray, yaw_deg: float) -> np.ndarray:
    """Reflect normals through the face's midplane: the plane holding the up axis and the
    direction the face points (yaw_deg), so a point on one cheek maps to the other cheek."""
    y = np.radians(yaw_deg)
    m = np.array([np.cos(y), 0.0, -np.sin(y)], np.float32)  # unit normal of the midplane
    return n - 2.0 * (n @ m)[..., None] * m


def soft_clip(r: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Bound a ratio to (lo, hi) smoothly in the log domain: up to half the bound (in log) it is
    unchanged, beyond that it bends toward the bound instead of stopping at it, so no flat
    clipped patches appear and a strong side light still gets most of its correction."""
    t = np.log(np.maximum(r, 1e-6))
    out = t.copy()
    for sign, bound in ((1.0, np.log(hi)), (-1.0, -np.log(lo))):
        knee = 0.5 * bound
        x = sign * t
        over = x > knee
        out[over] = sign * (knee + knee * np.tanh((x[over] - knee) / knee))
    return np.exp(out).astype(np.float32)


def mirror_balance(Y: np.ndarray, smooth, sample, cov: np.ndarray) -> np.ndarray:
    """Left-right correction measured from the frame: sqrt(Ys(mirror p) / Ys(p)) at each covered
    grid pixel p, where Ys = smooth(Y) is the skin brightness averaged over a wide window and
    sample reads a grid at the mirror points (the same points of the face, other side).

    After it both points have the geometric mean of the two, so a side light is evened where
    the face actually is darker, whatever caused it (the light, the nose's cast shadow, a
    cheek's curvature). Nine SH terms could not follow a real side light closely: they
    corrected the area next to the nose most and the darker outer cheek least.
    """
    Ys = smooth(Y)
    gain = np.sqrt(np.maximum(sample(Ys), 1e-6) / np.maximum(Ys[cov], 1e-6)).astype(np.float32)
    gain[~_supported(smooth, sample, cov)] = 1.0
    return gain


def mirror_color(rgb: np.ndarray, smooth, sample, cov: np.ndarray) -> np.ndarray:
    """(N, 3) per-channel gains that move the widely averaged skin color at each covered grid
    pixel halfway (geometric mean) to the color at its mirror point, at equal luminance.

    Shadowed skin is redder than lit skin, and a highlight is paler; evening only the
    brightness left a brightened shadow orange and a darkened highlight gray.
    """
    Cs = smooth(rgb)
    q = Cs / np.maximum(color.luminance(Cs), 1e-6)[..., None]  # color at luminance 1
    qp = q[cov]
    target = np.sqrt(np.maximum(qp, 1e-6) * np.maximum(sample(q), 1e-6))
    target /= np.maximum(color.luminance(target), 1e-6)[..., None]
    gain = (target / np.maximum(qp, 1e-6)).astype(np.float32)
    gain[~_supported(smooth, sample, cov)] = 1.0
    return gain


def _supported(smooth, sample, cov: np.ndarray) -> np.ndarray:
    """(N,) True where both a covered grid pixel and its mirror point have enough skin around
    them to compare. A face cut by the frame edge, or a cheek under hair, leaves none on one
    side; its average is then empty, and reading it as a dark cheek drove the visible side
    toward the lower bound. Such points are left uncorrected."""
    s = smooth.support
    return (s[cov] >= config.MIRROR_MIN_SUPPORT) & (sample(s) >= config.MIRROR_MIN_SUPPORT)


def _smoother(weight: np.ndarray, radius: int):
    """f(a): a (S, S[, C]) averaged over the weight > 0 pixels by two box passes of radius."""
    w = weight.astype(np.float32)
    support = npimg.box_mean(npimg.box_mean(w, radius), radius)  # share of the window that counts
    den = np.maximum(support, 1e-6)

    def smooth(a):
        if a.ndim == 3:
            return npimg.box_mean(npimg.box_mean(a * w[..., None], radius), radius) / den[..., None]
        return npimg.box_mean(npimg.box_mean(a * w, radius), radius) / den
    smooth.support = support
    return smooth


def _sampler(xy: np.ndarray, size: int):
    """f(a): bilinear samples of a (size, size[, C]) at grid positions xy (N, 2), pixel centers
    at +0.5; the taps are computed once and reused for every grid."""
    x = np.clip(xy[:, 0] - 0.5, 0.0, size - 1.001)
    y = np.clip(xy[:, 1] - 0.5, 0.0, size - 1.001)
    x0, y0 = x.astype(np.int64), y.astype(np.int64)
    fx, fy = (x - x0).astype(np.float32), (y - y0).astype(np.float32)
    idx = np.stack([y0 * size + x0, y0 * size + x0 + 1, (y0 + 1) * size + x0, (y0 + 1) * size + x0 + 1])
    wts = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy])

    def sample(a):
        flat = a.reshape(size * size, -1)
        out = (flat[idx] * wts[..., None]).sum(axis=0)
        return out if a.ndim == 3 else out[:, 0]
    return sample


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
        A, b = H.T @ Hw + np.diag(ridge), Hw.T @ y
        try:
            coef = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:  # normals that do not vary: take the smallest solution
            coef = np.linalg.lstsq(A, b, rcond=None)[0]
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
        if not mesh.usable(ctx.landmarks):
            self._lost()
            ctx.relit_frame, ctx.ratio_map = ctx.linear_rgb, np.ones((h, w), np.float32)
            return ctx

        S = self.fit_size
        lm = ctx.landmarks
        box = mesh.face_box(lm)
        vertex_normals = mesh.vertex_normals(lm)
        normals, cov, raster = mesh.mesh_normals(lm, box, S, vertex_normals, with_raster=True)
        regions = mesh.region_masks(mesh.to_grid(lm, box, S), (S, S))
        Yc = npimg.resize(ctx.Y, S, S, kernel="linear", box=box)
        core = npimg.erode(cov, 1)
        lo, hi = config.FIT_Y_RANGE
        # Where the face box sticks out of the frame, resizing repeats the edge pixels; those
        # grid pixels are not skin and must not be fitted.
        centers = (np.arange(S) + 0.5) * (box[2] - box[0]) / S
        inside = ((box[1] + centers >= 0) & (box[1] + centers < h))[:, None] \
            & ((box[0] + centers >= 0) & (box[0] + centers < w))[None, :]
        skin = core & ~regions["features"] & inside & (Yc > lo) & (Yc < hi)
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

        # Shading ratio on the face grid, in two parts:
        # - left-right: measured from the frame, each point against its mirror point;
        # - shape: the fitted light made even across the midline, turned into the target
        #   (frontal, slightly from above, wide fill) with the same mean brightness.
        n_cov = normals[cov]
        H = sh_basis(n_cov)
        even_fit = 0.5 * (H @ light + sh_basis(mirror_normals(n_cov, target_yaw)) @ light)
        on_skin = skin[cov]
        mean_fit = max(float(even_fit[on_skin].mean()), 1e-4)
        target = H @ target_light(yaw_deg=target_yaw)
        target *= mean_fit / max(float(target[on_skin].mean()), 1e-6)  # keep mean brightness
        r_shape = target / np.maximum(even_fit, 0.02 * mean_fit)
        # Hair inside the face outline (bangs) is far darker than any shadowed skin.
        bright = Yc > config.MIRROR_MIN_SKIN * float(np.median(Yc[skin]))
        sample = _sampler(mesh.mirror_points(lm, box, S, raster), S)
        skin_smooth = _smoother(skin & bright, config.MIRROR_SMOOTH_RADIUS)
        face_smooth = _smoother(cov, config.RATIO_SMOOTH_RADIUS) if config.RATIO_SMOOTH_RADIUS > 0 else None
        y_skin = Yc[cov][on_skin]
        fade = npimg.box_mean(core.astype(np.float32), 2)
        shape = soft_clip(r_shape, *config.SHAPE_RATIO_CLIP)

        def to_grid(r_lr):
            """The full-strength ratio on the grid, and its face mean.

            Separate bounds: the shape part only reshapes an even light and stays gentle (its
            largest values sit on the face rim, where the mesh normals are least reliable);
            the left-right part may need a side light's full range. Light changes slowly over
            a face, so the ratio is smoothed (two box passes, over the face only: averaging in
            the face mean outside pulled the outer cheeks toward it). The face mean is
            weighted by brightness so hair, neck and shoulders keep the exposure the face
            keeps; outside the mesh the ratio fades to it.
            """
            r = shape * soft_clip(r_lr, *config.RATIO_CLIP)
            mean = float((r[on_skin] * y_skin).sum() / max(float(y_skin.sum()), 1e-6))
            g = np.full((S, S), mean, np.float32)
            g[cov] = r
            if face_smooth is not None:
                g[cov] = face_smooth(g)[cov]
            return mean + fade * (g - mean), mean

        # Each pass measures what is still uneven in the frame relit with the complete ratio
        # (shape, bounds and smoothing included), so the passes converge on even cheeks.
        r_lr = np.ones(int(cov.sum()), np.float32)
        for _ in range(config.MIRROR_PASSES):
            grid, _ = to_grid(r_lr)
            r_lr = r_lr * mirror_balance(Yc * grid, skin_smooth, sample, cov)
        grid, face_mean = to_grid(r_lr)
        # Everything after the bounds is linear in r, so the strength blends the grid itself.
        grid = 1.0 + strength * (grid - 1.0)
        face_mean = 1.0 + strength * (face_mean - 1.0)

        # Color: each point's skin color halfway to its mirror point's, at equal luminance,
        # smoothed and faded to no change outside the mesh like the ratio.
        cgrid = None
        if config.MIRROR_COLOR > 0 and strength > 0:  # at strength 0 the frame stays untouched
            Cc = npimg.resize(ctx.linear_rgb, S, S, kernel="linear", box=box)
            cg = mirror_color(Cc, skin_smooth, sample, cov)
            cg = soft_clip(cg, *config.COLOR_GAIN_CLIP)
            cg = 1.0 + config.MIRROR_COLOR * strength * (cg - 1.0)
            cgrid = np.ones((S, S, 3), np.float32)
            cgrid[cov] = cg
            if face_smooth is not None:
                cgrid = np.where(cov[..., None], face_smooth(cgrid), 1.0)
            cgrid = 1.0 + fade[..., None] * (cgrid - 1.0)

        # Guided upsampling: coefficients on the grid, bilinear to the frame, a * guide + b.
        a, b = npimg.guided_filter_coeffs(np.sqrt(Yc), grid, config.RATIO_GF_RADIUS, config.RATIO_GF_EPS)
        fx0, fy0 = max(int(np.floor(box[0])), 0), max(int(np.floor(box[1])), 0)
        fx1, fy1 = min(int(np.ceil(box[2])), w), min(int(np.ceil(box[3])), h)
        ratio_map = np.full((h, w), face_mean, np.float32)
        color_gain = None
        if fx1 > fx0 and fy1 > fy0:
            g = S / (box[2] - box[0])  # grid pixels per frame pixel
            on_grid = ((fx0 - box[0]) * g, (fy0 - box[1]) * g, (fx1 - box[0]) * g, (fy1 - box[1]) * g)
            ab = npimg.resize(np.stack([a, b], axis=-1), fy1 - fy0, fx1 - fx0, "linear", box=on_grid)
            guide = np.sqrt(ctx.Y[fy0:fy1, fx0:fx1])
            lo_r, hi_r = 1.0 + strength * (np.array(config.RATIO_CLIP) - 1.0)
            ratio_map[fy0:fy1, fx0:fx1] = np.clip(ab[..., 0] * guide + ab[..., 1], lo_r, hi_r)
            if cgrid is not None:
                color_gain = np.ones((h, w, 3), np.float32)
                color_gain[fy0:fy1, fx0:fx1] = npimg.resize(cgrid, fy1 - fy0, fx1 - fx0, "linear", box=on_grid)
            gy = np.clip(((np.arange(fy0, fy1) + 0.5 - box[1]) * g).astype(np.int64), 0, S - 1)
            gx = np.clip(((np.arange(fx0, fx1) + 0.5 - box[0]) * g).astype(np.int64), 0, S - 1)
            ctx.face_mask[fy0:fy1, fx0:fx1] = cov[np.ix_(gy, gx)]

        ctx.normals, ctx.face_box, ctx.light_coeffs = normals, box, light
        ctx.ratio_map, ctx.color_gain = ratio_map, color_gain
        ctx.relit_frame = ctx.linear_rgb * ratio_map[..., None]
        if color_gain is not None:
            ctx.relit_frame *= color_gain
        np.minimum(ctx.relit_frame, 1.0, out=ctx.relit_frame)

        # Measured on what is shown: with the color gains, the relit luminance is not exactly
        # Yc * grid (the gains are bounded and smoothed, so they keep luminance only roughly).
        relit_grid = Yc * grid if cgrid is None else color.luminance(Cc * (grid[..., None] * cgrid))
        fitted_grid = np.zeros((S, S), np.float32)
        target_grid = np.zeros((S, S), np.float32)
        fitted_grid[cov], target_grid[cov] = H @ light, target
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
