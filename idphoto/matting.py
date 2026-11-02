"""Module B: single-frame person separation and background replacement (plan B.3).

Without a light we control, the person is separated from one frame:
1. a color model of the wall from the upper corners (outside the head, placed by the landmarks);
2. a per-pixel score from the color distance D to the wall, made shadow-tolerant (a pixel
   that is the wall color scaled darker or brighter, i.e. keeps the wall's chromaticity,
   stays background), times a head-and-shoulders prior from the landmarks; module A's
   face mask is sure foreground;
3. Otsu's threshold on the score, then opening and closing;
4. guided-filter edge refinement with the frame luminance as the guide;
5. wall-color spill removal and compositing on the new background:
   F = (I_relit - (1 - alpha) mu_B) / max(alpha, 0.1),  I_out = alpha F + (1 - alpha) B_new.
"""

import numpy as np

from . import config, mesh, npimg
from .context import FrameContext


def face_measures(lm: np.ndarray) -> dict:
    """Eye center, chin, eye-chin length, face width and midline x from the landmarks."""
    eye = 0.5 * (lm[mesh.IRIS_R, :2] + lm[mesh.IRIS_L, :2])
    chin = lm[mesh.CHIN, :2]
    return {
        "eye": eye,
        "chin": chin,
        "ec": float(np.linalg.norm(chin - eye)),
        "fw": float(np.linalg.norm(lm[mesh.FACE_EDGE_L, :2] - lm[mesh.FACE_EDGE_R, :2])),
        "cx": float(0.5 * (eye[0] + chin[0])),
    }


def wall_model(lin: np.ndarray, lm: np.ndarray):
    """Mean and spread (3,) of the wall in the upper corners, outside the head.

    The spread is measured after scaling each sample to the mean brightness: a wall is
    rarely lit evenly (one corner darker), and wall_distance already allows for brightness
    changes, so only the color spread and noise should widen the model.
    """
    h, w = lin.shape[:2]
    cw, ch = max(int(config.CORNER_W * w), 4), max(int(config.CORNER_H * h), 4)
    sel = np.zeros((h, w), bool)
    sel[:ch, :cw] = True
    sel[:ch, w - cw :] = True
    f = face_measures(lm)
    x0 = int(max(f["cx"] - 1.1 * f["fw"], 0))
    x1 = int(min(f["cx"] + 1.1 * f["fw"], w))
    sel[: int(min(f["chin"][1], h)), x0:x1] = False  # the head and its hair
    if sel.sum() < 64:  # head fills the corners: fall back to the outermost columns
        sel[:, : max(w // 20, 2)] = True
        sel[:, w - max(w // 20, 2) :] = True
        sel[int(f["chin"][1]) :, :] = False
    samples = lin[sel]
    mu = samples.mean(axis=0)
    scale = (samples @ mu) / max(float(mu @ mu), 1e-9)
    spread = (samples - scale[:, None] * mu).std(axis=0)
    sigma = np.maximum(spread, np.maximum(config.WALL_SIGMA_FLOOR, config.WALL_SIGMA_REL * mu))
    return mu.astype(np.float32), sigma.astype(np.float32)


def wall_distance(lin: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Shadow-tolerant distance to the wall, in wall spreads.

    D = || (c - mu) / sigma ||  is the plan's distance. Here mu is first scaled by the factor
    s in SHADOW_SCALE that best matches the pixel, so a darker (shadowed) or brighter wall
    pixel with the wall's chromaticity has a small distance; very dark pixels (hair) can't
    be explained by a shadow and keep a large one.
    """
    inv = (1.0 / (sigma * sigma)).astype(np.float32)
    cm = lin @ (mu * inv)  # c . mu / sigma^2
    mm = float(mu @ (mu * inv))
    s = np.clip(cm / mm, *config.SHADOW_SCALE)
    # || (c - s mu) / sigma ||^2 = c.c/sigma^2 - 2 s c.mu/sigma^2 + s^2 mu.mu/sigma^2
    d2 = (lin * lin) @ inv - 2.0 * s * cm + s * s * mm
    return np.sqrt(np.maximum(d2, 0.0))


def person_prior(shape, lm: np.ndarray) -> np.ndarray:
    """Head-and-shoulders prior from the landmarks: 1 where the person can be, fading to 0
    outside. It is generous (hair volume, long hair, wide shoulders); it only has to keep
    clutter far from the person out of the mask."""
    h, w = shape
    f = face_measures(lm)
    fw, ec, cx = f["fw"], f["ec"], f["cx"]
    chin_y = f["chin"][1]
    yy = (np.arange(h, dtype=np.float32) + 0.5)[:, None]
    xx = (np.arange(w, dtype=np.float32) + 0.5)[None, :]
    soft = 0.3 * fw

    # Head: an ellipse from well above the crown (hair volume) to the chin.
    crown_y = chin_y - 1.3 * config.CROWN_K * ec
    cy, ay, ax = 0.5 * (crown_y + chin_y), 0.5 * (chin_y - crown_y), 1.0 * fw
    e = np.sqrt(((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2)
    head = np.clip((1.25 - e) / 0.25, 0.0, 1.0)
    # Neck and shoulders: widening downward from just above the chin.
    y0 = chin_y - 0.3 * ec
    half = 0.8 * fw + 2.0 * fw * np.sqrt(np.clip((yy - y0) / (1.6 * ec), 0.0, 1.0))
    body = np.clip(1.0 + (half - np.abs(xx - cx)) / soft, 0.0, 1.0) * np.clip(1.0 + (yy - y0) / soft, 0.0, 1.0)
    return np.maximum(head, body).astype(np.float32)


class Matter:
    """Stateful matting: the wall model is averaged over frames for stability."""

    def __init__(self, smoothing: float = 0.8):
        self.smoothing = smoothing
        self.mu = None
        self.sigma = None

    def reset(self):
        self.mu = self.sigma = None

    def process(self, ctx: FrameContext, background=None, update: bool = True,
                fast: bool = True) -> FrameContext:
        """Fill alpha, output_frame and wall_color. background: linear RGB (3,), or None
        to keep the original wall. fast uses the subsampled guided filter (live view)."""
        lin, relit = ctx.linear_rgb, ctx.relit_frame
        h, w = lin.shape[:2]
        if ctx.landmarks is None:
            ctx.alpha = np.zeros((h, w), np.float32)
            ctx.output_frame = relit
            return ctx
        k = w / config.REF_WIDTH

        # Steps 1-3 and the guided-filter coefficients run on a grid s times coarser in the
        # live view (s = 2); the shutter frame uses s = 1, the exact filter of B.3.
        s = 2 if fast else 1
        face = ctx.face_mask if ctx.face_mask is not None else np.zeros((h, w), bool)
        lin_s, relit_s = npimg.downscale_mean(lin, s), npimg.downscale_mean(relit, s)
        face_s = face[: lin_s.shape[0] * s : s, : lin_s.shape[1] * s : s]
        lm_s = ctx.landmarks / np.float32(s)

        mu, sigma = wall_model(lin_s, lm_s)
        if update and self.mu is not None:
            mu = self.smoothing * self.mu + (1 - self.smoothing) * mu
            sigma = self.smoothing * self.sigma + (1 - self.smoothing) * sigma
        if update:
            self.mu, self.sigma = mu, sigma

        dist = wall_distance(lin_s, mu, sigma)
        p_color = 1.0 - np.exp(-((dist / config.COLOR_TAU) ** 2))
        prior = person_prior(lin_s.shape[:2], lm_s)
        score = p_color * prior
        score[face_s] = 1.0

        region = prior > 0.05
        t = float(np.clip(npimg.otsu_threshold(score[region]), 0.25, 0.75))
        mask = score > t
        ks = k / s
        mask = npimg.opening(mask, max(1, round(config.OPEN_RADIUS * ks)))
        mask = npimg.closing(mask, max(1, round(config.CLOSE_RADIUS * ks)))
        mask |= face_s

        # Edge refinement. The guide is the frame's color, not only its luminance: skin and a
        # wall of the same brightness differ only in color (He et al.'s color-guide variant).
        r = max(2, round(config.GF_RADIUS * ks))
        coeffs = npimg.guided_filter_color_coeffs(np.sqrt(relit_s), mask.astype(np.float32), r, config.GF_EPS)
        # Keep the soft band within r/2 of the coarse edge: sure foreground and background.
        near = npimg.box_sum(mask.astype(np.float32), r // 2)
        full = np.outer(npimg.window_counts(mask.shape[0], r // 2), npimg.window_counts(mask.shape[1], r // 2))
        sure = (near >= full - 0.5).astype(np.int8) - (near < 0.5)  # 1 fg, -1 bg, 0 band
        if s > 1:
            coeffs = npimg.resize(coeffs, h, w, kernel="linear", box=(0, 0, w / s, h / s))
            sure = npimg.upsample_nearest(sure, s, (h, w))
        guide = np.sqrt(relit)
        alpha = coeffs[..., 0] * guide[..., 0] + coeffs[..., 1] * guide[..., 1] + coeffs[..., 2] * guide[..., 2] + coeffs[..., 3]
        np.clip(alpha, 0.0, 1.0, out=alpha)
        alpha[(sure > 0) | face] = 1.0
        alpha[sure < 0] = 0.0

        ctx.alpha, ctx.wall_color = alpha.astype(np.float32), mu
        ctx.stats["matting"] = {"threshold": t, "wall_mu": mu, "wall_sigma": sigma}
        ctx.output_frame = relit if background is None else composite(ctx, mu, background)
        return ctx


def composite(ctx: FrameContext, mu: np.ndarray, background) -> np.ndarray:
    """Spill removal and compositing; only the soft band needs the full formula."""
    relit, alpha = ctx.relit_frame, ctx.alpha.ravel()
    bg = np.asarray(background, np.float32)
    out = relit.copy()
    flat = out.reshape(-1, 3)
    flat[alpha <= 0.0] = bg
    band = np.flatnonzero((alpha > 0.0) & (alpha < 1.0))
    a = alpha[band][:, None]
    ratio = ctx.ratio_map.ravel()[band][:, None] if ctx.ratio_map is not None else 1.0
    fg = np.clip((relit.reshape(-1, 3)[band] - (1.0 - a) * ratio * mu) / np.maximum(a, config.SPILL_ALPHA_FLOOR), 0.0, 1.0)
    flat[band] = a * fg + (1.0 - a) * bg
    return out
