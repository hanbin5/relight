"""Module B: single-frame person separation and background replacement (plan B.3).

The plan models the wall from the upper corners. Real rooms are rarely one plain wall (a
door, shelves, clothes, a ceiling light), so both sides get a small color model here:
1. Colors. Background: k-means clusters of the pixels the landmarks place well outside the
   person. Foreground: clusters of places the landmarks put inside the person (the face,
   the head between forehead and crown, the chest below the neck). In front of a plain
   wall the background model is one tight cluster: the plan's wall model.
2. Score. Each pixel's distance D to the nearest background color, in cluster spreads and
   allowing that color darker or brighter (a shadow on the wall keeps its chromaticity), is
   weighed against its distance to the nearest foreground color, then gated by the
   head-and-shoulders prior; module A's face mask is sure foreground.
3. Otsu's threshold on the score, opening and closing. Only the part connected to the face
   is kept, and holes inside it are filled.
4. Guided-filter edge refinement, with the frame's color as the guide.
5. Spill removal with the background color around each edge pixel, and compositing:
   F = (I_relit - (1 - alpha) B_local) / max(alpha, 0.1),  I_out = alpha F + (1 - alpha) B_new.
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


def _grid(shape):
    h, w = shape
    return (np.arange(h, dtype=np.float32) + 0.5)[:, None], (np.arange(w, dtype=np.float32) + 0.5)[None, :]


def person_prior(shape, lm: np.ndarray):
    """The head-and-shoulders prior from the landmarks, as two maps.

    gate: where the person can be, 1 inside and fading to 0 outside. Wide enough for hair at
    the sides and for the shoulders, but narrow at the neck and close above the crown, so
    clutter beside and above the head stays out.
    likely: the prior probability that a pixel inside the gate is the person. High where the
    person almost always is (the skull, the neck, the torso straight below the face), so a
    black top in front of dark clutter stays; neutral (0.5) at the margins, where color decides.
    """
    f = face_measures(lm)
    fw, ec, cx, chin_y, eye_y = f["fw"], f["ec"], f["cx"], f["chin"][1], f["eye"][1]
    yy, xx = _grid(shape)
    dx = np.abs(xx - cx)

    def step(d, soft):
        return np.clip(0.5 + d / (2.0 * soft), 0.0, 1.0)

    # Gate. Head: an ellipse from a little above the crown (hair) down to the chin, about
    # 0.15 face widths of hair at each side.
    crown_y = chin_y - 1.15 * config.CROWN_K * ec
    cy, ay, ax = 0.5 * (crown_y + chin_y), 0.5 * (chin_y - crown_y), 0.8 * fw
    e = np.sqrt(((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2)
    head = np.clip((1.15 - e) / 0.15, 0.0, 1.0)
    # Neck, then shoulders: they reach about 1.8 face widths from the midline within one
    # eye-chin length below the chin.
    shoulders_y = chin_y + 0.45 * ec
    drop = np.sqrt(np.clip((yy - shoulders_y) / (0.8 * ec), 0.0, 1.0))
    half = 0.6 * fw + 2.2 * fw * drop
    soft = 0.25 * fw
    body = np.clip(1.0 + (half - dx) / soft, 0.0, 1.0) * np.clip(1.0 + (yy - chin_y + 0.3 * ec) / soft, 0.0, 1.0)
    gate = np.maximum(head, body)

    # Likely. Skull (forehead up to just under the crown), neck, and the torso inside the
    # shoulder line.
    skull = step(1.0 - np.sqrt((dx / (0.42 * fw)) ** 2 + ((yy - (eye_y - 0.35 * ec)) / (0.6 * ec)) ** 2), 0.15)
    neck = step(0.3 * fw - dx, 0.1 * fw) * step(yy - chin_y, 0.1 * ec)
    torso = step(0.45 * fw + 1.75 * fw * drop - dx, 0.2 * fw) * step(yy - shoulders_y, 0.2 * ec)
    likely = 0.5 + np.maximum((config.PRIOR_CORE - 0.5) * np.maximum(skull, neck), (config.PRIOR_TORSO - 0.5) * torso)
    return gate.astype(np.float32), likely.astype(np.float32)


def foreground_seeds(shape, lm: np.ndarray, face: np.ndarray) -> np.ndarray:
    """Places that are the person: the face, the head between forehead and crown, a narrow
    column straight below the chin (neck, collar, top; it stays in the frame when the
    person sits close) and the chest."""
    f = face_measures(lm)
    fw, ec, cx, chin_y, eye_y = f["fw"], f["ec"], f["cx"], f["chin"][1], f["eye"][1]
    yy, xx = _grid(shape)
    dx = np.abs(xx - cx)
    head_core = (dx / (0.38 * fw)) ** 2 + ((yy - (eye_y - 0.62 * ec)) / (0.25 * ec)) ** 2 < 1.0
    below_chin = (yy > chin_y + 0.15 * ec) & (yy < chin_y + 1.8 * ec) & (dx < 0.25 * fw)
    chest = (yy > chin_y + 0.9 * ec) & (yy < chin_y + 1.8 * ec) & (dx < 0.65 * fw)
    return npimg.erode(face, 1) | head_core | below_chin | chest


def kmeans(x: np.ndarray, k: int, init=None, iterations: int = 6, seed: int = 0):
    """Cluster colors x (N, 3). Returns centers (k, 3), each cluster's RMS spread (k,) and
    its share of the samples (k,)."""
    k = min(k, len(x))
    rng = np.random.default_rng(seed)
    if init is not None and len(init) == k:
        centers = init.copy()
    else:  # k-means++ seeding
        centers = [x[rng.integers(len(x))]]
        d2 = ((x - centers[0]) ** 2).sum(axis=1)
        for _ in range(1, k):
            total = float(d2.sum())
            centers.append(x[rng.choice(len(x), p=d2 / total)] if total > 0 else x[rng.integers(len(x))])
            d2 = np.minimum(d2, ((x - centers[-1]) ** 2).sum(axis=1))
        centers = np.array(centers, np.float32)
    for it in range(iterations + 1):
        d2 = ((x[:, None, :] - centers[None]) ** 2).sum(axis=2)
        label = d2.argmin(axis=1)
        if it == iterations:
            break
        for j in range(k):
            members = label == j
            if members.any():
                centers[j] = x[members].mean(axis=0)
    spread = np.array([np.sqrt(d2[label == j, j].mean()) if (label == j).any() else 0.0 for j in range(k)])
    share = np.bincount(label, minlength=k) / len(x)
    return (centers.astype(np.float32), np.maximum(spread, config.CLUSTER_SPREAD_FLOOR).astype(np.float32),
            share.astype(np.float32))


def nearest_distance(x: np.ndarray, centers: np.ndarray, spread: np.ndarray, scale) -> np.ndarray:
    """Distance from each color in x (N, 3) to the nearest center, in that cluster's spreads.
    Each center may be scaled by a factor in scale first (a darker or brighter version)."""
    xc = x @ centers.T
    cc = (centers * centers).sum(axis=1)
    s = np.clip(xc / np.maximum(cc, 1e-9), scale[0], scale[1])
    d2 = np.maximum((x * x).sum(axis=1, keepdims=True) - 2.0 * s * xc + s * s * cc, 0.0)
    return np.sqrt((d2 / (spread * spread)).min(axis=1))


def connected(seed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Pixels of mask 4-connected to seed (flood fill by repeated one-pixel shifts)."""
    cur = seed & mask
    while True:
        before = cur.copy()
        for _ in range(8):  # grow 8 pixels between the (costlier) convergence checks
            nxt = cur.copy()
            nxt[1:] |= cur[:-1]
            nxt[:-1] |= cur[1:]
            nxt[:, 1:] |= cur[:, :-1]
            nxt[:, :-1] |= cur[:, 1:]
            cur = nxt & mask
        if np.array_equal(cur, before):
            return cur


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """mask plus every background region that does not reach the frame border."""
    border = np.zeros_like(mask)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    return ~connected(border, ~mask)


def _sample(img: np.ndarray, where: np.ndarray, n: int, seed: int) -> np.ndarray:
    idx = np.flatnonzero(where)
    if len(idx) > n:
        idx = np.random.default_rng(seed).choice(idx, n, replace=False)
    return img.reshape(-1, 3)[idx]


class Matter:
    """Stateful matting: the color clusters carry over between frames (warm start)."""

    def __init__(self):
        self.bg = None
        self.fg = None

    def reset(self):
        self.bg = self.fg = None

    def process(self, ctx: FrameContext, background=None, update: bool = True,
                fast: bool = True) -> FrameContext:
        """Fill alpha, output_frame and wall_color. background: linear RGB (3,), or None
        to keep the original wall. fast runs steps 1-3 on a half-resolution grid (live view)."""
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
        hs, ws = lin_s.shape[:2]
        face_s = face[: hs * s : s, : ws * s : s]
        lm_s = ctx.landmarks / np.float32(s)

        # 1. Color models, in sqrt-linear color (close to perceptual spacing).
        col = np.sqrt(lin_s)
        gate, likely = person_prior((hs, ws), lm_s)
        bg_seed = gate <= 0.0
        fg_seed = foreground_seeds((hs, ws), lm_s, face_s)
        n, warm = config.CLUSTER_SAMPLES, update and self.bg is not None
        if bg_seed.sum() < 50 or fg_seed.sum() < 50:  # person fills the frame: nothing to model
            bg_seed = np.zeros((hs, ws), bool)
            bg_seed[[0, -1], :] = bg_seed[:, [0, -1]] = True
        bg_samples = _sample(col, bg_seed, n, 1)
        bg = kmeans(bg_samples, config.BG_CLUSTERS, self.bg[0] if warm else None, 3 if warm else 8)
        fg = kmeans(_sample(col, fg_seed, n, 2), config.FG_CLUSTERS, self.fg[0] if warm else None,
                    3 if warm else 8)
        if update:
            self.bg, self.fg = bg, fg
        bg_scale, fg_scale = np.sqrt(config.BG_SHADOW_SCALE), np.sqrt(config.FG_SHADE_SCALE)
        # How plain the background is: the share of it within 3 spreads of its main color.
        main = int(np.argmax(bg[2]))
        plain = float((nearest_distance(bg_samples, bg[0][main : main + 1], bg[1][main : main + 1], bg_scale) < 3.0).mean())

        # 2. Score: the color evidence (background distance against foreground distance, as
        # log-odds) plus the prior's log-odds, gated by where the person can be.
        region = gate > 0.05
        x = col[region]
        cap = config.OUTLIER_DISTANCE  # a color seen by neither model should not count much for one
        evidence = np.zeros((hs, ws), np.float32)
        evidence[region] = (np.minimum(nearest_distance(x, *bg[:2], bg_scale), cap)
                            - np.minimum(nearest_distance(x, *fg[:2], fg_scale), cap)) / config.SCORE_TAU
        p_color = 1.0 / (1.0 + np.exp(-evidence))
        score = gate / (1.0 + np.exp(-(evidence + np.log(likely / (1.0 - likely)))))
        score[face_s] = 1.0

        # 3. Otsu, opening and closing, the part connected to the face, holes filled.
        t = float(np.clip(npimg.otsu_threshold(score[region]), 0.35, 0.65))
        mask = score > t
        ks = k / s
        mask = npimg.opening(mask, max(1, round(config.OPEN_RADIUS * ks)))
        mask = npimg.closing(mask, max(1, round(config.CLOSE_RADIUS * ks)))
        mask = self._connected_part(mask, face_s)

        # 4. Edge refinement with the frame's color as the guide (He et al.'s color variant:
        # skin and a wall of the same brightness differ only in color).
        r = max(2, round(config.GF_RADIUS * ks))
        soft_mask = npimg.box_mean(mask.astype(np.float32), 1)  # takes the stairs off diagonal edges
        coeffs = npimg.guided_filter_color_coeffs(np.sqrt(relit_s), soft_mask, r, config.GF_EPS)
        # Keep the soft band within r/2 of the coarse edge: sure foreground and background.
        band = max(1, r // 2)
        inside = npimg.box_mean(mask.astype(np.float32), band)  # share of the window that is person
        bg_local = self._local_background(lin_s, mask, ks, bg_seed)
        if s > 1:
            coeffs = npimg.resize(coeffs, h, w, kernel="linear", box=(0, 0, w / s, h / s))
            inside = npimg.resize(inside, h, w, kernel="linear", box=(0, 0, w / s, h / s))
            bg_local = npimg.upsample_nearest(bg_local, s, (h, w))
        guide = np.sqrt(relit)
        alpha = coeffs[..., 0] * guide[..., 0] + coeffs[..., 1] * guide[..., 1] + coeffs[..., 2] * guide[..., 2] + coeffs[..., 3]
        np.clip(alpha, 0.0, 1.0, out=alpha)
        alpha[(inside > 0.999) | face] = 1.0
        alpha[inside < 0.001] = 0.0

        wall = lin_s[bg_seed].mean(axis=0) if bg_seed.any() else lin_s.reshape(-1, 3).mean(axis=0)
        ctx.alpha, ctx.wall_color = alpha.astype(np.float32), wall.astype(np.float32)
        ctx.stats["matting"] = {"threshold": t, "wall_mu": ctx.wall_color, "plain_share": plain,
                                "p_color": p_color, "prior": gate * likely, "coarse_mask": mask}
        # 5. Spill removal and compositing.
        ctx.output_frame = relit if background is None else composite(ctx, bg_local, background)
        return ctx

    @staticmethod
    def _connected_part(mask: np.ndarray, face: np.ndarray, q: int = 4) -> np.ndarray:
        """Keep what is connected to the face and fill holes, on a grid q times coarser
        (it only has to find blobs; the edges stay those of mask)."""
        small = npimg.downscale_mean(mask.astype(np.float32), q) > 0.5
        face_small = npimg.downscale_mean(face.astype(np.float32), q) > 0.5
        if not face_small.any():
            return mask | face
        kept = connected(face_small, small)
        holes = fill_holes(kept) & ~kept
        keep = npimg.upsample_nearest(npimg.dilate(kept, 1), q, mask.shape)
        return (mask & keep) | npimg.upsample_nearest(holes, q, mask.shape) | face

    @staticmethod
    def _local_background(lin: np.ndarray, mask: np.ndarray, ks: float, bg_seed: np.ndarray) -> np.ndarray:
        """Background color around each pixel: the mean of nearby background pixels (a smooth
        field, so it is computed on a grid twice as coarse)."""
        bgm = npimg.downscale_mean((~npimg.dilate(mask, 2)).astype(np.float32), 2)
        small = npimg.downscale_mean(lin, 2)
        r = max(2, round(6 * ks))
        num = npimg.box_sum(small * bgm[..., None], r)
        den = npimg.box_sum(bgm, r)[..., None]
        fallback = lin[bg_seed].mean(axis=0) if bg_seed.any() else lin.reshape(-1, 3).mean(axis=0)
        field = np.where(den > 0.5, num / np.maximum(den, 1e-6), fallback).astype(np.float32)
        return npimg.upsample_nearest(field, 2, mask.shape)


def composite(ctx: FrameContext, wall: np.ndarray, background) -> np.ndarray:
    """Spill removal and compositing; only the soft band needs the full formula.
    wall is the background color behind each pixel, (H, W, 3), or one color (3,)."""
    relit, alpha = ctx.relit_frame, ctx.alpha.ravel()
    bg = np.asarray(background, np.float32)
    out = relit.copy()
    flat = out.reshape(-1, 3)
    flat[alpha <= 0.0] = bg
    band = np.flatnonzero((alpha > 0.0) & (alpha < 1.0))
    a = alpha[band][:, None]
    ratio = ctx.ratio_map.ravel()[band][:, None] if ctx.ratio_map is not None else 1.0
    behind = wall.reshape(-1, 3)[band] if wall.ndim == 3 else wall
    fg = np.clip((relit.reshape(-1, 3)[band] - (1.0 - a) * ratio * behind) / np.maximum(a, config.SPILL_ALPHA_FLOOR), 0.0, 1.0)
    flat[band] = a * fg + (1.0 - a) * bg
    return out
