"""Module D: the capture-time checker and the auto shutter (plan B.4).

Seven rules run on every output frame. Each returns a measured value, a pass or fail
against its bound and, when it fails, the key of a coaching message (see i18n.py). The
shutter fires once all seven rules pass for 5 frames in a row.
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import color, config, mesh, npimg
from .context import FrameContext
from .matting import face_measures

RULES = {
    1: "even_light",
    2: "glare",
    3: "sharpness",
    4: "head_pose",
    5: "eyes_open",
    6: "expression",
    7: "clean_cutout",
}


@dataclass
class Verdict:
    rule: int
    key: str
    passed: bool
    value: float | None
    bound: str
    message: str | None = None  # i18n key of the coaching message when the rule fails
    detail: dict = field(default_factory=dict)


def _ear(lm: np.ndarray, ids) -> float:
    p = lm[list(ids), :2]
    return float((np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2 * np.linalg.norm(p[0] - p[3]) + 1e-9))


def _box_slices(box, size):
    w, h = size
    x0, y0 = max(int(np.floor(box[0])), 0), max(int(np.floor(box[1])), 0)
    x1, y1 = min(int(np.ceil(box[2])), w), min(int(np.ceil(box[3])), h)
    return slice(y0, y1), slice(x0, x1), (x0, y0)


class Checker:
    def __init__(self):
        self.sharpness_seen: deque = deque(maxlen=config.SHARPNESS_WINDOW)

    def sharpness_bound(self) -> float:
        """Rule 3's bound, calibrated per camera (B.4). A camera whose sharp frames stay below
        120 is held to 70% of them (the 80th percentile of recent frames), so motion blur still
        fails; never below SHARPNESS_FLOOR."""
        if len(self.sharpness_seen) < config.SHARPNESS_WARMUP:
            return config.SHARPNESS_MIN
        best = float(np.percentile(self.sharpness_seen, 80))
        return min(config.SHARPNESS_MIN, max(config.SHARPNESS_FLOOR, config.SHARPNESS_REL * best))

    def evaluate(self, ctx: FrameContext) -> dict:
        lm = ctx.landmarks
        if lm is None:
            ctx.rule_verdicts = {
                n: Verdict(n, k, False, None, "", "no_face") for n, k in RULES.items()
            }
            return ctx.rule_verdicts
        v = {
            1: self.even_light(ctx),
            2: self.glare(ctx),
            3: self.sharpness(ctx),
            4: self.head_pose(ctx),
            5: self.eyes_open(lm),
            6: self.expression(lm),
            7: self.clean_cutout(ctx),
        }
        ctx.rule_verdicts = v
        return v

    # Rule 1 -------------------------------------------------------------------------
    def even_light(self, ctx: FrameContext) -> Verdict:
        S = config.FIT_SIZE
        box = ctx.face_box or mesh.face_box(ctx.landmarks)
        g = mesh.to_grid(ctx.landmarks, box, S)
        reg = mesh.region_masks(g, (S, S))
        Y = color.luminance(npimg.resize(ctx.relit_frame, S, S, "linear", box=box))
        skin = npimg.erode(reg["oval"], 1) & ~reg["features"]
        r_lr = mesh.cheek_balance(Y, reg, skin)
        shadow = mesh.shadow_share(Y, skin, config.SHADOW_FACTOR)
        lo, hi = config.EVEN_LIGHT_RANGE
        ok = bool(lo <= r_lr <= hi and shadow < config.SHADOW_MAX)
        return Verdict(1, RULES[1], ok, r_lr, f"R_LR {lo:.2f}–{hi:.2f}; shadow < {config.SHADOW_MAX:.0%}",
                       None if ok else "turn_to_light", {"r_lr": r_lr, "shadow_share": shadow})

    # Rule 2 -------------------------------------------------------------------------
    def glare(self, ctx: FrameContext) -> Verdict:
        lm = ctx.landmarks
        p = lm[:468, :2]
        f = face_measures(lm)
        pad = 0.05 * f["fw"]
        box = (p[:, 0].min() - pad, p[:, 1].min() - pad, p[:, 0].max() + pad, p[:, 1].max() + pad)
        ys, xs, (x0, y0) = _box_slices(box, ctx.size)
        lin = ctx.relit_frame[ys, xs]
        shape = lin.shape[:2]
        local = lm.copy()
        local[:, 0] -= x0
        local[:, 1] -= y0
        bright = color.luminance(lin) > config.GLARE_Y
        glare = np.zeros(shape, bool)
        if bright.any():
            sat = color.hsv_saturation(color.linear_to_srgb(lin[bright]))
            glare[bright] = sat < config.GLARE_SAT
        eyes = mesh.polygon(local, mesh.EYE_R, shape) | mesh.polygon(local, mesh.EYE_L, shape)
        brows = mesh.polygon(local, mesh.BROW_R, shape) | mesh.polygon(local, mesh.BROW_L, shape)
        lips = mesh.polygon(local, mesh.LIPS, shape)
        skin = mesh.polygon(local, mesh.FACE_OVAL, shape) & ~npimg.dilate(eyes | brows | lips, 1)
        lens = np.zeros(shape, bool)
        for outer, inner in ((33, 133), (263, 362)):  # glasses lens around each eye
            a, b = local[outer, :2], local[inner, :2]
            ew = float(np.linalg.norm(a - b))
            cx, cy = 0.5 * (a + b)
            ly0, ly1 = int(max(cy - 0.75 * ew, 0)), int(min(cy + 0.6 * ew, shape[0]))
            lx0, lx1 = int(max(cx - 0.95 * ew, 0)), int(min(cx + 0.95 * ew, shape[1]))
            lens[ly0:ly1, lx0:lx1] = True
        lens &= ~npimg.dilate(eyes, 1)  # catchlights in the eyes are fine
        face_share = float(glare[skin].mean()) if skin.any() else 0.0
        lens_share = float(glare[lens].mean()) if lens.any() else 0.0
        lens_ok = lens_share < config.GLARE_LENS_MAX
        ok = bool(lens_ok and face_share < config.GLARE_FACE_MAX)
        message = None if ok else ("glare_glasses" if not lens_ok else "glare_skin")
        return Verdict(2, RULES[2], ok, face_share, f"no lens glare; < {config.GLARE_FACE_MAX:.0%} of face",
                       message, {"face_share": face_share, "lens_share": lens_share})

    # Rule 3 -------------------------------------------------------------------------
    def sharpness(self, ctx: FrameContext) -> Verdict:
        n = config.SHARPNESS_CROP
        box = ctx.face_box or mesh.face_box(ctx.landmarks)
        if ctx.full_frame is not None:  # the camera's own pixels, before any downscale
            k = ctx.full_frame.shape[1] / ctx.size[0]
            crop = npimg.resize(ctx.full_frame, n, n, "linear", box=tuple(c * k for c in box))
            gray = crop @ np.array([0.299, 0.587, 0.114], np.float32)
        else:
            gray = color.gray_u8(color.encode_u8(npimg.resize(ctx.output_frame, n, n, "linear", box=box)))
        score = float(npimg.laplacian3(gray).var())
        self.sharpness_seen.append(score)
        bound = self.sharpness_bound()
        ok = score >= bound
        return Verdict(3, RULES[3], ok, score, f"≥ {bound:.0f}", None if ok else "hold_still", {"bound": bound})

    # Rule 4 -------------------------------------------------------------------------
    def head_pose(self, ctx: FrameContext) -> Verdict:
        geo = ctx.geometry
        roll, yaw, pitch = geo["roll_deg"], geo["yaw_ratio"], geo["pitch_ratio"]
        roll_ok = abs(roll) <= config.ROLL_MAX_DEG
        lo, hi = config.PITCH_RATIO_RANGE
        facing_ok = abs(yaw) <= config.YAW_RATIO_MAX and lo <= pitch <= hi
        ok = roll_ok and facing_ok
        message = None if ok else ("straighten_head" if not roll_ok else "face_camera")
        return Verdict(4, RULES[4], ok, roll,
                       f"roll ≤ {config.ROLL_MAX_DEG}°; yaw ≤ {config.YAW_RATIO_MAX}; pitch {lo}–{hi}",
                       message, {"roll_deg": roll, "yaw_ratio": yaw, "pitch_ratio": pitch})

    # Rule 5 -------------------------------------------------------------------------
    def eyes_open(self, lm: np.ndarray) -> Verdict:
        right, left = _ear(lm, mesh.EAR_R), _ear(lm, mesh.EAR_L)
        value = min(right, left)
        ok = value >= config.EAR_MIN
        return Verdict(5, RULES[5], ok, value, f"EAR ≥ {config.EAR_MIN}", None if ok else "open_eyes",
                       {"ear_right": right, "ear_left": left})

    # Rule 6 -------------------------------------------------------------------------
    def expression(self, lm: np.ndarray) -> Verdict:
        gap = np.linalg.norm(lm[mesh.LIP_LOWER_INNER, :2] - lm[mesh.LIP_UPPER_INNER, :2])
        width = np.linalg.norm(lm[mesh.MOUTH_L, :2] - lm[mesh.MOUTH_R, :2])
        value = float(gap / (width + 1e-9))
        ok = value < config.LIP_GAP_MAX
        return Verdict(6, RULES[6], ok, value, f"< {config.LIP_GAP_MAX}", None if ok else "close_mouth")

    # Rule 7 -------------------------------------------------------------------------
    def clean_cutout(self, ctx: FrameContext) -> Verdict:
        f = face_measures(ctx.landmarks)
        alpha = ctx.alpha
        w, h = ctx.size
        box = ctx.geometry.get("crop_box") or (0, 0, w, h)
        ys, xs, (x0, y0) = _box_slices(box, ctx.size)
        a = alpha[ys, xs]
        yy = (np.arange(ys.start, ys.stop) + 0.5)[:, None]
        xx = (np.arange(xs.start, xs.stop) + 0.5)[None, :]
        mouth_y = 0.5 * (ctx.landmarks[mesh.MOUTH_R, 1] + ctx.landmarks[mesh.MOUTH_L, 1])
        hair_band = (yy < mouth_y) & (np.abs(xx - f["cx"]) < 1.1 * f["fw"])
        uncertain = (a > 0.1) & (a < 0.9) & ~hair_band
        share = float(uncertain.mean()) if a.size else 0.0

        # Person-wall contrast at the shoulder line, on each side of the neck: the top against
        # the background in the same rows on that side (or the background's mean color if
        # none shows there). The weaker side counts: that is where the cutout fails.
        y_s = f["chin"][1] + 0.9 * f["ec"]
        rows = slice(int(np.clip(y_s - 0.08 * f["ec"], 0, h - 1)), int(np.clip(y_s + 0.08 * f["ec"], 1, h)))
        offset = np.arange(w) + 0.5 - f["cx"]
        contrast = None
        if rows.stop > rows.start:
            band, behind = ctx.linear_rgb[rows], alpha[rows] < 0.1
            for sign in (-1, 1):
                d = sign * offset
                top = (d > 0.45 * f["fw"]) & (d < 1.0 * f["fw"])
                wall_side = behind & (d > 0)[None, :]
                if not top.any():
                    continue
                person = band[:, top].reshape(-1, 3).mean(axis=0)
                wall = band[wall_side].mean(axis=0) if wall_side.sum() >= 20 else ctx.wall_color
                c = float(np.linalg.norm(color.linear_to_srgb(person) - color.linear_to_srgb(wall)) / np.sqrt(3))
                contrast = c if contrast is None else min(contrast, c)
        contrast_ok = contrast is None or contrast >= config.CONTRAST_MIN
        ok = bool(share < config.UNCERTAIN_MAX and contrast_ok)
        plain = ctx.stats.get("matting", {}).get("plain_share", 1.0)
        message = None
        if not ok:  # what to change: the top, the background, or the distance to the wall
            message = ("top_vs_wall" if not contrast_ok
                       else "plain_wall" if plain < config.PLAIN_WALL_SHARE else "move_from_wall")
        return Verdict(7, RULES[7], ok, share, f"< {config.UNCERTAIN_MAX:.0%}; contrast ≥ {config.CONTRAST_MIN}",
                       message, {"uncertain_share": share, "contrast": contrast, "plain_background": plain})


class AutoShutter:
    """Fires once every rule passes for SHUTTER_FRAMES frames in a row, then waits to be rearmed."""

    def __init__(self, frames: int = config.SHUTTER_FRAMES):
        self.frames = frames
        self.count = 0
        self.armed = True

    def update(self, all_pass: bool, enabled: bool = True) -> bool:
        self.count = self.count + 1 if all_pass else 0
        if enabled and self.armed and self.count >= self.frames:
            self.armed = False
            return True
        return False

    def rearm(self):
        self.armed = True
        self.count = 0
