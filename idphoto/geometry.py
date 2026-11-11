"""Module C: head geometry, crop and scale, print sheet and 300 DPI output (plan B.5, A.1).

Landmark 10 sits on the upper forehead, not the top of the head, so the crown is estimated
from two cues: the proportion rule crown = chin - k (chin - eye line) with k = 2, and the top
of the foreground mask on the face midline. When the two agree, the crown is their weighted
mean (config.CROWN_MASK_WEIGHT is the mask cue's share).
The crop keeps the head at its default share of the photo height:
    H_crop = H_head / f_head,  W_crop = H_crop * W_doc / H_doc
and is resampled with separable bicubic interpolation; scaling is always uniform.
"""

import json
import struct
import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from . import color, config, mesh, npimg


@dataclass(frozen=True)
class PhotoFormat:
    key: str
    size_mm: tuple | None  # (w, h) in mm; None for the square profile photo
    size_px: tuple  # (w, h) of the output at 300 DPI
    head_share: float  # default head length as a share of the photo height (A.1)
    top_margin: float  # crown distance from the top edge, as a share of the photo height
    sheet_grid: tuple | None  # (columns, rows) on the 6 x 4 in landscape print sheet
    face_share: float  # face mode: chin to skull top as a share of the photo height
    skull_top: float  # face mode: skull top distance from the top edge, as a share of the height


# Face-mode defaults: a typical head of hair adds about 5% to chin-to-skull, so these put
# typical hair where head mode puts it (top margin and head share) and keep the chin in place.
FORMATS = {
    "resume": PhotoFormat("resume", (30, 40), (354, 472), 0.75, 0.10, (5, 2), 0.71, 0.14),
    "id": PhotoFormat("id", (35, 45), (413, 531), 0.75, 0.10, (4, 2), 0.71, 0.14),
    "profile": PhotoFormat("profile", None, (800, 800), 0.50, 0.15, None, 0.47, 0.18),
}


def default_share(fmt: PhotoFormat, mode: str | None = None) -> float:
    """The format's default size share in the given crop mode (config.CROP_MODE if None)."""
    return fmt.face_share if (mode or config.CROP_MODE) == "face" else fmt.head_share

NOTICE = ("Edited photo (relit, background replaced). Not for passports or other government "
          "IDs: they require unedited photos.")


def head_pose(lm: np.ndarray) -> dict:
    """Roll from the iris centers, yaw from the nose tip to the face edges, pitch from 10, 1, 152.

    Distances are measured along the eye axis (yaw) and across it (pitch), so roll does not
    leak into them. pitch_ratio is d(10, 1) / d(1, 152) divided by its value on the
    canonical face, so 1.0 means a level head.
    """
    ir, il = lm[mesh.IRIS_R, :2], lm[mesh.IRIS_L, :2]
    axis = il - ir
    roll = float(np.degrees(np.arctan2(axis[1], axis[0])))
    u = axis / (np.linalg.norm(axis) + 1e-9)
    v = np.array([-u[1], u[0]])  # points down the face
    nose = lm[mesh.NOSE_TIP, :2]
    d_r = float((nose - lm[mesh.FACE_EDGE_R, :2]) @ u)
    d_l = float((lm[mesh.FACE_EDGE_L, :2] - nose) @ u)
    yaw_ratio = (d_r - d_l) / max(d_r + d_l, 1e-6)
    upper = float((nose - lm[mesh.FOREHEAD, :2]) @ v)
    lower = float((lm[mesh.CHIN, :2] - nose) @ v)
    pitch_ratio = (upper / max(lower, 1e-6)) / mesh.canonical_pitch_ratio()
    return {"roll_deg": roll, "yaw_ratio": yaw_ratio, "pitch_ratio": pitch_ratio}


def crown_cues(lm: np.ndarray, alpha: np.ndarray | None = None) -> dict:
    """Both crown cues before they are combined.

    proportion: chin + k (eye line - chin). mask: where the foreground mask ends on the
    midline above the forehead mark, or None if it never ends inside the frame. agree:
    the mask cue's distance from the chin over the proportion cue's (1.0 = same point).
    up and ec are the unit chin-to-eye direction and the chin-to-eye distance.
    """
    eye = 0.5 * (lm[mesh.IRIS_R, :2] + lm[mesh.IRIS_L, :2])
    chin = lm[mesh.CHIN, :2]
    up = eye - chin
    ec = float(np.linalg.norm(up))
    up = up / max(ec, 1e-6)
    cues = {"proportion": chin + config.CROWN_K * ec * up, "mask": None, "agree": None, "up": up, "ec": ec}
    if alpha is None:
        return cues
    # Scan up the midline from the forehead mark until the mask ends.
    start = lm[mesh.FOREHEAD, :2]
    steps = np.arange(0.0, 1.2 * ec, 0.5)
    pts = start[None, :] + steps[:, None] * up[None, :]
    inside = (pts[:, 1] >= 0) & (pts[:, 0] >= 0) & (pts[:, 0] < alpha.shape[1])
    a = np.where(inside, npimg.sample_bilinear(alpha, pts[:, 0], pts[:, 1]), 0.0)
    below = a < 0.5
    run = np.convolve(below.astype(np.int32), np.ones(3, np.int32), "valid") == 3
    if not run.any() or not inside[np.argmax(run)]:
        return cues  # mask runs off the frame or never ends
    cues["mask"] = pts[np.argmax(run)]
    cues["agree"] = float((cues["mask"] - chin) @ up) / (config.CROWN_K * ec)
    return cues


def estimate_crown(lm: np.ndarray, alpha: np.ndarray | None = None):
    """Crown point (x, y) and which cues were used."""
    return _combine(crown_cues(lm, alpha))


def _combine(cues: dict):
    lo, hi = config.CROWN_MASK_AGREE
    if cues["mask"] is not None and lo <= cues["agree"] <= hi:
        w = config.CROWN_MASK_WEIGHT
        return (1.0 - w) * cues["proportion"] + w * cues["mask"], "mask+proportion"
    return cues["proportion"], "proportion"


def measure(lm: np.ndarray, alpha: np.ndarray | None = None) -> dict:
    """Crown (hair top), skull top, chin, eye center, head length, midline and pose angles.

    The skull top is the proportion cue alone: the eyes-halfway rule describes the skull, not
    the hair. Face-mode crops are sized from it.
    """
    cues = crown_cues(lm, alpha)
    crown, source = _combine(cues)
    chin = lm[mesh.CHIN, :2]
    eye = 0.5 * (lm[mesh.IRIS_R, :2] + lm[mesh.IRIS_L, :2])
    geo = {
        "crown": crown.astype(np.float32),
        "skull": cues["proportion"].astype(np.float32),
        "chin": chin.astype(np.float32),
        "eye_center": eye.astype(np.float32),
        "head_length": float(np.linalg.norm(chin - crown)),
        "center_x": float(0.5 * (eye[0] + chin[0])),
        "crown_source": source,
    }
    geo.update(head_pose(lm))
    return geo


def crop_box(geo: dict, fmt: PhotoFormat, share: float | None = None, mode: str | None = None):
    """(x0, y0, x1, y1) of the crop in frame pixels; may extend past the frame.

    share is the head share in head mode and the face share in face mode (config.CROP_MODE
    when mode is None); None takes the format's default.
    """
    if (mode or config.CROP_MODE) == "face":
        return face_crop(geo, fmt, share)["box"]
    share = float(share or fmt.head_share)
    h_crop = geo["head_length"] / share
    w_crop = h_crop * fmt.size_px[0] / fmt.size_px[1]
    top_margin = min(fmt.top_margin, 0.5 * (1.0 - share))
    y0 = float(geo["crown"][1]) - top_margin * h_crop
    x0 = geo["center_x"] - 0.5 * w_crop
    return (x0, y0, x0 + w_crop, y0 + h_crop)


def face_crop(geo: dict, fmt: PhotoFormat, face_share: float | None = None) -> dict:
    """Face-mode crop: sized by chin to skull top, then kept clear of the hair.

    The skull top goes at fmt.skull_top. If the hair top would come closer to the top edge
    than HAIR_MARGIN_MIN, the crop moves up ("moved"); if that pushes the chin below
    CHIN_MAX, it zooms out until both fit ("zoomed"). Returns the box, what was done, and the
    face and hair positions as shares of the photo height.
    """
    share = float(face_share or fmt.face_share)
    skull_y, chin_y = float(geo["skull"][1]), float(geo["chin"][1])
    hair_y = min(float(geo["crown"][1]), skull_y)
    h = float(np.linalg.norm(geo["chin"] - geo["skull"])) / share
    y0 = skull_y - fmt.skull_top * h
    action = "fits"
    m, c = config.HAIR_MARGIN_MIN, config.CHIN_MAX
    if hair_y - y0 < m * h:
        y0, action = hair_y - m * h, "moved"
        if chin_y - y0 > c * h:
            h = (chin_y - hair_y) / (c - m)
            y0, action = hair_y - m * h, "zoomed"
    w = h * fmt.size_px[0] / fmt.size_px[1]
    x0 = geo["center_x"] - 0.5 * w
    return {"box": (x0, y0, x0 + w, y0 + h), "action": action,
            "face_share": float(np.linalg.norm(geo["chin"] - geo["skull"])) / h,
            "hair_top": (hair_y - y0) / h, "chin": (chin_y - y0) / h}


def crop_fits(box, size, tolerance: float = 0.02) -> bool:
    """True if the crop lies inside the frame (bottom may run past it by the tolerance)."""
    w, h = size
    x0, y0, x1, y1 = box
    t = tolerance * (y1 - y0)
    return x0 >= -t and y0 >= -t and x1 <= w + t and y1 <= h + t


class GeometryTracker:
    """Smooths the measured geometry over frames so the crop guide does not jitter."""

    def __init__(self, smoothing: float = 0.6):
        self.smoothing = smoothing
        self.state = None

    def reset(self):
        self.state = None

    def update(self, geo: dict) -> dict:
        keys = ("crown", "skull", "chin", "eye_center", "head_length", "center_x")
        if self.state is None:
            self.state = {k: geo[k] for k in keys}
        else:
            s = self.smoothing
            self.state = {k: s * np.asarray(self.state[k]) + (1 - s) * np.asarray(geo[k]) for k in keys}
            self.state["head_length"] = float(self.state["head_length"])
            self.state["center_x"] = float(self.state["center_x"])
        return {**geo, **self.state}


def render_photo(frame_lin: np.ndarray, box, fmt: PhotoFormat, pad_color) -> np.ndarray:
    """Crop and scale to the format's 300 DPI size (separable bicubic), as sRGB uint8.

    Parts of the crop above or beside the frame get the background color; below the frame
    the last row is extended (the shoulders continue).
    """
    h, w = frame_lin.shape[:2]
    out_w, out_h = fmt.size_px
    x0, y0, x1, y1 = box
    photo = np.clip(npimg.resize(frame_lin, out_h, out_w, "cubic", box=box), 0.0, 1.0)
    ys = y0 + (np.arange(out_h) + 0.5) * (y1 - y0) / out_h
    xs = x0 + (np.arange(out_w) + 0.5) * (x1 - x0) / out_w
    outside = (ys[:, None] < 0) | (xs[None, :] < 0) | (xs[None, :] >= w)
    photo[outside] = np.asarray(pad_color, np.float32)
    return color.encode_u8(photo)


def print_sheet(photo: np.ndarray, fmt: PhotoFormat) -> np.ndarray:
    """6 x 4 in landscape sheet at 300 DPI, tiled with copies of the photo and cut lines."""
    sheet_w, sheet_h = config.SHEET_PX
    cols, rows = fmt.sheet_grid
    ph, pw = photo.shape[:2]
    gx = (sheet_w - cols * pw) / (cols + 1)
    gy = (sheet_h - rows * ph) / (rows + 1)
    sheet = np.full((sheet_h, sheet_w, 3), 255, np.uint8)
    line = np.array([200, 200, 200], np.uint8)
    for r in range(rows):
        for c in range(cols):
            x = int(round(gx + c * (pw + gx)))
            y = int(round(gy + r * (ph + gy)))
            sheet[max(y - 1, 0) : y + ph + 1, max(x - 1, 0) : x + pw + 1] = line
            sheet[y : y + ph, x : x + pw] = photo
    return sheet


# ---------------------------------------------------------------------------------------
# Output files: OpenCV encodes; we only patch in the 300 DPI density and a notice.

FILE_NAMES = {"resume": "resume_3x4cm", "id": "id_3.5x4.5cm", "profile": "profile_800px"}


def save_outputs(result: dict, folder, ext: str = "jpg"):
    """Write the three photos, the print sheets and a JSON report into a new subfolder.
    Returns (subfolder, files)."""
    out = Path(folder) / datetime.now().strftime("idphoto_%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for key, photo in result["photos"].items():
        files.append(save_image(out / f"{FILE_NAMES[key]}.{ext}", photo))
    for key, sheet in result["sheets"].items():
        files.append(save_image(out / f"print_sheet_{FILE_NAMES[key]}_6x4in.{ext}", sheet))
    report = out / "report.json"
    report.write_text(json.dumps(result["report"], indent=2, ensure_ascii=False))
    files.append(report)
    return out, files


def save_image(path, rgb: np.ndarray, dpi: int = config.PRINT_DPI, comment: str = NOTICE) -> Path:
    """Write a JPEG or PNG that records its print density and the notice."""
    import cv2

    path = Path(path)
    ext = path.suffix.lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, np.ascontiguousarray(rgb[:, :, ::-1]), params)
    if not ok:
        raise OSError(f"could not encode {path.name}")
    data = buf.tobytes()
    if ext in (".jpg", ".jpeg"):
        data = _jpeg_with_density(data, dpi, comment)
    elif ext == ".png":
        data = _png_with_density(data, dpi, comment)
    path.write_bytes(data)
    return path


def _jpeg_with_density(data: bytes, dpi: int, comment: str) -> bytes:
    out = bytearray(data)
    if out[2:4] == b"\xff\xe0" and out[6:11] == b"JFIF\x00":
        out[13] = 1  # density unit: dots per inch
        out[14:18] = struct.pack(">HH", dpi, dpi)
        insert_at = 4 + struct.unpack(">H", bytes(out[4:6]))[0]
    else:
        insert_at = 2
    text = comment.encode("utf-8")[:65000]
    com = b"\xff\xfe" + struct.pack(">H", len(text) + 2) + text
    return bytes(out[:insert_at]) + com + bytes(out[insert_at:])


def _png_chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)


def _png_with_density(data: bytes, dpi: int, comment: str) -> bytes:
    ppm = int(round(dpi / 0.0254))
    phys = _png_chunk(b"pHYs", struct.pack(">IIB", ppm, ppm, 1))
    text = _png_chunk(b"tEXt", b"Comment\x00" + comment.encode("latin-1", "replace"))
    end_of_ihdr = 8 + 25  # signature + IHDR chunk
    return data[:end_of_ihdr] + phys + text + data[end_of_ihdr:]
