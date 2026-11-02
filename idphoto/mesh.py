"""Face-mesh topology and per-pixel geometry from MediaPipe's 478 landmarks.

Landmark numbers follow the MediaPipe face mesh (468 face points + 10 iris points).
"Right" and "left" are the subject's: in an unmirrored camera image the subject's left
side appears on the image right. Triangles come from MediaPipe's canonical face mesh,
extracted by tools/extract_canonical_mesh.py.

Normals live in the light frame used by the spherical-harmonic model: x to the image
right, y up, z toward the camera.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np

from . import npimg

_MESH_FILE = Path(__file__).parent / "data" / "canonical_face_mesh.npz"

NOSE_TIP, FOREHEAD, CHIN = 1, 10, 152
FACE_EDGE_R, FACE_EDGE_L = 234, 454
IRIS_R, IRIS_L = 468, 473
EAR_R = (33, 160, 158, 133, 153, 144)  # p1..p6, p1 and p4 are the eye corners
EAR_L = (362, 385, 387, 263, 373, 380)
LIP_UPPER_INNER, LIP_LOWER_INNER = 13, 14
MOUTH_R, MOUTH_L = 61, 291

FACE_OVAL = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
             400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
             54, 103, 67, 109)
EYE_R = (33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7)
EYE_L = (263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249)
BROW_R = (70, 63, 105, 66, 107, 55, 65, 52, 53, 46)
BROW_L = (300, 293, 334, 296, 336, 285, 295, 282, 283, 276)
LIPS = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39,
        40, 185)
# Cheek polygons sit below the lower orbital rim so glasses frames stay outside them.
CHEEK_R = (123, 50, 36, 205, 187, 147)  # appears on the image left
CHEEK_L = (352, 280, 266, 425, 411, 376)  # appears on the image right


@lru_cache(maxsize=1)
def canonical_mesh():
    """(vertices (468, 3) in the light frame, triangles (898, 3)) of the canonical face."""
    data = np.load(_MESH_FILE)
    return data["vertices"].astype(np.float32), data["triangles"].astype(np.int64)


def triangles() -> np.ndarray:
    return canonical_mesh()[1]


@lru_cache(maxsize=1)
def canonical_pitch_ratio() -> float:
    """d(10, 1) / d(1, 152) on the canonical face: the value of a level head."""
    v = canonical_mesh()[0]
    return float((v[FOREHEAD, 1] - v[NOSE_TIP, 1]) / (v[NOSE_TIP, 1] - v[CHIN, 1]))


def face_box(lm: np.ndarray, pad: float = 1.08):
    """Square box (x0, y0, x1, y1) around the 468 face points."""
    p = lm[:468, :2]
    (x0, y0), (x1, y1) = p.min(axis=0), p.max(axis=0)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    half = 0.5 * pad * max(x1 - x0, y1 - y0)
    return (float(cx - half), float(cy - half), float(cx + half), float(cy + half))


def to_grid(lm: np.ndarray, box, size: int) -> np.ndarray:
    """Landmarks mapped onto a size x size grid covering box; z is scaled alike."""
    s = size / (box[2] - box[0])
    out = lm.astype(np.float32).copy()
    out[:, 0] = (lm[:, 0] - box[0]) * s
    out[:, 1] = (lm[:, 1] - box[1]) * s
    out[:, 2] = lm[:, 2] * s
    return out


def vertex_normals(lm: np.ndarray) -> np.ndarray:
    """Unit normals (468, 3) at the mesh vertices, in the light frame."""
    q = np.stack([lm[:468, 0], -lm[:468, 1], -lm[:468, 2]], axis=1).astype(np.float64)
    t = triangles()
    fn = np.cross(q[t[:, 1]] - q[t[:, 0]], q[t[:, 2]] - q[t[:, 0]])  # area-weighted
    vn = np.zeros_like(q)
    for i in range(3):
        np.add.at(vn, t[:, i], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
    if vn[:, 2].mean() < 0:  # mirrored input flips the winding
        vn = -vn
    return vn.astype(np.float32)


def mesh_normals(lm: np.ndarray, box, size: int):
    """Per-pixel normals (size, size, 3) on the face grid and the mesh coverage mask."""
    g = to_grid(lm, box, size)
    tri_id, bary = npimg.rasterize_triangles(g[:468, :2], g[:468, 2], triangles(), (size, size))
    cov = tri_id >= 0
    vn = vertex_normals(lm)
    n = np.einsum("pk,pkc->pc", bary[cov], vn[triangles()[tri_id[cov]]])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    normals = np.zeros((size, size, 3), np.float32)
    normals[cov] = n
    return normals, cov


def polygon(lm: np.ndarray, ids, shape) -> np.ndarray:
    return npimg.fill_polygon(shape, lm[list(ids), :2])


def region_masks(g: np.ndarray, shape) -> dict:
    """Face regions on a grid, from landmarks already mapped onto it (see to_grid)."""
    eyes = polygon(g, EYE_R, shape) | polygon(g, EYE_L, shape)
    brows = polygon(g, BROW_R, shape) | polygon(g, BROW_L, shape)
    lips = polygon(g, LIPS, shape)
    return {
        "oval": polygon(g, FACE_OVAL, shape),
        "eyes": eyes,
        "features": npimg.dilate(eyes | brows | lips, 1),
        "cheek_r": polygon(g, CHEEK_R, shape),
        "cheek_l": polygon(g, CHEEK_L, shape),
    }


def cheek_balance(Y: np.ndarray, regions: dict, valid=None) -> float:
    """R_LR = mean Y of the subject's left cheek / (mean of the right cheek + eps)."""
    left, right = regions["cheek_l"], regions["cheek_r"]
    if valid is not None:
        left, right = left & valid, right & valid
    if left.sum() < 4 or right.sum() < 4:
        return float("nan")
    return float(Y[left].mean() / (Y[right].mean() + 1e-6))


def shadow_share(Y: np.ndarray, skin: np.ndarray, factor: float) -> float:
    """Share of skin pixels darker than factor x the skin mean."""
    if skin.sum() == 0:
        return float("nan")
    v = Y[skin]
    return float((v < factor * v.mean()).mean())
