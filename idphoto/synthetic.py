"""Synthetic test frames rendered from MediaPipe's canonical face mesh.

Like Figure 3 of the plan, a matte face is lit by a chosen light (for example a side
window) in front of a plain wall, with hair, neck and shoulders. The renderer returns the
frame, its exact 478 landmarks and the true person mask, so every module can be tested
without a camera or the landmark detector, and staged faults (closed eyes, open mouth,
glare, blur, a top the color of the wall) can be produced on demand.
"""

from dataclasses import dataclass, field

import numpy as np

from . import color, mesh, npimg


@dataclass
class Scene:
    width: int = 640
    height: int = 480
    face_height: float = 0.36  # chin to forehead mark, as a share of the frame height
    center: tuple = (0.5, 0.42)  # face center (x, y) as shares of the frame
    yaw: float = 0.0  # degrees; + turns the face toward the image right
    pitch: float = 0.0  # degrees; + tilts the chin down
    roll: float = 0.0  # degrees; + tilts clockwise in the image
    light: tuple = (-0.75, 0.25, 0.6)  # key light direction (x right, y up, z to the camera)
    key: float = 0.85
    ambient: float = 0.12
    skin: str = "#e0b49a"
    hair: str = "#2b2120"
    top: str = "#2f3b52"
    wall: str = "#dcdad6"
    noise: float = 1.2  # sensor noise, in 8-bit levels
    eyes_open: bool = True
    mouth_open: float = 0.0  # inner-lip gap as a share of mouth width
    glare: bool = False  # bright reflection on the left lens
    blur: int = 0  # box-blur radius in pixels
    seed: int = 0
    extras: dict = field(default_factory=dict)


def _rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    y, p, r = np.radians([yaw, pitch, roll])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    rz = np.array([[np.cos(r), np.sin(r), 0], [-np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _canonical_with_iris() -> np.ndarray:
    """(478, 3) canonical points: 468 mesh vertices plus the two iris centers and rings."""
    v, _ = mesh.canonical_mesh()
    pts = [v]
    for eye in (mesh.EYE_R, mesh.EYE_L):  # 468-472 right iris, 473-477 left iris
        c = v[list(eye)].mean(axis=0) + np.array([0.0, 0.05, 0.25])
        rad = 0.55
        pts.append(np.array([c, c + [rad, 0, 0], c + [0, rad, 0], c - [rad, 0, 0], c - [0, rad, 0]]))
    return np.concatenate(pts).astype(np.float64)


def landmarks_for(scene: Scene) -> np.ndarray:
    """Exact landmarks (478, 3) of the scene's face, in pixels (MediaPipe convention)."""
    v = _canonical_with_iris() @ _rotation(scene.yaw, scene.pitch, scene.roll).T
    can = mesh.canonical_mesh()[0]
    s = scene.face_height * scene.height / float(can[mesh.FOREHEAD, 1] - can[mesh.CHIN, 1])
    cx, cy = scene.center[0] * scene.width, scene.center[1] * scene.height
    lm = np.stack([cx + s * v[:, 0], cy - s * v[:, 1], -s * v[:, 2]], axis=1)
    if not scene.eyes_open:  # upper lids down onto the lower lids
        for upper, lower in ((160, 144), (158, 153), (159, 145), (161, 163), (157, 154),
                             (385, 380), (387, 373), (386, 374), (384, 381), (388, 390)):
            lm[upper, :2] = lm[lower, :2] + 0.15 * (lm[upper, :2] - lm[lower, :2])
    # The canonical mesh has parted lips; close them, then open by mouth_open if asked.
    for upper, lower in ((13, 14), (82, 87), (81, 178), (80, 88), (191, 95),
                         (312, 317), (311, 402), (310, 318), (415, 324)):
        mid = 0.5 * (lm[upper, :2] + lm[lower, :2])
        lm[upper, :2] = lm[lower, :2] = mid
    if scene.mouth_open > 0:
        width = np.linalg.norm(lm[mesh.MOUTH_L, :2] - lm[mesh.MOUTH_R, :2])
        gap = scene.mouth_open * width
        for i in (14, 87, 178, 88, 95, 317, 402, 318, 324, 17, 84, 181, 91, 146, 314, 405, 321, 375):
            lm[i, 1] += gap
    return lm.astype(np.float32)


def render(scene: Scene):
    """Render a scene. Returns (rgb uint8 (H, W, 3), landmarks (478, 3), person mask bool (H, W))."""
    h, w = scene.height, scene.width
    rng = np.random.default_rng(scene.seed)
    lm = landmarks_for(scene)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) + 0.5
    light = np.asarray(scene.light, np.float64)
    light /= np.linalg.norm(light)

    eye_y = 0.5 * (lm[mesh.IRIS_R, 1] + lm[mesh.IRIS_L, 1])
    chin = lm[mesh.CHIN, :2]
    ec = chin[1] - eye_y
    fw = lm[mesh.FACE_EDGE_L, 0] - lm[mesh.FACE_EDGE_R, 0]
    cx = 0.5 * (lm[mesh.FACE_EDGE_L, 0] + lm[mesh.FACE_EDGE_R, 0])

    def shade(nx, ny, nz):
        return scene.ambient + scene.key * np.maximum(nx * light[0] + ny * light[1] + nz * light[2], 0.0)

    # Wall: brighter on the side the light comes from.
    wall = color.hex_to_linear(scene.wall)
    wall_shade = 0.62 + 0.25 * light[0] * (xx / w - 0.5) * 2 + 0.05 * (1 - yy / h)
    img = wall[None, None, :] * wall_shade[..., None]

    # Shoulders and torso (an ellipsoid-like surface below the chin).
    top_y = chin[1] + 0.62 * ec
    half = 0.32 * fw + 1.55 * fw * np.sqrt(np.clip((yy - top_y) / (1.1 * ec), 0, 1))
    torso = (yy > top_y) & (np.abs(xx - cx) < half)
    tx = np.clip((xx - cx) / (2.2 * fw), -0.95, 0.95)
    tz = np.sqrt(1 - tx**2)
    top_col = color.hex_to_linear(scene.top)
    img[torso] = (top_col * shade(tx, 0.25 * tz, tz)[..., None])[torso]

    # Neck.
    neck = (yy > chin[1] - 0.45 * ec) & (yy < top_y + 0.35 * ec) & (np.abs(xx - cx) < 0.3 * fw)
    nx = np.clip((xx - cx) / (0.34 * fw), -0.95, 0.95)
    skin = color.hex_to_linear(scene.skin)
    img[neck] = (skin * 0.85 * shade(nx, -0.15, np.sqrt(1 - nx**2))[..., None])[neck]

    # Hair: an ellipse around the upper head, about 1 cm above the skull (crown ~ eye - ec).
    hcy = eye_y - 0.2 * ec
    ax, ay = 0.6 * fw, 0.95 * ec
    e = ((xx - cx) / ax) ** 2 + ((yy - hcy) / ay) ** 2
    hair = (e < 1.0) & (yy < eye_y + 0.35 * ec)
    hx = np.clip((xx - cx) / ax, -0.95, 0.95)
    hy = np.clip((hcy - yy) / ay, -0.95, 0.95)
    hz = np.sqrt(np.clip(1 - hx**2 - hy**2, 0.05, 1))
    hair_col = color.hex_to_linear(scene.hair)
    img[hair] = (hair_col * shade(hx, hy, hz)[..., None])[hair]

    # Face: rasterized mesh, matte shading from interpolated vertex normals.
    tri_id, bary = npimg.rasterize_triangles(lm[:468, :2], lm[:468, 2], mesh.triangles(), (h, w))
    face = tri_id >= 0
    vn = mesh.vertex_normals(lm)
    n = np.einsum("pk,pkc->pc", bary[face], vn[mesh.triangles()[tri_id[face]]])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    albedo = np.empty((h, w, 3), np.float32)
    albedo[:] = skin
    texture = npimg.box_mean(rng.normal(0, 1, (h, w)).astype(np.float32), 1)
    albedo *= (1.0 + 0.06 * texture)[..., None]
    brows = mesh.polygon(lm, mesh.BROW_R, (h, w)) | mesh.polygon(lm, mesh.BROW_L, (h, w))
    albedo[brows] = hair_col * 1.6
    lips = mesh.polygon(lm, mesh.LIPS, (h, w))
    albedo[lips] = color.hex_to_linear("#b5615c")
    if scene.mouth_open > 0:
        inner = mesh.polygon(lm, (78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95), (h, w))
        albedo[inner] = color.hex_to_linear("#3a1414")
    if scene.eyes_open:
        for eye, iris in ((mesh.EYE_R, mesh.IRIS_R), (mesh.EYE_L, mesh.IRIS_L)):
            opening = mesh.polygon(lm, eye, (h, w))
            albedo[opening] = color.hex_to_linear("#e8e4e0")
            r_iris = np.linalg.norm(lm[iris + 1, :2] - lm[iris, :2])
            disk = ((xx - lm[iris, 0]) ** 2 + (yy - lm[iris, 1]) ** 2 < r_iris**2) & opening
            albedo[disk] = color.hex_to_linear("#4a3226")
    shading = np.zeros((h, w), np.float32)
    shading[face] = shade(n[:, 0], n[:, 1], n[:, 2])
    img[face] = albedo[face] * shading[face][:, None]

    if scene.glare:  # a bright patch on the subject's left lens
        gx, gy = lm[mesh.IRIS_L, 0] + 0.12 * fw, lm[mesh.IRIS_L, 1] - 0.05 * ec
        spot = ((xx - gx) / (0.07 * fw)) ** 2 + ((yy - gy) / (0.045 * fw)) ** 2 < 1
        img[spot] = 1.0

    if scene.blur > 0:
        img = npimg.box_mean(npimg.box_mean(img.astype(np.float32), scene.blur), scene.blur)

    rgb = color.encode_u8(img.astype(np.float32)).astype(np.float32)
    rgb += rng.normal(0, scene.noise, rgb.shape)
    rgb = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
    person = torso | neck | hair | face
    return rgb, lm, person
