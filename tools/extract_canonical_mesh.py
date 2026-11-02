"""Extract MediaPipe's canonical face mesh into idphoto/data/canonical_face_mesh.npz.

    uv run python tools/extract_canonical_mesh.py

The Face Landmarker bundle (models/face_landmarker.task) ships the canonical 468-vertex
face mesh used by MediaPipe's face-geometry pipeline as a binary protobuf. We need its
triangle list to turn landmarks into per-pixel normals (module A) and its 3D vertices to
render synthetic test faces. The protobuf is decoded with a minimal wire-format reader so
no proto definitions are needed. The output is checked in, so this only reruns when the
model changes.
"""

import struct
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "models" / "face_landmarker.task"
OUT = ROOT / "idphoto" / "data" / "canonical_face_mesh.npz"
MEMBER = "geometry_pipeline_metadata_landmarks.binarypb"


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = buf[i]
        i += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if byte < 0x80:
            return result, i


def _fields(buf: bytes):
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _varint(buf, i)
        elif wire == 1:
            value, i = buf[i : i + 8], i + 8
        elif wire == 2:
            n, i = _varint(buf, i)
            value, i = buf[i : i + n], i + n
        elif wire == 5:
            value, i = buf[i : i + 4], i + 4
        else:
            raise ValueError(f"unsupported wire type {wire}")
        yield field, wire, value


def _repeated_floats(wire: int, value) -> list[float]:
    if wire == 5:
        return [struct.unpack("<f", value)[0]]
    return list(struct.unpack(f"<{len(value) // 4}f", value))  # packed


def _repeated_uints(wire: int, value) -> list[int]:
    if wire == 0:
        return [value]
    out, j = [], 0  # packed
    while j < len(value):
        v, j = _varint(value, j)
        out.append(v)
    return out


def main() -> int:
    data = zipfile.ZipFile(TASK).read(MEMBER)
    # GeometryPipelineMetadata: field 1 = canonical_mesh (Mesh3d), field 2 = procrustes basis.
    mesh = next(v for f, w, v in _fields(data) if f == 1 and w == 2)
    verts: list[float] = []
    index: list[int] = []
    for f, w, v in _fields(mesh):
        if f == 3:  # vertex_buffer: x, y, z, u, v per vertex
            verts += _repeated_floats(w, v)
        elif f == 4:  # index_buffer: triangles
            index += _repeated_uints(w, v)
    vb = np.asarray(verts, np.float32).reshape(-1, 5)
    tris = np.asarray(index, np.int32).reshape(-1, 3)

    # Canonical frame: x to the image right, y up, z toward the camera. Make every
    # triangle wind so its normal points toward the camera.
    p = vb[:, :3]
    n = np.cross(p[tris[:, 1]] - p[tris[:, 0]], p[tris[:, 2]] - p[tris[:, 0]])
    if n[:, 2].sum() < 0:
        tris = tris[:, ::-1].copy()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, vertices=p, uv=vb[:, 3:], triangles=tris)
    print(f"{len(p)} vertices, {len(tris)} triangles -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
