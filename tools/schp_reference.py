"""Local, offline SCHP-LIP reference inference; not an app segmentation backend.

    uv run --extra parsing python -m tools.schp_reference --download

The converted checkpoint's model card labels it MIT. The original SCHP source
is MIT; training-data and weight provenance should be reviewed before deployment.
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

import numpy as np

from idphoto import matting, npimg

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "models" / "schp"
REVISION = "7c0028fe34dcd8bd534ffa291a9086cdc856e4a9"
MODEL_URL = f"https://huggingface.co/pirocheto/schp-lip-20/resolve/{REVISION}/onnx"
FILES = {
    "schp-lip-20.onnx": "41c9d6c753a387f794bf3cbd9aed5e20c81a934fdc91731475bd46138937f84c",
    "schp-lip-20.onnx.data": "be931b91c76ed7707b5426abbccee98620a122b777c7cbd1f771a122803666e7",
}
SIZE = 473
CLOTHING_CLASSES = (5, 6, 7, 10, 11)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_assets(directory: Path = ASSETS):
    for name, expected in FILES.items():
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(
                "SCHP assets missing. Run: uv run --extra parsing python "
                "-m tools.schp_reference --download"
            )
        if sha256(path) != expected:
            raise ValueError(f"SCHP checksum mismatch: {path}")


def fetch_assets(directory: Path = ASSETS):
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        path = directory / name
        if path.exists():
            if sha256(path) != expected:
                raise ValueError(f"SCHP checksum mismatch: {path}")
            continue
        part = path.with_suffix(path.suffix + ".part")
        print(f"Downloading {name}...", flush=True)
        try:
            with urllib.request.urlopen(f"{MODEL_URL}/{name}") as response, part.open("wb") as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
            if sha256(part) != expected:
                raise ValueError(f"Downloaded SCHP checksum mismatch: {part}")
            part.replace(path)
        finally:
            part.unlink(missing_ok=True)
    verify_assets(directory)
    print(f"Verified SCHP model revision {REVISION}.")


class SCHPReference:
    def __init__(self, directory: Path = ASSETS, threads: int = 4):
        import onnxruntime as ort

        if threads < 1:
            raise ValueError("SCHP needs at least one CPU thread.")
        verify_assets(directory)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(directory / "schp-lip-20.onnx"), sess_options=options,
            providers=["CPUExecutionProvider"])

    def predict(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from PIL import Image

        if (rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8
                or min(rgb.shape[:2]) == 0):
            raise ValueError("Expected a nonempty RGB uint8 frame.")
        h, w = rgb.shape[:2]
        side = max(h, w)
        top, left = (side - h) // 2, (side - w) // 2
        bgr = rgb[..., ::-1]
        square = np.pad(bgr, ((top, side - h - top), (left, side - w - left), (0, 0)))
        resized = np.asarray(Image.fromarray(square).resize((SIZE, SIZE), Image.Resampling.BILINEAR))
        mean = np.array([0.406, 0.456, 0.485], np.float32)
        std = np.array([0.225, 0.224, 0.229], np.float32)
        tensor = ((resized.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)[None]
        logits = self.session.run(["logits"], {"pixel_values": tensor})[0]
        if logits.shape != (1, 20, SIZE, SIZE) or not np.isfinite(logits).all():
            raise ValueError("SCHP returned invalid LIP logits.")
        logits = logits[0]
        probabilities = np.exp(logits - logits.max(axis=0))
        probabilities /= probabilities.sum(axis=0)
        maps = np.stack((probabilities[list(CLOTHING_CLASSES)].sum(axis=0), probabilities[0]), axis=-1)
        maps = npimg.resize(maps, side, side, kernel="linear")[top:top + h, left:left + w]
        maps = np.clip(maps, 0.0, 1.0)
        return maps[..., 0], maps[..., 1]


def combine_with_main(result, lm: np.ndarray, clothing: np.ndarray,
                      background: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return garment-only and whole-person SCHP combinations with a main alpha."""
    alpha, face = result.alpha, result.face_mask
    if alpha is None or alpha.ndim != 2 or face is None or face.shape != alpha.shape or not face.any():
        raise ValueError("Main alpha and a nonempty face mask must match.")
    if (clothing.shape != alpha.shape or background.shape != alpha.shape
            or not np.isfinite(alpha).all() or not np.isfinite(clothing).all()
            or not np.isfinite(background).all()
            or any(a.min() < 0 or a.max() > 1 for a in (alpha, clothing, background))):
        raise ValueError("Alpha and SCHP probabilities must match and be finite in [0, 1].")
    f = matting.face_measures(lm)
    fw, ec, cx, chin_y = f["fw"], f["ec"], f["cx"], f["chin"][1]
    if not np.isfinite([fw, ec, cx, chin_y]).all() or fw <= 0 or ec <= 0:
        raise ValueError("Valid face dimensions and position are required.")
    yy, xx = matting._grid(alpha.shape)
    dx = np.abs(xx - cx)
    drop = np.sqrt(np.clip((yy - chin_y + 0.5 * ec) / ec, 0.0, 1.0))
    garment_gate = (yy > chin_y - 0.5 * ec) & (dx < (0.85 + 2.2 * drop) * fw)
    neck = (dx < 0.35 * fw) & (yy < chin_y + 1.15 * ec)
    body = (yy > chin_y + 0.1 * ec) & ~neck
    background_gate = np.clip((background - 0.65) / 0.3, 0.0, 1.0)
    main = np.where(body, alpha * (1.0 - background_gate), alpha)
    garment = np.clip((clothing - 0.15) / 0.7, 0.0, 1.0) * garment_gate
    person = np.clip((1.0 - background - 0.15) / 0.7, 0.0, 1.0)
    proposals = (person > 0.5) | (alpha > 0.5)
    small_face = npimg.downscale_mean(face.astype(np.float32), 4) > 0.5
    association = (matting.Matter._connected_part(proposals, face) if small_face.any()
                   else matting.connected(face, proposals))
    garment_result = np.maximum(main, garment * association)
    person_result = np.maximum(main, person * association)
    garment_result[face] = person_result[face] = 1.0
    return garment_result.astype(np.float32), person_result.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--download", action="store_true", help="download the pinned ~255 MiB model")
    args = parser.parse_args()
    if not args.download:
        parser.error("Use --download to fetch the local research model.")
    fetch_assets()
    return 0


if __name__ == "__main__":
    sys.exit(main())
