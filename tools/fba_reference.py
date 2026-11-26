"""Pinned, local FBA Matting reference inference; not a production app backend.

    uv run --extra fba python tools/fba_reference.py --accept-noncommercial

The authors' code is MIT licensed; their released weights are noncommercial only.
Code and weights stay in the ignored models/ directory. Images never leave the machine.
"""

import argparse
import hashlib
import importlib
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "models" / "fba"
REVISION = "8dd100d760186150710510ee8d78ed916cdb17a4"
WEIGHTS_ID = "1T_oiKDE_biWf2kqexMEN7ObWqtXAzbB1"
WEIGHTS_SHA256 = "f3f8a2b6cefd7200368b6008a712468ed1debc6658693db31b0f719f658a2967"
SOURCE_URL = "https://github.com/MarcoForte/FBA_Matting.git"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_assets(directory: Path = ASSETS):
    source, weights = directory / "source", directory / "FBA.pth"
    if not source.is_dir() or not weights.is_file():
        raise FileNotFoundError(
            "FBA assets missing. Run: uv run --extra fba python "
            "tools/fba_reference.py --accept-noncommercial"
        )
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if revision != REVISION or dirty:
        raise ValueError(f"FBA source must be unmodified at {REVISION}: {source}")
    if sha256(weights) != WEIGHTS_SHA256:
        raise ValueError(f"FBA weight checksum mismatch: {weights}")


def fetch_assets(directory: Path = ASSETS):
    import gdown

    directory.mkdir(parents=True, exist_ok=True)
    source, weights = directory / "source", directory / "FBA.pth"
    if not source.exists():
        subprocess.run(["git", "clone", "--quiet", "--no-checkout", SOURCE_URL, str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--quiet", "--detach", REVISION], check=True)
    if not weights.exists():
        part = directory / "FBA.pth.part"
        print("Downloading official research-only FBA weights (139 MB)...", flush=True)
        if gdown.download(id=WEIGHTS_ID, output=str(part), quiet=True) is None:
            raise RuntimeError("Official FBA weight download failed.")
        if sha256(part) != WEIGHTS_SHA256:
            raise ValueError(f"Downloaded weight checksum mismatch; not installing {part}")
        part.replace(weights)
    verify_assets(directory)
    print(f"Verified FBA source {REVISION} and weights {WEIGHTS_SHA256}.")


def validate_trimap(rgb: np.ndarray, trimap: np.ndarray):
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected RGB uint8 (H, W, 3).")
    if trimap.shape != rgb.shape[:2] or trimap.dtype != np.uint8 or not np.isin(trimap, (0, 128, 255)).all():
        raise ValueError("Trimap must match the input: uint8, 0=BG, 128=unknown, 255=FG.")
    if not (trimap == 0).any() or not (trimap == 255).any():
        raise ValueError("Trimap needs both known foreground and known background.")


def native_tiles(rgb: np.ndarray, trimap: np.ndarray, predict, tile_size: int = 768,
                 overlap: int = 96) -> tuple[np.ndarray, int]:
    """Blend native-resolution predictions. Never downsample the image or trimap."""
    validate_trimap(rgb, trimap)
    if tile_size < 32 or overlap < 0 or 2 * overlap >= tile_size:
        raise ValueError("Tiles require size >=32 and 0 <= overlap < size/2.")
    h, w = trimap.shape
    step = tile_size - 2 * overlap

    def starts(length):
        return sorted(set([*range(0, max(1, length - tile_size + 1), step), max(0, length - tile_size)]))

    output = np.zeros((h, w, 7), np.float32)
    norm = np.zeros((h, w), np.float32)
    count = 0
    for y in starts(h):
        for x in starts(w):
            part = trimap[y:y + tile_size, x:x + tile_size]
            if not (part == 128).any():
                continue
            image = rgb[y:y + tile_size, x:x + tile_size]
            prediction = predict(image, part)
            if prediction.shape != (*part.shape, 7) or not np.isfinite(prediction).all():
                raise ValueError("FBA predictor returned invalid native-resolution output.")
            if prediction.min() < -1e-4 or prediction.max() > 1.0001:
                raise ValueError("FBA output is outside the expected [0, 1] range.")
            ph, pw = part.shape
            window = np.maximum(np.hanning(ph), 0.01)[:, None] * np.maximum(np.hanning(pw), 0.01)[None, :]
            output[y:y + ph, x:x + pw] += prediction * window[..., None]
            norm[y:y + ph, x:x + pw] += window
            count += 1
    unknown = trimap == 128
    if (norm[unknown] <= 0).any():
        raise RuntimeError("Native tiles did not cover every unknown pixel.")
    output[unknown] /= norm[unknown, None]
    original = rgb.astype(np.float32) / 255.0
    fg, bg = trimap == 255, trimap == 0
    output[fg, 0], output[bg, 0] = 1.0, 0.0
    output[fg, 1:4], output[bg, 4:7] = original[fg], original[bg]
    return np.clip(output, 0.0, 1.0), count


def _exact_adaptive_pool(size: int):
    """MPS lacks adaptive pooling for non-divisible dimensions; use exact bin averages."""
    import torch

    class ExactPool(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cache = {}

        def forward(self, x):
            h, w = x.shape[-2:]
            key = (h, w, x.device, x.dtype)
            if key not in self.cache:
                def weights(length):
                    result = torch.zeros((size, length), dtype=x.dtype)
                    for i in range(size):
                        start = i * length // size
                        end = ((i + 1) * length + size - 1) // size
                        result[i, start:end] = 1 / (end - start)
                    return result.to(x.device)

                if len(self.cache) >= 16:
                    self.cache.clear()
                self.cache[key] = weights(h), weights(w).T
            left, right = self.cache[key]
            return torch.matmul(left, torch.matmul(x, right))

    return ExactPool()


class FBAReference:
    def __init__(self, directory: Path = ASSETS, device: str = "cpu"):
        import torch

        if device not in ("cpu", "mps"):
            raise ValueError(f"Unsupported FBA device: {device}")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("FBA MPS acceleration is unavailable on this system; select CPU.")
        self.device = torch.device(device)
        verify_assets(directory)
        source = (directory / "source").resolve()
        sys.path.insert(0, str(source))
        try:
            models = importlib.import_module("networks.models")
        finally:
            sys.path.remove(str(source))
        if not Path(models.__file__).resolve().is_relative_to(source):
            raise RuntimeError("A different 'networks' package is already imported; use a fresh process.")
        self.model = models.build_model("default")
        state = torch.load(directory / "FBA.pth", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        if device == "mps":
            for block in self.model.decoder.ppm:
                block[0] = _exact_adaptive_pool(block[0].output_size)
            self.model.to(self.device)

    def predict(self, rgb: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        import torch

        # Padding, unlike the upstream demo's resize-to-eight, keeps the original pixel grid.
        h, w = trimap.shape
        image = np.pad(rgb, ((0, -h % 8), (0, -w % 8), (0, 0)), mode="edge").astype(np.float32) / 255.0
        labels = np.pad(trimap, ((0, -h % 8), (0, -w % 8)), mode="edge")
        known = np.stack([labels == 0, labels == 255], axis=-1).astype(np.float32)
        features = []
        for label in (0, 255):
            distance = cv2.distanceTransform((labels != label).astype(np.uint8), cv2.DIST_L2, 0)
            for fraction in (0.02, 0.08, 0.16):
                features.append(np.exp(-0.5 * (distance / (fraction * 320)) ** 2))

        def tensor(array):
            return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)[None].to(self.device)

        image_t, known_t = tensor(image), tensor(known)
        mean = image_t.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = image_t.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        with torch.inference_mode():
            output = self.model(image_t, known_t, (image_t - mean) / std,
                                tensor(np.stack(features, axis=-1)))
        return output[0, :, :h, :w].permute(1, 2, 0).cpu().numpy()

    def process(self, rgb: np.ndarray, trimap: np.ndarray, tile_size: int = 768,
                overlap: int = 96) -> tuple[np.ndarray, int]:
        return native_tiles(rgb, trimap, self.predict, tile_size, overlap)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--accept-noncommercial", action="store_true",
                        help="acknowledge the official weights' noncommercial research restriction")
    args = parser.parse_args()
    if not args.accept_noncommercial:
        parser.error("Released weights are noncommercial only; acknowledge with --accept-noncommercial.")
    fetch_assets()
    return 0


if __name__ == "__main__":
    sys.exit(main())
