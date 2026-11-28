"""Local, offline Sapiens-0.3B parsing; not an app segmentation backend.

    uv run --extra fba python -m tools.sapiens_reference --accept-noncommercial

The official model is licensed CC BY-NC 4.0. Photos never leave this machine.
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from tools.schp_reference import sha256

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "models" / "sapiens"
REVISION = "18e07d7474a00d30728f22de33325741ce27046f"
FILENAME = "sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_torchscript.pt2"
WEIGHTS_SHA256 = "735a9a8d63fe8f3f6a4ca3d787de07e69b1f9708ad550e09bb33c9854b7eafbc"
MODEL_URL = f"https://huggingface.co/facebook/sapiens-seg-0.3b-torchscript/resolve/{REVISION}/{FILENAME}"
CLOTHING_CLASSES = (1, 12, 22)
SIZE = (1024, 768)
MEAN = (123.5, 116.5, 103.5)
STD = (58.5, 57.0, 57.5)


def verify_assets(directory: Path = ASSETS):
    weights = directory / FILENAME
    if not weights.is_file():
        raise FileNotFoundError(
            "Sapiens assets missing. Run: uv run --extra fba python "
            "-m tools.sapiens_reference --accept-noncommercial"
        )
    if sha256(weights) != WEIGHTS_SHA256:
        raise ValueError(f"Sapiens weight checksum mismatch: {weights}")


def fetch_assets(directory: Path = ASSETS):
    directory.mkdir(parents=True, exist_ok=True)
    weights = directory / FILENAME
    if not weights.exists():
        part = weights.with_suffix(weights.suffix + ".part")
        print("Downloading official CC BY-NC 4.0 Sapiens weights (1.36 GB)...", flush=True)
        try:
            with urllib.request.urlopen(MODEL_URL) as response, part.open("wb") as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
            if sha256(part) != WEIGHTS_SHA256:
                raise ValueError(f"Downloaded Sapiens checksum mismatch: {part}")
            part.replace(weights)
        finally:
            part.unlink(missing_ok=True)
    verify_assets(directory)
    print(f"Verified Sapiens model revision {REVISION}.")


class SapiensReference:
    def __init__(self, directory: Path = ASSETS, device: str = "cpu",
                 layout: str = "stretch", gamma: float = 1.0):
        import torch

        if device not in ("cpu", "mps", "cuda"):
            raise ValueError("Sapiens device must be cpu, mps, or cuda.")
        if layout not in ("stretch", "letterbox"):
            raise ValueError("Sapiens layout must be stretch or letterbox.")
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError("Sapiens gamma must be finite and positive.")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("The MPS device is not available.")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("The CUDA device is not available.")
        verify_assets(directory)
        self.device = torch.device(device)
        self.layout = layout
        self.gamma = gamma
        self.model = torch.jit.load(str(directory / FILENAME), map_location="cpu").eval().to(self.device)

    def predict(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import torch
        import torch.nn.functional as functional

        if (rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8
                or min(rgb.shape[:2]) == 0):
            raise ValueError("Expected a nonempty RGB uint8 frame.")
        if self.gamma != 1.0:
            rgb = np.rint(255 * (rgb.astype(np.float32) / 255) ** self.gamma).astype(np.uint8)
        if self.layout == "letterbox":
            h, w = rgb.shape[:2]
            scale = min(SIZE[1] / w, SIZE[0] / h)
            rh = min(SIZE[0], max(1, round(h * scale)))
            rw = min(SIZE[1], max(1, round(w * scale)))
            top, left = (SIZE[0] - rh) // 2, (SIZE[1] - rw) // 2
            resized = np.zeros((*SIZE, 3), np.uint8)
            resized[top:top + rh, left:left + rw] = cv2.resize(
                rgb, (rw, rh), interpolation=cv2.INTER_LINEAR)
        else:
            resized = cv2.resize(rgb, (SIZE[1], SIZE[0]), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float()[None]
        mean = tensor.new_tensor(MEAN)[None, :, None, None]
        std = tensor.new_tensor(STD)[None, :, None, None]
        with torch.inference_mode():
            logits = self.model(((tensor - mean) / std).to(self.device))
            if (not isinstance(logits, torch.Tensor) or logits.ndim != 4
                    or logits.shape[0] != 1 or logits.shape[1] != 28
                    or not torch.isfinite(logits).all().item()):
                raise ValueError("Sapiens returned invalid 28-class logits.")
            if self.layout == "letterbox":
                logits = functional.interpolate(logits, size=SIZE, mode="bilinear", align_corners=False)
                logits = logits[:, :, top:top + rh, left:left + rw]
            logits = functional.interpolate(
                logits, size=rgb.shape[:2], mode="bilinear", align_corners=False)
            probabilities = logits.softmax(dim=1)
            maps = torch.stack(
                (probabilities[0, list(CLOTHING_CLASSES)].sum(dim=0), probabilities[0, 0]))
        maps = maps.cpu().numpy()
        return maps[0], maps[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--accept-noncommercial", action="store_true",
                        help="acknowledge the official model's CC BY-NC 4.0 restriction")
    args = parser.parse_args()
    if not args.accept_noncommercial:
        parser.error("The model is noncommercial only; acknowledge with --accept-noncommercial.")
    fetch_assets()
    return 0


if __name__ == "__main__":
    sys.exit(main())
