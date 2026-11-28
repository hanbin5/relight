"""Evaluate a conservative HSV/background-connectivity hypothesis offline."""

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from idphoto import color, config, matting, npimg
from idphoto.pipeline import to_live
from tools.eval_clothing import BASELINE, NAMES, ROOT, baseline_module, process, write_sheets

HSV_DISTANCE = 0.08


def hsv_features(rgb: np.ndarray) -> np.ndarray:
    """Encode hue as a saturation-weighted circle, with value as the third axis."""
    high, low = rgb.max(axis=2), rgb.min(axis=2)
    delta = high - low
    saturation = np.divide(delta, high, out=np.zeros_like(high), where=high > 0)
    hue = np.zeros_like(high)
    chromatic = delta > 0
    red, green, blue = (rgb[..., i] for i in range(3))
    red_max = chromatic & (high == red)
    green_max = chromatic & ~red_max & (high == green)
    blue_max = chromatic & ~red_max & ~green_max
    hue[red_max] = ((green - blue)[red_max] / delta[red_max]) % 6
    hue[green_max] = (blue - red)[green_max] / delta[green_max] + 2
    hue[blue_max] = (red - green)[blue_max] / delta[blue_max] + 4
    angle = hue * (np.pi / 3)
    return np.stack((saturation * np.cos(angle), saturation * np.sin(angle), high), axis=2)


class HSVProbeMatter(matting.Matter):
    """Test a foreground cut based on frame-connected, background-like colors."""

    def process(self, ctx, background=None, update=True, fast=True):
        if ctx.raw_frame is None or ctx.landmarks is None:
            raise ValueError("The HSV probe requires a frame and face landmarks.")
        scale = 2 if fast else 1
        self._rgb = npimg.downscale_mean(ctx.raw_frame.astype(np.float32) / 255, scale)
        self._lm = ctx.landmarks / np.float32(scale)
        self._removed = 0
        result = super().process(ctx, background, update, fast)
        result.stats["matting"]["hsv_probe_removed_coarse"] = self._removed
        return result

    def _connected_part(self, mask, face, q=4):
        gate, _ = matting.person_prior(mask.shape, self._lm)
        _, _, _, guard, _ = matting.clothing_regions(mask.shape, self._lm)
        seeds = (gate <= 0) & ~guard
        if seeds.sum() < 50:
            return super()._connected_part(mask, face, q)
        features = hsv_features(self._rgb)
        centers, _, _ = matting.kmeans(
            matting._sample(features, seeds, config.CLUSTER_SAMPLES, 3),
            config.BG_CLUSTERS, iterations=8)
        flat = features.reshape(-1, 3)
        distance = np.full(len(flat), np.inf, np.float32)
        for center in centers:
            distance = np.minimum(distance, np.linalg.norm(flat - center, axis=1))
        background_like = distance.reshape(mask.shape) < HSV_DISTANCE
        f = matting.face_measures(self._lm)
        yy, xx = matting._grid(mask.shape)
        drop = np.sqrt(np.clip((yy - f["chin"][1] - 0.2 * f["ec"]) / (0.8 * f["ec"]), 0, 1))
        core = ((yy > f["chin"][1] - 0.1 * f["ec"])
                & (np.abs(xx - f["cx"]) < (0.65 + 0.9 * drop) * f["fw"]))
        connected_background = matting.connected(seeds, background_like & ~core & ~face)
        cut = connected_background & (yy > f["chin"][1] + 0.2 * f["ec"]) & mask
        self._removed = int(cut.sum())
        return super()._connected_part(mask & ~cut, face, q)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=ROOT / "out/clothing-landmarks")
    parser.add_argument("--full-grid", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    main_module = baseline_module(BASELINE)
    candidate_module = SimpleNamespace(Matter=HSVProbeMatter)
    panels, rows = [], []
    for name in NAMES:
        path = ROOT / "docs/evaluation/test" / f"{name}.jpg"
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(path)
        rgb, _ = to_live(bgr[..., ::-1].copy())
        lm = np.load(args.cache / f"{name}-live.npy")
        main_result, _ = process(main_module, rgb, lm, not args.full_grid)
        current, _ = process(matting, rgb, lm, not args.full_grid)
        candidate, candidate_ms = process(candidate_module, rgb, lm, not args.full_grid)
        row = {
            "photo": name, "mode": "full-grid" if args.full_grid else "live",
            "candidate_vs_numpy_added": int(((candidate.alpha > .5) & (current.alpha <= .5)).sum()),
            "candidate_vs_numpy_removed": int(((current.alpha > .5) & (candidate.alpha <= .5)).sum()),
            "removed_before_cleanup_coarse": candidate.stats["matting"]["hsv_probe_removed_coarse"],
            "candidate_ms": candidate_ms,
        }
        rows.append(row)
        panels.append((name, rgb, color.encode_u8(main_result.output_frame),
                       color.encode_u8(current.output_frame), color.encode_u8(candidate.output_frame)))
        print(row, flush=True)
    write_sheets(panels, ["Original", "Main", "Clothing (NumPy)", "HSV + pose + connectivity"],
                 "comparison", args.out)
    with (args.out / "measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
