"""Compare fixed, local FBA/clothing combinations on the twelve evaluation photos."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from idphoto import color, matting
from idphoto.landmarks import FaceLandmarkDetector
from idphoto.pipeline import to_live
from tools.eval_clothing import (
    BASELINE, NAMES, ROOT, baseline_module, fba_composite, fba_trimap,
    guarded_fba_trimap, process,
)
from tools.fba_reference import FBAReference, validate_trimap


def refine_alpha(rgb, lm, main_trimap, clothing_trimap, fba_alpha, clothing_alpha):
    """Raise only mutually supported, unknown torso pixels; retain all known labels."""
    validate_trimap(rgb, main_trimap)
    validate_trimap(rgb, clothing_trimap)
    for alpha in (fba_alpha, clothing_alpha):
        if (alpha.shape != rgb.shape[:2] or not np.isfinite(alpha).all()
                or alpha.min() < 0 or alpha.max() > 1):
            raise ValueError("Alpha must match the frame and be finite in [0, 1].")
    gate = matting.clothing_regions(rgb.shape[:2], lm)[0]
    eligible = ((main_trimap == 128) & (clothing_trimap == 255)
                & (gate >= 0.95) & (clothing_alpha >= 0.9) & (fba_alpha > 0.75))
    result = fba_alpha.copy()
    result[eligible] = np.maximum(result[eligible], clothing_alpha[eligible])
    return result, eligible


def sheet(rows, titles, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), len(titles), figsize=(3 * len(titles), 2.4 * len(rows)),
                             squeeze=False, layout="constrained")
    for row, (name, images) in enumerate(rows):
        for col, image in enumerate(images):
            ax = axes[row, col]
            vmax = 1 if np.issubdtype(image.dtype, np.floating) else 255
            ax.imshow(image, cmap="gray" if image.ndim == 2 else None, vmin=0, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(titles[col], fontsize=10)
            if col == 0:
                ax.set_ylabel(name, fontsize=10)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=ROOT / "out/clothing-landmarks")
    parser.add_argument("--full-grid", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    import torch
    torch.set_num_threads(4)
    fba = FBAReference()
    baseline = baseline_module(BASELINE)
    detector = None
    measurements, panels, alphas = [], [], []
    try:
        for name in NAMES:
            path = ROOT / "docs/evaluation/test" / f"{name}.jpg"
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise FileNotFoundError(path)
            rgb, _ = to_live(bgr[..., ::-1].copy())
            cache = args.cache / f"{name}-live.npy"
            if cache.exists():
                lm = np.load(cache)
            else:
                if detector is None:
                    detector = FaceLandmarkDetector(video=False)
                lm = detector.detect(rgb)
                if lm is None:
                    raise RuntimeError(f"No face detected: {path}")
                np.save(cache, lm)
            before, _ = process(baseline, rgb, lm, not args.full_grid)
            clothing, _ = process(matting, rgb, lm, not args.full_grid)
            main_trimap = fba_trimap(rgb, lm, before)
            clothing_trimap = fba_trimap(rgb, lm, clothing)
            main_prediction, _ = fba.process(rgb, main_trimap, tile_size=1024, overlap=96)
            clothing_prediction, _ = fba.process(rgb, clothing_trimap, tile_size=1024, overlap=96)
            main_alpha = main_prediction[..., 0]
            post_alpha, eligible = refine_alpha(
                rgb, lm, main_trimap, clothing_trimap, main_alpha, clothing.alpha)
            post_prediction = main_prediction.copy()
            post_prediction[..., 0] = post_alpha
            guarded = guarded_fba_trimap(rgb, main_trimap, clothing_trimap, main_alpha)
            guarded_prediction, _ = fba.process(rgb, guarded, tile_size=1024, overlap=96)
            methods = {
                "main": before.alpha,
                "clothing": clothing.alpha,
                "fba_main": main_alpha,
                "fba_clothing": clothing_prediction[..., 0],
                "post_refine": post_alpha,
                "guarded_fba": guarded_prediction[..., 0],
            }
            row = {
                "photo": name, "mode": "full-grid" if args.full_grid else "live",
                "width": rgb.shape[1], "height": rgb.shape[0],
                "clothing_applied": clothing.stats["matting"]["clothing_applied"],
                "post_eligible_pixels": int(eligible.sum()),
                "post_changed_pixels": int((post_alpha > main_alpha).sum()),
                "guarded_promoted_pixels": int(((main_trimap == 128) & (guarded == 255)).sum()),
                "main_trimap_background_pixels": int((main_trimap == 0).sum()),
            }
            for label, alpha in methods.items():
                row[f"{label}_mean_alpha"] = float(alpha.mean())
                row[f"{label}_foreground_pixels_gt_0_5"] = int((alpha > 0.5).sum())
                row[f"{label}_vs_fba_added_gt_0_5"] = int(((alpha > 0.5) & (main_alpha <= 0.5)).sum())
                row[f"{label}_vs_fba_removed_gt_0_5"] = int(((alpha <= 0.5) & (main_alpha > 0.5)).sum())
                row[f"{label}_vs_fba_mean_abs_alpha_difference"] = float(np.abs(alpha - main_alpha).mean())
            measurements.append(row)
            panels.append((name, [rgb, color.encode_u8(before.output_frame),
                                  fba_composite(main_prediction), color.encode_u8(clothing.output_frame),
                                  fba_composite(clothing_prediction), fba_composite(post_prediction),
                                  fba_composite(guarded_prediction)]))
            alphas.append((name, [main_trimap, clothing_trimap, main_alpha, clothing.alpha,
                                  clothing_prediction[..., 0], post_alpha, guarded_prediction[..., 0]]))
            print(f"{row['mode']} {name}: post changed {row['post_changed_pixels']}, "
                  f"guarded promoted {row['guarded_promoted_pixels']}", flush=True)
    finally:
        if detector is not None:
            detector.close()
    with (args.out / "measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=measurements[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(measurements)
    sheet(panels, ["Original", "Main", "FBA: Main", "Clothing",
                   "FBA: clothing", "FBA + post", "FBA: guarded"],
          args.out / "comparison.jpg")
    sheet(alphas, ["Main trimap", "Clothing trimap", "FBA: Main", "Clothing alpha",
                   "FBA: clothing", "FBA + post", "FBA: guarded"],
          args.out / "alpha-detail.jpg")


if __name__ == "__main__":
    main()
