"""Compare fixed color-supported FBA constraints; local research only."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from idphoto import color, config, matting, npimg
from idphoto.pipeline import to_live
from tools.eval_clothing import NAMES, ROOT, fba_trimap, process
from tools.eval_fba_combinations import sheet
from tools.eval_fba_leakage import compose
from tools.fba_reference import FBAReference


def background_evidence(rgb, lm, trimap):
    """Compare color samples from known background with the central torso."""
    f = matting.face_measures(lm)
    if not np.isfinite([f["fw"], f["ec"], f["cx"], f["chin"][1]]).all() or min(f["fw"], f["ec"]) <= 0:
        raise ValueError("Valid face geometry is required.")
    if trimap.shape != rgb.shape[:2]:
        raise ValueError("Trimap must match frame.")
    yy, xx = matting._grid(trimap.shape)
    torso = ((yy > f["chin"][1] + .25 * f["ec"])
             & (np.abs(xx - f["cx"]) < .65 * f["fw"]) & (trimap == 255))
    bg = trimap == 0
    sample_counts = (int(bg.sum()), int(torso.sum()))
    if min(sample_counts) < 50:
        return np.zeros(trimap.shape, bool), sample_counts
    col = np.sqrt(color.decode_u8(rgb))
    bg_model = matting.kmeans(matting._sample(col, bg, 2000, 3), 8, iterations=8)
    fg_model = matting.kmeans(matting._sample(col, torso, 2000, 4), 5, iterations=8)
    flat = col.reshape(-1, 3)
    db = matting.nearest_distance(flat, *bg_model[:2], np.sqrt(config.BG_SHADOW_SCALE))
    df = matting.nearest_distance(flat, *fg_model[:2], np.sqrt(config.CLOTHING_SHADE_SCALE))
    evidence = ((db < 1.0) & (df > 2.5)).reshape(trimap.shape)
    return evidence & (yy > f["chin"][1] + .2 * f["ec"]), sample_counts


def candidates(trimap, evidence):
    """Unknown -> BG, or additionally demote conflicting FG to unknown."""
    if trimap.shape != evidence.shape:
        raise ValueError("Evidence must match trimap.")
    unknown_bg = trimap.copy()
    unknown_bg[(trimap == 128) & evidence] = 0
    demote = unknown_bg.copy()
    demote[(trimap == 255) & evidence] = 128
    return unknown_bg, demote


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=ROOT / "out/clothing-landmarks")
    parser.add_argument("--full-grid", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(4)
    fba = FBAReference()
    rows, panels, masks = [], [], []
    for name in NAMES:
        path = ROOT / "docs/evaluation/test" / f"{name}.jpg"
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(path)
        rgb, _ = to_live(bgr[..., ::-1].copy())
        lm = np.load(args.cache / f"{name}-live.npy")
        ctx, _ = process(matting, rgb, lm, not args.full_grid)
        trimap = fba_trimap(rgb, lm, ctx)
        evidence, (bg_samples, torso_samples) = background_evidence(rgb, lm, trimap)
        unknown_bg, demote = candidates(trimap, evidence)
        base, _ = fba.process(rgb, trimap, tile_size=768, overlap=96)
        radius = max(1, round(.25 * matting.face_measures(lm)["fw"]))
        protected = npimg.dilate((base[..., 0] > .8) | (trimap == 255), radius)
        buffered, _ = candidates(trimap, evidence & ~protected)
        bg_result, _ = fba.process(rgb, unknown_bg, tile_size=768, overlap=96)
        demote_result, _ = fba.process(rgb, demote, tile_size=768, overlap=96)
        buffered_result, _ = fba.process(rgb, buffered, tile_size=768, overlap=96)
        soft_result = base.copy()
        soft_result[..., 0] *= np.where(evidence & (trimap == 128), .5, 1)
        predictions = {"unknown_bg": bg_result, "demote": demote_result,
                       "buffered": buffered_result, "soft": soft_result}
        row = {"photo": name, "mode": "full-grid" if args.full_grid else "live",
               "width": rgb.shape[1], "height": rgb.shape[0],
               "background_sample_pixels": bg_samples, "torso_sample_pixels": torso_samples,
               "evidence_model_applied": min(bg_samples, torso_samples) >= 50,
               "evidence_pixels": int(evidence.sum()),
               "unknown_to_bg": int(((trimap == 128) & evidence).sum()),
               "buffered_unknown_to_bg": int(((trimap == 128) & evidence & ~protected).sum()),
               "fg_demoted": int(((trimap == 255) & evidence).sum())}
        alpha = base[..., 0]
        for label, prediction in predictions.items():
            other = prediction[..., 0]
            row[f"{label}_added_gt_0_5"] = int(((other > .5) & (alpha <= .5)).sum())
            row[f"{label}_removed_gt_0_5"] = int(((other <= .5) & (alpha > .5)).sum())
            row[f"{label}_mean_abs_alpha_difference"] = float(np.abs(other - alpha).mean())
        rows.append(row)
        panels.append((name, [rgb, compose(base), compose(bg_result),
                              compose(demote_result), compose(buffered_result), compose(soft_result)]))
        masks.append((name, [rgb, trimap, evidence.astype(np.float32), alpha,
                             bg_result[..., 0], demote_result[..., 0], buffered_result[..., 0],
                             soft_result[..., 0]]))
        print(f"{row['mode']} {name}: unknown {row['unknown_to_bg']}, fg {row['fg_demoted']}, "
              f"model applied {row['evidence_model_applied']}", flush=True)
    with (args.out / "measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    titles = ["Original", "FBA: Clothing", "Color BG trimap", "BG + FG demotion",
              "Buffered BG trimap", "Color soft post"]
    sheet(panels, titles, args.out / "comparison.jpg")
    sheet(masks, ["Original", "Clothing trimap", "BG evidence", "Baseline alpha",
                  "BG trimap alpha", "BG + demote alpha", "Buffered alpha",
                  "Soft post alpha"], args.out / "alpha.jpg")
    focus = {"shot3", "shot5", "test1", "test2", "test3", "test4", "test5", "test6", "test9"}
    sheet([panel for panel in panels if panel[0] in focus], titles, args.out / "detail.jpg")
    for panel in panels:
        if panel[0] in {"shot3", "test3", "test4", "test5"}:
            sheet([panel], titles, args.out / f"{panel[0]}-detail.jpg")


if __name__ == "__main__":
    main()
