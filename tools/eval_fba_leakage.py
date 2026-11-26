"""Reproduce FBA demo leakage and guarded alternative on the fixed 12-photo set."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from idphoto import color, matting
from idphoto.pipeline import to_live
from tools.eval_clothing import (
    BASELINE, NAMES, ROOT, baseline_module, fba_trimap, guarded_fba_trimap, process,
)
from tools.eval_fba_combinations import sheet
from tools.fba_reference import FBAReference


def trimap_changes(main, clothing):
    """Red: added FG; blue: released BG; magenta: BG promoted directly to FG."""
    if main.shape != clothing.shape:
        raise ValueError("Trimaps must match.")
    image = np.zeros((*main.shape, 3), np.uint8)
    image[(main == 128) & (clothing == 255)] = (255, 60, 60)
    image[(main == 0) & (clothing == 128)] = (70, 150, 255)
    image[(main == 0) & (clothing == 255)] = (255, 0, 255)
    return image


def compose(prediction):
    alpha = prediction[..., :1]
    return color.encode_u8(alpha * color.srgb_to_linear(prediction[..., 1:4]) + (1 - alpha))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=ROOT / "out/clothing-landmarks")
    parser.add_argument("--full-grid", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(4)
    fba, baseline = FBAReference(), baseline_module(BASELINE)
    rows, panels, diagnostics = [], [], []
    for name in NAMES:
        path = ROOT / "docs/evaluation/test" / f"{name}.jpg"
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(path)
        rgb, _ = to_live(bgr[..., ::-1].copy())
        lm = np.load(args.cache / f"{name}-live.npy")
        before, _ = process(baseline, rgb, lm, not args.full_grid)
        clothing, _ = process(matting, rgb, lm, not args.full_grid)
        main_trimap = fba_trimap(rgb, lm, before)
        clothing_trimap = fba_trimap(rgb, lm, clothing)
        main_pred, main_tiles = fba.process(rgb, main_trimap, tile_size=768, overlap=96)
        clothing_pred, clothing_tiles = fba.process(rgb, clothing_trimap, tile_size=768, overlap=96)
        guarded_trimap = guarded_fba_trimap(rgb, main_trimap, clothing_trimap, main_pred[..., 0])
        guarded_pred, guarded_tiles = fba.process(rgb, guarded_trimap, tile_size=768, overlap=96)
        base_alpha = main_pred[..., 0]
        row = {
            "photo": name, "mode": "full-grid" if args.full_grid else "live",
            "width": rgb.shape[1], "height": rgb.shape[0],
            "clothing_added_vs_main_gt_0_5": int(((clothing.alpha > .5) & (before.alpha <= .5)).sum()),
            "clothing_removed_vs_main_gt_0_5": int(((clothing.alpha <= .5) & (before.alpha > .5)).sum()),
            "trimap_main_unknown_to_clothing_foreground": int(((main_trimap == 128) & (clothing_trimap == 255)).sum()),
            "trimap_main_background_to_clothing_unknown": int(((main_trimap == 0) & (clothing_trimap == 128)).sum()),
            "trimap_main_background_to_clothing_foreground": int(((main_trimap == 0) & (clothing_trimap == 255)).sum()),
            "guarded_promoted_foreground": int(((main_trimap == 128) & (guarded_trimap == 255)).sum()),
            "main_tiles": main_tiles, "clothing_tiles": clothing_tiles, "guarded_tiles": guarded_tiles,
        }
        for label, alpha in (("main", before.alpha), ("clothing", clothing.alpha),
                             ("fba_main", base_alpha), ("fba_clothing", clothing_pred[..., 0]),
                             ("fba_guarded", guarded_pred[..., 0])):
            row[f"{label}_mean_alpha"] = float(alpha.mean())
            row[f"{label}_foreground_gt_0_5"] = int((alpha > .5).sum())
            row[f"{label}_vs_fba_main_added_gt_0_5"] = int(((alpha > .5) & (base_alpha <= .5)).sum())
            row[f"{label}_vs_fba_main_removed_gt_0_5"] = int(((alpha <= .5) & (base_alpha > .5)).sum())
            row[f"{label}_vs_fba_main_mean_abs_alpha"] = float(np.abs(alpha - base_alpha).mean())
        ca, ga = clothing_pred[..., 0], guarded_pred[..., 0]
        row["fba_main_foreground_in_main_unknown_gt_0_5"] = int(
            ((main_trimap == 128) & (base_alpha > .5)).sum())
        row["fba_clothing_foreground_in_clothing_unknown_gt_0_5"] = int(
            ((clothing_trimap == 128) & (ca > .5)).sum())
        row["guarded_vs_clothing_added_gt_0_5"] = int(((ga > .5) & (ca <= .5)).sum())
        row["guarded_vs_clothing_removed_gt_0_5"] = int(((ga <= .5) & (ca > .5)).sum())
        rows.append(row)
        panels.append((name, [rgb, color.encode_u8(before.output_frame),
                              color.encode_u8(clothing.output_frame), compose(main_pred),
                              compose(clothing_pred), compose(guarded_pred)]))
        diagnostics.append((name, [rgb, before.alpha, clothing.alpha, main_trimap, clothing_trimap,
                                   trimap_changes(main_trimap, clothing_trimap), base_alpha,
                                   clothing_pred[..., 0], guarded_pred[..., 0]]))
        print(f"{row['mode']} {name}: added FG {row['trimap_main_unknown_to_clothing_foreground']}; "
              f"released BG {row['trimap_main_background_to_clothing_unknown']}", flush=True)
    with (args.out / "measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    sheet(panels, ["Original", "Main", "Clothing", "FBA: Main",
                   "FBA: Clothing", "FBA: Guarded"], args.out / "comparison.jpg")
    sheet(diagnostics, ["Original", "Main alpha", "Clothing alpha", "Main trimap",
                        "Clothing trimap", "Trimap changes\nred FG / blue BG / purple both",
                        "FBA: Main alpha", "FBA: Clothing alpha", "FBA: Guarded alpha"],
          args.out / "pathways.jpg")
    focus = {"shot5", "test1", "test4", "test9"}
    sheet([panel for panel in panels if panel[0] in focus],
          ["Original", "Main", "Clothing", "FBA: Main", "FBA: Clothing", "FBA: Guarded"],
          args.out / "detail.jpg")
    sheet([panel for panel in diagnostics if panel[0] in focus],
          ["Original", "Main alpha", "Clothing alpha", "Main trimap", "Clothing trimap",
           "Trimap changes\nred FG / blue BG / purple both", "FBA: Main alpha",
           "FBA: Clothing alpha", "FBA: Guarded alpha"], args.out / "pathways-detail.jpg")


if __name__ == "__main__":
    main()
