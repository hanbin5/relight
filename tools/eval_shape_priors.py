"""Evaluate local, fixed soft shape and luminance-gradient priors with FBA."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from idphoto import color, config, matting
from idphoto.pipeline import to_live
from tools.eval_clothing import (
    BASELINE, NAMES, ROOT, baseline_module, fba_composite, fba_trimap,
    guarded_fba_trimap, process,
)
from tools.eval_fba_combinations import sheet
from tools.fba_reference import FBAReference, validate_trimap


def prior_maps(rgb, lm, face, alpha):
    """Torso plausibility and alpha component attached to the detected face."""
    shape = rgb.shape[:2]
    if face.shape != shape or alpha.shape != shape:
        raise ValueError("Face and alpha must match the input frame.")
    if not np.isfinite(alpha).all() or alpha.min() < 0 or alpha.max() > 1:
        raise ValueError("Alpha must be finite and in [0, 1].")
    f = matting.face_measures(lm)
    if not np.isfinite([f["fw"], f["ec"], f["chin"][1]]).all() or f["fw"] <= 0 or f["ec"] <= 0:
        raise ValueError("Valid face geometry is required.")
    likely = matting.clothing_regions(shape, lm)[1]
    torso = np.clip((likely - 0.5) / (config.PRIOR_TORSO - 0.5), 0, 1)
    y = np.arange(shape[0], dtype=np.float32)[:, None] + 0.5
    body = np.broadcast_to(np.clip((y - f["chin"][1]) / (0.4 * f["ec"]), 0, 1), shape)
    foreground = (alpha > 0.5).astype(np.uint8)
    count, labels = cv2.connectedComponents(foreground, connectivity=8)
    face_labels = np.unique(labels[face & (foreground != 0)])
    face_labels = face_labels[face_labels != 0]
    if count <= 1 or not face_labels.size:
        raise ValueError("FBA alpha must contain a foreground component overlapping the face.")
    connected = np.isin(labels, face_labels)
    return torso, body, connected


def edge_safety(rgb):
    """Near-zero at strong linear-luminance gradients; this is not depth estimation."""
    luminance = color.luminance(color.decode_u8(rgb))
    gx = cv2.Sobel(luminance, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8)
    gy = cv2.Sobel(luminance, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8)
    return 1 - np.clip(np.hypot(gx, gy) / 0.04, 0, 1)


def shape_results(rgb, main_trimap, clothing_trimap, alpha, torso, body, connected):
    validate_trimap(rgb, main_trimap)
    validate_trimap(rgb, clothing_trimap)
    shape = rgb.shape[:2]
    if any(item.shape != shape for item in (alpha, torso, body, connected)):
        raise ValueError("Prior maps must match the input frame.")
    if not all(np.isfinite(item).all() for item in (alpha, torso, body)):
        raise ValueError("Alpha and prior maps must be finite.")
    if any(item.min() < 0 or item.max() > 1 for item in (alpha, torso, body)):
        raise ValueError("Alpha and prior maps must be in [0, 1].")
    editable = (main_trimap == 128) & (body > 0)
    shape_penalty = 0.2 * body * (1 - torso)
    connected_penalty = 0.15 * body * ~connected
    geometry = alpha.copy()
    geometry[editable] *= 1 - shape_penalty[editable]
    combined = alpha.copy()
    combined[editable] *= 1 - (shape_penalty + connected_penalty)[editable]
    edge = alpha.copy()
    edge[editable] *= 1 - ((shape_penalty + connected_penalty) * edge_safety(rgb))[editable]
    trimap = clothing_trimap.copy()
    demote = ((clothing_trimap == 255) & editable & (body >= 0.5)
              & ((torso < 0.5) | ~connected))
    trimap[demote] = 128
    validate_trimap(rgb, trimap)
    return geometry, combined, edge, trimap, demote


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
    baseline = baseline_module(BASELINE)
    rows, panels, details = [], [], []
    for name in NAMES:
        path = ROOT / "docs/evaluation/test" / f"{name}.jpg"
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(path)
        rgb, _ = to_live(bgr[..., ::-1].copy())
        cache = args.cache / f"{name}-live.npy"
        if not cache.is_file():
            raise FileNotFoundError(f"Existing landmark cache required: {cache}")
        lm = np.load(cache)
        before, _ = process(baseline, rgb, lm, not args.full_grid)
        clothing, _ = process(matting, rgb, lm, not args.full_grid)
        main_trimap = fba_trimap(rgb, lm, before)
        clothing_trimap = fba_trimap(rgb, lm, clothing)
        main, _ = fba.process(rgb, main_trimap, tile_size=1024, overlap=96)
        initial, _ = fba.process(rgb, clothing_trimap, tile_size=1024, overlap=96)
        guarded_trimap = guarded_fba_trimap(rgb, main_trimap, clothing_trimap, main[..., 0])
        guarded, _ = fba.process(rgb, guarded_trimap, tile_size=1024, overlap=96)
        torso, body, connected = prior_maps(rgb, lm, before.face_mask, initial[..., 0])
        geometry, combined, edge, revised, demote = shape_results(
            rgb, main_trimap, clothing_trimap, initial[..., 0], torso, body, connected)
        pre, _ = fba.process(rgb, revised, tile_size=1024, overlap=96)
        candidates = {"geometry_post": geometry, "connected_post": combined,
                      "edge_post": edge, "uncertain_pre": pre[..., 0]}
        row = {"photo": name, "mode": "full-grid" if args.full_grid else "live",
               "width": rgb.shape[1], "height": rgb.shape[0],
               "demoted_foreground_pixels": int(demote.sum())}
        for label, alpha in (("fba_main", main[..., 0]), ("fba_clothing", initial[..., 0]),
                             ("fba_guarded", guarded[..., 0])):
            row[f"{label}_mean_alpha"] = float(alpha.mean())
            row[f"{label}_foreground_pixels_gt_0_5"] = int((alpha > .5).sum())
        for label, alpha in candidates.items():
            base = initial[..., 0]
            row[f"{label}_mean_alpha"] = float(alpha.mean())
            row[f"{label}_foreground_pixels_gt_0_5"] = int((alpha > .5).sum())
            row[f"{label}_vs_clothing_added_gt_0_5"] = int(((alpha > .5) & (base <= .5)).sum())
            row[f"{label}_vs_clothing_removed_gt_0_5"] = int(((alpha <= .5) & (base > .5)).sum())
            row[f"{label}_mean_abs_alpha_difference"] = float(np.abs(alpha - base).mean())
            row[f"{label}_changed_pixels_gt_0_001"] = int((np.abs(alpha - base) > .001).sum())
        rows.append(row)
        output = []
        for alpha in (geometry, combined, edge):
            prediction = initial.copy()
            prediction[..., 0] = alpha
            output.append(fba_composite(prediction))
        panels.append((name, [rgb, fba_composite(main), fba_composite(initial),
                              fba_composite(guarded), fba_composite(pre), *output]))
        details.append((name, [clothing_trimap, revised, torso, connected.astype(np.float32),
                               initial[..., 0], pre[..., 0], combined, edge]))
        print(f"{row['mode']} {name}: demoted {row['demoted_foreground_pixels']}, "
              f"post removed {row['connected_post_vs_clothing_removed_gt_0_5']}", flush=True)
    with (args.out / "measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    sheet(panels, ["Original", "FBA: Main", "FBA: clothing", "FBA: guarded",
                   "Shape + connect\ntrimap / FBA", "Shape\npost", "Shape + connect\npost",
                   "Shape + connect\n+ edge post"], args.out / "comparison.jpg")
    sheet(details, ["Clothing\ntrimap", "Revised\ntrimap", "Torso\nsupport", "Face\nconnected",
                    "Clothing FBA\nalpha", "Trimap / FBA\nalpha", "Shape + connect\nalpha",
                    "Edge-safe\nalpha"], args.out / "prior-detail.jpg")


if __name__ == "__main__":
    main()
