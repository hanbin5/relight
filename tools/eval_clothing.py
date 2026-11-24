"""Compare Module B against a fixed Git revision on the twelve evaluation photos."""

import argparse
import csv
import subprocess
import time
import types
from pathlib import Path

import cv2
import numpy as np

from idphoto import color, matting, npimg
from idphoto.context import FrameContext
from idphoto.landmarks import FaceLandmarkDetector
from idphoto.pipeline import to_live
from idphoto.relight import Relighter

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "e96bb08cb678634207ea3aa45f086152c4c96f11"
NAMES = ["shot2", "shot3", "shot4", "shot5", "test1", "test2",
         "test3", "test4", "test5", "test6", "test8", "test9"]
FBA_BAND = 0.5
FBA_EROSION = 0.06


def baseline_module(ref):
    source = subprocess.check_output(
        ["git", "show", f"{ref}:idphoto/matting.py"], cwd=ROOT, text=True)
    module = types.ModuleType("idphoto._clothing_baseline")
    module.__package__ = "idphoto"
    exec(compile(source, f"{ref}:idphoto/matting.py", "exec"), module.__dict__)
    return module


def process(module, rgb, lm, fast):
    ctx = FrameContext(raw_frame=rgb, landmarks=lm.copy())
    ctx.linear_rgb = color.decode_u8(rgb)
    ctx.Y = color.luminance(ctx.linear_rgb)
    Relighter().process(ctx, strength=0.0)
    start = time.perf_counter()
    module.Matter().process(ctx, np.ones(3, np.float32), update=False, fast=fast)
    return ctx, 1000 * (time.perf_counter() - start)


def fba_trimap(rgb, lm, before):
    from tools.fba_reference import validate_trimap

    if before.alpha is None or before.alpha.shape != rgb.shape[:2]:
        raise ValueError("The baseline alpha must match the input frame.")
    if not np.isfinite(before.alpha).all() or before.alpha.min() < 0 or before.alpha.max() > 1:
        raise ValueError("The baseline alpha must be finite and in [0, 1].")
    if before.face_mask is None or before.face_mask.shape != rgb.shape[:2]:
        raise ValueError("The baseline face mask must match the input frame.")
    f = matting.face_measures(lm)
    fw, ec, cx, chin_y = f["fw"], f["ec"], f["cx"], f["chin"][1]
    if not np.isfinite([fw, ec, cx, chin_y]).all() or fw <= 0 or ec <= 0:
        raise ValueError("Valid face dimensions and position are required for the FBA trimap.")
    foreground = before.alpha > 0.5
    sure_fg = npimg.erode(foreground, max(1, round(FBA_EROSION * fw))) | before.face_mask
    sure_bg = ~npimg.dilate(foreground, max(1, round(FBA_BAND * fw)))
    yy = np.arange(rgb.shape[0], dtype=np.float32)[:, None] + 0.5
    xx = np.arange(rgb.shape[1], dtype=np.float32)[None, :] + 0.5
    body_width = fw * (1 + 2.5 * np.sqrt(np.clip((yy - chin_y) / ec, 0, 1)))
    body = (yy >= chin_y) & (np.abs(xx - cx) < body_width)
    sure_bg &= ~body
    trimap = np.full(foreground.shape, 128, np.uint8)
    trimap[sure_bg] = 0
    trimap[sure_fg] = 255
    validate_trimap(rgb, trimap)
    return trimap


def fba_composite(prediction):
    alpha = prediction[..., :1]
    return np.rint(np.clip(alpha * prediction[..., 1:4] + 1 - alpha, 0, 1) * 255).astype(np.uint8)


def sample_views(rgb, lm, ctx, fast):
    h, w = rgb.shape[:2]
    s = 2 if fast else 1
    hs, ws = h // s, w // s
    lm_s = lm / np.float32(s)
    face = ctx.face_mask if ctx.face_mask is not None else np.zeros((h, w), bool)
    face_s = face[:hs * s:s, :ws * s:s]
    old_bg = matting.person_prior((hs, ws), lm_s)[0] <= 0
    old_fg = matting.foreground_seeds((hs, ws), lm_s, face_s)
    if old_bg.sum() < 50 or old_fg.sum() < 50:
        old_bg = np.zeros((hs, ws), bool)
        old_bg[[0, -1], :] = old_bg[:, [0, -1]] = True
    _, _, clothing_fg, guard = matting.clothing_regions((hs, ws), lm_s)
    guarded_bg = old_bg & ~guard
    stats = ctx.stats["matting"]
    if (clothing_fg.sum() != stats["clothing_samples"]
            or guarded_bg.sum() != stats["clothing_background_samples"]):
        raise RuntimeError("Sample regions do not match the matting statistics")

    def overlay(layers):
        image = rgb.copy()
        for region, shade in layers:
            pixels = npimg.upsample_nearest(region, s, (h, w))
            image[pixels] = (0.55 * rgb[pixels] + 0.45 * np.asarray(shade)).astype(np.uint8)
        return image

    orange, blue, green = (255, 130, 0), (0, 170, 255), (0, 230, 70)
    return [rgb, overlay([(old_bg, orange)]),
            overlay([(guarded_bg, orange), (old_bg & guard, blue)]),
            overlay([(clothing_fg, green)])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/evaluation/clothing")
    parser.add_argument("--cache", type=Path, default=ROOT / "out/clothing-landmarks")
    parser.add_argument("--full-grid", action="store_true",
                        help="shutter algorithm (s=1) on the same 640-px input, not original resolution")
    parser.add_argument("--sample-figure", action="store_true",
                        help="also visualize sample candidates for shot2 and test5")
    parser.add_argument("--fba", action="store_true",
                        help="include the full-frame, baseline-trimap-conditioned FBA research reference")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    before_module = baseline_module(args.baseline)
    fba = None
    if args.fba:
        import torch
        from tools.fba_reference import FBAReference

        torch.set_num_threads(4)
        fba = FBAReference()
    rows, panels, fba_panels, sample_panels = [], [], [], {}
    detector = None
    try:
        for name in NAMES:
            path = ROOT / "docs/evaluation/test" / f"{name}.jpg"
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise FileNotFoundError(path)
            rgb = bgr[..., ::-1].copy()
            rgb, _ = to_live(rgb)
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
            before, before_ms = process(before_module, rgb, lm, not args.full_grid)
            after, after_ms = process(matting, rgb, lm, not args.full_grid)
            row = {
                "photo": name, "baseline": args.baseline, "mode": "full-grid" if args.full_grid else "live",
                "width": rgb.shape[1], "height": rgb.shape[0],
                "before_threshold": before.stats["matting"]["threshold"],
                "after_threshold": after.stats["matting"]["threshold"],
                "added_pixels": int(((after.alpha > .5) & (before.alpha <= .5)).sum()),
                "removed_pixels": int(((before.alpha > .5) & (after.alpha <= .5)).sum()),
                "clothing_applied": after.stats["matting"]["clothing_applied"],
                "clothing_samples": after.stats["matting"]["clothing_samples"],
                "clothing_background_samples": after.stats["matting"]["clothing_background_samples"],
                "before_ms": before_ms, "after_ms": after_ms,
            }
            if fba is not None:
                trimap = fba_trimap(rgb, lm, before)
                start = time.perf_counter()
                prediction, tiles = fba.process(rgb, trimap, tile_size=1024, overlap=96)
                fba_ms = 1000 * (time.perf_counter() - start)
                fba_alpha = prediction[..., 0]
                row.update({
                    "fba_trimap_foreground": int((trimap == 255).sum()),
                    "fba_trimap_background": int((trimap == 0).sum()),
                    "fba_trimap_unknown": int((trimap == 128).sum()),
                    "fba_added_pixels": int(((fba_alpha > .5) & (before.alpha <= .5)).sum()),
                    "fba_removed_pixels": int(((before.alpha > .5) & (fba_alpha <= .5)).sum()),
                    "fba_tiles": tiles, "fba_ms": fba_ms,
                })
                panels.append((name, rgb, color.encode_u8(before.output_frame),
                               fba_composite(prediction), color.encode_u8(after.output_frame)))
                fba_panels.append((name, rgb, trimap, fba_alpha))
            else:
                panels.append((name, rgb, color.encode_u8(before.output_frame),
                               color.encode_u8(after.output_frame), before.alpha, after.alpha))
            rows.append(row)
            if args.sample_figure and name in ("shot2", "test5"):
                sample_panels[name] = sample_views(rgb, lm, after, not args.full_grid)
            print(row, flush=True)
    finally:
        if detector is not None:
            detector.close()
    with (args.out / "measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def write_sheets(items, titles, prefix):
        for page in range(3):
            fig, axes = plt.subplots(4, len(titles), figsize=(3 * len(titles), 10), layout="constrained")
            for r, (name, *images) in enumerate(items[page * 4:page * 4 + 4]):
                for c, image in enumerate(images):
                    vmax = 1 if image.ndim == 2 and np.issubdtype(image.dtype, np.floating) else 255
                    axes[r, c].imshow(image, cmap="gray", vmin=0, vmax=vmax)
                    axes[r, c].set_xticks([])
                    axes[r, c].set_yticks([])
                    if r == 0:
                        axes[r, c].set_title(titles[c], fontsize=10)
                    if c == 0:
                        axes[r, c].set_ylabel(name)
            fig.savefig(args.out / f"{prefix}-{page + 1}.jpg", dpi=130)
            plt.close(fig)

    if fba is None:
        write_sheets(panels, ["Input", "Before: white background", "After: white background",
                              "Before: alpha", "After: alpha"], "comparison")
    else:
        write_sheets(panels, ["Input", "Before: main", "FBA: main trimap", "After: ours"], "comparison")
        write_sheets(fba_panels, ["Input", "FBA trimap", "FBA alpha"], "trimap")
    if args.sample_figure:
        fig, axes = plt.subplots(2, 4, figsize=(12, 8), layout="constrained")
        for r, name in enumerate(("shot2", "test5")):
            for ax, image in zip(axes[r], sample_panels[name]):
                ax.imshow(image)
                ax.set_axis_off()
        fig.savefig(args.out / "sample-regions.jpg", dpi=150, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)


if __name__ == "__main__":
    main()
