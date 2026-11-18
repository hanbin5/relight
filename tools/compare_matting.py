"""Compare the production mask with GrabCut and Random Walker research baselines.

    uv run --extra research python tools/compare_matting.py --synthetic
    uv run --extra research python tools/compare_matting.py photo.jpg --out out/comparison

These third-party reference algorithms are offline experiments, not app backends.
No relighting, silhouette clipping, morphology, or edge matting is applied to their masks.
"""

import argparse
import importlib.metadata
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from idphoto import color, mesh, npimg, synthetic
from idphoto.context import FrameContext
from idphoto.landmarks import FaceLandmarkDetector
from idphoto.matting import Matter, _sample, face_measures, kmeans, nearest_distance
from idphoto.relight import Relighter

UNKNOWN, BACKGROUND, FOREGROUND = 0, 1, 2
SEED_VALUES = np.array([0, 128, 255], np.uint8)
METHODS = ("current", "grabcut", "random_walker")
RANDOM_WALKER_TOL = 1e-6


@dataclass
class Result:
    mask: np.ndarray
    seconds: float
    probability: np.ndarray | None = None
    notices: tuple[str, ...] = ()


def validate_markers(rgb: np.ndarray, markers: np.ndarray):
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected an RGB uint8 image (H, W, 3).")
    if markers.shape != rgb.shape[:2] or not np.isin(markers, (0, 1, 2)).all():
        raise ValueError("Markers must match the image, with 0=unknown, 1=background, 2=foreground.")
    if any(np.count_nonzero(markers == label) < 5 for label in (BACKGROUND, FOREGROUND)):
        raise ValueError("Provide at least five foreground and five background seed pixels.")


def automatic_markers(face: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Small seed patches, not a head-and-shoulders silhouette. They can still be wrong."""
    h, w = face.shape
    f = face_measures(landmarks)
    if f["fw"] <= 0 or f["ec"] <= 0:
        raise ValueError("Cannot place seeds with degenerate face landmarks.")
    yy, xx = np.mgrid[:h, :w] + 0.5
    core = npimg.erode(face, max(1, round(w / 320)))
    half_width = max(1.0, 0.10 * f["fw"])
    half_height = max(1.0, 0.08 * f["ec"])
    for cy in (landmarks[mesh.FOREHEAD, 1] - 0.18 * f["ec"],
               f["chin"][1] + 1.25 * f["ec"]):
        core |= (np.abs(xx - f["cx"]) < half_width) & (np.abs(yy - cy) < half_height)
    markers = np.zeros((h, w), np.uint8)
    border = max(2, round(min(h, w) * 0.04))
    upper_end = max(border, min(h, round(f["chin"][1] - 0.1 * f["ec"])))
    markers[:border, :] = BACKGROUND
    markers[:upper_end, :border] = BACKGROUND
    markers[:upper_end, -border:] = BACKGROUND
    # Do not seed the lower border (clothes), or a generous box around the head.
    near_head = (np.abs(xx - f["cx"]) < f["fw"]) & (
        yy > landmarks[mesh.FOREHEAD, 1] - f["ec"])
    markers[near_head] = UNKNOWN
    markers[core] = FOREGROUND
    return markers


def grabcut(rgb: np.ndarray, markers: np.ndarray, iterations: int = 5) -> Result:
    validate_markers(rgb, markers)
    if iterations < 1:
        raise ValueError("GrabCut iterations must be positive.")
    started = time.perf_counter()
    features = np.sqrt(color.decode_u8(rgb))
    bg = kmeans(_sample(features, markers == BACKGROUND, 2000, 1), 5, iterations=8)
    fg = kmeans(_sample(features, markers == FOREGROUND, 2000, 2), 5, iterations=8)
    x = features.reshape(-1, 3)
    probable_fg = nearest_distance(x, *fg[:2], (1.0, 1.0)) < nearest_distance(x, *bg[:2], (1.0, 1.0))
    labels = np.where(probable_fg.reshape(markers.shape), cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    labels[markers == BACKGROUND] = cv2.GC_BGD
    labels[markers == FOREGROUND] = cv2.GC_FGD
    bg_model, fg_model = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.setRNGSeed(0)
    cv2.grabCut(np.ascontiguousarray(rgb), labels, None, bg_model, fg_model,
                iterations, cv2.GC_INIT_WITH_MASK)
    seconds = time.perf_counter() - started
    return Result((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD), seconds)


def random_walker(rgb: np.ndarray, markers: np.ndarray, beta: float = 130.0) -> Result:
    from skimage.segmentation import random_walker as solve

    validate_markers(rgb, markers)
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("Random Walker beta must be finite and positive.")
    if not (markers == UNKNOWN).any():
        probability = (markers == FOREGROUND).astype(np.float32)
        return Result(probability > 0.5, 0.0, probability)
    features = np.sqrt(color.decode_u8(rgb))
    if float(features.std()) < 1e-8:
        raise ValueError("Random Walker's reference solver requires a nonconstant image.")
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        probabilities = solve(features, markers.astype(np.int32), beta=beta, mode="cg_j",
                              tol=RANDOM_WALKER_TOL, return_full_prob=True, channel_axis=-1)
    seconds = time.perf_counter() - started
    probability = probabilities[1]
    if not np.isfinite(probability).all() or probability.min() < -1e-3 or probability.max() > 1.001:
        raise RuntimeError("Random Walker returned invalid foreground probabilities.")
    probability = np.clip(probability, 0.0, 1.0).astype(np.float32)
    return Result(probability > 0.5, seconds, probability,
                  tuple(str(item.message) for item in caught))


def compare(rgb: np.ndarray, landmarks: np.ndarray, markers: np.ndarray | None = None,
            iterations: int = 5, beta: float = 130.0):
    ctx = FrameContext(raw_frame=rgb, landmarks=landmarks)
    ctx.linear_rgb = color.decode_u8(rgb)
    ctx.Y = color.luminance(ctx.linear_rgb)
    Relighter().process(ctx, strength=0.0, update=False)
    if not ctx.face_mask.any():
        raise ValueError("No usable face mask; cannot compare against the production method.")
    if markers is None:
        markers = automatic_markers(ctx.face_mask, landmarks)
    validate_markers(rgb, markers)
    started = time.perf_counter()
    Matter().process(ctx, update=False, fast=False)
    results = {"current": Result(ctx.stats["matting"]["coarse_mask"].copy(),
                                 time.perf_counter() - started)}
    results["grabcut"] = grabcut(rgb, markers, iterations)
    results["random_walker"] = random_walker(rgb, markers, beta)
    return markers, results


def mask_metrics(mask: np.ndarray, truth: np.ndarray | None = None) -> dict:
    metrics = {"foreground_share": float(mask.mean()), "iou": None,
               "precision": None, "recall": None}
    if truth is not None:
        if truth.shape != mask.shape or truth.dtype != bool:
            raise ValueError("Ground truth must be a boolean mask matching the working image.")
        tp = int((mask & truth).sum())
        union, predicted, actual = int((mask | truth).sum()), int(mask.sum()), int(truth.sum())
        metrics.update(iou=tp / union if union else 1.0,
                       precision=tp / predicted if predicted else None,
                       recall=tp / actual if actual else None)
    return metrics


def write_image(path: Path, image: np.ndarray):
    data = image[..., ::-1] if image.ndim == 3 else image
    if not cv2.imwrite(str(path), np.ascontiguousarray(data)):
        raise OSError(f"Could not write image: {path}")


def read_markers(path: Path, shape: tuple[int, int]) -> np.ndarray:
    values = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if values is None:
        raise FileNotFoundError(f"Could not read seed image: {path}")
    if values.shape != shape or values.dtype != np.uint8 or not np.isin(values, SEED_VALUES).all():
        raise ValueError(f"{path}: expected working-size grayscale uint8 PNG with only 0, 128, 255.")
    return np.searchsorted(SEED_VALUES, values).astype(np.uint8)


def comparison_panel(rgb: np.ndarray, markers: np.ndarray, results: dict[str, Result]) -> np.ndarray:
    _, w = rgb.shape[:2]
    overlay = rgb.copy()
    for label, tint in ((BACKGROUND, (255, 140, 0)), (FOREGROUND, (0, 255, 130))):
        where = markers == label
        overlay[where] = (0.25 * rgb[where] + 0.75 * np.array(tint)).astype(np.uint8)

    def caption(image, text):
        title = np.full((28, w, 3), 245, np.uint8)
        cv2.putText(title, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (15, 15, 15), 1)
        return np.concatenate([title, image], axis=0)

    upper, lower = [caption(rgb, "Input (no relighting)")], [caption(overlay, "Seeds: BG orange / FG green")]
    background = color.hex_to_linear("#2f5d9a")
    linear = color.decode_u8(rgb)
    for name in METHODS:
        mask = results[name].mask
        composite = color.encode_u8(np.where(mask[..., None], linear, background))
        upper.append(caption(composite, name))
        lower.append(caption(np.repeat((mask * 255).astype(np.uint8)[..., None], 3, axis=2), name + " mask"))
    return np.concatenate([np.concatenate(upper, axis=1), np.concatenate(lower, axis=1)], axis=0)


def save_comparison(folder: Path, rgb: np.ndarray, markers: np.ndarray,
                    results: dict[str, Result], truth: np.ndarray | None = None) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    write_image(folder / "input.png", rgb)
    write_image(folder / "seeds.png", SEED_VALUES[markers])
    write_image(folder / "comparison.jpg", comparison_panel(rgb, markers, results))
    if truth is not None:
        write_image(folder / "truth.png", truth.astype(np.uint8) * 255)
    report = {
        "size_wh": [rgb.shape[1], rgb.shape[0]],
        "seed_counts": {name: int((markers == label).sum()) for name, label in
                        (("background", BACKGROUND), ("foreground", FOREGROUND))},
        "methods": {},
    }
    for name, result in results.items():
        write_image(folder / f"{name}_mask.png", result.mask.astype(np.uint8) * 255)
        if result.probability is not None:
            write_image(folder / f"{name}_probability.png",
                        np.rint(result.probability * 65535).astype(np.uint16))
        report["methods"][name] = {
            **mask_metrics(result.mask, truth), "seconds": result.seconds,
            "warnings": list(result.notices),
        }
        for notice in result.notices:
            print(f"WARNING {folder.name}/{name}: {notice}", file=sys.stderr)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("photos", nargs="*", type=Path)
    parser.add_argument("--synthetic", action="store_true", help="also run three scenes with ground truth")
    parser.add_argument("--max-side", type=int, default=480, help="working resolution limit, no upscaling")
    parser.add_argument("--iterations", type=int, default=5, help="GrabCut iterations")
    parser.add_argument("--beta", type=float, default=130.0, help="Random Walker edge penalty")
    parser.add_argument("--seeds-dir", type=Path, help="edited <photo-stem>.png seeds at working resolution")
    parser.add_argument("--out", type=Path, default=Path("out/matting-comparison"))
    args = parser.parse_args(argv)
    if not args.photos and not args.synthetic:
        parser.error("Provide photos and/or --synthetic.")
    if args.max_side < 64 or args.iterations < 1 or not np.isfinite(args.beta) or args.beta <= 0:
        parser.error("--max-side must be >=64; --iterations and --beta must be positive.")
    names = [path.stem for path in args.photos]
    scenes = {
        "synthetic_navy": synthetic.Scene(width=320, height=240),
        "synthetic_red": synthetic.Scene(width=320, height=240, wall="#e8dcc8", top="#9c2f2f"),
        "synthetic_similar": synthetic.Scene(width=320, height=240, top="#d8d6d2"),
    } if args.synthetic else {}
    if len(set(names + list(scenes))) != len(names) + len(scenes):
        parser.error("Input stems must be unique (including synthetic case names).")
    if args.seeds_dir and args.synthetic:
        parser.error("--seeds-dir applies to photos only; run synthetic scenes separately.")
    try:
        importlib.metadata.version("scikit-image")
    except importlib.metadata.PackageNotFoundError:
        parser.error("Random Walker requires: uv sync --extra research")

    report = {
        "settings": {"max_side": args.max_side, "grabcut_iterations": args.iterations,
                     "grabcut_initialization": "seed-color k-means, 5 clusters per class, no shade scaling",
                     "random_walker_beta": args.beta, "random_walker_mode": "cg_j",
                     "random_walker_tol": RANDOM_WALKER_TOL},
        "versions": {name: importlib.metadata.version(name)
                     for name in ("numpy", "opencv-contrib-python", "scikit-image", "scipy")},
        "cases": {},
    }

    def run_case(name, rgb, landmarks, truth=None, source=None):
        original_size = [rgb.shape[1], rgb.shape[0]]
        scale = min(1.0, args.max_side / max(rgb.shape[:2]))
        if scale < 1:
            h, w = [max(1, round(n * scale)) for n in rgb.shape[:2]]
            rgb = np.clip(npimg.resize(rgb, h, w, kernel="linear") + 0.5, 0, 255).astype(np.uint8)
            landmarks = landmarks.copy()
            landmarks[:, (0, 2)] *= w / original_size[0]
            landmarks[:, 1] *= h / original_size[1]
            if truth is not None:
                truth = cv2.resize(truth.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        seed_path = args.seeds_dir / f"{name}.png" if args.seeds_dir else None
        markers = read_markers(seed_path, rgb.shape[:2]) if seed_path else None
        markers, results = compare(rgb, landmarks, markers, args.iterations, args.beta)
        case = save_comparison(args.out / name, rgb, markers, results, truth)
        case.update(source=source or name, original_size_wh=original_size,
                    seeds=str(seed_path) if seed_path else "automatic")
        report["cases"][name] = case
        for method, result in results.items():
            metrics = case["methods"][method]
            iou = f", IoU {metrics['iou']:.3f}" if truth is not None else ""
            print(f"{name:20s} {method:14s} {result.seconds:.3f}s, "
                  f"foreground {metrics['foreground_share']:.1%}{iou}")

    for name, scene in scenes.items():
        run_case(name, *synthetic.render(scene))
    detector = FaceLandmarkDetector(video=False) if args.photos else None
    try:
        for path in args.photos:
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise FileNotFoundError(f"Could not read photo: {path}")
            rgb = np.ascontiguousarray(bgr[..., ::-1])
            # Detect on the working image, independently for each photo (no VIDEO history).
            scale = min(1.0, args.max_side / max(rgb.shape[:2]))
            h, w = [max(1, round(n * scale)) for n in rgb.shape[:2]]
            working = np.clip(npimg.resize(rgb, h, w, kernel="linear") + 0.5, 0, 255).astype(np.uint8)
            landmarks = detector.detect(working)
            if landmarks is None:
                raise ValueError(f"No face detected: {path}")
            landmarks[:, (0, 2)] *= rgb.shape[1] / w
            landmarks[:, 1] *= rgb.shape[0] / h
            run_case(path.stem, rgb, landmarks, source=str(path))
    finally:
        if detector is not None:
            detector.close()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved comparisons and report to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
