"""Original-resolution head matting experiments with a shared editable trimap.

    uv run --extra hair-matting python tools/matte_hair.py photo.jpg
    uv run --extra hair-matting python tools/matte_hair.py photo.jpg --methods fba cf knn

FBA uses the authors' pretrained, noncommercial reference weights, without training.
Only landmarks and initial masks are computed on a reduced image; matting is native.
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

from idphoto import color, mesh, npimg
from idphoto.context import FrameContext
from idphoto.landmarks import FaceLandmarkDetector
from idphoto.matting import Matter, face_measures
from idphoto.relight import Relighter
from tools.compare_matting import write_image
from tools.fba_reference import ASSETS, REVISION, WEIGHTS_SHA256, FBAReference, sha256, validate_trimap


@dataclass
class Matte:
    alpha: np.ndarray
    foreground: np.ndarray
    seconds: float
    tiles: int = 0
    notices: tuple[str, ...] = ()


def head_box(shape, landmarks: np.ndarray) -> tuple[int, int, int, int]:
    h, w = shape
    f = face_measures(landmarks)
    if not np.isfinite(landmarks).all() or f["fw"] <= 0 or f["ec"] <= 0:
        raise ValueError("Cannot crop a head from invalid landmarks.")
    return check_box((int(np.floor(f["cx"] - 1.05 * f["fw"])),
                      int(np.floor(f["eye"][1] - 1.7 * f["ec"])),
                      int(np.ceil(f["cx"] + 1.05 * f["fw"])),
                      int(np.ceil(f["chin"][1] + 0.2 * f["ec"]))), shape, clamp=True)


def check_box(box, shape, clamp=False) -> tuple[int, int, int, int]:
    h, w = shape
    x0, y0, x1, y1 = box
    if clamp:
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError("Crop box must be nonempty and inside the original photo.")
    return x0, y0, x1, y1


def prepare_crop(rgb: np.ndarray, landmarks: np.ndarray, small_alpha: np.ndarray,
                 box=None, band_fraction: float = 0.25):
    box = head_box(rgb.shape[:2], landmarks) if box is None else check_box(box, rgb.shape[:2])
    x0, y0, x1, y1 = box
    crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
    h, w = rgb.shape[:2]
    sh, sw = small_alpha.shape
    alpha = npimg.resize(small_alpha, y1 - y0, x1 - x0, kernel="linear",
                         box=(x0 * sw / w, y0 * sh / h, x1 * sw / w, y1 * sh / h))
    f = face_measures(landmarks)
    if not np.isfinite(band_fraction) or band_fraction <= 0:
        raise ValueError("Unknown-band fraction must be finite and positive.")
    inner, outer = max(1, round(0.06 * f["fw"])), max(1, round(band_fraction * f["fw"]))
    coarse = (alpha > 0.5).astype(np.uint8)
    fg = cv2.erode(coarse, np.ones((2 * inner + 1, 2 * inner + 1), np.uint8)) > 0
    bg = cv2.dilate(coarse, np.ones((2 * outer + 1, 2 * outer + 1), np.uint8)) == 0
    trimap = np.full(coarse.shape, 128, np.uint8)
    trimap[bg], trimap[fg] = 0, 255
    local = landmarks.copy()
    local[:, :2] -= (x0, y0)
    face = mesh.polygon(local, mesh.FACE_OVAL, crop.shape[:2]).astype(np.uint8)
    face = cv2.erode(face, np.ones((2 * inner + 1, 2 * inner + 1), np.uint8)) > 0
    trimap[face] = 255
    validate_trimap(crop, trimap)
    return crop, np.clip(alpha, 0, 1).astype(np.float32), trimap, box


def load_trimap(path: Path, rgb: np.ndarray) -> np.ndarray:
    trimap = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if trimap is None:
        raise FileNotFoundError(f"Cannot read trimap: {path}")
    validate_trimap(rgb, trimap)
    return trimap


def classical_matte(rgb: np.ndarray, trimap: np.ndarray, method: str) -> Matte:
    from pymatting import estimate_alpha_cf, estimate_alpha_knn, estimate_foreground_ml

    validate_trimap(rgb, trimap)
    if trimap.size > 1_000_000:
        raise ValueError("CF/KNN experiments are limited to 1 MP for memory safety. Use --methods fba.")
    if method not in ("cf", "knn"):
        raise ValueError(f"Unknown classical matting method: {method}")
    image, labels = rgb.astype(np.float64) / 255.0, trimap.astype(np.float64) / 255.0
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimate = estimate_alpha_cf if method == "cf" else estimate_alpha_knn
        alpha = estimate(image, labels)
        foreground = estimate_foreground_ml(image, alpha)
    alpha[trimap == 0], alpha[trimap == 255] = 0.0, 1.0
    foreground[trimap == 255] = image[trimap == 255]
    if not np.isfinite(alpha).all() or not np.isfinite(foreground).all():
        raise RuntimeError(f"{method} returned nonfinite matting estimates.")
    return Matte(np.clip(alpha, 0, 1).astype(np.float32),
                 np.clip(foreground, 0, 1).astype(np.float32),
                 time.perf_counter() - started, notices=tuple(str(item.message) for item in caught))


def preview_composite(matte: Matte, background=(47, 93, 154), original=None) -> np.ndarray:
    foreground = matte.foreground if original is None else original.astype(np.float32) / 255.0
    alpha = matte.alpha[..., None]
    output = alpha * foreground + (1 - alpha) * (np.asarray(background, np.float32) / 255.0)
    return np.rint(np.clip(output, 0, 1) * 255).astype(np.uint8)


def panel(rgb: np.ndarray, trimap: np.ndarray, mattes: dict[str, Matte],
          max_side: int | None = None, trimap_title: str = "Shared trimap") -> np.ndarray:
    h, w = trimap.shape
    scale = min(1.0, max_side / max(h, w)) if max_side else 1.0
    dh, dw = max(1, round(h * scale)), max(1, round(w * scale))

    def label(image, text):
        if scale < 1:
            image = cv2.resize(image, (dw, dh), interpolation=cv2.INTER_AREA)
        title = np.full((28, dw, 3), 245, np.uint8)
        cv2.putText(title, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (15, 15, 15), 1)
        return np.concatenate([title, image])

    upper, lower = [label(rgb, "Native head crop")], [label(np.repeat(trimap[..., None], 3, axis=2), trimap_title)]
    for name, matte in mattes.items():
        title = name + (" / input RGB" if name == "current" else " / recovered FG")
        upper.append(label(preview_composite(matte), title))
        mask = np.repeat(np.rint(matte.alpha[..., None] * 255).astype(np.uint8), 3, axis=2)
        lower.append(label(mask, name + " alpha"))
    return np.concatenate([np.concatenate(upper, axis=1), np.concatenate(lower, axis=1)], axis=0)


def save_results(folder: Path, rgb: np.ndarray, trimap: np.ndarray, mattes: dict[str, Matte],
                 trimap_title: str = "Shared trimap") -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    write_image(folder / "crop.png", rgb)
    write_image(folder / "trimap.png", trimap)
    write_image(folder / "comparison.jpg",
                panel(rgb, trimap, mattes, max_side=min(480, 1600 // (len(mattes) + 1)),
                      trimap_title=trimap_title))
    # A 256-pixel patch at the upper central mask boundary, kept at 1:1 pixel scale.
    h, w = trimap.shape
    middle = np.flatnonzero(mattes["current"].alpha[:, w // 2] > 0.5)
    cy = int(middle[0]) if len(middle) else h // 3
    size = min(256, h, w)
    x0, y0 = min(max(0, w // 2 - size // 2), w - size), min(max(0, cy - size // 2), h - size)
    detail_box = [x0, y0, x0 + size, y0 + size]
    selection = np.s_[y0:y0 + size, x0:x0 + size]
    details = {name: Matte(matte.alpha[selection], matte.foreground[selection], matte.seconds)
               for name, matte in mattes.items()}
    write_image(folder / "detail_1to1.png",
                panel(rgb[selection], trimap[selection], details, trimap_title=trimap_title))
    report = {"native_size_wh": [w, h], "detail_box_xyxy": detail_box, "methods": {}}
    for name, matte in mattes.items():
        write_image(folder / f"{name}_alpha.png", np.rint(matte.alpha * 65535).astype(np.uint16))
        fg = np.rint(matte.foreground * 255).astype(np.uint8)
        rgba = np.dstack([fg, np.rint(matte.alpha * 255).astype(np.uint8)])
        if not cv2.imwrite(str(folder / f"{name}_rgba.png"), rgba[..., [2, 1, 0, 3]]):
            raise OSError("Could not save foreground RGBA.")
        write_image(folder / f"{name}_blue.png", preview_composite(matte))
        write_image(folder / f"{name}_white.png", preview_composite(matte, (255, 255, 255)))
        if name != "current":
            write_image(folder / f"{name}_naive.png", preview_composite(matte, original=rgb))
        touching = {edge: bool((values > 0.1).any()) for edge, values in
                    (("top", matte.alpha[0]), ("left", matte.alpha[:, 0]), ("right", matte.alpha[:, -1]))}
        report["methods"][name] = {
            "seconds": matte.seconds, "native_tiles": matte.tiles,
            "unknown_fraction": float((trimap == 128).mean()),
            "partial_alpha_fraction": float(((matte.alpha > 0.01) & (matte.alpha < 0.99)).mean()),
            "foreground_touches_crop": touching, "warnings": list(matte.notices),
            "ground_truth_metrics": None,
        }
        for notice in matte.notices:
            print(f"WARNING {folder.name}/{name}: {notice}", file=sys.stderr)
        if any(touching.values()):
            print(f"WARNING {folder.name}/{name}: foreground reaches crop edge: {touching}", file=sys.stderr)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("photos", nargs="+", type=Path)
    parser.add_argument("--methods", nargs="+", choices=("fba", "cf", "knn"), default=["fba"])
    parser.add_argument("--trimap-dir", type=Path, help="edited native-crop <photo-stem>.png trimaps")
    parser.add_argument("--box", nargs=4, type=int, help="original-pixel x0 y0 x1 y1; one photo only")
    parser.add_argument("--band", type=float, default=0.25, help="unknown outer margin in face widths")
    parser.add_argument("--tile-size", type=int, default=768)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument("--prepare-only", action="store_true", help="save crop and trimap without matting")
    parser.add_argument("--out", type=Path, default=Path("out/hair-matting"))
    args = parser.parse_args(argv)
    if args.box and len(args.photos) != 1:
        parser.error("--box requires exactly one photo.")
    if len({path.stem for path in args.photos}) != len(args.photos):
        parser.error("Photo stems must be unique.")
    if args.tile_size < 32 or args.overlap < 0 or 2 * args.overlap >= args.tile_size:
        parser.error("Require tile size >=32 and 0 <= overlap < tile size/2.")
    if not np.isfinite(args.band) or args.band <= 0:
        parser.error("--band must be finite and positive.")
    model = None
    if "fba" in args.methods and not args.prepare_only:
        import torch

        torch.set_num_threads(4)
        model = FBAReference(ASSETS)
    detector = FaceLandmarkDetector(video=False)
    report = {
        "settings": {"methods": args.methods, "device": "cpu", "tile_size": args.tile_size,
                     "overlap": args.overlap, "unknown_band_face_widths": args.band,
                     "prepare_only": args.prepare_only, "matting_color_space": "sRGB, per reference"},
        "fba": {"source_revision": REVISION, "weights_sha256": WEIGHTS_SHA256,
                "weights_license": "noncommercial only"},
        "cases": {},
    }
    try:
        for path in args.photos:
            source = cv2.imread(str(path))
            if source is None:
                raise FileNotFoundError(f"Cannot read photo: {path}")
            rgb = np.ascontiguousarray(source[..., ::-1])
            h, w = rgb.shape[:2]
            scale = min(1.0, 640 / max(h, w))
            small = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
            landmarks = detector.detect(small)
            if landmarks is None:
                raise ValueError(f"No face detected: {path}")
            ctx = FrameContext(raw_frame=small, landmarks=landmarks)
            ctx.linear_rgb = color.decode_u8(small)
            ctx.Y = color.luminance(ctx.linear_rgb)
            Relighter().process(ctx, strength=0.0, update=False)
            if not ctx.face_mask.any():
                raise ValueError(f"No usable face mask: {path}")
            started = time.perf_counter()
            Matter().process(ctx, update=False, fast=False)
            baseline_seconds = time.perf_counter() - started
            landmarks = landmarks.copy()
            landmarks[:, (0, 2)] *= w / small.shape[1]
            landmarks[:, 1] *= h / small.shape[0]
            crop, baseline, trimap, box = prepare_crop(rgb, landmarks, ctx.alpha, args.box, args.band)
            trimap_path = args.trimap_dir / f"{path.stem}.png" if args.trimap_dir else None
            if trimap_path:
                trimap = load_trimap(trimap_path, crop)
            if not args.prepare_only and crop.shape[0] * crop.shape[1] > 1_000_000 and any(
                method in args.methods for method in ("cf", "knn")
            ):
                raise ValueError("CF/KNN experiments are limited to 1 MP for memory safety. Use --methods fba.")
            mattes = {"current": Matte(baseline, crop.astype(np.float32) / 255.0, baseline_seconds)}
            if not args.prepare_only:
                for method in dict.fromkeys(args.methods):
                    if method == "fba":
                        started = time.perf_counter()
                        result, tiles = model.process(crop, trimap, args.tile_size, args.overlap)
                        matte = Matte(result[..., 0], result[..., 1:4], time.perf_counter() - started, tiles)
                    else:
                        matte = classical_matte(crop, trimap, method)
                    mattes[method] = matte
                    print(f"{path.stem}/{method}: {crop.shape[1]}x{crop.shape[0]} native, "
                          f"{matte.seconds:.2f}s, {matte.tiles} FBA tiles", flush=True)
            folder = args.out / path.stem
            case = save_results(folder, crop, trimap, mattes)
            case.update(source=str(path), source_sha256=sha256(path), crop_box_xyxy=list(box),
                        trimap_source=str(trimap_path) if trimap_path else "automatic from current alpha",
                        trimap_sha256=sha256(folder / "trimap.png"))
            report["cases"][path.stem] = case
    finally:
        detector.close()
    packages = ["numpy", "opencv-contrib-python", "mediapipe"]
    if not args.prepare_only:
        packages += ["pymatting"] if any(method in args.methods for method in ("cf", "knn")) else []
        packages += ["torch"] if model is not None else []
    report["versions"] = {name: importlib.metadata.version(name) for name in packages}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved native crops, trimaps, mattes, and 1:1 details to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
