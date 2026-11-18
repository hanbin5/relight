"""Suppress broad FBA alpha leakage without narrowing the original trimap.

    uv run python tools/refine_hair.py out/hair-native-large \
        --narrow out/hair-native-narrow --out out/hair-color-refined

Reads saved native 16-bit alpha and 8-bit foreground estimates; no model inference.
This is an experimental appearance constraint, not semantic hair recognition.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from tools.fba_reference import sha256, validate_trimap
from tools.matte_hair import Matte, load_trimap, save_results, write_image


def connected_to(mask: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    _, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(int(labels.max()) + 1, bool)
    keep[np.unique(labels[seeds & mask])] = True
    keep[0] = False
    return keep[labels]


def refine_foreground(rgb: np.ndarray, trimap: np.ndarray, matte: Matte,
                      radius: int | None = None) -> tuple[Matte, np.ndarray]:
    validate_trimap(rgb, trimap)
    alpha, foreground = matte.alpha, matte.foreground
    if (alpha.shape != trimap.shape or foreground.shape != rgb.shape
            or not np.isfinite(alpha).all() or not np.isfinite(foreground).all()
            or alpha.min() < 0 or alpha.max() > 1
            or foreground.min() < 0 or foreground.max() > 1):
        raise ValueError("Expected finite native alpha and foreground in [0, 1].")
    radius = max(2, round(min(trimap.shape) / 64)) if radius is None else radius
    if not isinstance(radius, int) or radius < 1:
        raise ValueError("Color-neighborhood radius must be a positive integer.")
    started = time.perf_counter()
    known_fg, known_bg = trimap == 255, trimap == 0
    if (alpha[known_fg] != 1).any() or (alpha[known_bg] != 0).any():
        raise ValueError("Input alpha must respect the known foreground/background trimap.")
    opaque = connected_to((alpha >= 0.98) | known_fg, known_fg)
    image = rgb.astype(np.float32) / 255.0
    support = opaque.astype(np.float32)
    kernel = (2 * radius + 1,) * 2
    count = cv2.boxFilter(support, -1, kernel, normalize=False)
    mean = cv2.boxFilter(image * support[..., None], -1, kernel, normalize=False)
    mean /= np.maximum(count[..., None], 1)
    second = cv2.boxFilter(image * image * support[..., None], -1, kernel, normalize=False)
    second /= np.maximum(count[..., None], 1)
    sigma = np.sqrt(np.maximum(second - mean * mean, 0) + 0.08 ** 2)

    # Compare recovered F, not mixed input RGB: a genuine translucent hair can look like B.
    _, labels = cv2.distanceTransformWithLabels(
        (~opaque).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    means = np.zeros((int(labels.max()) + 1, 3), np.float32)
    sigmas = np.ones_like(means)
    means[labels[opaque]], sigmas[labels[opaque]] = mean[opaque], sigma[opaque]
    distance = np.sqrt(np.mean(((foreground - means[labels]) / sigmas[labels]) ** 2, axis=2))
    transition = np.clip((distance - 2) / 2, 0, 1)
    confidence = 1 - transition * transition * (3 - 2 * transition)
    opacity = np.clip((alpha - 0.25) / 0.55, 0, 1)
    confidence = np.clip(np.maximum(confidence, opacity * opacity * (3 - 2 * opacity)), 0, 1)
    confidence[opaque] = 1
    detail_radius = max(1, round(min(trimap.shape) / 800))
    floor = cv2.erode(alpha, np.ones((2 * detail_radius + 1,) * 2, np.uint8))
    # Remove a broad pedestal, not the native thin-ridge residual above that pedestal.
    refined = np.clip(alpha - (1 - confidence) * floor, 0, alpha)
    supported = connected_to((refined > 0.005) | opaque, opaque)
    confidence[~supported] = 0
    refined[~supported] = 0
    refined = refined.astype(np.float32)
    refined[known_fg], refined[known_bg] = 1, 0
    confidence[known_fg], confidence[known_bg] = 1, 0
    return Matte(refined, foreground.copy(), time.perf_counter() - started), confidence


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read RGB crop: {path}")
    return np.ascontiguousarray(image[..., ::-1])


def read_matte(folder: Path, name: str, rgb: np.ndarray) -> Matte:
    alpha_path, rgba_path = folder / f"{name}_alpha.png", folder / f"{name}_rgba.png"
    alpha = cv2.imread(str(alpha_path), cv2.IMREAD_UNCHANGED)
    rgba = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
    if alpha is None or rgba is None:
        raise FileNotFoundError(f"Missing native alpha/RGBA: {alpha_path}, {rgba_path}")
    if (alpha.dtype != np.uint16 or alpha.shape != rgb.shape[:2]
            or rgba.dtype != np.uint8 or rgba.shape != (*rgb.shape[:2], 4)):
        raise ValueError(f"Expected matching native 16-bit alpha and 8-bit RGBA: {folder}")
    return Matte(alpha.astype(np.float32) / 65535,
                 rgba[..., [2, 1, 0]].astype(np.float32) / 255, 0.0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wide", type=Path, help="saved matte_hair run containing report.json")
    parser.add_argument("--narrow", type=Path, help="optional aligned narrow-trimap reference run")
    parser.add_argument("--out", type=Path, default=Path("out/hair-color-refined"))
    args = parser.parse_args(argv)
    roots = [args.wide] + ([args.narrow] if args.narrow else [])
    if any(args.out.resolve() == root.resolve() for root in roots):
        parser.error("--out must not overwrite an input run.")
    source = json.loads((args.wide / "report.json").read_text())
    narrow = json.loads((args.narrow / "report.json").read_text()) if args.narrow else None
    report = {
        "settings": {"opaque_alpha": 0.98, "color_sigma_floor_srgb": 0.08,
                     "color_distance_full": 2, "color_distance_zero": 4,
                     "opacity_protection_start": 0.25, "opacity_protection_full": 0.8,
                     "connectivity": 8, "connectivity_alpha_floor": 0.005},
        "input_report_sha256": sha256(args.wide / "report.json"),
        "narrow_report_sha256": sha256(args.narrow / "report.json") if narrow else None,
        "refiner_sha256": sha256(Path(__file__)),
        "versions": {"numpy": np.__version__, "opencv": cv2.__version__},
        "timing_scope": "postprocessing only; loaded references have zero runtime",
        "cases": {},
    }
    for name, metadata in source["cases"].items():
        if Path(name).name != name or name in (".", ".."):
            raise ValueError(f"Invalid saved photo stem: {name}")
        folder = args.wide / name
        rgb = read_rgb(folder / "crop.png")
        trimap = load_trimap(folder / "trimap.png", rgb)
        wide = read_matte(folder, "fba", rgb)
        mattes = {"current": read_matte(folder, "current", rgb), "fba_wide": wide}
        if narrow:
            reference = narrow["cases"][name]
            if any(metadata[key] != reference[key] for key in ("source_sha256", "crop_box_xyxy")):
                raise ValueError(f"Narrow reference has a different photo or crop: {name}")
            narrow_rgb = read_rgb(args.narrow / name / "crop.png")
            if not np.array_equal(rgb, narrow_rgb):
                raise ValueError(f"Narrow reference crop pixels differ: {name}")
            narrow_trimap = load_trimap(args.narrow / name / "trimap.png", rgb)
            mattes["fba_narrow"] = read_matte(args.narrow / name, "fba", rgb)
        refined, confidence = refine_foreground(rgb, trimap, wide)
        mattes["fba_refined"] = refined
        output = args.out / name
        title = "Wide trimap; narrow differs" if narrow else "Original trimap (unchanged)"
        case = save_results(output, rgb, trimap, mattes, trimap_title=title)
        write_image(output / "support_confidence.png", np.rint(confidence * 65535).astype(np.uint16))
        case.update(
            source=metadata["source"], source_sha256=metadata["source_sha256"],
            crop_box_xyxy=metadata["crop_box_xyxy"],
            color_radius_pixels=max(2, round(min(trimap.shape) / 64)),
            detail_radius_pixels=max(1, round(min(trimap.shape) / 800)),
            input_alpha_sha256=sha256(folder / "fba_alpha.png"),
            input_foreground_sha256=sha256(folder / "fba_rgba.png"),
            input_crop_sha256=sha256(folder / "crop.png"),
            original_trimap_sha256=sha256(folder / "trimap.png"),
        )
        if narrow:
            write_image(output / "narrow_trimap.png", narrow_trimap)
            case["methods"]["fba_narrow"]["unknown_fraction"] = float((narrow_trimap == 128).mean())
            case["methods"]["fba_narrow"]["trimap_file"] = "narrow_trimap.png"
            case["narrow_alpha_sha256"] = sha256(args.narrow / name / "fba_alpha.png")
            case["narrow_foreground_sha256"] = sha256(args.narrow / name / "fba_rgba.png")
            case["narrow_trimap_sha256"] = sha256(args.narrow / name / "trimap.png")
        report["cases"][name] = case
        print(f"{name}: color-connected refinement {refined.seconds:.2f}s", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
