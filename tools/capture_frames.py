"""Save webcam frames, their landmarks and the pipeline's steps, for debugging and test sets.

    uv run python tools/capture_frames.py
    uv run python tools/capture_frames.py --seconds 10 --every 15 --out out/frames

Opens the camera on the main thread (so macOS can ask for access), lets exposure settle,
runs the live pipeline on every frame and saves every Nth one into a new folder:
- frame_NN.png: the camera frame, lossless (input for offline experiments);
- frame_NN.npz: its landmarks, in camera-frame pixels;
- frame_NN_steps.png: live frame | relit | wall-color evidence | person prior |
  coarse mask | alpha | output, side by side.
Frames stay on this computer.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from idphoto import color, config, npimg
from idphoto.capture import Camera
from idphoto.landmarks import FaceLandmarkDetector
from idphoto.pipeline import Pipeline, Settings, to_live

ROOT = Path(__file__).resolve().parents[1]


def gray(a: np.ndarray, size) -> np.ndarray:
    """A [0, 1] map as an RGB tile of the given (h, w)."""
    a = npimg.resize(np.asarray(a, np.float32), size[0], size[1], "linear")
    return np.repeat((np.clip(a, 0, 1) * 255).astype(np.uint8)[..., None], 3, axis=2)


def steps_image(ctx) -> np.ndarray:
    h, w = ctx.raw_frame.shape[:2]
    m = ctx.stats.get("matting", {})
    tiles = [ctx.raw_frame, color.encode_u8(ctx.relit_frame)]
    for key in ("p_color", "prior", "coarse_mask"):
        tiles.append(gray(m[key], (h, w)) if key in m else np.zeros((h, w, 3), np.uint8))
    tiles += [gray(ctx.alpha, (h, w)), color.encode_u8(ctx.output_frame)]
    return np.concatenate(tiles, axis=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--every", type=int, default=10, help="save every Nth frame")
    parser.add_argument("--out", default=str(ROOT / "out" / "debug"))
    args = parser.parse_args()
    import cv2

    try:
        camera = Camera(args.camera)
    except RuntimeError:
        time.sleep(3)  # the first attempt may only have shown the permission request
        camera = Camera(args.camera)
    folder = Path(args.out) / datetime.now().strftime("%Y%m%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    detector = FaceLandmarkDetector()
    pipeline, settings = Pipeline(detector), Settings()
    warmup = config.WARMUP_S + 0.5  # let auto exposure settle first
    start, n, saved = time.monotonic(), 0, 0
    while time.monotonic() - start < warmup + args.seconds:
        frame, _ = camera.read()
        if frame is None:
            continue
        if time.monotonic() - start < warmup:
            continue
        live, _ = to_live(frame)
        ctx, _ = pipeline.process(live, settings, timestamp_ms=int(1000 * time.monotonic()), full_frame=frame)
        n += 1
        if n % args.every:
            continue
        cv2.imwrite(str(folder / f"frame_{saved:02d}.png"), frame[:, :, ::-1])
        lm = None if ctx.landmarks is None else ctx.landmarks * np.float32(frame.shape[1] / live.shape[1])
        np.savez(folder / f"frame_{saved:02d}.npz", landmarks=lm if lm is not None else np.zeros((0, 3)))
        cv2.imwrite(str(folder / f"frame_{saved:02d}_steps.png"), steps_image(ctx)[:, :, ::-1])
        verdicts = ctx.rule_verdicts or {}
        fails = [f"{k}:{v.key}" for k, v in verdicts.items() if not v.passed]
        print(f"frame_{saved:02d}: face={'yes' if ctx.landmarks is not None else 'no'}"
              f" R_LR={ctx.stats.get('r_lr_before', float('nan')):.2f}->{ctx.stats.get('r_lr_after', float('nan')):.2f}"
              f" fails={fails or '-'}")
        saved += 1
    camera.release()
    detector.close()
    print(f"{n} frames processed ({camera.backend}, {frame.shape[1]}x{frame.shape[0]}, exposure lock: {camera.lock_state});"
          f" {saved} saved to {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
