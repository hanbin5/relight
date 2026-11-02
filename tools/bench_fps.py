"""Frames per second of the live pipeline on this machine (plan C.1: >= 10 fps on a laptop CPU).

    uv run python tools/bench_fps.py
    uv run python tools/bench_fps.py --frames 200 --camera-size 1920x1080

Feeds synthetic camera frames through exactly what the app does per frame: downscale to
the live width, MediaPipe landmarks, relighting, matting, geometry and the checker.
Rendering the frames is not timed.
"""

import argparse
import sys
import time

import numpy as np

from idphoto import color, synthetic
from idphoto.pipeline import Pipeline, Settings, to_live


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--camera-size", default="1280x720")
    parser.add_argument("--no-detector", action="store_true", help="use the renderer's landmarks instead of MediaPipe")
    args = parser.parse_args()
    w, h = (int(v) for v in args.camera_size.split("x"))

    frames = []
    for i in range(8):
        rgb, lm, _ = synthetic.render(synthetic.Scene(width=w, height=h, yaw=2.0 * np.sin(i), seed=i))
        frames.append((rgb, lm))
    detector = None
    if not args.no_detector:
        from idphoto.landmarks import FaceLandmarkDetector

        detector = FaceLandmarkDetector()
    pipeline = Pipeline(detector)
    settings = Settings()
    totals: dict = {}
    t0 = None
    for i in range(args.frames + 5):
        if i == 5:  # warm-up done
            t0 = time.perf_counter()
            totals = {}
        rgb, lm = frames[i % len(frames)]
        start = time.perf_counter()
        live, lm_live = to_live(rgb, None if detector else lm)
        ctx, _ = pipeline.process(live, settings, landmarks=lm_live, timestamp_ms=33 * i, full_frame=rgb)
        color.encode_u8(ctx.output_frame)  # the live view's display conversion
        totals["total"] = totals.get("total", 0.0) + 1000 * (time.perf_counter() - start)
        for k, v in ctx.stats["timings_ms"].items():
            totals[k] = totals.get(k, 0.0) + v
    elapsed = time.perf_counter() - t0
    print(f"camera {w}x{h} -> live {live.shape[1]}x{live.shape[0]}, {args.frames} frames")
    for k, v in totals.items():
        print(f"  {k:10s} {v / args.frames:6.1f} ms")
    fps = args.frames / elapsed
    print(f"  => {fps:.1f} fps ({'meets' if fps >= 10 else 'below'} the 10 fps target)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
