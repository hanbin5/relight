"""Checks this machine's webcam exposure / white-balance lock (plan B.2: test on every laptop).

Opens the camera, lets it warm up and lock, then watches the mean frame brightness for a
while. Keep the scene still (sit still, no one walking by, same light) while it runs. A
working lock keeps the brightness within a few percent; a camera that still adjusts itself
drifts. On macOS run it from a terminal that has camera access.

    uv run python tools/check_lock.py [--camera 0] [--seconds 6]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from idphoto import config  # noqa: E402
from idphoto.capture import open_camera  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=6.0, help="how long to watch after the lock")
    args = parser.parse_args()

    try:
        camera = open_camera(args.camera)
    except RuntimeError as exc:
        print(f"{exc}\nWaiting 5 s in case this was the permission request...")
        time.sleep(5)
        camera = open_camera(args.camera)
    print(f"camera: {camera.name}")
    if hasattr(camera, "capabilities"):
        print(f"controls offered: {camera.capabilities()}")

    start = time.monotonic()
    locked_at, before, after = None, [], []
    while True:
        now = time.monotonic()
        if locked_at is None and now - start > config.WARMUP_MAX_S + 10:
            camera.release()
            print("The camera gave no frames, or never finished warming up.")
            return 1
        frame, _ = camera.read()
        if frame is None:
            continue
        brightness = float(frame[::8, ::8].mean())
        if camera.lock_state == "warming":
            before.append(brightness)
            continue
        if locked_at is None:
            locked_at = now
            print(f"lock state after {now - start:.1f} s: {camera.lock_state}"
                  f"  ({frame.shape[1]}x{frame.shape[0]})")
        after.append(brightness)
        if now - locked_at >= args.seconds:
            break
    camera.release()

    def spread(values):
        return (max(values) - min(values)) / max(float(np.mean(values)), 1e-6)

    if len(before) >= 5:
        print(f"brightness while warming: {np.mean(before):.1f}, spread {spread(before):.1%}")
    drift = spread(after)
    verdict = "steady" if drift <= config.LOCK_MAX_DRIFT else "DRIFTING"
    print(f"brightness after the lock: {np.mean(after):.1f}, spread {drift:.1%} over "
          f"{len(after)} frames (limit {config.LOCK_MAX_DRIFT:.0%}): {verdict}")
    if camera.lock_state in ("unsupported", "failed"):
        print("The app will average the light over "
              f"{config.SMOOTH_FRAMES_FALLBACK} frames instead of {config.SMOOTH_FRAMES} (C.4 fallback).")
    return 0 if camera.lock_state in ("locked", "partial") and verdict == "steady" else 2


if __name__ == "__main__":
    sys.exit(main())
