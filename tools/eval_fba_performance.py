"""Compare local CPU/MPS FBA output and offscreen worker throughput.

Requires the FBA extra, local research assets and cached evaluation landmarks.
This script writes numeric measurements only; no input or output photos are sent anywhere.
"""

import argparse
import csv
import functools
import os
import time
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import cv2
import numpy as np
import torch
from PySide6.QtWidgets import QApplication

from idphoto.capture import DemoSource
from idphoto.gui import Worker
from idphoto.pipeline import Pipeline, Settings, to_live
from tools.eval_clothing import NAMES, ROOT, fba_trimap
from tools.fba_reference import FBAReference
from tools.presentation_fba import PresentationFBA


def load(name, cache):
    bgr = cv2.imread(str(ROOT / "docs/evaluation/test" / f"{name}.jpg"))
    if bgr is None:
        raise FileNotFoundError(f"Missing evaluation image: {name}")
    rgb, _ = to_live(bgr[:, :, ::-1].copy())
    return rgb, np.load(cache / f"{name}-live.npy")


def compare(cache, output):
    cpu, mps = FBAReference(device="cpu"), FBAReference(device="mps")
    rows = []
    for name in NAMES:
        rgb, lm = load(name, cache)
        ctx, _ = Pipeline().process(rgb, Settings(strength=0, auto_shutter=False), landmarks=lm)
        trimap = fba_trimap(rgb, lm, ctx)
        baseline, _ = cpu.process(rgb, trimap)
        candidate, _ = mps.process(rgb, trimap)
        diff = np.abs(baseline - candidate)
        rows.append({"photo": name, "alpha_mae": float(diff[..., 0].mean()),
                     "alpha_max_difference": float(diff[..., 0].max()),
                     "all_channels_max_difference": float(diff.max()),
                     "alpha_threshold_05_changed": int(
                         ((baseline[..., 0] > .5) != (candidate[..., 0] > .5)).sum())})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Compared {len(rows)} photos; max alpha difference "
          f"{max(row['alpha_max_difference'] for row in rows):.8f}; CSV: {output}", flush=True)


def measure_worker(device, rgb, landmarks, warmup, duration):
    class FixedSource(DemoSource):
        name = "fixed shot3"
        lock_state = "locked"

        def read(self):
            time.sleep(1 / 30)
            return rgb, landmarks

    settings = Settings(clothing_enabled=True, fba_demo=True, strength=0, auto_shutter=False)
    worker = Worker(FixedSource(), settings, functools.partial(PresentationFBA, device=device))
    frames, snapshots, errors = [], [], []
    start = time.monotonic()
    worker.frame_ready.connect(lambda _: frames.append(time.monotonic()))
    worker.fba_snapshot.connect(lambda result: snapshots.append(
        (time.monotonic(), time.monotonic() - result["timestamp_ms"] / 1000)))
    worker.failed.connect(errors.append)
    worker.fba_failed.connect(errors.append)
    worker.start()
    app = QApplication.instance()
    try:
        while time.monotonic() - start < warmup + duration:
            app.processEvents()
            time.sleep(.002)
    finally:
        worker.running = False
        worker.wait()
        app.processEvents()
    if errors:
        raise RuntimeError(f"{device} worker failed: {errors}")
    frame_times = [t for t in frames if start + warmup <= t < start + warmup + duration]
    samples = [s for s in snapshots if start + warmup <= s[0] < start + warmup + duration]
    if not samples:
        raise RuntimeError(f"No {device} snapshots completed during measurement.")
    print(f"{device}: Clothing {len(frame_times) / duration:.2f} fps, "
          f"FBA {len(samples) / duration:.2f} fps, "
          f"mean frame age {1000 * np.mean([s[1] for s in samples]):.1f} ms", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="numeric CPU/MPS comparison CSV")
    parser.add_argument("--cache", type=Path, default=ROOT / "out/clothing-landmarks")
    parser.add_argument("--warmup", type=float, default=3)
    parser.add_argument("--duration", type=float, default=9)
    args = parser.parse_args()
    if args.warmup < 0 or args.duration <= 0:
        parser.error("warmup must be nonnegative and duration must be positive")
    if not torch.backends.mps.is_available():
        parser.error("MPS is unavailable; this comparison requires an Apple GPU.")
    torch.set_num_threads(4)
    compare(args.cache, args.out)
    app = QApplication.instance() or QApplication([])
    rgb, lm = load("shot3", args.cache)
    for device in ("cpu", "mps"):
        measure_worker(device, rgb, lm, args.warmup, args.duration)
    app.processEvents()


if __name__ == "__main__":
    main()
