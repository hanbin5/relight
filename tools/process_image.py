"""Run the whole pipeline on still photos: three formats, print sheets and a report.

    uv run python tools/process_image.py photo.jpg
    uv run python tools/process_image.py photos/*.jpg --background light_blue --out out/

For each photo it prints the seven checks and saves the outputs the app's Save button
would. Useful for collecting test results (plan C.1) from photos taken elsewhere.
"""

import argparse
import sys
from pathlib import Path

from idphoto import geometry
from idphoto.capture import ImageSource
from idphoto.landmarks import FaceLandmarkDetector
from idphoto.pipeline import BACKGROUNDS, Pipeline, Settings, to_live

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("photos", nargs="+")
    parser.add_argument("--background", choices=list(BACKGROUNDS), default="white")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--out", default=str(ROOT / "out"))
    parser.add_argument("--png", action="store_true", help="save PNG instead of JPG")
    args = parser.parse_args()

    settings = Settings(background=args.background, strength=args.strength)
    detector = FaceLandmarkDetector()
    timestamp = 0  # VIDEO-mode landmarks need increasing timestamps, across photos too
    status = 0
    for path in args.photos:
        frame = ImageSource(path).frame
        live, _ = to_live(frame)
        pipeline = Pipeline(detector)
        for _ in range(5):  # a still photo stands in for 5 identical frames
            timestamp += 33
            ctx, _ = pipeline.process(live, settings, timestamp_ms=timestamp, full_frame=frame)
        print(f"\n{path}")
        if ctx.landmarks is None:
            print("  no face found")
            status = 1
            continue
        for n, v in ctx.rule_verdicts.items():
            value = "-" if v.value is None else f"{v.value:.3f}"
            print(f"  {n} {v.key:13s} {'pass' if v.passed else 'FAIL'}  {value:>9s}  ({v.bound})"
                  + (f"  -> {v.message}" if v.message else ""))
        print(f"  R_LR {ctx.stats['r_lr_before']:.2f} -> {ctx.stats['r_lr_after']:.2f}")
        result = pipeline.render_final(frame, ctx, settings)
        folder, files = geometry.save_outputs(result, Path(args.out), "png" if args.png else "jpg")
        print(f"  saved {len(files)} files to {folder}")
    detector.close()
    return status


if __name__ == "__main__":
    sys.exit(main())
