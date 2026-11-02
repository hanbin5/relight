"""Re-run the pipeline on frames saved by capture_frames.py (offline debugging).

    uv run python tools/replay_frames.py out/debug/20261102_225728

For each frame_NN.png with its landmarks, runs the live pipeline a few times (as the app
would on a still scene) and writes frame_NN_replay.png with the same panels as
capture_frames.py, so a change to the code can be compared on real frames.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from idphoto.capture import ImageSource
from idphoto.pipeline import Pipeline, Settings, to_live

from capture_frames import steps_image  # noqa: E402  (same folder)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder")
    parser.add_argument("--frames", type=int, default=4, help="pipeline runs per frame (warm start)")
    args = parser.parse_args()
    import cv2

    folder = Path(args.folder)
    for png in sorted(folder.glob("frame_[0-9][0-9].png")):
        frame = ImageSource(png).frame
        lm = np.load(png.with_suffix(".npz"))["landmarks"]
        live, lm_live = to_live(frame, lm if len(lm) else None)
        pipeline, settings = Pipeline(), Settings()
        for i in range(args.frames):
            ctx, _ = pipeline.process(live, settings, landmarks=lm_live, timestamp_ms=33 * i, full_frame=frame)
        out = png.with_name(png.stem + "_replay.png")
        cv2.imwrite(str(out), steps_image(ctx)[:, :, ::-1])
        v = ctx.rule_verdicts[7]
        t = ctx.stats["timings_ms"]
        print(f"{png.name}: rule 7 {'pass' if v.passed else 'FAIL'} {v.detail}  matting {t['matting']:.1f} ms")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    sys.exit(main())
