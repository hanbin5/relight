"""Make relit vs unrelit photo pairs for the blind rating (plan C.1, Appendix D).

    uv run python tools/make_rating_pairs.py docs/evaluation/test/*.jpg

For each photo, the résumé photo (3 x 4 cm) is rendered twice from the same frame: once
relit, once with relighting off (strength 0). Both get the same new background and crop,
so only the light differs. The two go side by side as "A" and "B" in a random order, for
a form that asks "Which photo looks better?".

Writes to out/rating/:
    pairs/pair_01.jpg ...      the images for the form, in this order
    singles/<photo>_relit.png  each version on its own (used by identity_check.py)
    key.csv                    which side is relit in each pair: keep it away from raters
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np

from idphoto import geometry
from idphoto.capture import ImageSource
from idphoto.landmarks import FaceLandmarkDetector
from idphoto.pipeline import Pipeline, Settings, to_live

GAP = 40  # white pixels between the two photos
MAX_WIDTH = 1600


def label(img: np.ndarray, text: str) -> np.ndarray:
    import cv2

    out = np.full((img.shape[0] + 60, img.shape[1], 3), 255, np.uint8)
    out[60:] = img
    cv2.putText(out, text, (img.shape[1] // 2 - 15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("photos", nargs="+")
    parser.add_argument("--out", default="out/rating")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    out = Path(args.out)
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    (out / "singles").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    photos = list(args.photos)
    rng.shuffle(photos)  # pair order shouldn't follow the file names

    detector = FaceLandmarkDetector()
    timestamp, rows = 0, []
    for path in photos:
        # Phone photos (4000 x 3000) use too much memory at full size, and the résumé
        # photo is only 354 x 472 px, so shrink anything wider than MAX_WIDTH first.
        frame, _ = to_live(ImageSource(path).frame, width=MAX_WIDTH)
        live, _ = to_live(frame)
        pipeline = Pipeline(detector)
        for _ in range(5):  # same as process_image.py: a still photo stands in for 5 frames
            timestamp += 33
            ctx, _ = pipeline.process(live, Settings(), timestamp_ms=timestamp, full_frame=frame)
        if ctx.landmarks is None:
            print(f"{path}: no face found, skipped")
            continue
        relit = pipeline.render_final(frame, ctx, Settings(strength=1.0))["photos"]["resume"]
        unrelit = pipeline.render_final(frame, ctx, Settings(strength=0.0))["photos"]["resume"]
        stem = Path(path).stem
        geometry.save_image(out / "singles" / f"{stem}_relit.png", relit)
        geometry.save_image(out / "singles" / f"{stem}_unrelit.png", unrelit)

        relit_side = rng.choice("AB")
        a, b = (relit, unrelit) if relit_side == "A" else (unrelit, relit)
        gap = np.full((a.shape[0] + 60, GAP, 3), 255, np.uint8)
        pair = np.hstack([label(a, "A"), gap, label(b, "B")])
        n = len(rows) + 1
        geometry.save_image(out / "pairs" / f"pair_{n:02d}.jpg", pair)
        rows.append({"pair": n, "photo": stem, "relit_side": relit_side})
        print(f"pair {n:2d}: {stem} (relit is {relit_side})")
    detector.close()

    with open(out / "key.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair", "photo", "relit_side"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} pairs in {out / 'pairs'}; answer key in {out / 'key.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
