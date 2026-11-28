"""Make staged-fault copies of good photos, for eval_checker.py (plan C.1).

    uv run python tools/make_faults.py docs/evaluation/test/*.jpg --out testset/

We don't have real fault photos yet, so this makes three faults that are easy to fake
from a good photo: blur, a tilted head and a dark side of the face. Each photo is
copied to testset/good/ and a faulty copy goes to testset/blur/, tilt/ and
side_shadow/. Glare, closed eyes, an open mouth and a top like the wall can't be faked
this way; those still need real photos (or the synthetic faces in eval_synthetic.py).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from idphoto import color, npimg
from idphoto.capture import ImageSource

TILT_DEG = 8.0  # rule 4 allows 2.5 degrees of roll
DARK_SIDE = 0.15  # the dark edge of the frame keeps 15% of its light (linear)


def blur(img):
    r = max(2, img.shape[1] // 300)  # about 4 px on a 1280 px photo
    out = npimg.box_mean(npimg.box_mean(img.astype(np.float32), r), r)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def tilt(img, deg=TILT_DEG):
    """Rotate about the center; corners are filled from the nearest edge."""
    h, w = img.shape[:2]
    a = np.radians(deg)
    ys, xs = np.mgrid[0:h, 0:w] + 0.5
    cx, cy = w / 2, h / 2
    src_x = cx + (xs - cx) * np.cos(a) - (ys - cy) * np.sin(a)
    src_y = cy + (xs - cx) * np.sin(a) + (ys - cy) * np.cos(a)
    out = np.stack([npimg.sample_bilinear(img[:, :, c].astype(np.float32), src_x, src_y) for c in range(3)], axis=-1)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def side_shadow(img, dark=DARK_SIDE):
    """Light falls off from left to right, like a window on one side. The fall-off is
    across the middle third of the frame, where the face usually is."""
    lin = color.decode_u8(img)
    t = np.clip((np.arange(img.shape[1]) / img.shape[1] - 1 / 3) * 3, 0, 1)
    ramp = (1.0 - (1.0 - dark) * t).astype(np.float32)
    return color.encode_u8(lin * ramp[None, :, None])


FAULTS = {"blur": blur, "tilt": tilt, "side_shadow": side_shadow}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("photos", nargs="+")
    parser.add_argument("--out", default="testset")
    args = parser.parse_args()
    import cv2

    out = Path(args.out)
    for name in ["good", *FAULTS]:
        (out / name).mkdir(parents=True, exist_ok=True)
    for path in args.photos:
        img = ImageSource(path).frame  # RGB
        stem = Path(path).stem + ".png"
        cv2.imwrite(str(out / "good" / stem), img[:, :, ::-1])
        for name, fault in FAULTS.items():
            cv2.imwrite(str(out / name / stem), fault(img)[:, :, ::-1])
        print(f"{path}: good + {', '.join(FAULTS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
