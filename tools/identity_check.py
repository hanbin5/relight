"""Does relighting keep the face? ArcFace similarity to the unrelit photo (plan C.1).

    uv run python tools/make_rating_pairs.py docs/evaluation/test/*.jpg   # makes the photos
    uv run --with onnxruntime python tools/identity_check.py out/rating/singles

For each <name>_relit.png in the folder, compares its ArcFace embedding with
<name>_unrelit.png (cosine similarity, 1 = same face). If <name>_ai.png or .jpg is there
too (an AI-generated profile photo of the same person, consenting team members only),
it is compared with the unrelit photo as well: the plan expects ours to be closer.

ArcFace is used for evaluation only; the app never runs it. The model is insightface's
w600k_mbf.onnx (from the buffalo_s pack, non-commercial research use), downloaded once to
models/. `--with onnxruntime` adds onnxruntime for this run only, so the project's
dependencies don't change.
"""

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from idphoto import npimg
from idphoto.capture import ImageSource
from idphoto.landmarks import FaceLandmarkDetector

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "arcface_w600k_mbf.onnx"
MODEL_ZIP = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip"

# Where ArcFace expects the 5 points in its 112 x 112 input: the eye on the image's left,
# the other eye, the nose tip, and the mouth corners (image left, then right).
TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                     [41.5493, 92.3655], [70.7299, 92.2041]])
POINTS = [468, 473, 1, 61, 291]  # the same 5 points in MediaPipe's mesh (iris centers)


def fetch_model():
    if MODEL.exists():
        return
    print(f"downloading the ArcFace model (127 MB pack, keeps 13 MB) to {MODEL.parent} ...")
    data = urllib.request.urlopen(MODEL_ZIP).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        name = next(n for n in z.namelist() if n.endswith("w600k_mbf.onnx"))
        MODEL.parent.mkdir(exist_ok=True)
        MODEL.write_bytes(z.read(name))


def similarity_transform(src, dst):
    """Least-squares scale + rotation + shift mapping src points onto dst (Umeyama)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    u, sig, vt = np.linalg.svd(d.T @ s / len(src))
    flip = np.diag([1.0, np.sign(np.linalg.det(u @ vt))])
    rot = u @ flip @ vt
    scale = np.trace(np.diag(sig) @ flip) / s.var(0).sum()
    return scale * rot, mu_d - scale * rot @ mu_s


def align(img, lm):
    """Warp the face to ArcFace's 112 x 112 template."""
    A, t = similarity_transform(lm[POINTS, :2].astype(float), TEMPLATE)
    inv = np.linalg.inv(A)
    ys, xs = np.mgrid[0:112, 0:112] + 0.5
    pts = np.stack([xs.ravel(), ys.ravel()], 1) - t
    src = pts @ inv.T
    out = [npimg.sample_bilinear(img[:, :, c].astype(np.float32), src[:, 0], src[:, 1]) for c in range(3)]
    return np.stack(out, -1).reshape(112, 112, 3)


class ArcFace:
    def __init__(self):
        import onnxruntime

        fetch_model()
        self.session = onnxruntime.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def embed(self, face112):
        x = ((face112 - 127.5) / 127.5).transpose(2, 0, 1)[None].astype(np.float32)
        e = self.session.run(None, {self.input: x})[0][0]
        return e / np.linalg.norm(e)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", nargs="?", default="out/rating/singles")
    args = parser.parse_args()

    folder = Path(args.folder)
    names = sorted(p.name[: -len("_relit.png")] for p in folder.glob("*_relit.png"))
    if not names:
        print(f"No *_relit.png photos in {folder}. Run make_rating_pairs.py first.")
        return 1

    model, detector = ArcFace(), FaceLandmarkDetector()
    timestamp = 0

    def embedding(path):
        nonlocal timestamp
        img = ImageSource(path).frame
        timestamp += 33
        lm = detector.detect(img, timestamp)
        return None if lm is None else model.embed(align(img, lm))

    print("photo     relit vs unrelit   AI profile vs unrelit")
    sims, wins, ai_count = [], 0, 0
    for name in names:
        ref = embedding(folder / f"{name}_unrelit.png")
        relit = embedding(folder / f"{name}_relit.png")
        if ref is None or relit is None:
            print(f"{name:9s} no face found")
            continue
        sim = float(relit @ ref)
        sims.append(sim)
        ai_path = next((p for p in (folder / f"{name}_ai.png", folder / f"{name}_ai.jpg") if p.exists()), None)
        ai_text = "-"
        if ai_path is not None:
            ai = embedding(ai_path)
            if ai is not None:
                ai_sim = float(ai @ ref)
                ai_count += 1
                wins += sim > ai_sim
                ai_text = f"{ai_sim:.3f}"
        print(f"{name:9s} {sim:.3f}              {ai_text}")
    detector.close()

    if sims:
        print(f"\nrelit vs unrelit: mean {np.mean(sims):.3f}, lowest {min(sims):.3f} ({len(sims)} photos)")
    if ai_count:
        print(f"ours closer to the person than the AI profile: {wins} of {ai_count} (plan: at least 9 of 10)")
    else:
        print("no AI profile photos (<name>_ai.png), so no baseline comparison")
    return 0


if __name__ == "__main__":
    sys.exit(main())
