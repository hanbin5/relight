"""Download the MediaPipe Face Landmarker model (the app's only pretrained part).

    uv run python tools/fetch_model.py

Source: Google's MediaPipe model storage, as linked from the Face Landmarker guide
(plan reference [21]). The file is about 3.6 MB and is saved to models/face_landmarker.task.
"""

import sys
import urllib.request
from pathlib import Path

URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEST = Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"


def main() -> int:
    if DEST.exists():
        print(f"Already present: {DEST}")
        return 0
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    tmp = DEST.with_suffix(".part")
    urllib.request.urlretrieve(URL, tmp)
    tmp.rename(DEST)
    print(f"Saved {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
