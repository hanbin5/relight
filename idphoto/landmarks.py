"""Face landmarks with MediaPipe Face Landmarker: the app's only pretrained component.

Returns the 478-point mesh (468 face + 10 iris points) as float32 (478, 3) in pixels of
the input frame. z is MediaPipe's relative depth (smaller = closer), scaled like x.

Each detect() also keeps the head's pitch and yaw from MediaPipe's facial transformation
matrix in `pitch_deg` and `yaw_deg` (degrees; None when no face was found). Rule 4 (head
pose) uses them: the landmark ratios it used before depend on face shape, the matrix
angles do not. Roll needs no matrix: the iris line already gives it in degrees.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("GLOG_minloglevel", "2")  # silence MediaPipe's C++ logging

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"


@dataclass(frozen=True)
class FaceCandidate:
    landmarks: np.ndarray
    pitch_deg: float | None
    yaw_deg: float | None


class FaceLandmarkDetector:
    def __init__(self, model_path=None, video: bool = True, num_faces: int = 1):
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        path = Path(model_path or MODEL_PATH)
        if not path.exists():
            raise FileNotFoundError(
                f"Face landmark model not found at {path}. "
                "Provide a local model with --face-model; in a checkout, "
                "uv run python tools/fetch_model.py can prepare the default path."
            )
        mode = vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=mode,
            num_faces=num_faces,
            output_facial_transformation_matrixes=True,  # for pitch_deg, yaw_deg (rule 4)
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._mp = mp
        self._video = video
        self._model_path = str(path)
        self._num_faces = num_faces
        self._last_ts = -1
        self.pitch_deg = None  # angles of the last detect(), see pitch_from_matrix
        self.yaw_deg = None

    def detect(self, rgb: np.ndarray, timestamp_ms=None):
        """Landmarks of the most prominent face in an RGB uint8 frame, or None."""
        candidates = self.detect_all(rgb, timestamp_ms)
        self.pitch_deg = self.yaw_deg = None
        if not candidates:
            return None
        candidate = candidates[0]
        self.pitch_deg, self.yaw_deg = candidate.pitch_deg, candidate.yaw_deg
        return candidate.landmarks

    def detect_all(self, rgb: np.ndarray, timestamp_ms=None):
        """Return every detected mesh and its corresponding transformation angles."""
        h, w = rgb.shape[:2]
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if self._video:
            ts = int(timestamp_ms) if timestamp_ms is not None else self._last_ts + 33
            ts = max(ts, self._last_ts + 1)  # VIDEO mode needs increasing timestamps
            self._last_ts = ts
            result = self._landmarker.detect_for_video(image, ts)
        else:
            result = self._landmarker.detect(image)
        candidates = []
        matrices = result.facial_transformation_matrixes or []
        for index, face in enumerate(result.face_landmarks):
            matrix = matrices[index] if index < len(matrices) else None
            pts = np.array([(p.x, p.y, p.z) for p in face], np.float32)
            pts[:, 0] *= w
            pts[:, 1] *= h
            pts[:, 2] *= w
            candidates.append(FaceCandidate(
                pts, pitch_from_matrix(matrix) if matrix is not None else None,
                yaw_from_matrix(matrix) if matrix is not None else None))
        return candidates

    def close(self):
        self._landmarker.close()

    def reset(self):
        """Replace VIDEO tracking state while keeping this detector's configuration."""
        fresh = type(self)(self._model_path, video=self._video, num_faces=self._num_faces)
        old = self._landmarker
        self._landmarker = fresh._landmarker
        fresh._landmarker = None
        self._last_ts = -1
        self.pitch_deg = self.yaw_deg = None
        old.close()


def pitch_from_matrix(matrix) -> float:
    """Head pitch in degrees from MediaPipe's 4 x 4 facial transformation matrix:
    the rotation about the x axis, 0 when the face looks straight at the camera."""
    R = np.asarray(matrix, np.float64)[:3, :3]
    return float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))


def yaw_from_matrix(matrix) -> float:
    """Head yaw in degrees from the same matrix: the rotation about the y axis, 0 when the
    face looks straight at the camera, + when it turns toward the image right (the same
    sign as geometry.head_pose's yaw_ratio)."""
    R = np.asarray(matrix, np.float64)[:3, :3]
    return float(np.degrees(np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2]))))
