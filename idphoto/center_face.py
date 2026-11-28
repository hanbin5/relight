"""Frame-local, centre-based face selection for experimental MODNet."""

import math

import numpy as np

from .landmarks import FaceCandidate, FaceLandmarkDetector, MODEL_PATH
from .reacquiring_landmarks import _extent, _square, _tiles


def _duplicate(a, b):
    ax, ay, size_a = _extent(a.landmarks)
    bx, by, size_b = _extent(b.landmarks)
    return (math.hypot(ax - bx, ay - by) < 0.3 * min(size_a, size_b)
            and 0.5 <= size_a / max(size_b, 1.0) <= 2.0)


def _crop_boxes(width, height):
    half = width // 2
    return ((0, 0, half, height), (half, 0, width, height),
            (width // 8, 0, width - width // 8, height))


class CenterFaceDetector:
    """Select the detected face closest to the frame centre, with no hysteresis."""

    def __init__(self, model_path=MODEL_PATH, *, factory=FaceLandmarkDetector):
        self._factory, self._model = factory, model_path
        self._full = factory(model_path=model_path, video=True, num_faces=5)
        try:
            self._crop = factory(model_path=model_path, video=False, num_faces=5)
        except BaseException:
            self._full.close()
            raise
        self._closed = False
        self._clear()

    def _clear(self):
        self._shape = self._ts = self._last_face = None
        self._search = []
        self._next_tile = 0
        self.pitch_deg = self.yaw_deg = None
        self.candidates = ()
        self.selected_index = None
        self.track_break = False
        self.source = "none"
        self.calls = 0

    def reset(self):
        if self._closed:
            raise RuntimeError("detector is closed")
        self._full.reset()
        self._clear()

    def close(self):
        if not self._closed:
            self._closed = True
            self._full.close()
            self._crop.close()
            self._clear()

    def detect(self, rgb, timestamp_ms=None):
        if self._closed:
            raise RuntimeError("detector is closed")
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
            raise ValueError("expected nonempty RGB uint8 [height,width,3]")
        height, width = rgb.shape[:2]
        if self._shape is not None and self._shape != (height, width):
            self.reset()
        if self._shape is None:
            self._shape = height, width
            self._search = _tiles(width, height)
        timestamp = (0 if self._ts is None else self._ts + 33) \
            if timestamp_ms is None else int(timestamp_ms)
        if timestamp < 0 or (self._ts is not None and timestamp <= self._ts):
            raise ValueError("timestamp must increase; reset on stream change or seek")
        self._ts = timestamp
        found = []
        self.calls = 1

        def append(candidate, x=0, y=0, crop_width=width, crop_height=height, cropped=False):
            points = candidate.landmarks
            if points.shape != (478, 3) or not np.isfinite(points).all():
                raise ValueError("invalid face landmark mesh")
            cx, cy, side = _extent(points)
            if side < 8:
                return
            if cropped and not (
                side * .15 <= cx <= crop_width - side * .15
                and side * .15 <= cy <= crop_height - side * .15
            ):
                return
            mapped = points.copy()
            mapped[:, 0] += x
            mapped[:, 1] += y
            candidate = FaceCandidate(mapped, candidate.pitch_deg, candidate.yaw_deg)
            if not any(_duplicate(candidate, existing) for existing in found):
                found.append(candidate)

        for item in self._full.detect_all(rgb, timestamp):
            append(item)
        def detect_crop(box):
            x0, y0, x1, y1 = box
            if x1 <= x0:
                return
            self.calls += 1
            for item in self._crop.detect_all(rgb[y0:y1, x0:x1]):
                append(item, x0, y0, x1 - x0, y1 - y0, True)

        for box in _crop_boxes(width, height):
            detect_crop(box)
        if not found:
            hint = None
            if self._last_face is not None:
                cx, cy, side = self._last_face
                hint = _square(cx, cy, max(64, 2.8 * side), width, height)
                detect_crop(hint)
            for _ in range(min(2, len(self._search))):
                box = self._search[self._next_tile]
                self._next_tile = (self._next_tile + 1) % len(self._search)
                if box != hint:
                    detect_crop(box)

        self.candidates = tuple(found)
        self.pitch_deg = self.yaw_deg = None
        self.source = "none"
        self.selected_index = None
        if not found:
            self.track_break = self._last_face is not None
            self._last_face = None
            return None

        def distance_to_center(index):
            x, y, _ = _extent(found[index].landmarks)
            return ((x - width / 2) ** 2 + (y - height / 2) ** 2, x, y)

        selected = min(range(len(found)), key=distance_to_center)
        face = _extent(found[selected].landmarks)
        old = self._last_face
        self.track_break = old is None
        if old is not None:
            nearest_old = min(range(len(found)), key=lambda i: math.hypot(
                _extent(found[i].landmarks)[0] - old[0],
                _extent(found[i].landmarks)[1] - old[1]))
            self.track_break = (selected != nearest_old
                                or math.hypot(face[0] - old[0], face[1] - old[1])
                                > .7 * max(old[2], face[2])
                                or not .55 <= face[2] / max(old[2], 1.0) <= 1.8)
        self._last_face = face
        self.selected_index = selected
        self.source = "native"
        chosen = found[selected]
        self.pitch_deg, self.yaw_deg = chosen.pitch_deg, chosen.yaw_deg
        return chosen.landmarks
