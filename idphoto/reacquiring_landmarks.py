"""Opt-in live face reacquisition for the MODNet experiment."""

import math

import numpy as np

from .landmarks import FaceLandmarkDetector, MODEL_PATH


def _square(cx, cy, side, w, h):
    side = min(max(1, int(round(side))), w, h)
    x = min(max(0, int(round(cx - side * 0.5))), w - side)
    y = min(max(0, int(round(cy - side * 0.5))), h - side)
    return x, y, x + side, y + side


def _tiles(w, h):
    side = min(min(w, h), max(64, round(0.55 * min(w, h))))

    def centers(length):
        return np.linspace(
            side / 2, length - side / 2,
            1 + math.ceil((length - side) / (side * 0.65)),
        )

    boxes = [_square(x, y, side, w, h) for y in centers(h) for x in centers(w)]
    return sorted(boxes, key=lambda box: (
        ((box[0] + side / 2) / w - 0.5) ** 2
        + ((box[1] + side / 2) / h - 0.35) ** 2
    ))


def _extent(landmarks):
    lo = landmarks[:468, :2].min(axis=0)
    hi = landmarks[:468, :2].max(axis=0)
    return (
        float((lo[0] + hi[0]) / 2),
        float((lo[1] + hi[1]) / 2),
        float(max(hi - lo)),
    )


class ReacquiringFaceDetector:
    """Run native full-frame VIDEO detection first, then current-frame IMAGE crops."""

    def __init__(self, model_path=MODEL_PATH, *, factory=FaceLandmarkDetector):
        self._factory = factory
        self._model = model_path
        self._full = factory(model_path=model_path, video=True)
        try:
            self._crop = factory(model_path=model_path, video=False)
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
        self.track_break = True
        self.source = "none"
        self.calls = 0

    def reset(self):
        if self._closed:
            raise RuntimeError("detector is closed")
        full = self._factory(model_path=self._model, video=True)
        self._full.close()
        self._full = full
        self._clear()

    def close(self):
        if not self._closed:
            self._closed = True
            self._full.close()
            self._crop.close()
            self._clear()

    def _accepted(self, landmarks, source, previous_source):
        face = _extent(landmarks)
        old = self._last_face
        self.track_break = (
            old is None
            or source == "search"
            or (source == "native") != (previous_source == "native")
            or math.hypot(face[0] - old[0], face[1] - old[1]) > 0.7 * max(old[2], face[2])
            or not 0.55 <= face[2] / max(old[2], 1.0) <= 1.8
        )
        self._last_face = face
        self.source = source
        return landmarks

    def _fit_crop(self, rgb, box):
        x, y, right, bottom = box
        self.calls += 1
        points = self._crop.detect(rgb[y:bottom, x:right])
        if points is None:
            return None
        if points.shape != (478, 3) or not np.isfinite(points).all():
            raise ValueError("invalid landmark mesh returned by crop detector")
        mid_x, mid_y, side = _extent(points)
        margin = 0.35 * side
        if side < 8 or not (
            margin <= mid_x <= right - x - margin
            and margin <= mid_y <= bottom - y - margin
        ):
            return None
        landmarks = points.copy()
        landmarks[:, 0] += x
        landmarks[:, 1] += y
        return landmarks

    def detect(self, rgb, timestamp_ms=None):
        if self._closed:
            raise RuntimeError("detector is closed")
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
            raise ValueError("expected nonempty RGB uint8 [height,width,3]")
        height, width = rgb.shape[:2]
        if self._shape is not None and self._shape != (height, width):
            self.reset()
        if self._shape is None:
            self._shape = (height, width)
            self._search = _tiles(width, height)
        timestamp = (0 if self._ts is None else self._ts + 33) \
            if timestamp_ms is None else int(timestamp_ms)
        if timestamp < 0 or (self._ts is not None and timestamp <= self._ts):
            raise ValueError("timestamp must increase; reset on stream change or seek")
        self._ts = timestamp
        self.pitch_deg = self.yaw_deg = None
        previous_source = self.source
        self.source = "none"
        self.calls = 1
        landmarks = self._full.detect(rgb, timestamp)
        if landmarks is not None:
            self.pitch_deg, self.yaw_deg = self._full.pitch_deg, self._full.yaw_deg
            return self._accepted(landmarks, "native", previous_source)
        hint = None
        if self._last_face is not None:
            cx, cy, side = self._last_face
            hint = _square(cx, cy, max(64, 2.8 * side), width, height)
            landmarks = self._fit_crop(rgb, hint)
            if landmarks is not None:
                return self._accepted(landmarks, "hint", previous_source)
        for _ in range(min(2, len(self._search))):
            box = self._search[self._next_tile]
            self._next_tile = (self._next_tile + 1) % len(self._search)
            if box != hint:
                landmarks = self._fit_crop(rgb, box)
                if landmarks is not None:
                    return self._accepted(landmarks, "search", previous_source)
        self._last_face = None
        self.track_break = True
        return None
