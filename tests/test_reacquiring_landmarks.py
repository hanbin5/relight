import numpy as np

from idphoto.reacquiring_landmarks import ReacquiringFaceDetector


def _landmarks(x0=40, y0=40):
    points = np.zeros((478, 3), np.float32)
    points[:468, 0] = np.linspace(x0, x0 + 20, 468)
    points[:468, 1] = np.linspace(y0, y0 + 20, 468)
    return points


class _Detector:
    def __init__(self, detections):
        self.detections = list(detections)
        self.pitch_deg = self.yaw_deg = None

    def detect(self, _rgb, _timestamp_ms=None):
        return self.detections.pop(0)

    def close(self):
        pass


def test_native_crop_transition_and_search_mark_track_breaks():
    full_results = [_landmarks(), None, _landmarks(), _landmarks(), None]
    crop_results = [_landmarks(20, 20), None, _landmarks(20, 20)]
    created = []

    def factory(*, model_path, video):
        created.append((model_path, video))
        return _Detector(full_results if video else crop_results)

    detector = ReacquiringFaceDetector(model_path="unused", factory=factory)
    rgb = np.zeros((100, 100, 3), np.uint8)
    try:
        assert detector.detect(rgb, 1) is not None
        assert detector.source == "native" and detector.track_break
        assert detector.detect(rgb, 34) is not None
        assert detector.source == "hint" and detector.track_break
        assert detector.detect(rgb, 67) is not None
        assert detector.source == "native" and detector.track_break
        assert detector.detect(rgb, 100) is not None
        assert detector.source == "native" and not detector.track_break
        assert detector.detect(rgb, 133) is not None
        assert detector.source == "search" and detector.track_break
        detector.reset()
        assert created[-1] == ("unused", True)
        assert all(path == "unused" for path, _ in created)
    finally:
        detector.close()
