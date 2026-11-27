import numpy as np

from idphoto.center_face import CenterFaceDetector
from idphoto.landmarks import FaceCandidate
from idphoto.subject_matte import select_subject_alpha


def face(cx, cy, side=20, pitch=0, yaw=0):
    p = np.zeros((478, 3), np.float32)
    p[:468, 0] = np.linspace(cx - side / 2, cx + side / 2, 468)
    p[:468, 1] = np.linspace(cy - side / 2, cy + side / 2, 468)
    return FaceCandidate(p, pitch, yaw)


class Detector:
    def __init__(self, results):
        self.results = list(results)

    def detect_all(self, *_args):
        return self.results.pop(0)

    def close(self):
        pass

    def reset(self):
        pass


def test_crop_candidates_are_deduplicated_and_nearest_pose_is_used():
    full = [[face(60, 100, pitch=1, yaw=2)]]
    crop = [
        [face(60, 100, pitch=9, yaw=9)],
        [face(10, 100, pitch=3, yaw=4)],
        [face(122.5, 100, pitch=8, yaw=8)],
    ]

    def factory(*, model_path, video, num_faces):
        assert model_path == "local" and num_faces == 5
        return Detector(full if video else crop)

    detector = CenterFaceDetector("local", factory=factory)
    try:
        selected = detector.detect(np.zeros((200, 300, 3), np.uint8), 1)
        assert detector.calls == 4
        assert len(detector.candidates) == 2
        assert detector.selected_index == 1
        assert detector.track_break
        assert detector.pitch_deg == 3 and detector.yaw_deg == 4
        assert np.mean(selected[:468, 0]) == 160
    finally:
        detector.close()


def test_face_switch_and_loss_break_the_track_even_if_previous_face_still_exists():
    full = [
        [face(80, 100), face(170, 100)],
        [face(145, 100), face(170, 100)],
        [],
    ]
    crop = [[] for _ in range(12)]

    def factory(*, model_path, video, num_faces):
        return Detector(full if video else crop)

    detector = CenterFaceDetector("local", factory=factory)
    try:
        frame = np.zeros((200, 300, 3), np.uint8)
        detector.detect(frame, 1)
        assert np.mean(detector.candidates[detector.selected_index].landmarks[:468, 0]) == 170
        detector.detect(frame, 34)
        assert np.mean(detector.candidates[detector.selected_index].landmarks[:468, 0]) == 145
        assert detector.track_break
        assert detector.detect(frame, 67) is None
        assert detector.track_break
    finally:
        detector.close()


def test_selection_tie_prefers_left_face_independent_of_detection_order():
    def factory(*, model_path, video, num_faces):
        return Detector([[face(180, 100), face(120, 100)]] if video else [[], [], []])

    detector = CenterFaceDetector("local", factory=factory)
    try:
        selected = detector.detect(np.zeros((200, 300, 3), np.uint8), 1)
        assert np.mean(selected[:468, 0]) == 120
    finally:
        detector.close()


def test_centre_crop_can_supply_a_missing_more_central_face():
    def factory(*, model_path, video, num_faces):
        return Detector([[face(40, 100)]] if video else [
            [], [], [face(113, 100, yaw=7)]])

    detector = CenterFaceDetector("local", factory=factory)
    try:
        chosen = detector.detect(np.zeros((200, 300, 3), np.uint8), 1)
        assert np.mean(chosen[:468, 0]) == 150
        assert len(detector.candidates) == 2
        assert detector.yaw_deg == 7
    finally:
        detector.close()


def test_reacquisition_checks_current_search_tiles_even_if_old_hint_matches():
    from idphoto.reacquiring_landmarks import _tiles

    x, y, _, _ = _tiles(300, 200)[0]
    full = [[face(60, 100)], []]
    crop = [[], [], [], [], [], [], [face(32, 32, yaw=1)],
            [face(150 - x, 100 - y, yaw=7)], []]

    def factory(*, model_path, video, num_faces):
        return Detector(full if video else crop)

    detector = CenterFaceDetector("local", factory=factory)
    try:
        frame = np.zeros((200, 300, 3), np.uint8)
        detector.detect(frame, 1)
        chosen = detector.detect(frame, 34)
        assert np.mean(chosen[:468, 0]) == 150
        assert detector.yaw_deg == 7
        assert detector.track_break
    finally:
        detector.close()


def test_subject_matte_keeps_only_selected_disconnected_person_and_soft_alpha():
    alpha = np.zeros((100, 160), np.float32)
    alpha[20:80, 10:60] = .9
    alpha[20:80, 100:150] = .8
    alpha[40, 9] = .01
    alpha[40, 150] = .01
    faces = [face(35, 40).landmarks, face(125, 40).landmarks]
    selected, status = select_subject_alpha(alpha, faces, 1)
    assert status == "ready"
    assert np.isclose(selected[40, 125], .8)
    assert selected[40, 35] == 0
    assert np.isclose(selected[40, 150], .01)
    assert selected[40, 9] == 0


def test_subject_matte_abstains_for_shared_or_unexplained_foreground():
    alpha = np.zeros((100, 160), np.float32)
    alpha[20:80, 10:150] = .9
    faces = [face(35, 40).landmarks, face(125, 40).landmarks]
    assert select_subject_alpha(alpha, faces, 1) == (None, "ambiguous")
    assert select_subject_alpha(alpha, [], None) == (None, "no_face")
    alpha[20:80, 60:100] = 0
    assert select_subject_alpha(alpha, faces[:1], 0) == (None, "ambiguous")


def test_one_person_preserves_original_alpha_exactly():
    alpha = np.zeros((100, 160), np.float32)
    alpha[20:80, 10:60] = .9
    alpha[5, 5] = .01
    result, status = select_subject_alpha(alpha, (face(35, 40).landmarks,), 0)
    assert status == "ready"
    np.testing.assert_array_equal(result, alpha)


def test_missing_background_samples_abstains_instead_of_sampling_people():
    alpha = np.ones((100, 160), np.float32)
    assert select_subject_alpha(alpha, (face(35, 40).landmarks,), 0) == (None, "ambiguous")
