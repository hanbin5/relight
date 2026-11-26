import sys
from pathlib import Path

from idphoto import synthetic

from .conftest import FRONT, needs_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import eval_checker  # noqa: E402


def test_score_counts_caught_faults_and_false_alarms():
    results = [
        ("good", set()),
        ("good", {3}),  # false alarm for rule 3
        ("blur", {3}),  # caught
        ("blur", set()),  # missed
        ("eyes_closed", {5}),  # caught
    ]
    table = eval_checker.score(results)
    assert table[3] == (1, 2, 1, 0.5, 0.5)
    assert table[5] == (1, 1, 0, 1.0, 1.0)
    assert table[1] == (0, 0, 0, None, None)  # no side-shadow photos


def test_find_photos_reads_only_known_folders(tmp_path):
    for name in ("good", "blur", "holiday"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "a.jpg").touch()
    (tmp_path / "good" / "notes.txt").touch()
    found = eval_checker.find_photos(tmp_path)
    assert sorted(name for _, name in found) == ["blur", "good"]


@needs_model
def test_synthetic_photos_end_to_end(tmp_path):
    import cv2

    from idphoto.landmarks import FaceLandmarkDetector

    scenes = {"good": synthetic.Scene(**FRONT), "mouth_open": synthetic.Scene(mouth_open=0.12, **FRONT)}
    detector, timestamp, results = FaceLandmarkDetector(), 0, []
    for name, scene in scenes.items():
        rgb, _, _ = synthetic.render(scene)
        path = tmp_path / f"{name}.png"
        cv2.imwrite(str(path), rgb[:, :, ::-1])
        ctx, timestamp = eval_checker.check_photo(path, detector, timestamp)
        results.append((name, {n for n, v in ctx.rule_verdicts.items() if not v.passed}))
    detector.close()
    assert eval_checker.score(results)[6][4] == 1.0  # the open mouth is caught


def test_camera_bounds_follow_the_app_calibration():
    from idphoto import config

    bounds = eval_checker.camera_bounds({"webcam": [14, 15, 13, 1], "blurry": [5, 6, 4, 1],
                                         "phone": [200, 210, 190, 4]})
    assert abs(bounds["webcam"] - 0.7 * 14.4) < 1e-6  # 70% of the 80th percentile
    assert bounds["blurry"] == config.SHARPNESS_FLOOR  # never below the floor
    assert bounds["phone"] == config.SHARPNESS_MIN  # never above 120
