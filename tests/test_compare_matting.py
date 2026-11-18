import json

import numpy as np
import pytest

from idphoto import synthetic
from tools import compare_matting


@pytest.fixture
def two_regions():
    rgb = np.empty((48, 64, 3), np.uint8)
    rgb[:] = (205, 215, 225)
    truth = np.zeros((48, 64), bool)
    truth[8:48, 18:55] = True
    rgb[truth] = (170, 45, 50)
    markers = np.zeros(truth.shape, np.uint8)
    markers[1:4, 1:5] = compare_matting.BACKGROUND
    markers[20:24, 30:34] = compare_matting.FOREGROUND
    return rgb, markers, truth


@pytest.mark.parametrize("method", ["grabcut", "random_walker"])
def test_reference_methods_keep_foreground_on_the_bottom_border(two_regions, method):
    if method == "random_walker":
        pytest.importorskip("skimage")
    rgb, markers, truth = two_regions
    before = markers.copy()
    result = getattr(compare_matting, method)(rgb, markers)
    assert compare_matting.mask_metrics(result.mask, truth)["iou"] > 0.99
    assert result.mask[-1, 40]
    assert not result.mask[0, 40]
    assert result.mask[markers == compare_matting.FOREGROUND].all()
    assert not result.mask[markers == compare_matting.BACKGROUND].any()
    np.testing.assert_array_equal(markers, before)


@pytest.mark.parametrize("method", ["grabcut", "random_walker"])
def test_reference_methods_reject_missing_background_seeds(two_regions, method):
    if method == "random_walker":
        pytest.importorskip("skimage")
    rgb, markers, _ = two_regions
    markers[markers == compare_matting.BACKGROUND] = 0
    with pytest.raises(ValueError, match="five foreground and five background"):
        getattr(compare_matting, method)(rgb, markers)


def test_grabcut_initializes_a_small_shirt_seed_as_probable_foreground():
    rgb = np.full((64, 64, 3), 220, np.uint8)
    truth = np.zeros((64, 64), bool)
    truth[8:, 20:48] = True
    rgb[8:30, 20:48] = (170, 105, 85)
    rgb[30:, 20:48] = (35, 50, 65)
    markers = np.zeros(truth.shape, np.uint8)
    markers[:3, :8] = compare_matting.BACKGROUND
    markers[10:28, 24:42] = compare_matting.FOREGROUND
    markers[44:46, 32:35] = compare_matting.FOREGROUND
    result = compare_matting.grabcut(rgb, markers)
    assert compare_matting.mask_metrics(result.mask, truth)["iou"] > 0.99


def test_automatic_seeds_do_not_mark_the_bottom_border_as_background():
    rgb, landmarks, _ = synthetic.render(synthetic.Scene(width=160, height=120))
    face = np.zeros(rgb.shape[:2], bool)
    face[35:60, 70:90] = True
    markers = compare_matting.automatic_markers(face, landmarks)
    assert not (markers[-1] == compare_matting.BACKGROUND).any()
    assert (markers == compare_matting.FOREGROUND).any()
    assert (markers == compare_matting.BACKGROUND).any()
    assert (markers == compare_matting.UNKNOWN).mean() > 0.8


def test_seed_image_roundtrip_and_invalid_values(two_regions, tmp_path):
    rgb, markers, _ = two_regions
    path = tmp_path / "seeds.png"
    compare_matting.write_image(path, compare_matting.SEED_VALUES[markers])
    np.testing.assert_array_equal(compare_matting.read_markers(path, rgb.shape[:2]), markers)
    values = compare_matting.SEED_VALUES[markers]
    values[0, 0] = 42
    compare_matting.write_image(path, values)
    with pytest.raises(ValueError, match="only 0, 128, 255"):
        compare_matting.read_markers(path, rgb.shape[:2])


def test_random_walker_probability_is_finite_and_saved_as_16_bit(two_regions, tmp_path):
    pytest.importorskip("skimage")
    import cv2

    rgb, markers, truth = two_regions
    result = compare_matting.random_walker(rgb, markers)
    assert result.probability.shape == truth.shape
    assert np.isfinite(result.probability).all()
    assert 0 <= result.probability.min() <= result.probability.max() <= 1
    results = {name: result for name in compare_matting.METHODS}
    report = compare_matting.save_comparison(tmp_path, rgb, markers, results, truth)
    encoded = cv2.imread(str(tmp_path / "random_walker_probability.png"), cv2.IMREAD_UNCHANGED)
    assert encoded.dtype == np.uint16
    np.testing.assert_allclose(encoded / 65535, result.probability, atol=1 / 65535)
    assert report["methods"]["random_walker"]["iou"] > 0.99
    assert report["methods"]["current"]["precision"] is not None
    json.dumps(report, allow_nan=False)


def test_metrics_do_not_invent_ground_truth(two_regions):
    _, _, truth = two_regions
    assert compare_matting.mask_metrics(truth)["iou"] is None
    with pytest.raises(ValueError, match="Ground truth"):
        compare_matting.mask_metrics(truth, truth[:-1])


def test_synthetic_comparison_cli_saves_all_methods_and_real_metrics(tmp_path):
    pytest.importorskip("skimage")
    assert compare_matting.main(["--synthetic", "--max-side", "160", "--out", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert len(report["cases"]) == 3
    for name, case in report["cases"].items():
        assert case["size_wh"] == [160, 120]
        assert (tmp_path / name / "comparison.jpg").is_file()
        assert (tmp_path / name / "seeds.png").is_file()
        for method in compare_matting.METHODS:
            assert (tmp_path / name / f"{method}_mask.png").is_file()
            assert 0 <= case["methods"][method]["iou"] <= 1
            assert not case["methods"][method]["warnings"]


def test_comparison_does_not_use_the_silhouette_for_reference_methods(monkeypatch):
    pytest.importorskip("skimage")
    from idphoto import matting

    rgb, landmarks, _ = synthetic.render(synthetic.Scene(width=160, height=120))
    calls = []
    original = matting.person_prior

    def track_prior(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(matting, "person_prior", track_prior)
    _, results = compare_matting.compare(rgb, landmarks)
    assert len(calls) == 1  # Only the unchanged production method.
    assert all(result.mask.shape == rgb.shape[:2] for result in results.values())
    assert all(result.mask.dtype == bool for result in results.values())
