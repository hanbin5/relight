import numpy as np
import pytest

from idphoto import color, matting
from tools.eval_clothing import process
from tools.eval_hsv_probe import HSVProbeMatter, hsv_features

from .test_matting_checker import clothing_scene


def test_hsv_features_keep_gray_hue_neutral_and_wrap_red():
    rgb = np.array([[[0.5, 0.5, 0.5], [1, 0.01, 0], [1, 0, 0.01]]], np.float32)
    features = hsv_features(rgb)
    np.testing.assert_array_equal(features[0, 0], [0, 0, 0.5])
    assert np.linalg.norm(features[0, 1] - features[0, 2]) < 0.03


@pytest.mark.parametrize("fast", [True, False], ids=["live", "shutter"])
def test_hsv_probe_preserves_face_and_produces_valid_composite(fast):
    rgb, lm, _, _ = clothing_scene("jacket")
    module = type("ProbeModule", (), {"Matter": HSVProbeMatter})
    result, _ = process(module, rgb, lm, fast)
    assert np.all(result.alpha[result.face_mask] == 1)
    assert np.isfinite(result.alpha).all()
    assert result.alpha.min() >= 0 and result.alpha.max() <= 1
    assert color.encode_u8(result.output_frame).shape == rgb.shape


@pytest.mark.parametrize("fast", [True, False], ids=["live", "shutter"])
def test_hsv_probe_can_remove_labeled_clothing_similar_to_the_wall(fast):
    rgb, lm, person, _ = clothing_scene("white")
    f = matting.face_measures(lm)
    yy, xx = np.mgrid[:rgb.shape[0], :rgb.shape[1]] + 0.5
    sleeve = (person & (yy > f["chin"][1] + 0.75 * f["ec"])
              & (xx > f["cx"] + 1.4 * f["fw"]))
    rgb[sleeve] = np.rint(0.75 * rgb[0, 0] + 0.25 * rgb[sleeve]).astype(np.uint8)
    current, _ = process(matting, rgb, lm, fast)
    module = type("ProbeModule", (), {"Matter": HSVProbeMatter})
    candidate, _ = process(module, rgb, lm, fast)
    current_fg, candidate_fg = current.alpha > 0.5, candidate.alpha > 0.5
    assert current_fg[sleeve].mean() > 0.9
    assert (current_fg & ~candidate_fg)[sleeve].mean() > 0.5
    assert not (current_fg & ~candidate_fg & ~person).any()
