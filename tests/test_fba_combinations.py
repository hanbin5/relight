import numpy as np
import pytest

from tools import eval_fba_combinations as combinations


def test_post_refinement_only_raises_supported_unknown_pixels(monkeypatch):
    rgb = np.zeros((2, 5, 3), np.uint8)
    main = np.array([[0, 255, 128, 128, 128], [128, 128, 128, 128, 0]], np.uint8)
    clothing = np.array([[255, 255, 255, 255, 255], [128, 255, 255, 255, 0]], np.uint8)
    fba = np.array([[0, 1, .8, .75, .8], [.8, .8, .8, .95, 0]], np.float32)
    garment = np.array([[1, 1, .95, .95, .89], [.95, .95, .95, .92, 0]], np.float32)
    gate = np.ones((2, 5), np.float32)
    gate[1, 1] = .94
    monkeypatch.setattr(combinations.matting, "clothing_regions",
                        lambda *_: (gate, None, None, None, None))
    result, eligible = combinations.refine_alpha(rgb, None, main, clothing, fba, garment)
    np.testing.assert_array_equal(eligible, [[False, False, True, False, False],
                                             [False, False, True, True, False]])
    assert result[0, 2] == pytest.approx(.95)
    assert result[1, 3] == pytest.approx(.95)
    assert np.all(result[main == 0] == 0)
    assert np.all(result[main == 255] == fba[main == 255])
    np.testing.assert_allclose(fba, [[0, 1, .8, .75, .8], [.8, .8, .8, .95, 0]])
    garment[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        combinations.refine_alpha(rgb, None, main, clothing, fba, garment)
