import numpy as np
import pytest

from idphoto import matting
from tests.test_matting_checker import clothing_scene
from tools.eval_clothing import fba_trimap, process
from tools.eval_fba_color_constraints import background_evidence, candidates


def test_candidate_changes_only_background_supported_labels():
    trimap = np.array([[0, 128, 128, 255, 255]], np.uint8)
    evidence = np.array([[True, True, False, True, False]])
    bg, demote = candidates(trimap, evidence)
    np.testing.assert_array_equal(bg, [[0, 0, 128, 255, 255]])
    np.testing.assert_array_equal(demote, [[0, 0, 128, 128, 255]])
    np.testing.assert_array_equal(trimap, [[0, 128, 128, 255, 255]])
    with pytest.raises(ValueError, match="match"):
        candidates(trimap, evidence[:, :-1])


def test_background_colored_sleeve_can_be_marked_as_background():
    rgb, lm, _, target = clothing_scene("jacket")
    f = matting.face_measures(lm)
    yy, xx = np.mgrid[:rgb.shape[0], :rgb.shape[1]] + .5
    sleeve = (target & (xx > f["cx"] + 1.05 * f["fw"])
              & (yy > f["chin"][1] + .65 * f["ec"]))
    rgb[sleeve] = rgb[0, 0]
    ctx, _ = process(matting, rgb, lm, True)
    trimap = fba_trimap(rgb, lm, ctx)
    evidence, _ = background_evidence(rgb, lm, trimap)
    bg, _ = candidates(trimap, evidence)
    assert sleeve.sum() > 100
    assert ((bg == 0) & sleeve).sum() / sleeve.sum() > .9
