import numpy as np
import pytest

from tools.eval_fba_leakage import trimap_changes


def test_trimap_change_colors_distinguish_the_three_risk_paths():
    main = np.array([[0, 0, 128, 128, 255]], np.uint8)
    clothing = np.array([[128, 255, 255, 128, 255]], np.uint8)
    image = trimap_changes(main, clothing)
    np.testing.assert_array_equal(image[0], [[70, 150, 255], [255, 0, 255],
                                             [255, 60, 60], [0, 0, 0], [0, 0, 0]])
    with pytest.raises(ValueError, match="match"):
        trimap_changes(main, clothing[:, :3])
