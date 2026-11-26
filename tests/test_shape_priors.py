import numpy as np
import pytest

from tools.eval_shape_priors import edge_safety, prior_maps, shape_results


def test_soft_shape_and_connectivity_preserve_known_and_only_demote_trimap():
    rgb = np.zeros((2, 5, 3), np.uint8)
    main = np.array([[0, 255, 128, 128, 128], [128, 128, 128, 128, 0]], np.uint8)
    clothing = np.array([[0, 255, 255, 255, 128], [255, 255, 255, 128, 0]], np.uint8)
    alpha = np.array([[0, 1, .9, .8, .7], [.8, .9, .8, .5, 0]], np.float32)
    torso = np.array([[0, 1, 1, 0, 0], [0, 1, 0, .5, 0]], np.float32)
    body = np.ones((2, 5), np.float32)
    connected = np.array([[0, 1, 1, 1, 1], [0, 0, 1, 1, 0]], bool)
    geometry, both, edge, trimap, demote = shape_results(
        rgb, main, clothing, alpha, torso, body, connected)
    assert geometry[0, 3] == pytest.approx(.64)
    assert both[1, 0] == pytest.approx(.52)
    assert both[1, 1] == pytest.approx(.765)
    np.testing.assert_array_equal(edge, both)
    np.testing.assert_array_equal(geometry[main != 128], alpha[main != 128])
    assert np.all(both[main != 128] == alpha[main != 128])
    assert np.all(both >= alpha * .65 - 1e-6)
    np.testing.assert_array_equal(demote, [[False, False, False, True, False],
                                           [True, True, True, False, False]])
    assert np.all(trimap[demote] == 128)
    assert np.all(trimap[~demote] == clothing[~demote])


def test_invalid_shape_values_are_rejected():
    rgb = np.zeros((1, 3, 3), np.uint8)
    trimap = np.array([[0, 128, 255]], np.uint8)
    alpha = np.array([[0, np.nan, 1]], np.float32)
    with pytest.raises(ValueError, match="finite"):
        shape_results(rgb, trimap, trimap, alpha, np.ones((1, 3)),
                      np.ones((1, 3)), np.ones((1, 3), bool))


def test_face_connectivity_distinguishes_detached_foreground(monkeypatch):
    from tools import eval_shape_priors as experiment

    shape = (6, 8)
    alpha = np.zeros(shape, np.float32)
    alpha[1:5, 1:3] = 1
    alpha[4:6, 6:8] = 1
    face = np.zeros(shape, bool)
    face[1:2, 1:3] = True
    monkeypatch.setattr(experiment.matting, "face_measures",
                        lambda _: {"fw": 2, "ec": 2, "chin": [2, 2]})
    monkeypatch.setattr(experiment.matting, "clothing_regions",
                        lambda *_: (None, np.full(shape, .5), None, None, False))
    torso, body, connected = prior_maps(np.zeros((*shape, 3), np.uint8), None, face, alpha)
    assert not torso.any()
    assert body[1, 0] == 0 and body[4, 0] == 1
    assert connected[4, 1] and not connected[4, 6]
    with pytest.raises(ValueError, match="overlapping the face"):
        prior_maps(np.zeros((*shape, 3), np.uint8), None, np.zeros(shape, bool), alpha)


def test_luminance_edge_reduces_attenuation_near_strong_gradient():
    rgb = np.zeros((5, 8, 3), np.uint8)
    rgb[:, 4:] = 255
    safety = edge_safety(rgb)
    assert safety[2, 1] == pytest.approx(1)
    assert safety[2, 3] == pytest.approx(0)
