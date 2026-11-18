import json

import numpy as np
import pytest

from idphoto import synthetic
from tools import fba_reference, matte_hair


@pytest.fixture
def mixed_edge():
    h, w = 48, 64
    alpha = np.broadcast_to(np.clip((np.arange(w) - 24) / 12, 0, 1), (h, w)).copy()
    fg, bg = np.array([0.1, 0.15, 0.2]), np.array([0.85, 0.8, 0.75])
    rgb = np.rint((alpha[..., None] * fg + (1 - alpha[..., None]) * bg) * 255).astype(np.uint8)
    trimap = np.full((h, w), 128, np.uint8)
    trimap[:, :24], trimap[:, 37:] = 0, 255
    return rgb, trimap, alpha


def test_native_crop_keeps_the_original_pixels_and_trimap_resolution():
    rgb, landmarks, _ = synthetic.render(synthetic.Scene(width=320, height=240))
    coarse = np.zeros((60, 80), np.float32)
    coarse[10:55, 25:55] = 1
    crop, alpha, trimap, box = matte_hair.prepare_crop(rgb, landmarks, coarse)
    x0, y0, x1, y1 = box
    np.testing.assert_array_equal(crop, rgb[y0:y1, x0:x1])
    assert crop.shape[:2] == alpha.shape == trimap.shape
    assert (trimap == 128).any() and (trimap == 255).any() and (trimap == 0).any()
    with pytest.raises(ValueError, match="inside"):
        matte_hair.prepare_crop(rgb, landmarks, coarse, box=(-1, 0, 200, 200))


def test_edited_trimap_is_not_rescaled_or_treated_as_seed_map(mixed_edge, tmp_path):
    rgb, trimap, _ = mixed_edge
    path = tmp_path / "trimap.png"
    matte_hair.write_image(path, trimap)
    np.testing.assert_array_equal(matte_hair.load_trimap(path, rgb), trimap)
    matte_hair.write_image(path, trimap[:-1])
    with pytest.raises(ValueError, match="native crop"):
        matte_hair.load_trimap(path, rgb)
    matte_hair.write_image(path, np.full(trimap.shape, 128, np.uint8))
    with pytest.raises(ValueError, match="both known"):
        matte_hair.load_trimap(path, rgb)


def test_overlapping_native_tiles_preserve_fine_pixels_and_known_regions(mixed_edge):
    rgb, trimap, _ = mixed_edge
    rgb = rgb.copy()
    rgb[::2, ::2, 0] = 255
    calls = []

    def predict(image, labels):
        calls.append(image.shape)
        output = np.empty((*labels.shape, 7), np.float32)
        output[..., 0] = image[..., 0] / 255
        output[..., 1:4] = image / 255
        output[..., 4:7] = 1 - image / 255
        return output

    output, count = fba_reference.native_tiles(rgb, trimap, predict, tile_size=32, overlap=8)
    unknown = trimap == 128
    np.testing.assert_allclose(output[..., 0][unknown], rgb[..., 0][unknown] / 255, atol=1e-6)
    np.testing.assert_allclose(output[..., 1:4][unknown], rgb[unknown] / 255, atol=1e-6)
    np.testing.assert_array_equal(output[..., 0][trimap == 0], 0)
    np.testing.assert_array_equal(output[..., 0][trimap == 255], 1)
    np.testing.assert_allclose(output[..., 1:4][trimap == 255], rgb[trimap == 255] / 255, atol=1e-7)
    assert count == len(calls) > 1
    assert all(max(shape[:2]) <= 32 for shape in calls)


@pytest.mark.parametrize("method", ["cf", "knn"])
def test_classical_matting_recovers_a_mixed_color_edge(mixed_edge, method):
    pytest.importorskip("pymatting")
    rgb, trimap, truth = mixed_edge
    matte = matte_hair.classical_matte(rgb, trimap, method)
    unknown = trimap == 128
    assert np.abs(matte.alpha[unknown] - truth[unknown]).mean() < 0.03
    np.testing.assert_array_equal(matte.alpha[trimap == 0], 0)
    np.testing.assert_array_equal(matte.alpha[trimap == 255], 1)
    assert np.isfinite(matte.foreground).all()


def test_saving_mattes_preserves_native_alpha_and_foreground_rgba(mixed_edge, tmp_path):
    import cv2

    rgb, trimap, alpha = mixed_edge
    fg = np.full(rgb.shape, 0.2, np.float32)
    matte = matte_hair.Matte(alpha.astype(np.float32), fg, 0.25)
    report = matte_hair.save_results(tmp_path, rgb, trimap, {"current": matte, "fba": matte})
    encoded = cv2.imread(str(tmp_path / "fba_alpha.png"), cv2.IMREAD_UNCHANGED)
    rgba = cv2.imread(str(tmp_path / "fba_rgba.png"), cv2.IMREAD_UNCHANGED)
    assert encoded.shape == trimap.shape and encoded.dtype == np.uint16
    np.testing.assert_allclose(encoded / 65535, alpha, atol=1 / 65535)
    np.testing.assert_array_equal(rgba[..., :3], 51)
    np.testing.assert_array_equal(rgba[..., 3], np.rint(alpha * 255).astype(np.uint8))
    detail = cv2.imread(str(tmp_path / "detail_1to1.png"))
    assert detail.shape[1] == min(trimap.shape) * 3
    assert report["methods"]["fba"]["ground_truth_metrics"] is None
    json.dumps(report, allow_nan=False)


def test_fba_missing_assets_are_an_explicit_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="accept-noncommercial"):
        fba_reference.verify_assets(tmp_path)


@pytest.mark.skipif(not (fba_reference.ASSETS / "FBA.pth").exists(), reason="download FBA research weights")
def test_official_fba_weights_run_on_cpu_at_native_resolution(mixed_edge):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    rgb, trimap, _ = mixed_edge
    rgb, trimap = rgb[:-1, :-3], trimap[:-1, :-3]
    model = fba_reference.FBAReference()
    output, count = model.process(rgb, trimap, tile_size=64, overlap=8)
    assert output.shape == (*trimap.shape, 7) and count == 1
    assert np.isfinite(output).all()
    np.testing.assert_array_equal(output[..., 0][trimap == 0], 0)
    np.testing.assert_array_equal(output[..., 0][trimap == 255], 1)
    unknown = output[..., 0][trimap == 128]
    assert ((unknown > 0) & (unknown < 1)).any()
