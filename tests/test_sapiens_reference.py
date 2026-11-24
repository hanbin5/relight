import numpy as np
import pytest

from tools.sapiens_reference import (
    ASSETS, CLOTHING_CLASSES, FILENAME, MEAN, SIZE, STD, SapiensReference, verify_assets,
)


def test_sapiens_missing_and_corrupt_assets_fail_explicitly(tmp_path):
    with pytest.raises(FileNotFoundError, match="sapiens_reference --accept-noncommercial"):
        verify_assets(tmp_path)
    (tmp_path / FILENAME).write_bytes(b"not the expected model")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_assets(tmp_path)


def test_sapiens_preprocessing_and_probability_classes():
    torch = pytest.importorskip("torch")
    rgb = np.full((3, 5, 3), [20, 100, 220], np.uint8)
    logits = torch.zeros((1, 28, 2, 2))
    logits[0, 0] = 1
    logits[0, 1] = 2
    logits[0, 12] = 3
    logits[0, 22] = 4
    seen = []

    def model(tensor):
        seen.append(tensor)
        return logits

    reference = SapiensReference.__new__(SapiensReference)
    reference.device, reference.model = torch.device("cpu"), model
    reference.layout, reference.gamma = "stretch", 1.0
    clothing, background = reference.predict(rgb)
    assert tuple(seen[0].shape) == (1, 3, *SIZE)
    np.testing.assert_allclose(seen[0][0, :, 100, 100].numpy(),
                               (rgb[0, 0] - np.array(MEAN)) / np.array(STD), atol=1e-6)
    probabilities = logits.softmax(dim=1)[0, :, 0, 0]
    np.testing.assert_allclose(clothing, probabilities[list(CLOTHING_CLASSES)].sum().item(), atol=1e-6)
    np.testing.assert_allclose(background, probabilities[0].item(), atol=1e-6)
    assert clothing.shape == background.shape == rgb.shape[:2]


def test_sapiens_letterbox_pads_and_restores_the_original_frame():
    torch = pytest.importorskip("torch")
    rgb = np.full((3, 5, 3), [20, 100, 220], np.uint8)
    logits = torch.zeros((1, 28, 8, 6))
    logits[:, 0] = 5
    logits[:, 22, 2:6] = 8
    seen = []

    def model(tensor):
        seen.append(tensor)
        return logits

    reference = SapiensReference.__new__(SapiensReference)
    reference.device, reference.model = torch.device("cpu"), model
    reference.layout, reference.gamma = "letterbox", 1.0
    clothing, background = reference.predict(rgb)
    np.testing.assert_allclose(seen[0][0, :, 0, 0].numpy(),
                               -np.array(MEAN) / np.array(STD), atol=1e-6)
    np.testing.assert_allclose(seen[0][0, :, SIZE[0] // 2, SIZE[1] // 2].numpy(),
                               (rgb[0, 0] - np.array(MEAN)) / np.array(STD), atol=1e-6)
    assert clothing.shape == background.shape == rgb.shape[:2]
    assert clothing[1, 2] > 0.9 and background[1, 2] < 0.1
    assert (clothing[:, 2] > 0.5).all()


def test_sapiens_gamma_changes_only_the_parser_input():
    torch = pytest.importorskip("torch")
    rgb = np.full((2, 3, 3), [20, 100, 220], np.uint8)
    original = rgb.copy()
    seen = []

    def model(tensor):
        seen.append(tensor)
        return torch.zeros((1, 28, 1, 1))

    reference = SapiensReference.__new__(SapiensReference)
    reference.device, reference.model = torch.device("cpu"), model
    reference.layout, reference.gamma = "stretch", 0.55
    reference.predict(rgb)
    corrected = np.rint(255 * (original[0, 0].astype(np.float32) / 255) ** 0.55)
    np.testing.assert_allclose(seen[0][0, :, 100, 100].numpy(),
                               (corrected - np.array(MEAN)) / np.array(STD), atol=1e-6)
    np.testing.assert_array_equal(rgb, original)


def test_sapiens_rejects_bad_input_and_logits():
    torch = pytest.importorskip("torch")
    reference = SapiensReference.__new__(SapiensReference)
    reference.device = torch.device("cpu")
    reference.layout = "stretch"
    reference.gamma = 1.0
    reference.model = lambda _: torch.full((1, 28, 1, 1), float("nan"))
    with pytest.raises(ValueError, match="nonempty RGB uint8"):
        reference.predict(np.zeros((1, 2, 3), np.float32))
    with pytest.raises(ValueError, match="invalid 28-class logits"):
        reference.predict(np.zeros((1, 2, 3), np.uint8))


def test_sapiens_rejects_invalid_gamma_before_loading_assets(tmp_path):
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="finite and positive"):
        SapiensReference(tmp_path, gamma=float("nan"))


def test_sapiens_reference_runs_on_cpu_at_input_resolution():
    if not (ASSETS / FILENAME).exists():
        pytest.skip("download Sapiens research weights")
    torch = pytest.importorskip("torch")
    torch.set_num_threads(4)
    rgb = np.full((90, 160, 3), [80, 110, 150], np.uint8)
    clothing, background = SapiensReference().predict(rgb)
    assert clothing.shape == background.shape == rgb.shape[:2]
    assert np.isfinite(clothing).all() and np.isfinite(background).all()
    assert ((clothing >= 0) & (background >= 0) & (clothing + background <= 1.00001)).all()
