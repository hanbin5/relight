import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from idphoto import presentation_birefnet as birefnet


def test_rembg_288_uint8_only_mask_contract_with_fake_session():
    pytest.importorskip("PIL")
    from PIL import Image

    raw_rgb = np.arange(9 * 13 * 3, dtype=np.uint8).reshape(9, 13, 3)

    class FakeSession:
        def __init__(self):
            self.input = None

        def run(self, _outputs, inputs):
            self.input = next(iter(inputs.values()))
            axis = np.linspace(-4, 4, 1024, dtype=np.float32)
            logits = np.broadcast_to(axis[None, None, None, :], (1, 1, 1024, 1024)).copy()
            return [logits]

    session = FakeSession()
    adapter = object.__new__(birefnet.BirefNetPortrait)
    adapter.session = session
    adapter.input_name = "input"
    adapter.versions = {"onnxruntime": "test", "pillow": "test", "numpy": np.__version__}
    adapter.session_create_ms = 0.0
    adapter.inference_count = 0
    result = adapter.predict_alpha(raw_rgb, capture=True)

    resized = Image.fromarray(raw_rgb).convert("RGB").resize(
        (1024, 1024), Image.Resampling.LANCZOS
    )
    normalized = np.asarray(resized) / max(np.max(np.asarray(resized)), 1e-6)
    expected_tensor = np.zeros((1024, 1024, 3))
    for channel, (mean, std) in enumerate(zip(birefnet.MEAN, birefnet.STD)):
        expected_tensor[:, :, channel] = (normalized[:, :, channel] - mean) / std
    expected_tensor = np.expand_dims(expected_tensor.transpose((2, 0, 1)), 0).astype(np.float32)
    np.testing.assert_array_equal(session.input, expected_tensor)

    logits = session.run(None, {"input": expected_tensor})[0][:, 0, :, :]
    prediction = 1 / (1 + np.exp(-logits))
    prediction = (prediction - np.min(prediction)) / (np.max(prediction) - np.min(prediction))
    expected_mask = Image.fromarray((np.squeeze(prediction) * 255).astype("uint8"), mode="L")
    expected_mask = np.asarray(
        expected_mask.resize((raw_rgb.shape[1], raw_rgb.shape[0]), Image.Resampling.LANCZOS)
    )
    np.testing.assert_array_equal(result.mask_u8, expected_mask)
    np.testing.assert_array_equal(
        result.alpha, expected_mask.astype(np.float32) / np.float32(255)
    )
    assert result.alpha.shape == raw_rgb.shape[:2]
    assert result.mask_u8.dtype == np.uint8
    assert result.stats["provider"] == "CPUExecutionProvider"
    assert result.stats["inference_count"] == 1
    for key in ("session_create_ms", "preprocess_ms", "inference_ms",
                "alpha_postprocess_ms", "total_predict_ms"):
        assert result.stats[key] >= 0
    with pytest.raises(ValueError, match="only for shutter captures"):
        adapter.predict_alpha(raw_rgb, capture=False)


def test_birefnet_requires_model_hash_and_optional_dependencies(monkeypatch, tmp_path):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    model_path = tmp_path / "portrait.onnx"
    model_path.write_bytes(b"synthetic test model")
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    monkeypatch.setattr(birefnet, "MODEL_SHA256", digest)
    monkeypatch.setattr(birefnet.importlib.util, "find_spec", lambda _: object())
    assert birefnet.check_availability(model_path) == model_path

    monkeypatch.setattr(birefnet, "MODEL_SHA256", "incorrect")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        birefnet.check_availability(model_path)

    monkeypatch.setattr(
        birefnet.importlib.util,
        "find_spec",
        lambda name: None if name == "onnxruntime" else object(),
    )
    with pytest.raises(ImportError, match="onnxruntime"):
        birefnet.check_availability(model_path)

    monkeypatch.setattr(birefnet.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with pytest.raises(ValueError, match="OMP_NUM_THREADS"):
        birefnet.check_availability(model_path)


def test_birefnet_capture_uses_full_raw_frame_and_real_save_path(tmp_path):
    from idphoto.geometry import save_outputs
    from idphoto.pipeline import Pipeline, Settings
    from tests.test_matting_checker import clothing_scene

    rgb, landmarks, _, _ = clothing_scene("jacket")
    raw_full = np.repeat(np.repeat(rgb, 2, axis=0), 2, axis=1)
    settings = Settings(
        auto_shutter=False,
        modnet_demo=True,
        capture_matte="birefnet-portrait",
    )
    calls = []

    class FakePortrait:
        def predict_alpha(self, raw_rgb, *, capture=True):
            calls.append((raw_rgb, capture))
            alpha = np.full(raw_rgb.shape[:2], 0.8, np.float32)
            alpha[0, :] = 0
            mask = np.full(raw_rgb.shape[:2], 204, np.uint8)
            return birefnet.BirefNetPrediction(alpha, {"provider": "fake"}, mask)

        @staticmethod
        def apply_prediction(*_args):
            raise AssertionError("capture pipeline must not apply the matte twice")

    pipeline = Pipeline(
        modnet_factory=lambda: (_ for _ in ()).throw(AssertionError("live MODNet used at capture")),
        birefnet_factory=FakePortrait,
    )
    live, _ = pipeline.process(
        rgb, settings, landmarks=landmarks, full_frame=raw_full, defer_modnet=True
    )
    assert calls == []

    result = pipeline.render_final(raw_full, live, settings)
    assert calls == [(raw_full, True)]
    assert result["ctx"].raw_frame is raw_full
    assert result["ctx"].full_frame is raw_full
    assert result["report"]["settings"]["capture_matte"] == "birefnet-portrait"
    assert set(result["report"]["rules"]) == set(range(1, 8))
    capture_timing = result["ctx"].stats["capture_timing_ms"]
    assert capture_timing["render_final_total_ms"] > 0
    assert capture_timing["render_final_started_monotonic_ms"] \
        <= capture_timing["render_final_finished_monotonic_ms"]

    output_dir, files = save_outputs(result, tmp_path, "jpg")
    assert len(files) == 6
    assert (output_dir / "report.json").is_file()
    assert len(list(output_dir.glob("*.jpg"))) == 5
