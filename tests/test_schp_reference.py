from types import SimpleNamespace

import numpy as np
import pytest

from idphoto import color, matting, synthetic
from tools.eval_clothing import composite_main_alpha, process
from tools.schp_reference import ASSETS, FILES, SCHPReference, combine_with_main, verify_assets


@pytest.fixture
def parsing_scene():
    rgb, lm, _ = synthetic.render(synthetic.Scene(width=320, height=280))
    shape = rgb.shape[:2]
    alpha = np.zeros(shape, np.float32)
    face = np.zeros(shape, bool)
    face[120:173, 145:175] = True
    alpha[face] = 0.8
    alpha[165:218, 150:170] = 0.65
    alpha[240:255, 15:30] = 0.8
    clothing = np.zeros(shape, np.float32)
    clothing[205:265, 165:225] = 0.95
    clothing[225:245, 20:35] = 0.95
    background = np.ones(shape, np.float32)
    background[205:265, 165:225] = 0
    background[225:245, 20:35] = 0
    background[75:120, 145:175] = 0
    return rgb, lm, SimpleNamespace(alpha=alpha, face_mask=face), clothing, background


def test_schp_fusion_connects_clothing_and_preserves_neck(parsing_scene):
    _, lm, ctx, clothing, background = parsing_scene
    garment, person = combine_with_main(ctx, lm, clothing, background)

    assert garment[220, 190] == 1
    assert person[220, 190] == 1
    assert garment[180, 160] == person[180, 160] == ctx.alpha[180, 160]
    assert garment[230, 25] == person[230, 25] == 0
    assert garment[250, 20] == person[250, 20] == 0
    assert garment[100, 160] == 0
    assert person[100, 160] == 1
    assert (garment[ctx.face_mask] == 1).all()


def test_schp_fusion_rejects_invalid_input(parsing_scene):
    _, lm, ctx, clothing, background = parsing_scene
    with pytest.raises(ValueError, match="must match"):
        combine_with_main(ctx, lm, clothing[:-1], background)
    with pytest.raises(ValueError, match="finite"):
        combine_with_main(ctx, lm, np.full_like(clothing, np.nan), background)
    ctx.face_mask[:] = False
    with pytest.raises(ValueError, match="face mask"):
        combine_with_main(ctx, lm, clothing, background)


def test_schp_fusion_handles_a_face_smaller_than_the_connectivity_grid(parsing_scene):
    _, lm, ctx, clothing, background = parsing_scene
    ctx.face_mask[:] = False
    ctx.face_mask[120, 160] = True
    ctx.alpha[:] = 0
    ctx.alpha[120:210, 160] = 1
    clothing[:] = 0
    clothing[205:220, 160:180] = 0.95
    clothing[230:240, 20:30] = 0.95
    background[:] = 1
    background[clothing > 0] = 0

    garment, person = combine_with_main(ctx, lm, clothing, background)
    assert garment[210, 170] == person[210, 170] == 1
    assert garment[235, 25] == person[235, 25] == 0


@pytest.mark.parametrize("fast", [False, True])
def test_schp_composite_uses_main_wall_estimate(parsing_scene, fast):
    rgb, lm, _, _, _ = parsing_scene
    result, _ = process(matting, rgb, lm, fast)
    np.testing.assert_array_equal(
        composite_main_alpha(matting, result, lm, result.alpha, fast),
        color.encode_u8(result.output_frame))


def test_schp_missing_assets_fail_explicitly(tmp_path):
    with pytest.raises(FileNotFoundError, match="schp_reference --download"):
        verify_assets(tmp_path)


def test_schp_invalid_checksum_fails_explicitly(tmp_path):
    (tmp_path / next(iter(FILES))).write_bytes(b"not the expected model")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_assets(tmp_path)


def test_schp_reference_runs_on_cpu_at_input_resolution(parsing_scene):
    if not (ASSETS / "schp-lip-20.onnx").exists():
        pytest.skip("download SCHP research weights")
    pytest.importorskip("onnxruntime")
    rgb, _, _, _, _ = parsing_scene
    clothing, background = SCHPReference(threads=2).predict(rgb)
    assert clothing.shape == background.shape == rgb.shape[:2]
    assert np.isfinite(clothing).all() and np.isfinite(background).all()
    assert ((clothing >= 0) & (background >= 0) & (clothing + background <= 1.00001)).all()


@pytest.mark.parametrize("full_grid", [False, True])
@pytest.mark.parametrize("with_fba", [False, True])
@pytest.mark.parametrize("with_schp,with_sapiens", [(True, False), (False, True), (True, True)])
def test_parser_evaluation_keeps_main_and_fba_baselines(
        parsing_scene, tmp_path, monkeypatch, full_grid, with_fba, with_schp, with_sapiens):
    import csv
    import sys

    from tools import eval_clothing, fba_reference, sapiens_reference, schp_reference

    rgb, lm, before, clothing, background = parsing_scene
    before.output_frame = rgb / 255
    before.stats = {"matting": {"threshold": 0.5}}
    previous = SimpleNamespace(
        alpha=before.alpha.copy(), output_frame=rgb / 255,
        stats={"matting": {"clothing_samples": 0}})
    after = SimpleNamespace(
        alpha=before.alpha.copy(), face_mask=before.face_mask, output_frame=rgb / 255,
        stats={"matting": {"threshold": 0.5, "clothing_applied": True,
                           "clothing_samples": 100, "clothing_background_samples": 100,
                           "clothing_cropped": True, "clothing_used_samples": 80}})
    after.alpha[205:265, 165:225] = 1
    main_module, previous_module = object(), object()
    calls, sheets, composites = [], {}, []

    class FakeSCHP:
        def predict(self, image):
            np.testing.assert_array_equal(image, rgb)
            return clothing, background

    class FakeSapiens(FakeSCHP):
        def __init__(self, device, layout, gamma):
            assert device == "cpu" and layout == "stretch" and gamma == 1.0

    class FakeFBA:
        def process(self, image, trimap, tile_size, overlap):
            calls.append(trimap.copy())
            output = np.zeros((*trimap.shape, 7), np.float32)
            output[..., 0] = trimap == 255
            return output, 1

    def fake_process(module, _rgb, _lm, fast):
        assert fast is not full_grid
        return (before if module is main_module else previous if module is previous_module else after), 1.0

    def fake_composite(module, source, _lm, alpha, fast):
        assert module is main_module and source is before and fast is not full_grid
        composites.append(alpha.copy())
        return rgb

    monkeypatch.setattr(eval_clothing, "NAMES", ["case"])
    monkeypatch.setattr(eval_clothing, "baseline_module",
                        lambda ref: main_module if ref == eval_clothing.BASELINE else previous_module)
    monkeypatch.setattr(eval_clothing.cv2, "imread", lambda _: rgb[..., ::-1].copy())
    monkeypatch.setattr(eval_clothing, "to_live", lambda image: (image, 1))
    monkeypatch.setattr(eval_clothing, "process", fake_process)
    monkeypatch.setattr(eval_clothing, "composite_main_alpha", fake_composite)
    monkeypatch.setattr(eval_clothing, "write_sheets",
                        lambda panels, titles, prefix, _out: sheets.update({prefix: (panels, titles)}))
    monkeypatch.setattr(schp_reference, "SCHPReference", FakeSCHP)
    monkeypatch.setattr(sapiens_reference, "SapiensReference", FakeSapiens)
    monkeypatch.setattr(fba_reference, "FBAReference", FakeFBA)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda _: None))
    cache, out = tmp_path / "cache", tmp_path / "out"
    cache.mkdir()
    np.save(cache / "case-live.npy", lm)
    args = ["eval_clothing.py", "--cache", str(cache), "--out", str(out)]
    if with_schp:
        args.append("--schp")
    if with_fba:
        args.append("--fba")
    if with_sapiens:
        args.append("--sapiens")
    if full_grid:
        args.append("--full-grid")
    monkeypatch.setattr(sys, "argv", args)

    eval_clothing.main()

    assert len(calls) == (2 if with_fba else 0)
    if with_fba:
        np.testing.assert_array_equal(calls[0], eval_clothing.fba_trimap(rgb, lm, before))
        np.testing.assert_array_equal(calls[1], eval_clothing.fba_trimap(rgb, lm, after))
    expected = combine_with_main(before, lm, clothing, background) * (with_schp + with_sapiens)
    assert len(composites) == len(expected)
    for actual, alpha in zip(composites, expected):
        np.testing.assert_array_equal(actual, alpha)
    panels, titles = sheets["comparison"]
    expected_titles = ["Original", "Main"] + (["FBA"] if with_fba else [])
    expected_titles += ["Previous clothing", "Clothing (NumPy)"]
    expected_titles += (["NumPy + FBA"] if with_fba else [])
    if with_schp:
        expected_titles += ["Main + SCHP\nclothing", "Main + SCHP\nperson"]
    if with_sapiens:
        expected_titles += ["Main + Sapiens\nclothing", "Main + Sapiens\nperson"]
    assert titles == expected_titles
    assert len(panels[0]) == len(titles) + 1
    with (out / "measurements.csv").open(newline="") as stream:
        row, = csv.DictReader(stream)
    assert row["previous_clothing"] == eval_clothing.PREVIOUS_CLOTHING
    assert int(row["crop_vs_previous_added"]) > 0
    assert int(row["clothing_used_samples"]) == 80
    if with_schp and with_sapiens:
        assert int(row["sapiens_clothing_vs_schp_added"]) == 0
        assert int(row["sapiens_person_vs_schp_removed"]) == 0
    elif with_sapiens:
        assert "sapiens_person_vs_schp_removed" not in row
