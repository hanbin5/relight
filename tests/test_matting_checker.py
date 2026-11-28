import copy

import numpy as np
import pytest

from idphoto import matting, npimg, synthetic
from idphoto.checker import AutoShutter

from .conftest import FRONT, SIDE, make_ctx, run_pipeline


def clothing_scene(kind):
    rgb, lm, person = synthetic.render(
        synthetic.Scene(width=320, height=280, noise=0, **FRONT))
    f = matting.face_measures(lm)
    yy, xx = np.mgrid[:280, :320] + 0.5
    dx, ec, fw, chin_y = np.abs(xx - f["cx"]), f["ec"], f["fw"], f["chin"][1]
    half = (0.55 + 1.7 * np.sqrt(np.clip((yy - chin_y - 0.25 * ec) / ec, 0, 1))) * fw
    body = (yy > chin_y + 0.25 * ec) & (dx < half)
    if kind == "jacket":
        rgb[body] = [90, 104, 75]
        rgb[body & (dx < 0.78 * fw)] = [24, 28, 30]
        target = body & (dx >= 0.78 * fw)
    elif kind == "white":
        rgb[body] = [230, 235, 240]
        target = body & (matting.person_prior(body.shape, lm)[0] < 0.05)
    else:
        rgb[body] = [25, 26, 27]
        rgb[(xx < 20) & (yy > 80) & (yy < 130)] = [27, 26, 25]
        target = body
    return rgb, lm, person | body, target


@pytest.mark.parametrize("kind", ["jacket", "white", "dark"])
@pytest.mark.parametrize("fast", [True, False], ids=["live", "shutter"])
def test_clothing_recovery_keeps_outer_fabric_without_filling_the_gate(kind, fast):
    from idphoto.relight import Relighter

    rgb, lm, person, target = clothing_scene(kind)
    ctx = make_ctx(rgb, lm)
    Relighter().process(ctx, strength=0.0)
    matting.Matter().process(ctx, fast=fast)
    assert ctx.stats["matting"]["clothing_applied"]
    assert (ctx.alpha[target] > 0.5).mean() > 0.98
    assert ((ctx.alpha > 0.5) & ~person).mean() < 0.005
    assert np.all(ctx.alpha[ctx.face_mask] == 1.0)


@pytest.mark.parametrize("fast", [True, False], ids=["live", "shutter"])
def test_disabled_clothing_matches_fixed_main_baseline(fast):
    from tools.eval_clothing import BASELINE, baseline_module, process

    rgb, lm, _, _ = clothing_scene("jacket")
    baseline, _ = process(baseline_module(BASELINE), rgb, lm, fast)
    ctx = make_ctx(rgb, lm)
    from idphoto.relight import Relighter

    Relighter().process(ctx, strength=0.0)
    matting.Matter().process(ctx, np.ones(3, np.float32), update=False,
                             fast=fast, clothing_enabled=False)
    np.testing.assert_array_equal(ctx.alpha, baseline.alpha)
    np.testing.assert_array_equal(ctx.output_frame, baseline.output_frame)
    assert not ctx.stats["matting"]["clothing_applied"]


def test_clothing_sampling_guard_excludes_out_of_gate_white_sleeves():
    rgb, lm, _, target = clothing_scene("white")
    gate, _, _, guard, _ = matting.clothing_regions(target.shape, lm)
    old_bg = matting.person_prior(target.shape, lm)[0] <= 0
    assert (old_bg & target).any()
    assert not (old_bg & ~guard & target).any()
    assert (gate[target] > 0.05).all()


def test_clothing_models_skip_cropped_chest_and_reset_warm_state():
    from idphoto.relight import Relighter

    rgb, lm, _, _ = clothing_scene("jacket")
    matter = matting.Matter()
    for frame in (rgb, rgb, rgb[:180]):
        ctx = make_ctx(frame, lm)
        Relighter().process(ctx, strength=0.0)
        matter.process(ctx)
    assert not ctx.stats["matting"]["clothing_applied"]
    assert ctx.stats["matting"]["clothing_samples"] < 50
    assert matter.clothing_bg is matter.clothing_fg is None
    matter.reset()
    assert matter.bg is matter.fg is None


@pytest.mark.parametrize("fast", [True, False], ids=["live", "shutter"])
def test_cropped_chest_recovers_jacket_without_learning_wall_samples(fast):
    from idphoto.relight import Relighter

    rgb, lm, person, target = clothing_scene("jacket")
    f = matting.face_measures(lm)
    h = int(f["chin"][1] + 0.55 * f["ec"])
    rgb, person, target = rgb[:h].copy(), person[:h].copy(), target[:h].copy()
    yy, xx = np.mgrid[:h, :rgb.shape[1]] + 0.5
    wall = ((yy > f["chin"][1] + 0.42 * f["ec"])
            & (xx > f["cx"] + 0.5 * f["fw"]) & (xx < f["cx"] + 0.85 * f["fw"]))
    rgb[wall] = rgb[0, 0]
    person[wall] = target[wall] = False

    ctx = make_ctx(rgb, lm)
    Relighter().process(ctx, strength=0.0)
    matting.Matter().process(ctx, fast=fast, update=False)
    stats = ctx.stats["matting"]
    assert stats["clothing_cropped"] and stats["clothing_applied"]
    assert 50 <= stats["clothing_used_samples"] < stats["clothing_samples"]
    assert (ctx.alpha[target] > 0.5).mean() > 0.9
    assert ((ctx.alpha > 0.5) & ~person).mean() < 0.005


@pytest.mark.parametrize("fast", [True, False], ids=["live", "shutter"])
def test_cropped_chest_skips_when_all_side_samples_match_background(fast):
    from idphoto.relight import Relighter

    rgb, lm, _, _ = clothing_scene("jacket")
    matter = matting.Matter()
    initial = make_ctx(rgb, lm)
    Relighter().process(initial, strength=0.0)
    matter.process(initial, fast=fast)
    assert matter.clothing_bg is not None and matter.clothing_fg is not None

    f = matting.face_measures(lm)
    h = int(f["chin"][1] + 0.55 * f["ec"])
    rgb = rgb[:h].copy()
    rgb[int(f["chin"][1] + 0.1 * f["ec"]):] = rgb[0, 0]

    ctx = make_ctx(rgb, lm)
    Relighter().process(ctx, strength=0.0)
    matter.process(ctx, fast=fast)
    stats = ctx.stats["matting"]
    assert stats["clothing_cropped"] and not stats["clothing_applied"]
    assert stats["clothing_used_samples"] < 50
    assert matter.clothing_bg is matter.clothing_fg is None


@pytest.mark.parametrize("fast", [True, False], ids=["live", "shutter"])
def test_clothing_sample_views_show_the_guard_and_candidates(fast):
    from idphoto.relight import Relighter
    from tools.eval_clothing import sample_views

    rgb, lm, _, _ = clothing_scene("white")
    rgb = rgb[:-1]
    ctx = make_ctx(rgb, lm)
    Relighter().process(ctx, strength=0.0)
    matting.Matter().process(ctx, fast=fast, update=False)
    original, old_view, guarded_view, clothing_view = sample_views(rgb, lm, ctx, fast)
    s = 2 if fast else 1
    shape = (rgb.shape[0] // s, rgb.shape[1] // s)
    lm_s = lm / np.float32(s)
    old_bg = matting.person_prior(shape, lm_s)[0] <= 0
    _, _, clothing_fg, guard, _ = matting.clothing_regions(shape, lm_s)
    excluded = npimg.upsample_nearest(old_bg & guard, s, rgb.shape[:2])
    candidates = npimg.upsample_nearest(clothing_fg, s, rgb.shape[:2])
    assert excluded.any() and candidates.any()
    assert np.array_equal(np.any(old_view != guarded_view, axis=2), excluded)
    assert np.array_equal(np.any(original != clothing_view, axis=2), candidates)

    ctx.stats["matting"]["clothing_background_samples"] += 1
    with pytest.raises(RuntimeError, match="Sample regions do not match"):
        sample_views(rgb, lm, ctx, fast)


def test_clothing_sample_views_show_frame_border_fallback(monkeypatch):
    from idphoto.context import FrameContext
    from tools.eval_clothing import sample_views

    shape = (20, 30)
    rgb = np.full((*shape, 3), 128, np.uint8)
    lm = np.zeros((478, 3), np.float32)
    guard = np.zeros(shape, bool)
    guard[:2, :4] = True
    clothing_fg = np.zeros(shape, bool)
    clothing_fg[5:15, 10:20] = True
    border = np.zeros(shape, bool)
    border[[0, -1], :] = border[:, [0, -1]] = True
    monkeypatch.setattr(matting, "person_prior", lambda *_: (np.ones(shape), None))
    monkeypatch.setattr(matting, "foreground_seeds", lambda *_: np.ones(shape, bool))
    monkeypatch.setattr(matting, "clothing_regions", lambda *_: (None, None, clothing_fg, guard, False))
    ctx = FrameContext(raw_frame=rgb, landmarks=lm)
    ctx.stats["matting"] = {"clothing_samples": int(clothing_fg.sum()),
                            "clothing_background_samples": int((border & ~guard).sum())}
    original, old_view, guarded_view, _ = sample_views(rgb, lm, ctx, fast=False)
    assert np.array_equal(np.any(old_view != original, axis=2), border)
    assert np.array_equal(np.any(old_view != guarded_view, axis=2), border & guard)


@pytest.mark.parametrize("columns", [3, 5, 6, 7, 8])
def test_clothing_sheet_shows_representative_jacket_and_other_photos_together(
        tmp_path, monkeypatch, columns):
    from matplotlib.figure import Figure
    from tools.eval_clothing import write_sheets

    names = ("shot2", "shot3", "shot4", "shot5", "test1", "test2",
             "test3", "test4", "test5", "test6", "test8", "test9")
    panels = [
        (name, *(np.full((2, 2), i * 10 + c, np.uint8) for c in range(columns)))
        for i, name in enumerate(names)
    ]
    saved = {}

    def capture(fig, path, **_kwargs):
        axes = fig.axes
        saved[path.name] = [
            (axes[r].get_ylabel(),
             [int(ax.images[0].get_array()[0, 0]) for ax in axes[r:r + columns]])
            for r in range(0, len(axes), columns)
        ]

    monkeypatch.setattr(Figure, "savefig", capture)
    write_sheets(panels, [f"Column {c}" for c in range(columns)], "comparison", tmp_path)
    expected_names = ("shot2", "shot5", "test1", "test2", "test3", "test4",
                      "test5", "test6", "test8", "test9")
    assert saved == {
        "comparison.jpg": [
            (name, [names.index(name) * 10 + c for c in range(columns)])
            for name in expected_names
        ]
    }


def test_fba_trimap_keeps_a_wide_unknown_band_around_the_whole_person():
    from types import SimpleNamespace

    from tools.eval_clothing import fba_trimap

    rgb, lm, _, target = clothing_scene("white")
    alpha = np.zeros(rgb.shape[:2], np.float32)
    alpha[60:250, 100:220] = 1
    face = np.zeros_like(alpha, dtype=bool)
    face[58, 145] = True
    before = SimpleNamespace(alpha=alpha, face_mask=face)
    trimap = fba_trimap(rgb, lm, before)

    assert trimap.shape == rgb.shape[:2]
    assert trimap[200, 150] == 255
    assert trimap[58, 145] == 255
    assert trimap[200, 95] == 128
    assert trimap[0, 0] == 0
    fw = matting.face_measures(lm)["fw"]
    outside_band = target & ~npimg.dilate(alpha > 0.5, max(1, round(0.5 * fw)))
    assert outside_band.any()
    assert (trimap[outside_band] == 128).all()
    with pytest.raises(ValueError, match="both known"):
        fba_trimap(rgb, lm, SimpleNamespace(alpha=np.ones_like(alpha), face_mask=face))
    alpha[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fba_trimap(rgb, lm, before)


def test_guarded_fba_trimap_requires_agreement_and_preserves_known_regions():
    from tools.eval_clothing import guarded_fba_trimap

    rgb = np.zeros((2, 4, 3), np.uint8)
    main = np.array([[0, 128, 128, 128], [255, 0, 128, 255]], np.uint8)
    clothing = np.array([[255, 255, 255, 128], [255, 0, 0, 255]], np.uint8)
    alpha = np.array([[1, 0.76, 0.75, 0.99], [1, 0, 0.99, 1]], np.float32)
    actual = guarded_fba_trimap(rgb, main, clothing, alpha)
    np.testing.assert_array_equal(actual, [[0, 255, 128, 128], [255, 0, 128, 255]])
    assert main[0, 1] == 128
    alpha[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        guarded_fba_trimap(rgb, main, clothing, alpha)


@pytest.mark.parametrize("changed", [False, True], ids=["unchanged-mask", "added-clothing"])
@pytest.mark.parametrize("full_grid", [False, True], ids=["live", "full-grid"])
def test_fba_comparison_uses_each_result_as_its_own_trimap_source(
        tmp_path, monkeypatch, changed, full_grid):
    import csv
    import sys
    from types import SimpleNamespace

    from tools import eval_clothing, fba_reference

    rgb, lm, _, _ = clothing_scene("jacket")
    alpha = np.zeros(rgb.shape[:2], np.float32)
    alpha[60:250, 100:170] = 1
    face = np.zeros_like(alpha, dtype=bool)
    face[65:100, 110:155] = True
    improved_alpha = alpha.copy()
    if changed:
        improved_alpha[150:250, 170:240] = 1
    before = SimpleNamespace(
        alpha=alpha, face_mask=face, output_frame=rgb / 255,
        stats={"matting": {"threshold": 0.5}})
    after = SimpleNamespace(
        alpha=improved_alpha, face_mask=face, output_frame=rgb / 255,
        stats={"matting": {"threshold": 0.5, "clothing_applied": changed,
                           "clothing_samples": 100, "clothing_background_samples": 100}})
    baseline = object()
    calls = []

    class FakeFBA:
        def process(self, image, trimap, tile_size, overlap):
            prediction = np.zeros((*trimap.shape, 7), np.float32)
            prediction[..., 0] = np.where(trimap == 255, 1, np.where(trimap == 0, 0, 0.25))
            prediction[..., 1:4] = image / 255
            calls.append((image.copy(), trimap.copy(), prediction.copy(), tile_size, overlap))
            return prediction, 1

    sheets = {}
    monkeypatch.setattr(eval_clothing, "NAMES", ["case"])
    monkeypatch.setattr(eval_clothing, "baseline_module", lambda _: baseline)
    monkeypatch.setattr(eval_clothing.cv2, "imread", lambda _: rgb[..., ::-1].copy())
    monkeypatch.setattr(eval_clothing, "to_live", lambda image: (image, 1))
    monkeypatch.setattr(eval_clothing, "process",
                        lambda module, *_args: (before if module is baseline else after, 1.0))
    monkeypatch.setattr(eval_clothing, "write_sheets",
                        lambda panels, titles, prefix, _out: sheets.update({prefix: (panels, titles)}))
    monkeypatch.setattr(fba_reference, "FBAReference", FakeFBA)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda _: None))
    cache, out = tmp_path / "cache", tmp_path / "out"
    cache.mkdir()
    np.save(cache / "case-live.npy", lm)
    args = ["eval_clothing.py", "--fba", "--cache", str(cache), "--out", str(out)]
    if full_grid:
        args.append("--full-grid")
    monkeypatch.setattr(sys, "argv", args)

    eval_clothing.main()

    assert len(calls) == 3
    for call, source in zip(calls, (before, after)):
        np.testing.assert_array_equal(call[0], rgb)
        np.testing.assert_array_equal(call[1], eval_clothing.fba_trimap(rgb, lm, source))
        assert call[3:] == (1024, 96)
    assert np.any(calls[0][1] != calls[1][1]) == changed
    np.testing.assert_array_equal(
        calls[2][1], eval_clothing.guarded_fba_trimap(rgb, calls[0][1], calls[1][1], calls[0][2][..., 0]))
    base_alpha, combined_alpha = calls[0][2][..., 0], calls[1][2][..., 0]
    with (out / "measurements.csv").open(newline="") as stream:
        row, = csv.DictReader(stream)
    assert int(row["combined_trimap_foreground"]) == int((calls[1][1] == 255).sum())
    assert int(row["combined_vs_fba_added_pixels"]) == int(((combined_alpha > 0.5) & (base_alpha <= 0.5)).sum())
    assert int(row["combined_vs_fba_removed_pixels"]) == int(((base_alpha > 0.5) & (combined_alpha <= 0.5)).sum())
    assert float(row["combined_vs_fba_mean_abs_alpha_difference"]) == pytest.approx(
        float(np.abs(combined_alpha - base_alpha).mean()))
    assert (int(row["combined_vs_fba_added_pixels"]) > 0) == changed
    guarded_alpha = calls[2][2][..., 0]
    assert int(row["guarded_promoted_foreground"]) == int(((calls[2][1] == 255) & (calls[0][1] == 128)).sum())
    assert int(row["guarded_vs_fba_added_pixels"]) == int(((guarded_alpha > .5) & (base_alpha <= .5)).sum())
    comparison, titles = sheets["comparison"]
    assert titles[-2:] == ["Clothing improvement + FBA", "Guarded FBA"]
    np.testing.assert_array_equal(comparison[0][-2], eval_clothing.fba_composite(calls[1][2]))
    np.testing.assert_array_equal(comparison[0][-1], eval_clothing.fba_composite(calls[2][2]))
    trimap_panels, _ = sheets["trimap"]
    np.testing.assert_array_equal(trimap_panels[0][-4], calls[1][1])
    np.testing.assert_array_equal(trimap_panels[0][-3], combined_alpha)
    np.testing.assert_array_equal(trimap_panels[0][-2], calls[2][1])
    np.testing.assert_array_equal(trimap_panels[0][-1], guarded_alpha)


def test_fba_tiles_keep_resolution_and_respect_known_trimap_regions():
    from tools.fba_reference import native_tiles

    rgb = np.zeros((48, 64, 3), np.uint8)
    rgb[..., 0] = np.arange(64)
    trimap = np.full(rgb.shape[:2], 128, np.uint8)
    trimap[:, :24], trimap[:, 40:] = 0, 255
    calls = []

    def predict(image, labels):
        calls.append(image.shape)
        output = np.empty((*labels.shape, 7), np.float32)
        output[..., 0] = image[..., 0] / 255
        output[..., 1:4] = image / 255
        output[..., 4:7] = 1 - image / 255
        return output

    output, count = native_tiles(rgb, trimap, predict, tile_size=32, overlap=8)
    unknown = trimap == 128
    np.testing.assert_allclose(output[..., 0][unknown], rgb[..., 0][unknown] / 255, atol=1e-6)
    np.testing.assert_array_equal(output[..., 0][trimap == 0], 0)
    np.testing.assert_array_equal(output[..., 0][trimap == 255], 1)
    np.testing.assert_allclose(output[..., 1:4][trimap == 255], rgb[trimap == 255] / 255, atol=1e-7)
    assert count == len(calls) > 1
    assert all(max(shape[:2]) <= 32 for shape in calls)


def test_fba_missing_assets_fail_explicitly(tmp_path):
    from tools.fba_reference import verify_assets

    with pytest.raises(FileNotFoundError, match="accept-noncommercial"):
        verify_assets(tmp_path)


def test_fba_download_requires_noncommercial_acknowledgment(monkeypatch):
    import sys

    from tools import fba_reference

    fetched = []
    monkeypatch.setattr(fba_reference, "fetch_assets", lambda: fetched.append(True))
    monkeypatch.setattr(sys, "argv", ["fba_reference.py"])
    with pytest.raises(SystemExit):
        fba_reference.main()
    assert not fetched
    monkeypatch.setattr(sys, "argv", ["fba_reference.py", "--accept-noncommercial"])
    assert fba_reference.main() == 0
    assert fetched == [True]


def test_fba_official_weights_run_on_cpu_at_input_resolution():
    from tools.fba_reference import ASSETS, FBAReference

    if not (ASSETS / "FBA.pth").exists():
        pytest.skip("download FBA research weights")
    torch = pytest.importorskip("torch")
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        rgb = np.zeros((47, 61, 3), np.uint8)
        rgb[..., :] = np.linspace(20, 230, 61).astype(np.uint8)[None, :, None]
        trimap = np.full(rgb.shape[:2], 128, np.uint8)
        trimap[:, :24], trimap[:, 37:] = 0, 255
        output, count = FBAReference().process(rgb, trimap, tile_size=64, overlap=8)
        assert output.shape == (*trimap.shape, 7) and count == 1
        assert np.isfinite(output).all()
        np.testing.assert_array_equal(output[..., 0][trimap == 0], 0)
        np.testing.assert_array_equal(output[..., 0][trimap == 255], 1)
        assert ((output[..., 0][trimap == 128] > 0) & (output[..., 0][trimap == 128] < 1)).any()
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize("scene", [
    synthetic.Scene(**FRONT),
    synthetic.Scene(**SIDE),
    synthetic.Scene(wall="#9fb6cf", top="#222222", **FRONT),
    synthetic.Scene(wall="#e8dcc8", top="#9c2f2f", **FRONT),
], ids=["light-wall", "side-light", "blue-wall", "beige-wall"])
def test_mask_matches_the_true_person(scene):
    ctx, person, _, _ = run_pipeline(scene, frames=3)
    a = ctx.alpha > 0.5
    assert (a & person).sum() / (a | person).sum() >= 0.95  # the plan's IoU target
    assert ctx.rule_verdicts[7].passed


def test_background_is_replaced_and_face_is_kept():
    ctx, person, _, _ = run_pipeline(synthetic.Scene(**FRONT), frames=2)
    out = ctx.output_frame
    assert np.allclose(out[5, 5], 1.0, atol=1e-3)  # wall corner -> white
    assert np.allclose(out[ctx.face_mask], ctx.relit_frame[ctx.face_mask])


def test_original_wall_option_keeps_the_relit_frame():
    from idphoto.pipeline import Settings

    ctx, _, _, _ = run_pipeline(synthetic.Scene(**FRONT), frames=2, settings=Settings(background="wall"))
    assert ctx.output_frame is ctx.relit_frame


@pytest.mark.parametrize("scene,rule,message", [
    (synthetic.Scene(light=(-0.8, 0.2, 0.55), ambient=0.12), 1, "turn_to_light"),
    (synthetic.Scene(glare=True, **FRONT), 2, "glare_glasses"),
    (synthetic.Scene(blur=3, **FRONT), 3, "hold_still"),
    (synthetic.Scene(roll=6.0, **FRONT), 4, "straighten_head"),
    (synthetic.Scene(yaw=12.0, **FRONT), 4, "face_camera"),
    (synthetic.Scene(pitch=15.0, **FRONT), 4, "face_camera"),
    (synthetic.Scene(eyes_open=False, **FRONT), 5, "open_eyes"),
    (synthetic.Scene(mouth_open=0.12, **FRONT), 6, "close_mouth"),
    (synthetic.Scene(top="#d8d6d2", **SIDE), 7, "top_vs_wall"),
], ids=["side-shadow", "glare", "blur", "tilt", "turn", "pitch", "closed-eyes", "open-mouth", "top-like-wall"])
def test_each_staged_fault_fails_its_rule(scene, rule, message):
    ctx, _, _, _ = run_pipeline(scene)
    v = ctx.rule_verdicts[rule]
    assert not v.passed and v.message == message


def test_good_frame_passes_every_rule_and_fires_the_shutter():
    ctx, _, pipeline, fired = run_pipeline(synthetic.Scene(**SIDE), frames=6)
    assert all(v.passed for v in ctx.rule_verdicts.values()), {n: v.value for n, v in ctx.rule_verdicts.items()}
    assert fired and not pipeline.shutter.armed


def test_auto_shutter_waits_while_the_crop_runs_past_the_frame():
    """Below the frame the photo would be filled by stretching the clothes' last row."""
    from idphoto.pipeline import Settings

    settings = Settings()
    settings.head_share[settings.format] = 0.4  # a small head: the crop runs past the frame
    ctx, _, pipeline, fired = run_pipeline(synthetic.Scene(**SIDE), frames=8, settings=settings)
    assert all(v.passed for v in ctx.rule_verdicts.values())
    assert not ctx.geometry["crop_fits"]
    assert not fired and pipeline.shutter.armed


def test_dark_nostrils_are_not_a_shadow_but_a_dark_cheek_is():
    """The nostrils are dark under any light; they took 3-4% of rule 1's 5% shadow share."""
    from idphoto import config, mesh
    from idphoto.checker import Checker

    ctx, _, _, _ = run_pipeline(synthetic.Scene(**SIDE), frames=6)
    S = config.FIT_SIZE
    box = ctx.face_box
    g = mesh.to_grid(ctx.landmarks, box, S)
    regions = mesh.region_masks(g, (S, S))

    def darken(region):
        """A copy of ctx whose relit frame is near black where region lies on the grid."""
        h, w = ctx.size[1], ctx.size[0]
        ys, xs = np.mgrid[0:h, 0:w]
        gx = ((xs + 0.5 - box[0]) * S / (box[2] - box[0])).astype(int)
        gy = ((ys + 0.5 - box[1]) * S / (box[3] - box[1])).astype(int)
        inside = (gx >= 0) & (gx < S) & (gy >= 0) & (gy < S)
        hit = np.zeros((h, w), bool)
        hit[inside] = region[gy[inside], gx[inside]]
        dark = ctx.relit_frame.copy()
        dark[hit] *= 0.05
        out = copy.copy(ctx)
        out.relit_frame = dark
        return Checker().even_light(out)

    base = Checker().even_light(ctx).detail["shadow_share"]
    nostrils = darken(npimg.erode(regions["nose_base"], 2))
    assert nostrils.detail["shadow_share"] == pytest.approx(base, abs=0.005)
    cheek = darken(npimg.dilate(regions["cheek_l"], 4))
    assert cheek.detail["shadow_share"] >= 0.05 and not cheek.passed


def test_sharpness_bound_is_calibrated_per_camera():
    from idphoto import config
    from idphoto.checker import Checker

    c = Checker()
    assert c.sharpness_bound() == config.SHARPNESS_MIN  # until the camera has been seen
    c.sharpness_seen.extend([50.0] * 40)  # a soft webcam
    assert c.sharpness_bound() == pytest.approx(0.7 * 50.0)
    c.sharpness_seen.extend([400.0] * 90)  # a sharp camera is never held above B.4's 120
    assert c.sharpness_bound() == config.SHARPNESS_MIN
    c.sharpness_seen.extend([5.0] * 90)  # nor a hopeless one below the floor
    assert c.sharpness_bound() == config.SHARPNESS_FLOOR


def test_auto_shutter_needs_five_passing_frames_in_a_row():
    s = AutoShutter(frames=5)
    assert not any(s.update(ok) for ok in [True, True, True, True, False, True, True, True, True])
    assert s.update(True)  # the 5th in a row
    assert not s.update(True)  # fires once until rearmed
    s.rearm()
    assert not s.update(True)


def test_auto_shutter_off_counts_but_never_fires():
    s = AutoShutter(frames=2)
    assert not any(s.update(True, enabled=False) for _ in range(5))
    assert s.count == 5 and s.armed


def test_no_face_fails_every_rule():
    from idphoto.pipeline import Pipeline, Settings

    blank = np.full((360, 640, 3), 200, np.uint8)
    ctx, fired = Pipeline().process(blank, Settings(), landmarks=None)
    assert not fired
    assert all(not v.passed and v.message == "no_face" for v in ctx.rule_verdicts.values())
    assert ctx.output_frame.shape == (360, 640, 3)
