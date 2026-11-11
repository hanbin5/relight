import json
import struct
import zlib

import numpy as np
import pytest

from idphoto import config, geometry, synthetic
from idphoto.pipeline import Pipeline, Settings, to_live

from .conftest import SIDE, needs_model


@pytest.mark.parametrize("key", list(geometry.FORMATS))
def test_crop_keeps_the_format_aspect_and_head_share(key):
    fmt = geometry.FORMATS[key]
    lm = synthetic.landmarks_for(synthetic.Scene(**SIDE))
    geo = geometry.measure(lm)
    x0, y0, x1, y1 = geometry.crop_box(geo, fmt)
    assert (x1 - x0) / (y1 - y0) == pytest.approx(fmt.size_px[0] / fmt.size_px[1])
    assert geo["head_length"] / (y1 - y0) == pytest.approx(fmt.head_share)
    assert (geo["crown"][1] - y0) / (y1 - y0) == pytest.approx(fmt.top_margin)


def test_crown_follows_the_proportion_rule():
    lm = synthetic.landmarks_for(synthetic.Scene(**SIDE))
    crown, source = geometry.estimate_crown(lm)
    eye = 0.5 * (lm[468, :2] + lm[473, :2])
    np.testing.assert_allclose(crown, lm[152, :2] + config.CROWN_K * (eye - lm[152, :2]), atol=1e-4)
    assert source == "proportion"



def test_mask_weight_moves_the_crown_between_the_two_cues(monkeypatch):
    rgb, lm, person = synthetic.render(synthetic.Scene(**SIDE))
    alpha = person.astype(np.float32)
    cues = geometry.crown_cues(lm, alpha)
    assert cues["mask"] is not None
    monkeypatch.setattr(config, "CROWN_MASK_AGREE", (0.0, 10.0))
    for w, expected in ((0.0, cues["proportion"]), (1.0, cues["mask"]),
                        (0.25, 0.75 * cues["proportion"] + 0.25 * cues["mask"])):
        monkeypatch.setattr(config, "CROWN_MASK_WEIGHT", w)
        crown, source = geometry.estimate_crown(lm, alpha)
        assert source == "mask+proportion"
        np.testing.assert_allclose(crown, expected, atol=1e-4)



@pytest.mark.parametrize("key", list(geometry.FORMATS))
def test_face_mode_sizes_the_face_from_the_skull(key):
    fmt = geometry.FORMATS[key]
    geo = geometry.measure(synthetic.landmarks_for(synthetic.Scene(**SIDE)))  # no mask: crown = skull
    crop = geometry.face_crop(geo, fmt)
    x0, y0, x1, y1 = crop["box"]
    assert crop["action"] == "fits"
    assert (x1 - x0) / (y1 - y0) == pytest.approx(fmt.size_px[0] / fmt.size_px[1])
    assert crop["face_share"] == pytest.approx(fmt.face_share)
    assert (geo["skull"][1] - y0) / (y1 - y0) == pytest.approx(fmt.skull_top)
    assert geometry.crop_box(geo, fmt, mode="face") == crop["box"]


def test_face_mode_keeps_big_hair_inside_the_photo():
    fmt = geometry.FORMATS["resume"]
    geo = geometry.measure(synthetic.landmarks_for(synthetic.Scene(**SIDE)))
    face = float(np.linalg.norm(geo["chin"] - geo["skull"]))
    for extra, action in ((0.18, "moved"), (0.40, "zoomed")):
        big = {**geo, "crown": geo["skull"] - np.float32([0, extra * face])}
        crop = geometry.face_crop(big, fmt)
        x0, y0, x1, y1 = crop["box"]
        assert crop["action"] == action
        assert crop["hair_top"] == pytest.approx(config.HAIR_MARGIN_MIN)
        assert crop["chin"] <= config.CHIN_MAX + 1e-6


def test_print_sheets_fit_ten_and_eight_photos():
    for key, count in (("resume", 10), ("id", 8)):
        fmt = geometry.FORMATS[key]
        photo = np.zeros((fmt.size_px[1], fmt.size_px[0], 3), np.uint8)  # black photos on white paper
        sheet = geometry.print_sheet(photo, fmt)
        assert sheet.shape == (1200, 1800, 3)
        assert (sheet.sum(axis=2) == 0).sum() == count * photo.shape[0] * photo.shape[1]


def test_saved_files_record_300_dpi(tmp_path):
    img = np.full((20, 30, 3), 128, np.uint8)
    jpg = geometry.save_image(tmp_path / "a.jpg", img).read_bytes()
    assert jpg[6:11] == b"JFIF\x00" and jpg[13] == 1 and struct.unpack(">HH", jpg[14:18]) == (300, 300)
    assert geometry.NOTICE.encode() in jpg
    png = geometry.save_image(tmp_path / "a.png", img).read_bytes()
    at = png.index(b"pHYs")
    ppm_x, ppm_y, unit = struct.unpack(">IIB", png[at + 4 : at + 13])
    assert unit == 1 and round(ppm_x * 0.0254) == 300 == round(ppm_y * 0.0254)
    assert zlib.crc32(png[at : at + 13]) & 0xFFFFFFFF == struct.unpack(">I", png[at + 13 : at + 17])[0]
    import cv2

    assert cv2.imread(str(tmp_path / "a.png")).shape == (20, 30, 3)  # still a valid PNG


def test_shutter_render_makes_every_output(tmp_path):
    rgb, lm, _ = synthetic.render(synthetic.Scene(width=1280, height=720, face_height=0.32, center=(0.5, 0.5), **SIDE))
    live, lm_live = to_live(rgb, lm)
    pipeline, settings = Pipeline(), Settings()
    for i in range(5):
        ctx, _ = pipeline.process(live, settings, landmarks=lm_live, timestamp_ms=33 * i, full_frame=rgb)
    result = pipeline.render_final(rgb, ctx, settings)
    for key, fmt in geometry.FORMATS.items():
        assert result["photos"][key].shape == (fmt.size_px[1], fmt.size_px[0], 3)
    assert set(result["sheets"]) == {"resume", "id"}
    assert not result["kept_wall"]
    folder, files = geometry.save_outputs(result, tmp_path)
    assert len(files) == 6
    report = json.loads((folder / "report.json").read_text())
    assert report["rules"]["1"]["passed"] and report["notice"] == geometry.NOTICE


def test_failed_cutout_keeps_the_original_wall():
    rgb, lm, _ = synthetic.render(synthetic.Scene(top="#d8d6d2", **SIDE))
    pipeline, settings = Pipeline(), Settings()
    for i in range(3):
        ctx, _ = pipeline.process(rgb, settings, landmarks=lm, timestamp_ms=33 * i)
    assert not ctx.rule_verdicts[7].passed
    assert pipeline.render_final(rgb, ctx, settings)["kept_wall"]


@needs_model
def test_mediapipe_finds_the_synthetic_face_where_the_renderer_put_it():
    from idphoto.landmarks import FaceLandmarkDetector

    rgb, lm, _ = synthetic.render(synthetic.Scene(width=1280, height=720, **SIDE))
    found = FaceLandmarkDetector(video=False).detect(rgb)
    assert found is not None and found.shape == (478, 3)
    face_width = lm[454, 0] - lm[234, 0]
    assert np.linalg.norm(found[:468, :2] - lm[:468, :2], axis=1).mean() < 0.04 * face_width
