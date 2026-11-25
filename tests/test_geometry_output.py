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


def test_crown_is_the_mask_top_when_the_proportion_agrees(monkeypatch):
    rgb, lm, person = synthetic.render(synthetic.Scene(**SIDE))
    alpha = person.astype(np.float32)
    cues = geometry.crown_cues(lm, alpha)
    assert cues["mask"] is not None
    monkeypatch.setattr(config, "CROWN_MASK_AGREE", (0.0, 10.0))
    crown, source = geometry.estimate_crown(lm, alpha)
    assert source == "mask"
    np.testing.assert_allclose(crown, cues["mask"])
    # A mask top far outside the band is not trusted: the proportion rule stands in.
    monkeypatch.setattr(config, "CROWN_MASK_AGREE", (cues["agree"] + 0.5, cues["agree"] + 1.0))
    crown, source = geometry.estimate_crown(lm, alpha)
    assert source == "proportion"
    np.testing.assert_allclose(crown, cues["proportion"])


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


@pytest.mark.parametrize("deg", [-20.0, 0.0, 12.5])
def test_pitch_is_the_rotation_about_the_x_axis(deg):
    from idphoto.landmarks import pitch_from_matrix

    a = np.radians(deg)
    m = np.eye(4)
    m[1:3, 1:3] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    m[:3, 3] = (1.0, -2.0, -40.0)  # the translation does not matter
    assert pitch_from_matrix(m.tolist()) == pytest.approx(deg)


class _FakeDetector:
    """Returns the renderer's landmarks with a fixed head angle, like FaceLandmarkDetector."""

    def __init__(self, lm, pitch_deg):
        self.lm, self.pitch_deg = lm, pitch_deg

    def detect(self, rgb, timestamp_ms=None):
        return self.lm


def test_geometry_carries_the_detector_pitch_to_the_report(tmp_path):
    rgb, lm, _ = synthetic.render(synthetic.Scene(width=1280, height=720, face_height=0.32, center=(0.5, 0.5), **SIDE))
    live, lm_live = to_live(rgb, lm)
    pipeline, settings = Pipeline(_FakeDetector(lm_live, 6.5)), Settings()
    ctx, _ = pipeline.process(live, settings, full_frame=rgb)
    assert ctx.geometry["pitch_deg"] == 6.5
    result = pipeline.render_final(rgb, ctx, settings)
    folder, _ = geometry.save_outputs(result, tmp_path)
    assert json.loads((folder / "report.json").read_text())["geometry"]["pitch_deg"] == 6.5


def test_pitch_is_none_without_a_detector():
    rgb, lm, _ = synthetic.render(synthetic.Scene(**SIDE))
    ctx, _ = Pipeline().process(rgb, Settings(), landmarks=lm)
    assert ctx.geometry["pitch_deg"] is None


@needs_model
def test_mediapipe_pitch_follows_the_head_tilt():
    from idphoto.landmarks import FaceLandmarkDetector

    detector = FaceLandmarkDetector(video=False)
    found = {}
    for deg in (-15.0, 0.0, 15.0):  # + tilts the chin down
        rgb, _, _ = synthetic.render(synthetic.Scene(width=1280, height=720, pitch=deg, **SIDE))
        assert detector.detect(rgb) is not None
        found[deg] = detector.pitch_deg
    assert detector.detect(np.zeros((480, 640, 3), np.uint8)) is None and detector.pitch_deg is None
    detector.close()
    assert found[-15.0] < found[0.0] < found[15.0]
    assert all(abs(found[d] - d) < 5.0 for d in found)
