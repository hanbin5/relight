"""color.py against known values (plan B.2): sRGB transfer function, luminance, gray, saturation."""

import warnings

import numpy as np
import pytest

from idphoto import color

# --- sRGB transfer function (IEC 61966-2-1) -------------------------------------------


@pytest.mark.parametrize("srgb, linear", [
    (0.0, 0.0),
    (1.0, 1.0),
    (0.04045, 0.04045 / 12.92),  # end of the linear segment
    (0.5, 0.21404114),  # the well-known mid-gray
    (0.2, 0.03310476),
    (0.8, 0.60382734),
])
def test_srgb_to_linear_known_values(srgb, linear):
    assert color.srgb_to_linear(np.float32(srgb)) == pytest.approx(linear, abs=1e-6)


@pytest.mark.parametrize("srgb, linear", [(0.0, 0.0), (1.0, 1.0), (0.5, 0.21404114), (0.8, 0.60382734)])
def test_linear_to_srgb_known_values(srgb, linear):
    assert color.linear_to_srgb(np.float32(linear)) == pytest.approx(srgb, abs=1e-6)


def test_transfer_function_is_continuous_at_the_knee():
    below = color.srgb_to_linear(np.float32(0.04045 - 1e-6))
    above = color.srgb_to_linear(np.float32(0.04045 + 1e-6))
    assert abs(float(above) - float(below)) < 1e-6
    below = color.linear_to_srgb(np.float32(0.0031308 - 1e-7))
    above = color.linear_to_srgb(np.float32(0.0031308 + 1e-7))
    assert abs(float(above) - float(below)) < 1e-5


def test_transfer_functions_are_monotonic_and_invert_each_other():
    x = np.linspace(0, 1, 2001, dtype=np.float32)
    lin = color.srgb_to_linear(x)
    assert np.all(np.diff(lin) >= 0)
    assert np.all(np.diff(color.linear_to_srgb(x)) >= 0)
    np.testing.assert_allclose(color.linear_to_srgb(lin), x, atol=2e-6)


def test_linear_to_srgb_clips_to_the_unit_range():
    out = color.linear_to_srgb(np.array([-0.5, 0.0, 1.0, 3.0], np.float32))
    np.testing.assert_array_equal(out, [0.0, 0.0, 1.0, 1.0])


def test_transfer_functions_keep_shape_and_float32():
    x = np.random.default_rng(0).random((5, 7, 3))  # float64 in
    for fn in (color.srgb_to_linear, color.linear_to_srgb):
        out = fn(x)
        assert out.shape == x.shape and out.dtype == np.float32


# --- 8-bit tables ---------------------------------------------------------------------


def test_decode_u8_matches_the_formula_for_every_value():
    values = np.arange(256, dtype=np.uint8)
    np.testing.assert_allclose(color.decode_u8(values), color.srgb_to_linear(values / 255.0), atol=1e-7)


def test_decode_u8_endpoints_and_dtype():
    out = color.decode_u8(np.array([[[0, 0, 0], [255, 255, 255]]], np.uint8))
    assert out.dtype == np.float32 and out.shape == (1, 2, 3)
    assert out[0, 0].tolist() == [0.0, 0.0, 0.0] and out[0, 1].tolist() == [1.0, 1.0, 1.0]


def test_decode_u8_result_can_be_modified_without_touching_the_table():
    img = np.full((2, 2, 3), 128, np.uint8)
    first = color.decode_u8(img)
    first[:] = 9.0
    assert color.decode_u8(img).max() < 1.0


def test_decode_u8_rejects_float_images_clearly():
    with pytest.raises(TypeError, match="uint8"):
        color.decode_u8(np.full((2, 2, 3), 0.5, np.float32))


def test_encode_u8_matches_the_exact_formula():
    lin = np.random.default_rng(1).random((200, 200)).astype(np.float32) ** 2  # dense in the shadows
    exact = np.round(color.linear_to_srgb(lin) * 255.0)
    assert np.abs(color.encode_u8(lin).astype(int) - exact).max() <= 1


def test_encode_u8_round_trips_every_value_exactly():
    values = np.arange(256, dtype=np.uint8)
    np.testing.assert_array_equal(color.encode_u8(color.decode_u8(values)), values)


def test_encode_u8_clips_and_handles_float64():
    out = color.encode_u8(np.array([-1.0, 0.0, 1.0, 5.0]))
    assert out.dtype == np.uint8 and out.tolist() == [0, 0, 255, 255]


def test_encode_u8_does_not_modify_its_input():
    lin = np.array([[-1.0, 0.25, 0.5, 7.0]], np.float32)
    before = lin.copy()
    color.encode_u8(lin)
    np.testing.assert_array_equal(lin, before)


def test_encode_u8_treats_nan_as_black_instead_of_crashing():
    out = color.encode_u8(np.array([np.nan, 1.0, np.inf, -np.inf], np.float32))
    assert out.tolist() == [0, 255, 255, 0]


# --- luminance, gray, saturation ------------------------------------------------------


def test_luminance_weights():
    assert color.luminance(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.2126)
    assert color.luminance(np.array([0.0, 1.0, 0.0])) == pytest.approx(0.7152)
    assert color.luminance(np.array([0.0, 0.0, 1.0])) == pytest.approx(0.0722)


def test_luminance_of_white_is_one_and_of_a_gray_is_that_gray():
    assert color.luminance(np.ones(3, np.float32)) == pytest.approx(1.0, abs=1e-6)
    gray = np.full((4, 4, 3), 0.3, np.float32)
    np.testing.assert_allclose(color.luminance(gray), 0.3, atol=1e-6)


def test_luminance_shape_and_dtype():
    out = color.luminance(np.random.default_rng(2).random((6, 9, 3)))
    assert out.shape == (6, 9) and out.dtype == np.float32


def test_luminance_of_srgb_mid_gray_is_the_linear_mid_gray():
    y = color.luminance(color.decode_u8(np.full((1, 1, 3), 128, np.uint8)))
    assert float(y[0, 0]) == pytest.approx(0.21586, abs=1e-4)


def test_gray_u8_weights_and_range():
    img = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0]]], np.uint8)
    np.testing.assert_allclose(color.gray_u8(img)[0], [76.245, 149.685, 29.07, 255.0, 0.0], atol=1e-3)


def test_gray_u8_does_not_overflow_on_uint8():
    out = color.gray_u8(np.full((2, 2, 3), 200, np.uint8))
    np.testing.assert_allclose(out, 200.0, atol=1e-3)


def test_hsv_saturation_known_values():
    rgb = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 0.5, 0.0], [0.0, 0.0, 0.0], [0.2, 0.2, 0.1]])
    np.testing.assert_allclose(color.hsv_saturation(rgb), [1.0, 0.0, 1.0, 0.0, 0.5], atol=1e-6)


def test_hsv_saturation_works_on_any_scale_and_on_uint8():
    np.testing.assert_allclose(color.hsv_saturation(np.array([[200, 100, 100]], np.uint8)), [0.5], atol=1e-6)
    np.testing.assert_allclose(color.hsv_saturation(np.array([[2.0, 1.0, 1.0]])), [0.5], atol=1e-6)


# --- hex colors -----------------------------------------------------------------------


def test_hex_to_linear_known_colors():
    np.testing.assert_allclose(color.hex_to_linear("#000000"), [0, 0, 0])
    np.testing.assert_allclose(color.hex_to_linear("#ffffff"), [1, 1, 1])
    np.testing.assert_allclose(color.hex_to_linear("#808080"), [0.21586] * 3, atol=1e-4)
    assert color.hex_to_linear("#ff0000").dtype == np.float32


def test_hex_to_linear_accepts_case_missing_hash_and_short_form():
    expected = color.hex_to_linear("#aabbcc")
    for spelling in ("AABBCC", "#AaBbCc", "abc", "#ABC"):
        np.testing.assert_allclose(color.hex_to_linear(spelling), expected)


@pytest.mark.parametrize("bad", ["", "#12", "#12345", "#gggggg", "red", "#1234567"])
def test_hex_to_linear_rejects_malformed_colors(bad):
    with pytest.raises(ValueError, match="hex color"):
        color.hex_to_linear(bad)


# --- numerical hygiene ----------------------------------------------------------------


def test_transfer_functions_raise_no_warnings_on_out_of_range_input():
    x = np.array([-1.0, -0.06, 0.0, 0.5, 1.0, 2.0, np.nan], np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        color.srgb_to_linear(x)
        color.linear_to_srgb(x)
