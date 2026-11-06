"""Item 773: the per-plate illumination model applies to a crop window.

A crop cut from a field at a known offset is corrected with the gain and
offset maps sliced to its place in the field, and every pixel matches the
whole-field correction the Measure hook applies.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr import illumination as ill
from spacr.measure_hooks import PreprocessingContext

SHAPE = (48, 64)


def _flat(seed):
    """A smooth, uneven gain field with mean 1."""
    yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1]].astype(np.float32)
    plane = 1.0 + 0.3 * np.sin(yy / 17.0 + seed) * np.cos(xx / 23.0 - seed)
    return (plane / plane.mean()).astype(np.float32)


def _model(darkfield=False):
    flats = np.stack([_flat(0), _flat(1), _flat(2)])
    item = ill.IlluminationField(
        plate="plate1", channels=(0, 2, 3), flatfield=flats,
        dark=np.asarray([10.0, 20.0, 5.0], dtype=np.float32),
        n_fields=1, estimator="polynomial", degree=4, bin_size=1,
        darkfield=(np.stack([np.linspace(0, 8, SHAPE[1], dtype=np.float32)
                             * np.ones(SHAPE, np.float32)] * 3)
                   if darkfield else None))
    return ill.IlluminationModel(fields={"plate1": item}, meta={})


def _whole_field(model, image, channels):
    corrector = ill.IlluminationCorrector(model, verbose=False)
    return corrector(image, PreprocessingContext(
        file_name="plate1_A01_F001", channels=list(channels), settings={}))


def _image(dtype, channels=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(50, 3000, SHAPE + (channels,)).astype(dtype)


@pytest.mark.parametrize("darkfield", [False, True])
def test_a_window_matches_the_whole_field_correction(darkfield):
    model = _model(darkfield)
    channels = [0, 2, 3]
    image = _image(np.float32)
    whole = _whole_field(model, image, channels)
    item = model.field_for("plate1")

    for y0, x0, h, w in [(0, 0, 48, 64), (5, 7, 20, 16), (30, 50, 18, 14)]:
        window = image[y0:y0 + h, x0:x0 + w]
        corrected = item.correct_window(window, channels, y0=y0, x0=x0)
        assert corrected.dtype == np.float32
        assert corrected.shape == window.shape
        np.testing.assert_array_equal(corrected, whole[y0:y0 + h, x0:x0 + w])


def test_a_uint16_window_gets_the_unrounded_corrected_values():
    model = _model()
    channels = [3, 0]
    image = _image(np.uint16, channels=2, seed=1)
    item = model.field_for("plate1")

    corrected = item.correct_window(image[10:30, 20:44], channels,
                                    y0=10, x0=20)
    whole = _whole_field(model, image.astype(np.float32), channels)
    np.testing.assert_array_equal(corrected, whole[10:30, 20:44])
    rounded = _whole_field(model, image, channels)[10:30, 20:44]
    np.testing.assert_array_equal(
        np.clip(np.rint(corrected), 0, 65535).astype(np.uint16), rounded)


def test_the_channel_order_of_the_window_is_honoured():
    model = _model()
    image = _image(np.float32, seed=2)
    item = model.field_for("plate1")
    window = image[4:20, 4:20]

    forward = item.correct_window(window, [0, 2, 3], y0=4, x0=4)
    backward = item.correct_window(window[..., ::-1], [3, 2, 0], y0=4, x0=4)
    np.testing.assert_array_equal(forward, backward[..., ::-1])


def test_a_single_channel_window_stays_two_dimensional():
    model = _model()
    image = _image(np.float32, channels=1, seed=3)[..., 0]
    item = model.field_for("plate1")
    corrected = item.correct_window(image[8:24, 8:40], [2], y0=8, x0=8)

    assert corrected.shape == (16, 32)
    expected = (image[8:24, 8:40] - np.float32(20.0)) * (
        np.float32(1.0) / _flat(1)[8:24, 8:40])
    np.testing.assert_allclose(corrected, expected, rtol=1e-6)


def test_padding_past_the_field_edge_comes_back_as_zero():
    model = _model()
    channels = [0, 2, 3]
    image = _image(np.float32, seed=4)
    whole = _whole_field(model, image, channels)
    item = model.field_for("plate1")

    window = np.zeros((20, 20, 3), np.float32)
    window[6:, 4:] = image[:14, :16]
    corrected = item.correct_window(window, channels, y0=-6, x0=-4)

    np.testing.assert_array_equal(corrected[6:, 4:], whole[:14, :16])
    assert not corrected[:6].any() and not corrected[:, :4].any()


def test_a_saved_model_corrects_a_window_after_loading(tmp_path):
    model = _model(darkfield=True)
    path = model.save(str(tmp_path / "illumination_model.npz"))
    loaded = ill.load_illumination_model(path)
    channels = [0, 2, 3]
    image = _image(np.float32, seed=5)

    item = loaded.field_for(ill.plate_of_field("plate1_B02_F003.npy"))
    corrected = item.correct_window(image[12:40, 2:30], channels,
                                    y0=12, x0=2)
    np.testing.assert_array_equal(
        corrected, _whole_field(model, image, channels)[12:40, 2:30])


@pytest.mark.parametrize("window, channels, offset, message", [
    (np.zeros((8, 8, 3)), [0, 2, 3], (60, 0), "outside"),
    (np.zeros((8, 8, 3)), [0, 2, 3], (0, -8), "outside"),
    (np.zeros((8, 8, 2)), [0, 2, 3], (0, 0), "does not match"),
    (np.zeros((8, 8)), [1], (0, 0), "covers channels"),
])
def test_a_window_that_cannot_be_corrected_is_refused(window, channels,
                                                      offset, message):
    item = _model().field_for("plate1")
    with pytest.raises(ill.IlluminationError, match=message):
        item.correct_window(window, channels, y0=offset[0], x0=offset[1])
