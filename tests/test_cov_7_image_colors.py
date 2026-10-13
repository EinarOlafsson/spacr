"""Encoder options must survive the RGB/BGR boundary.

``write_image_rgb`` has two calls into OpenCV: one with encoder parameters and
one without. The parameter form is what a caller reaches for when a PNG has to
be written at a specific compression level or a JPEG at a specific quality, and
it has to convert the array exactly like the plain form does -- a channel swap
that only happens on one of the two paths writes a blue cell in a red channel
for anybody who asked for compression.
"""

from __future__ import annotations

import numpy as np
import pytest

from spacr.image_colors import read_image_rgb, write_image_rgb

cv2 = pytest.importorskip("cv2")


def _swatch() -> np.ndarray:
    """A 2x2 image whose channels are all different, so a swap is visible."""
    image = np.zeros((2, 2, 3), np.uint8)
    image[..., 0] = 10   # R
    image[..., 1] = 120  # G
    image[..., 2] = 240  # B
    return image


def test_encoder_parameters_do_not_change_the_colours_written(tmp_path):
    """A PNG written with compression parameters holds the same RGB values."""
    target = tmp_path / "with_params.png"
    written = write_image_rgb(target, _swatch(),
                              [cv2.IMWRITE_PNG_COMPRESSION, 1])

    assert written is True
    assert target.is_file()
    back = read_image_rgb(target)
    assert np.array_equal(back, _swatch())


def test_the_parameter_form_matches_the_plain_form_byte_for_byte(tmp_path):
    """The two write paths differ only in encoder options, never in channels."""
    plain = tmp_path / "plain.png"
    tuned = tmp_path / "tuned.png"
    assert write_image_rgb(plain, _swatch()) is True
    assert write_image_rgb(tuned, _swatch(),
                           [cv2.IMWRITE_PNG_COMPRESSION, 9]) is True

    assert np.array_equal(read_image_rgb(plain), read_image_rgb(tuned))
