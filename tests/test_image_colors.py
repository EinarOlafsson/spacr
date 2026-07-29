"""RGB contract tests for the OpenCV image-I/O boundary."""
from __future__ import annotations

import numpy as np
from PIL import Image

from spacr.image_colors import (
    cv2_to_rgb,
    read_image_rgb,
    rgb_to_cv2,
    write_image_rgb,
)


def test_bgr_and_bgra_are_converted_without_moving_alpha():
    bgr = np.array([[[30, 20, 10]]], dtype=np.uint8)
    bgra = np.array([[[30, 20, 10, 77]]], dtype=np.uint8)

    assert cv2_to_rgb(bgr).tolist() == [[[10, 20, 30]]]
    assert cv2_to_rgb(bgra).tolist() == [[[10, 20, 30, 77]]]
    assert rgb_to_cv2(cv2_to_rgb(bgr)).tolist() == bgr.tolist()
    assert rgb_to_cv2(cv2_to_rgb(bgra)).tolist() == bgra.tolist()


def test_grayscale_and_multichannel_stain_stacks_pass_through():
    gray = np.arange(9, dtype=np.uint16).reshape(3, 3)
    stains = np.zeros((2, 2, 5), dtype=np.uint16)

    assert cv2_to_rgb(gray) is gray
    assert rgb_to_cv2(gray) is gray
    assert cv2_to_rgb(stains) is stains
    assert rgb_to_cv2(stains) is stains


def test_opencv_roundtrip_preserves_rgb_file_semantics(tmp_path):
    path = tmp_path / "colours.png"
    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    rgb[..., 0] = 211
    rgb[..., 1] = 37
    rgb[..., 2] = 9

    assert write_image_rgb(path, rgb) is True
    np.testing.assert_array_equal(read_image_rgb(path), rgb)
    np.testing.assert_array_equal(np.asarray(Image.open(path).convert("RGB")), rgb)


def test_missing_file_remains_none(tmp_path):
    assert read_image_rgb(tmp_path / "missing.png") is None
