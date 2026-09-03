"""Colour-order boundaries for OpenCV.

spaCR's in-memory colour contract is RGB (or RGBA).  OpenCV is the one major
dependency that decodes and encodes three-channel images as BGR, so conversions
belong directly beside ``imread``/``imwrite`` rather than in plotting or model
code.  Grayscale arrays pass through unchanged.
"""
from __future__ import annotations

from os import PathLike
from typing import Optional, Union

import numpy as np

PathValue = Union[str, PathLike[str]]


def cv2_to_rgb(image: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Convert an OpenCV BGR/BGRA array to spaCR's RGB/RGBA contract.

    :param image: decoded OpenCV array, grayscale array, or ``None``.
    :returns: contiguous RGB/RGBA data for three- or four-channel inputs;
        ``None``, grayscale, and other channel counts pass through unchanged.
    """
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.ndim != 3:
        return arr
    if arr.shape[-1] == 3:
        return np.ascontiguousarray(arr[..., ::-1])
    if arr.shape[-1] == 4:
        return np.ascontiguousarray(arr[..., [2, 1, 0, 3]])
    return arr


def rgb_to_cv2(image: np.ndarray) -> np.ndarray:
    """Convert an RGB/RGBA array only for an immediate OpenCV write call.

    :param image: spaCR RGB/RGBA array, or a grayscale array to pass through.
    :returns: contiguous BGR/BGRA data for three- or four-channel inputs;
        grayscale and other channel counts pass through unchanged.
    """
    arr = np.asarray(image)
    if arr.ndim != 3:
        return arr
    if arr.shape[-1] == 3:
        return np.ascontiguousarray(arr[..., ::-1])
    if arr.shape[-1] == 4:
        return np.ascontiguousarray(arr[..., [2, 1, 0, 3]])
    return arr


def read_image_rgb(path: PathValue, flags: int = -1) -> Optional[np.ndarray]:
    """Read with OpenCV and immediately return RGB/RGBA in memory.

    :param path: image path accepted by OpenCV.
    :param flags: OpenCV read mode; ``-1`` preserves the stored dtype and
        alpha channel.
    :returns: RGB/RGBA or grayscale image data, or ``None`` when OpenCV cannot
        read the path.
    """
    import cv2

    return cv2_to_rgb(cv2.imread(str(path), flags))


def write_image_rgb(path: PathValue, image: np.ndarray, params=None) -> bool:
    """Write an RGB/RGBA array through OpenCV without leaking BGR internally.

    :param path: output image path passed to OpenCV.
    :param image: spaCR RGB/RGBA array, converted immediately before encoding.
    :param params: optional OpenCV encoder parameter sequence.
    :returns: whether OpenCV encoded and wrote the image successfully.
    """
    import cv2

    encoded = rgb_to_cv2(image)
    if params is None:
        return bool(cv2.imwrite(str(path), encoded))
    return bool(cv2.imwrite(str(path), encoded, params))
