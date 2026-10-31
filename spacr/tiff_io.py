"""Canonical TIFF writing for spaCR's scientific image arrays."""

from __future__ import annotations

from os import PathLike
from typing import Any, Union

import numpy as np
import tifffile

TiffPath = Union[str, PathLike[str]]

DEFAULT_PHOTOMETRIC = "minisblack"
DEFAULT_PLANARCONFIG = "contig"


def write_tiff(path: TiffPath, array: np.ndarray, **kwargs: Any) -> None:
    """Write a TIFF with an explicit, stable sample interpretation.

    spaCR arrays are scientific intensity planes or label masks, including
    arrays whose leading dimension happens to contain three or four planes.
    Tifffile historically guessed RGB for those shapes and is changing that
    guess. Declaring ``minisblack`` prevents channel/z/time stacks from being
    mislabeled as color, while ``contig`` makes the planar choice explicit.

    Callers writing a true display RGB image may override either value through
    ``photometric=`` and ``planarconfig=``. All other tifffile options (axes
    metadata, compression, BigTIFF, ImageJ compatibility, and so on) pass
    through unchanged.

    :param path: destination TIFF path.
    :param array: image, stack, or label array.
    :param kwargs: additional :func:`tifffile.imwrite` options.
    :returns: None.
    """
    kwargs.setdefault("photometric", DEFAULT_PHOTOMETRIC)
    kwargs.setdefault("planarconfig", DEFAULT_PLANARCONFIG)
    from .portable_paths import _long_path

    tifffile.imwrite(_long_path(path), np.asarray(array), **kwargs)


__all__ = [
    "DEFAULT_PHOTOMETRIC",
    "DEFAULT_PLANARCONFIG",
    "write_tiff",
]


def _raster_with_axes(path, ext):
    """Read one raster file with the axis order it declares.

    TIFF files report tifffile's series axes. PNG, JPEG and BMP files are
    read with imageio; a colour picture is declared ``YXS`` so its colour
    samples become channels rather than being projected away.

    :param path: the image file.
    :param ext: its lowercase extension without the dot.
    :returns: ``(array, axes)``; ``axes`` is ``''`` when nothing is declared.
    """
    if ext in ('tif', 'tiff'):
        with tifffile.TiffFile(path) as tif:
            images = tif.asarray()
            try:
                axes = (tif.series[0].axes or '').upper()
            except Exception:
                axes = ''
        return images, axes
    import imageio.v2 as imageio

    images = np.asarray(imageio.imread(path))
    if images.ndim == 3 and images.shape[-1] in (3, 4):
        return images, 'YXS'
    return images, 'YX' if images.ndim == 2 else ''


def _declared_planes(images, axes):
    """Planes of an array whose time, depth and channel axes are declared.

    ``S`` (colour samples) and ``C`` are both channel axes, wherever they
    sit, so an RGB, a channel-last and a channel-first file of the same
    planes give the same channels. Depth is maximum-projected. A sample
    axis at least as long as a spatial one is tifffile's colour guess for a
    small undeclared stack, not colour, and is not used.

    :param images: the array.
    :param axes: one letter per axis, ending in or containing ``Y`` and ``X``.
    :returns: ``[(t, c, plane)]``, or None when the axes cannot be used.
    """
    if len(axes) != images.ndim or 'Y' not in axes or 'X' not in axes:
        return None
    letters = [a for a in axes if a not in 'YX']
    if any(a not in 'TZCS' for a in letters) or len(set(letters)) != len(letters):
        return None
    if 'C' in letters and 'S' in letters:
        return None
    channel = 'C' if 'C' in letters else 'S'
    if 'S' in letters and images.shape[axes.index('S')] >= min(
            images.shape[axes.index('Y')], images.shape[axes.index('X')]):
        return None
    order = [axes.index(a) for a in ('T', 'Z', channel) if a in axes]
    data = np.transpose(images, order + [axes.index('Y'), axes.index('X')])
    for at, a in enumerate(('T', 'Z', channel)):
        if a not in axes:
            data = np.expand_dims(data, at)
    planes = []
    for t in range(data.shape[0]):
        for c in range(data.shape[2]):
            stack = data[t, :, c]
            plane = stack[0] if stack.shape[0] == 1 else np.max(stack, axis=0)
            planes.append((t, c, np.ascontiguousarray(plane)))
    return planes


def _yokogawa_planes(images, axes, file):
    """The ``(t, c, plane)`` images :func:`convert_to_yokogawa` writes for one file.

    Declared time, depth and channel axes decide, so a multichannel
    ImageJ, OME-TIFF or RGB file becomes one plane per raw channel, in
    file order, exactly as if each channel had been saved separately. An
    undeclared 3-D array is channels when its first axis has at most four
    planes, channel-last when only its last axis does and that axis is
    shorter than both others, and a z-stack otherwise. An undeclared 4-D array is read as ``(T, Z, Y, X)``.

    :param images: the pixels.
    :param axes: the declared axes, or ``''``.
    :param file: the file name, for the warning about an undeclared 4-D file.
    :returns: ``[(t, c, plane)]`` in writing order.
    :raises ValueError: for an array spaCR cannot split into planes.
    """
    axes = (axes or '').upper()
    if images.ndim == 2:
        return [(0, 0, images)]
    planes = _declared_planes(images, axes)
    if planes is not None:
        return planes
    if images.ndim == 3:
        if images.shape[0] <= 4:
            return [(0, c, images[c, :, :]) for c in range(images.shape[0])]
        if images.shape[-1] <= 4 and images.shape[-1] < min(images.shape[:2]):
            return [(0, c, np.ascontiguousarray(images[:, :, c]))
                    for c in range(images.shape[-1])]
        return [(0, 0, np.max(images, axis=0))]
    if images.ndim == 4:
        t_axis = 0
        if axes[:2] == 'ZT':
            t_axis = 1
        elif axes[:2] != 'TZ':
            print(f"WARNING: {file} is 4-D but declares axes "
                  f"{axes or '(none)'}; reading it as "
                  f"(T, Z, Y, X). If it is really (Z, T, Y, X), "
                  f"every timepoint written below is a z-plane "
                  f"and every projection is over time.")
        return [(t, 0, np.max(images[t] if t_axis == 0 else images[:, t], axis=0))
                for t in range(images.shape[t_axis])]
    raise ValueError(f"Unsupported TIFF dimensions: {images.shape}")
