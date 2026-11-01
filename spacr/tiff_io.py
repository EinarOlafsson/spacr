"""Canonical TIFF writing for spaCR's scientific image arrays."""

from __future__ import annotations

from os import PathLike
from typing import Any, Union

import numpy as np
import tifffile

TiffPath = Union[str, PathLike[str]]

TIFF_FILE_ERROR: type = getattr(tifffile, "TiffFileError", ValueError)
TIFF_READ_ERRORS: tuple = (OSError, ValueError, TIFF_FILE_ERROR)

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
        return _tiff_with_axes(path)
    return _picture_with_axes(path)


def _tiff_with_axes(path):
    """A TIFF's pixels and its first series' axes, uppercase.

    :param path: the TIFF file.
    :returns: ``(array, axes)``; ``axes`` is ``''`` when none are recorded.
    """
    with tifffile.TiffFile(path) as tif:
        return tif.asarray(), _series_axes(tif)


def _series_axes(tif):
    """The first series' axis letters, or ``''`` when tifffile has none.

    :param tif: an open :class:`tifffile.TiffFile`.
    :returns: the axes, uppercase.
    """
    try:
        return (tif.series[0].axes or '').upper()
    except Exception:
        return ''


def _picture_with_axes(path):
    """A PNG, JPEG or BMP picture's pixels and the axes they stand for.

    :param path: the picture file.
    :returns: ``(array, axes)``.
    """
    import imageio.v2 as imageio

    images = np.asarray(imageio.imread(path))
    return images, _picture_axes(images)


def _picture_axes(images):
    """``YXS`` for a colour picture, ``YX`` for a grey one, else ``''``.

    :param images: the decoded picture.
    :returns: the axes.
    """
    if images.ndim == 3 and images.shape[-1] in (3, 4):
        return 'YXS'
    return 'YX' if images.ndim == 2 else ''


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
    channel = _declared_channel_axis(images, axes)
    if channel is None:
        return None
    return _projected_planes(_tzcyx(images, axes, channel))


def _declared_channel_axis(images, axes):
    """The letter of the declared channel axis, or None if the axes are unusable.

    :param images: the array.
    :param axes: its declared axes.
    :returns: ``'C'`` or ``'S'`` (also when neither is present), or None.
    """
    if not _axes_are_spatial(images, axes):
        return None
    letters = [a for a in axes if a not in 'YX']
    if not _letters_are_known(letters):
        return None
    if 'S' in letters and _samples_are_guessed(images, axes):
        return None
    return 'C' if 'C' in letters else 'S'


def _axes_are_spatial(images, axes):
    """Whether ``axes`` names every axis of ``images`` and includes Y and X.

    :param images: the array.
    :param axes: its declared axes.
    """
    return len(axes) == images.ndim and 'Y' in axes and 'X' in axes


def _letters_are_known(letters):
    """Whether the non-spatial axes are distinct T, Z and one channel kind.

    :param letters: the axis letters other than Y and X.
    """
    known = all(a in 'TZCS' for a in letters)
    distinct = len(set(letters)) == len(letters)
    return known and distinct and not ('C' in letters and 'S' in letters)


def _samples_are_guessed(images, axes):
    """Whether an ``S`` axis is at least as long as a spatial axis.

    tifffile reads the last axis of a small undeclared stack as colour
    samples; such an axis is not taken as channels.

    :param images: the array.
    :param axes: its declared axes, containing ``S``.
    """
    samples = images.shape[axes.index('S')]
    return samples >= min(images.shape[axes.index('Y')], images.shape[axes.index('X')])


def _tzcyx(images, axes, channel):
    """``images`` reordered to ``(T, Z, C, Y, X)``, missing axes of length 1.

    :param images: the array.
    :param axes: its declared axes.
    :param channel: the channel axis letter, ``'C'`` or ``'S'``.
    :returns: a five-dimensional view.
    """
    order = [axes.index(a) for a in ('T', 'Z', channel) if a in axes]
    data = np.transpose(images, order + [axes.index('Y'), axes.index('X')])
    for at, a in enumerate(('T', 'Z', channel)):
        if a not in axes:
            data = np.expand_dims(data, at)
    return data


def _projected_planes(data):
    """One plane per time point and channel, maximum-projected over depth.

    :param data: a ``(T, Z, C, Y, X)`` array.
    :returns: ``[(t, c, plane)]`` in time, then channel order.
    """
    return [(t, c, _project_depth(data[t, :, c]))
            for t in range(data.shape[0]) for c in range(data.shape[2])]


def _project_depth(stack):
    """The single plane of a one-plane stack, or its maximum projection.

    :param stack: a ``(Z, Y, X)`` array.
    :returns: a C-contiguous 2-D plane.
    """
    plane = stack[0] if stack.shape[0] == 1 else np.max(stack, axis=0)
    return np.ascontiguousarray(plane)


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
        return _undeclared_3d_planes(images)
    if images.ndim == 4:
        return _undeclared_4d_planes(images, axes, file)
    raise ValueError(f"Unsupported TIFF dimensions: {images.shape}")


def _undeclared_3d_planes(images):
    """Split an undeclared 3-D array into channels, or project it as a z-stack.

    :param images: the array.
    :returns: ``[(0, c, plane)]``.
    """
    if images.shape[0] <= 4:
        return [(0, c, images[c, :, :]) for c in range(images.shape[0])]
    if images.shape[-1] <= 4 and images.shape[-1] < min(images.shape[:2]):
        return [(0, c, np.ascontiguousarray(images[:, :, c]))
                for c in range(images.shape[-1])]
    return [(0, 0, np.max(images, axis=0))]


def _undeclared_4d_planes(images, axes, file):
    """Project each time point of a 4-D array over depth.

    :param images: the array.
    :param axes: its axes, read for a ``TZ`` or ``ZT`` start.
    :param file: the file name, for the warning about undeclared axes.
    :returns: ``[(t, 0, plane)]``.
    """
    t_axis = _time_axis(axes, file)
    return [(t, 0, np.max(images[t] if t_axis == 0 else images[:, t], axis=0))
            for t in range(images.shape[t_axis])]


def _time_axis(axes, file):
    """Which of a 4-D array's first two axes is time, warning when undeclared.

    :param axes: the declared axes.
    :param file: the file name, for the warning.
    :returns: 0 or 1.
    """
    if axes[:2] == 'ZT':
        return 1
    if axes[:2] != 'TZ':
        print(f"WARNING: {file} is 4-D but declares axes "
              f"{axes or '(none)'}; reading it as "
              f"(T, Z, Y, X). If it is really (Z, T, Y, X), "
              f"every timepoint written below is a z-plane "
              f"and every projection is over time.")
    return 0


_LENGTH_UNITS_NM = {"nm": 1.0, "µm": 1000.0, "μm": 1000.0, "um": 1000.0,
                    "mm": 1e6, "Å": 0.1, "A": 0.1, "pm": 0.001, "m": 1e9,
                    "cm": 1e7, "micron": 1000.0}
"""Length units met in EM and TIFF metadata, in nanometres."""


def _nm_per(unit):
    """How many nanometres one ``unit`` is, or None for a unit of no length."""
    unit = str(unit or "").strip()
    return _LENGTH_UNITS_NM.get(unit)


def _tiff_pixel_size_nm(path):
    """The pixel size a TIFF records, in nanometres, or None.

    What :func:`spacr.io.tiff_pixel_size_nm` answers, read with tifffile
    alone, so a mask save asking a TIFF's calibration never imports
    :mod:`spacr.io` and the deep-learning stack it brings in.

    :param path: the TIFF.
    """
    import os

    try:
        with tifffile.TiffFile(os.fspath(path)) as tif:
            page = tif.pages[0]
            tag = page.tags.get("XResolution")
            unit_tag = page.tags.get("ResolutionUnit")
            imagej = tif.imagej_metadata or {}
            shaped = (tif.shaped_metadata or ({},))[0] or {}
    except Exception:
        return None
    exact = shaped.get("pixel_size_nm") if isinstance(shaped, dict) else None
    if isinstance(exact, (int, float)) and exact > 0:
        return float(exact)
    if tag is None:
        return None
    numerator, denominator = tag.value
    if not numerator or not denominator:
        return None
    per_unit = numerator / denominator
    factor = _nm_per(imagej.get("unit"))
    if factor is None:
        unit = int(unit_tag.value) if unit_tag is not None else 1
        factor = 1e7 if unit == 3 else None
    if factor is None or per_unit <= 0:
        return None
    return factor / per_unit
