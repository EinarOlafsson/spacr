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
    tifffile.imwrite(path, np.asarray(array), **kwargs)


__all__ = [
    "DEFAULT_PHOTOMETRIC",
    "DEFAULT_PLANARCONFIG",
    "write_tiff",
]
