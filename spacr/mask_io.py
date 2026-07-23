"""
Cellpose mask I/O — write + read masks as EITHER TIFF or NumPy.

Introduces two helpers to replace scattered ``np.save``/``np.load``
calls across :mod:`spacr.object` and :mod:`spacr.io`:

* :func:`save_mask` — writes a single 2-D uint16 mask. Emits
  ``.tif`` by default (smaller on disk + openable in Fiji /
  napari) but ``.npy`` is still supported via ``fmt="npy"`` for
  users who prefer numpy round-trips.
* :func:`load_mask` — reads a mask regardless of on-disk format.
  Given a path with no suffix, or a stem, it probes ``.tif``,
  ``.tiff`` and ``.npy`` in that order and returns the first hit.

Both helpers preserve uint16 dtype (spaCR's mask convention) and
return arrays with the same shape ``(H, W)``. TIFF is written with
LZW compression via ``tifffile``.

Migration path for existing scripts:

    # BEFORE
    np.save("foo_mask.npy", mask.astype(np.uint16))
    m = np.load("foo_mask.npy")

    # AFTER
    save_mask("foo_mask", mask)       # writes foo_mask.tif
    m = load_mask("foo_mask")         # finds foo_mask.tif or foo_mask.npy

The env var ``SPACR_MASK_FORMAT`` overrides the default across
the whole process: set to ``npy`` to keep the old behaviour.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Union

import numpy as np

LOG = logging.getLogger("spacr.mask_io")

PathLike = Union[str, os.PathLike, Path]

DEFAULT_FORMAT = os.environ.get("SPACR_MASK_FORMAT", "tif").lower()
if DEFAULT_FORMAT not in ("tif", "tiff", "npy"):
    LOG.warning("SPACR_MASK_FORMAT=%r not recognised; using 'tif'",
                 DEFAULT_FORMAT)
    DEFAULT_FORMAT = "tif"


def save_mask(path: PathLike, mask: np.ndarray,
                fmt: str = None) -> Path:
    """Write a Cellpose mask to disk.

    :param path: destination — extension is optional. ``foo``,
        ``foo.tif``, ``foo.npy`` all work.
    :param mask: 2-D integer mask; will be cast to uint16.
    :param fmt: force a format (``"tif"``, ``"tiff"``, or ``"npy"``).
        Defaults to :data:`DEFAULT_FORMAT` (env-overridable).
    :returns: the resolved on-disk path.
    """
    fmt = (fmt or DEFAULT_FORMAT).lower().lstrip(".")
    p = Path(path)

    # If path already has a recognised suffix, that wins over `fmt`.
    if p.suffix.lower() in (".tif", ".tiff", ".npy"):
        fmt = p.suffix.lower().lstrip(".")

    if fmt in ("tif", "tiff"):
        p = p.with_suffix(f".{fmt}")
        try:
            import tifffile
        except Exception:
            LOG.warning("tifffile missing — falling back to npy for %s", p)
            return save_mask(path, mask, fmt="npy")
        tifffile.imwrite(
            str(p), mask.astype(np.uint16), compression="lzw",
        )
    elif fmt == "npy":
        p = p.with_suffix(".npy")
        np.save(str(p), mask.astype(np.uint16))
    else:
        raise ValueError(f"unknown mask format: {fmt!r}")
    return p


def load_mask(path: PathLike) -> np.ndarray:
    """Read a Cellpose mask regardless of on-disk format.

    Accepts a full path (``foo.tif`` / ``foo.npy``) OR a stem
    (``foo``) — in the stem case, tif → tiff → npy is tried and the
    first extant file is loaded.

    :returns: uint16 2-D array. Shape unchanged.
    :raises FileNotFoundError: when no matching file exists.
    """
    p = Path(path)
    candidates = []
    if p.suffix.lower() in (".tif", ".tiff", ".npy"):
        candidates.append(p)
    else:
        stem = p.with_suffix("")
        candidates.extend([
            stem.with_suffix(".tif"),
            stem.with_suffix(".tiff"),
            stem.with_suffix(".npy"),
        ])

    for c in candidates:
        if c.is_file():
            return _read_one(c)
    raise FileNotFoundError(f"no mask found for {path!r} (tried "
                              f"{[str(c.name) for c in candidates]})")


def _read_one(p: Path) -> np.ndarray:
    if p.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        arr = tifffile.imread(str(p))
    elif p.suffix.lower() == ".npy":
        arr = np.load(str(p), allow_pickle=False)
    else:
        raise ValueError(f"unsupported mask extension: {p}")
    return arr.astype(np.uint16, copy=False)
