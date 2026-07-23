"""
Multi-format dataset detection — handles single-file dataset drops.

Not every microscopy dataset comes as one image per file. spaCR must
also handle:

* ``.npz`` — one archive holding several named arrays (each usually
  a field / channel / stack).
* ``.npy`` — a single big ndarray whose axes carry the field / channel
  meaning.
* ``.lif`` / ``.nd2`` — vendor formats that pack a whole plate + its
  metadata into one file. spaCR uses ``readlif`` and ``nd2reader`` to
  crack them open.
* Multi-page ``.tif`` / ``.tiff`` — one file, many pages; typical
  meaning is a stack along Z, T, or C.

:func:`describe_file` returns a :class:`DatasetDescription` for any
supported file (or None). Callers can then either:

* Extract images out into the canonical filename format via
  :func:`explode_to_folder` (writes real .tif files + a
  ``filename_map.csv``), or
* Consume the arrays directly in a downstream analysis.

This module deliberately avoids importing heavy dependencies at
top-level — vendor libraries are imported lazily inside the
describers so users without them don't pay a cost.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LOG = logging.getLogger("spacr.qt.multi_format")


@dataclass
class DatasetDescription:
    """Structured summary of a single-file dataset drop.

    :ivar path: the source file.
    :ivar kind: one of ``"npz" | "npy" | "lif" | "nd2" | "tif_multi"``.
    :ivar n_fields: number of distinct fields (a.k.a. positions / series).
    :ivar n_channels: number of channels per field.
    :ivar n_timepoints: number of timepoints (1 if none).
    :ivar n_slices: number of z-slices (1 if none).
    :ivar shape: (H, W) of one plane if known, else None.
    :ivar dtype: dtype of the arrays if known.
    :ivar notes: anything else worth telling the user.
    """
    path:         Path
    kind:         str
    n_fields:     int = 1
    n_channels:   int = 1
    n_timepoints: int = 1
    n_slices:     int = 1
    shape:        Optional[Tuple[int, int]] = None
    dtype:        Optional[str] = None
    notes:        List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Human-readable one-line summary for the Console."""
        parts = [
            f"format={self.kind}",
            f"fields={self.n_fields}",
            f"channels={self.n_channels}",
        ]
        if self.n_timepoints > 1:
            parts.append(f"T={self.n_timepoints}")
        if self.n_slices > 1:
            parts.append(f"Z={self.n_slices}")
        if self.shape:
            parts.append(f"HxW={self.shape[0]}x{self.shape[1]}")
        if self.dtype:
            parts.append(f"dtype={self.dtype}")
        line = "  ".join(parts)
        if self.notes:
            line += "  · " + "; ".join(self.notes)
        return line


# ---------------------------------------------------------------------------
# describe_file — top-level dispatcher
# ---------------------------------------------------------------------------

def describe_file(path: Any) -> Optional[DatasetDescription]:
    """Return a :class:`DatasetDescription` for ``path`` or None.

    :param path: file to inspect.
    """
    p = Path(path)
    if not p.is_file():
        return None
    suf = p.suffix.lower()
    if suf == ".npz":     return _describe_npz(p)
    if suf == ".npy":     return _describe_npy(p)
    if suf in (".tif", ".tiff"): return _describe_tif(p)
    if suf == ".lif":     return _describe_lif(p)
    if suf == ".nd2":     return _describe_nd2(p)
    return None


# ---------------------------------------------------------------------------
# Backend describers
# ---------------------------------------------------------------------------

def _describe_npz(p: Path) -> Optional[DatasetDescription]:
    try:
        import numpy as np
        with np.load(p) as z:
            keys = list(z.files)
            if not keys:
                return None
            first = z[keys[0]]
        shape = tuple(first.shape)
        # Heuristics: (fields, H, W)  or  (fields, H, W, C)  or
        # (H, W, C)                      or  (H, W)
        n_fields = 1
        n_channels = 1
        img_shape: Optional[Tuple[int, int]] = None
        if len(shape) == 4:
            n_fields, H, W, n_channels = shape
            img_shape = (int(H), int(W))
        elif len(shape) == 3:
            # 3D — could be fields OR channels
            a, b, c = shape
            if a < 20 and b > 20 and c > 20:
                # Fields first, then H, W (channels = 1)
                n_fields, H, W = a, b, c
                img_shape = (int(H), int(W))
            else:
                # H, W, C
                img_shape = (int(a), int(b))
                n_channels = int(c)
        elif len(shape) == 2:
            img_shape = (int(shape[0]), int(shape[1]))
        # Each named key inside the npz is often ONE field
        if len(keys) > 1:
            n_fields = max(n_fields, len(keys))
        return DatasetDescription(
            path=p, kind="npz",
            n_fields=int(n_fields), n_channels=int(n_channels),
            shape=img_shape, dtype=str(first.dtype),
            notes=[f"arrays={keys[:5]}"
                   + ("…" if len(keys) > 5 else "")],
        )
    except Exception as e:
        LOG.debug("npz describe failed: %s", e)
        return None


def _describe_npy(p: Path) -> Optional[DatasetDescription]:
    try:
        import numpy as np
        # mmap_mode='r' → don't read the whole array into memory
        arr = np.load(p, mmap_mode="r")
        shape = tuple(arr.shape)
        n_fields = 1
        n_channels = 1
        img_shape: Optional[Tuple[int, int]] = None
        if len(shape) == 4:
            n_fields, H, W, n_channels = shape
            img_shape = (int(H), int(W))
        elif len(shape) == 3:
            a, b, c = shape
            if a < 20 and b > 20 and c > 20:
                n_fields, H, W = a, b, c
                img_shape = (int(H), int(W))
            else:
                img_shape = (int(a), int(b))
                n_channels = int(c)
        elif len(shape) == 2:
            img_shape = (int(shape[0]), int(shape[1]))
        return DatasetDescription(
            path=p, kind="npy",
            n_fields=int(n_fields), n_channels=int(n_channels),
            shape=img_shape, dtype=str(arr.dtype),
            notes=[f"npy_shape={shape}"],
        )
    except Exception as e:
        LOG.debug("npy describe failed: %s", e)
        return None


def _describe_tif(p: Path) -> Optional[DatasetDescription]:
    """Only interesting for MULTI-PAGE tiffs; single-page is treated
    as a normal image and left to the DnD handler."""
    try:
        import tifffile
        with tifffile.TiffFile(str(p)) as tf:
            n_pages = len(tf.pages)
            if n_pages <= 1:
                return None
            page = tf.pages[0]
            H, W = page.shape[:2] if hasattr(page, "shape") \
                                  else (page.imagelength, page.imagewidth)
            dtype = str(page.dtype)
            # Try imagej / ome metadata for axis meaning
            axes = None
            for tag_name in ("axes", "ImageJ", "OME"):
                if hasattr(tf, tag_name):
                    axes = getattr(tf, tag_name); break
            notes = [f"pages={n_pages}"]
            if axes:
                notes.append(f"axes={axes}")
        return DatasetDescription(
            path=p, kind="tif_multi",
            n_fields=n_pages,
            shape=(int(H), int(W)), dtype=dtype,
            notes=notes,
        )
    except Exception as e:
        LOG.debug("tif describe failed: %s", e)
        return None


def _describe_lif(p: Path) -> Optional[DatasetDescription]:
    try:
        from readlif.reader import LifFile        # type: ignore
        lif = LifFile(str(p))
        images = list(lif.get_iter_image())
        if not images:
            return None
        first = images[0]
        n_channels = int(getattr(first, "channels", 1) or 1)
        n_slices   = int(getattr(first, "nz", 1) or 1)
        n_time     = int(getattr(first, "nt", 1) or 1)
        H, W = (int(first.dims.y), int(first.dims.x))
        return DatasetDescription(
            path=p, kind="lif",
            n_fields=len(images), n_channels=n_channels,
            n_timepoints=n_time, n_slices=n_slices,
            shape=(H, W),
            notes=[f"series={[img.name for img in images[:5]]}"
                   + ("…" if len(images) > 5 else "")],
        )
    except Exception as e:
        LOG.debug("lif describe failed: %s", e)
        return None


def _describe_nd2(p: Path) -> Optional[DatasetDescription]:
    try:
        from nd2reader import ND2Reader             # type: ignore
        with ND2Reader(str(p)) as nd2:
            axes = nd2.axes
            sizes = nd2.sizes
            H = int(sizes.get("y", 0))
            W = int(sizes.get("x", 0))
            n_fields = int(sizes.get("v", 1))
            n_channels = int(sizes.get("c", 1))
            n_time = int(sizes.get("t", 1))
            n_slices = int(sizes.get("z", 1))
        return DatasetDescription(
            path=p, kind="nd2",
            n_fields=n_fields, n_channels=n_channels,
            n_timepoints=n_time, n_slices=n_slices,
            shape=(H, W) if H and W else None,
            notes=[f"axes={axes}"],
        )
    except Exception as e:
        LOG.debug("nd2 describe failed: %s", e)
        return None
