"""Ragged object crops in per-well Zarr pixel pools, with the ssCell_Crops table (items 983 and 984, alpha).

Each crop is its object's bounding box plus a margin, stored at its own size.
A well's crops of one object class go into flat pixel pools inside the
well's crop store, ``crops/plate=<P>/well=<W>.zarr`` (the per-well store of
item 974): ``<class>/pool/<k>`` holds the pixels ``(C, n)`` and
``<class>/mask/<k>`` the object's own mask pixels ``(n,)``, one pool array
per shard of ``crops_per_shard`` crops, written whole by the one writer. The
crop table gives each crop its pool, offset, height and width, so a crop is
one contiguous read.

This is spaCR's own crop format, not an OME-Zarr image: viewers open the
images, not the pools. :func:`export_crops` writes crops out as TIFF, PNG
or npy for other tools.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = ["CROP_COLUMNS", "CropPoolWriter", "cut_crops", "read_crop",
           "export_crops"]

#: The ssCell_Crops columns, in order (item 984).
CROP_COLUMNS: Tuple[str, ...] = (
    "object_id", "object_type", "plate", "row", "column", "well", "field",
    "timepoint", "object_label",
    "crop_store", "crop_array", "crop_index", "offset", "height", "width",
    "mask_offset",
    "bbox_y0", "bbox_x0", "bbox_y1", "bbox_x1", "margin_px", "clipped_at_edge",
    "png_path")

#: Crops per pool shard; each shard is one array written whole (16 chunks of 256).
DEFAULT_CROPS_PER_SHARD = 4096

#: Crops per chunk inside a shard.
CROPS_PER_CHUNK = 256


def cut_crops(image: np.ndarray, labels: np.ndarray, *, margin: int = 0,
              zero_outside: bool = False,
              objects: Optional[Sequence[int]] = None
              ) -> List[Dict[str, Any]]:
    """Cut each object's bounding box plus ``margin`` out of a field.

    :param image: the field ``(C, Y, X)``.
    :param labels: the object labels ``(Y, X)``.
    :param margin: pixels added on every side, clipped at the field edge.
    :param zero_outside: set pixels outside the object to 0 in the crop.
    :param objects: the labels to cut, ascending; every label when None.
    :returns: one record per object: ``object_label``, ``pixels``
        ``(C, h, w)``, ``mask`` ``(h, w)`` bool, ``bbox_y0/x0/y1/x1`` (the
        cut box, end-exclusive), ``margin_px`` and ``clipped_at_edge``.
    """
    from scipy import ndimage

    image = np.asarray(image)
    labels = np.asarray(labels)
    if image.ndim != 3 or image.shape[1:] != labels.shape:
        raise ValueError(f"image {image.shape} must be (C, Y, X) over labels {labels.shape}")
    found = np.unique(labels[labels > 0])
    wanted = found if objects is None else np.asarray(sorted(objects))
    boxes = ndimage.find_objects(labels.astype(np.int64, copy=False))
    out = []
    height, width = labels.shape
    for label in wanted:
        box = boxes[int(label) - 1] if int(label) - 1 < len(boxes) else None
        if box is None:
            continue
        y0 = box[0].start - margin
        x0 = box[1].start - margin
        y1 = box[0].stop + margin
        x1 = box[1].stop + margin
        clipped = y0 < 0 or x0 < 0 or y1 > height or x1 > width
        y0, x0, y1, x1 = max(y0, 0), max(x0, 0), min(y1, height), min(x1, width)
        mask = labels[y0:y1, x0:x1] == label
        pixels = np.array(image[:, y0:y1, x0:x1])
        if zero_outside:
            pixels[:, ~mask] = 0
        out.append({"object_label": int(label), "pixels": pixels, "mask": mask,
                    "bbox_y0": int(y0), "bbox_x0": int(x0), "bbox_y1": int(y1),
                    "bbox_x1": int(x1), "margin_px": int(margin),
                    "clipped_at_edge": bool(clipped)})
    return out


class CropPoolWriter:
    """The one writer of a well's ragged crop pools (item 984).

    :param project: the project folder (the 974 store's root).
    :param plate: the plate name.
    :param well: the well name.
    :param object_type: the object class, such as ``'cell'``.
    :param channels: the number of channels.
    :param dtype: the pixel dtype.
    :param crops_per_shard: crops per pool shard (default 4096). Each shard
        is buffered until full (or :meth:`close`) and written whole, so no
        shard is ever rewritten.
    """

    def __init__(self, project: Any, plate: str, well: str, object_type: str,
                 channels: int, dtype: Any = np.uint16, *,
                 crops_per_shard: int = DEFAULT_CROPS_PER_SHARD):
        """Open the well's crop store and start an empty shard buffer."""
        from .ome_zarr import require_zarr
        from .store import _encoded

        self.zarr = require_zarr()
        self.project = Path(project)
        self.plate, self.well = str(plate), str(well)
        self.object_type = str(object_type)
        self.channels = int(channels)
        self.dtype = np.dtype(dtype)
        self.crops_per_shard = max(1, int(crops_per_shard))
        self.relative = (Path("crops") / f"plate={_encoded(plate)}"
                         / f"well={_encoded(well)}.zarr")
        self.path = self.project / self.relative
        self.path.mkdir(parents=True, exist_ok=True)
        if not (self.path / "zarr.json").is_file():
            self.zarr.open_group(str(self.path), mode="w", zarr_format=3)
        self._shard = 0
        while (self.path / self.object_type / "pool" / str(self._shard)).exists():
            self._shard += 1
        self._buffer: List[Dict[str, Any]] = []
        self.rows: List[Dict[str, Any]] = []

    def add(self, crops: Iterable[Mapping[str, Any]], **identity: Any) -> None:
        """Queue crops from :func:`cut_crops`, with their identity columns.

        :param identity: ``field``, ``timepoint`` and any other columns every
            crop of this call shares.
        """
        for crop in crops:
            pixels = np.asarray(crop["pixels"])
            if pixels.shape[0] != self.channels:
                raise ValueError(f"crop has {pixels.shape[0]} channels, the pool {self.channels}")
            self._buffer.append({**identity, **crop})
            if len(self._buffer) >= self.crops_per_shard:
                self._flush()

    def close(self) -> List[Dict[str, Any]]:
        """Write the last, partly filled shard; return every crop's table row."""
        if self._buffer:
            self._flush()
        return self.rows

    def _flush(self) -> None:
        """Write the buffered crops as one pool shard and one mask shard."""
        from zarr.codecs import ZstdCodec

        crops, self._buffer = self._buffer, []
        sizes = [int(c["pixels"].shape[1] * c["pixels"].shape[2]) for c in crops]
        total = int(sum(sizes))
        pool = np.empty((self.channels, total), self.dtype)
        mask = np.empty(total, bool)
        offset = 0
        offsets = []
        for crop, size in zip(crops, sizes):
            pool[:, offset:offset + size] = crop["pixels"].reshape(self.channels, size)
            mask[offset:offset + size] = np.asarray(crop["mask"]).reshape(size)
            offsets.append(offset)
            offset += size
        name = str(self._shard)
        self._shard += 1
        chunk = max(1, -(-total // 16)) if total else 1
        for kind, array, chunks, shards in (
                ("pool", pool, (self.channels, chunk), (self.channels, chunk * 16)),
                ("mask", mask, (chunk,), (chunk * 16,))):
            parent = self.path / self.object_type / kind
            parent.mkdir(parents=True, exist_ok=True)
            tmp = parent / f".{name}.tmp"
            if tmp.exists():
                shutil.rmtree(tmp)
            group = self.zarr.open_group(str(tmp.parent), mode="a", zarr_format=3)
            out = group.create_array(
                tmp.name, shape=array.shape, dtype=array.dtype,
                chunks=chunks, shards=shards,
                compressors=[ZstdCodec(level=3)], fill_value=0,
                overwrite=True)
            if total:
                out[...] = array
            os.replace(tmp, parent / name)
        row_letter = "".join(ch for ch in self.well if ch.isalpha())
        column = "".join(ch for ch in self.well if ch.isdigit())
        for index, (crop, off) in enumerate(zip(crops, offsets)):
            height, width = crop["pixels"].shape[1:]
            self.rows.append({
                "object_type": self.object_type, "plate": self.plate,
                "row": row_letter, "column": int(column) if column else None,
                "well": self.well, "field": crop.get("field"),
                "timepoint": crop.get("timepoint"),
                "object_label": crop["object_label"],
                "object_id": crop.get("object_id"),
                "crop_store": self.relative.as_posix(),
                "crop_array": f"{self.object_type}/pool/{name}",
                "crop_index": index, "offset": off, "height": int(height),
                "width": int(width), "mask_offset": off,
                **{k: crop[k] for k in ("bbox_y0", "bbox_x0", "bbox_y1", "bbox_x1",
                                         "margin_px", "clipped_at_edge")},
                "png_path": None,
                **{k: v for k, v in crop.items() if k not in (
                    "pixels", "mask", "field", "timepoint", "object_label",
                    "object_id", "bbox_y0", "bbox_x0", "bbox_y1", "bbox_x1",
                    "margin_px", "clipped_at_edge")}})


def read_crop(project: Any, row: Mapping[str, Any], *, context: bool = True,
              size: Optional[Tuple[int, int]] = None
              ) -> Tuple[np.ndarray, np.ndarray]:
    """One crop and its mask from its ssCell_Crops row.

    :param project: the project folder.
    :param row: the crop's table row (``crop_store``, ``crop_array``,
        ``offset``, ``height``, ``width``, ``mask_offset``).
    :param context: keep the pixels around the object; False zeroes them.
    :param size: ``(h, w)`` to resize to, nearest-neighbour and therefore
        deterministic; the stored size when None.
    :returns: ``(pixels (C, h, w), mask (h, w) bool)``.
    """
    from .ome_zarr import require_zarr

    zarr = require_zarr()
    store = Path(project) / str(row["crop_store"])
    height, width = int(row["height"]), int(row["width"])
    n = height * width
    off = int(row["offset"])
    pool = zarr.open_array(str(store / str(row["crop_array"])), mode="r")
    pixels = np.asarray(pool[:, off:off + n]).reshape(pool.shape[0], height, width)
    mask_array = str(row["crop_array"]).replace("/pool/", "/mask/")
    moff = int(row["mask_offset"])
    mask = np.asarray(zarr.open_array(str(store / mask_array), mode="r")[
        moff:moff + n]).reshape(height, width)
    if not context:
        pixels = pixels.copy()
        pixels[:, ~mask] = 0
    if size is not None:
        ys = (np.arange(int(size[0])) * height // int(size[0])).astype(np.int64)
        xs = (np.arange(int(size[1])) * width // int(size[1])).astype(np.int64)
        pixels = pixels[:, ys][:, :, xs]
        mask = mask[ys][:, xs]
    return pixels, mask


def export_crops(project: Any, rows: Iterable[Mapping[str, Any]], out_dir: Any,
                 *, fmt: str = "tiff", context: bool = True) -> List[str]:
    """Write crops out of the pools as files other tools read (item 984).

    :param rows: ssCell_Crops rows.
    :param out_dir: the folder for the files.
    :param fmt: ``'tiff'`` (all channels, the mask as a second file),
        ``'npy'`` (``(C, h, w)``) or ``'png'`` (first three channels, 8-bit
        only, as PNG (Legacy) crops are).
    :returns: the files written.
    """
    fmt = str(fmt).lower()
    if fmt not in ("tiff", "npy", "png"):
        raise ValueError(f"fmt must be 'tiff', 'npy' or 'png'; got {fmt!r}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for row in rows:
        pixels, mask = read_crop(project, row, context=context)
        stem = (f"{row['plate']}_{row['well']}_{row.get('field')}_"
                f"{row['object_type']}_{row['object_label']}")
        if fmt == "npy":
            target = out / f"{stem}.npy"
            np.save(target, pixels)
        elif fmt == "tiff":
            import tifffile

            target = out / f"{stem}.tif"
            tifffile.imwrite(str(target), pixels, metadata={"axes": "CYX"})
            tifffile.imwrite(str(out / f"{stem}_mask.tif"), mask.astype(np.uint8))
        else:
            from PIL import Image

            if pixels.dtype != np.uint8:
                raise ValueError("png export takes 8-bit crops; use tiff or npy")
            rgb = np.moveaxis(pixels[:3], 0, -1)
            target = out / f"{stem}.png"
            Image.fromarray(rgb if rgb.shape[-1] == 3 else rgb[..., 0]).save(target)
        written.append(str(target))
    return written
