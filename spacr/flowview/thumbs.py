"""Qt-free thumbnail preparation and bounded on-disk caching."""

from __future__ import annotations

import hashlib
import io
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

MAX_THUMBNAIL_AXIS = 128
DEFAULT_CACHE_BYTES = 50 * 1024 * 1024


def _load_array(image: str | os.PathLike[str] | Image.Image | Any) -> np.ndarray:
    if isinstance(image, (str, os.PathLike)):
        with Image.open(image) as loaded:
            return np.asarray(loaded.convert("RGBA") if loaded.mode == "P" else loaded).copy()
    if isinstance(image, Image.Image):
        converted = image.convert("RGBA") if image.mode == "P" else image
        return np.asarray(converted).copy()
    return np.asarray(image)


def _stretch_channel(channel: np.ndarray) -> np.ndarray:
    values = np.asarray(channel, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, dtype=np.uint8)
    low, high = np.percentile(values[finite], (2.0, 98.0))
    if high > low:
        scaled = (values - low) * (255.0 / (high - low))
    elif 0.0 <= low <= 1.0:
        scaled = values * 255.0
    else:
        scaled = values
    scaled = np.where(finite, scaled, 0.0)
    return np.clip(np.rint(scaled), 0.0, 255.0).astype(np.uint8)


def _alpha_channel(channel: np.ndarray) -> np.ndarray:
    values = np.asarray(channel, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, dtype=np.uint8)
    usable = values[finite]
    if usable.min() >= 0.0 and usable.max() <= 1.0:
        values = values * 255.0
    values = np.where(finite, values, 0.0)
    return np.clip(np.rint(values), 0.0, 255.0).astype(np.uint8)


def _contrast_stretch(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return _stretch_channel(array)
    if array.ndim != 3 or array.shape[2] not in (1, 2, 3, 4):
        raise ValueError("thumbnail images must be HxW, HxWx1, HxWx2, HxWx3, or HxWx4")
    if array.shape[2] == 1:
        return _stretch_channel(array[..., 0])
    colour_channels = 1 if array.shape[2] in (1, 2) else 3
    channels = [
        _stretch_channel(array[..., index]) for index in range(colour_channels)
    ]
    if array.shape[2] in (2, 4):
        channels.append(_alpha_channel(array[..., -1]))
    return np.stack(channels, axis=-1)


def _outline_pixels(labels: np.ndarray) -> np.ndarray:
    padded = np.pad(labels, 1, mode="constant", constant_values=0)
    centre = padded[1:-1, 1:-1]
    return (centre != 0) & (
        (centre != padded[:-2, 1:-1])
        | (centre != padded[2:, 1:-1])
        | (centre != padded[1:-1, :-2])
        | (centre != padded[1:-1, 2:])
    )


def make_thumbnail(
    image: str | os.PathLike[str] | Image.Image | Any,
    *,
    outline_mask: Any | None = None,
    max_size: int = MAX_THUMBNAIL_AXIS,
) -> Image.Image:
    """Return a contrast-stretched thumbnail no larger than 128 pixels.

    ``outline_mask`` may contain binary or labelled segmentation data.  Its
    boundaries are overlaid as one-pixel neutral-white lines after nearest-
    neighbour downsampling; filled regions are never painted.
    """

    if not 1 <= int(max_size) <= MAX_THUMBNAIL_AXIS:
        raise ValueError("max_size must be between 1 and 128 pixels")
    array = _load_array(image)
    if array.ndim < 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("thumbnail images must have non-empty height and width")
    stretched = _contrast_stretch(array)
    thumbnail = Image.fromarray(stretched)
    thumbnail.thumbnail((int(max_size), int(max_size)), Image.Resampling.LANCZOS)

    if outline_mask is not None:
        labels = np.asarray(outline_mask)
        if labels.ndim != 2 or labels.shape != array.shape[:2]:
            raise ValueError("outline_mask must be two-dimensional and match the image")
        labels_image = Image.fromarray(labels.astype(np.int32), mode="I")
        labels_image = labels_image.resize(thumbnail.size, Image.Resampling.NEAREST)
        outline = _outline_pixels(np.asarray(labels_image))
        pixels = np.asarray(thumbnail).copy()
        if pixels.ndim == 2:
            pixels[outline] = 255
        else:
            pixels[outline, : min(3, pixels.shape[2])] = 255
            if pixels.shape[2] in (2, 4):
                pixels[outline, -1] = 255
        thumbnail = Image.fromarray(pixels)
    return thumbnail


def thumbnail_png(
    image: str | os.PathLike[str] | Image.Image | Any,
    *,
    outline_mask: Any | None = None,
    max_size: int = MAX_THUMBNAIL_AXIS,
) -> bytes:
    """Encode :func:`make_thumbnail` as deterministic PNG bytes."""

    output = io.BytesIO()
    make_thumbnail(
        image,
        outline_mask=outline_mask,
        max_size=max_size,
    ).save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


class ThumbnailCache:
    """A configurable oldest-first disk cache of PNG thumbnails."""

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        max_bytes: int = DEFAULT_CACHE_BYTES,
        max_size: int = MAX_THUMBNAIL_AXIS,
    ) -> None:
        if int(max_bytes) <= 0:
            raise ValueError("max_bytes must be greater than zero")
        if not 1 <= int(max_size) <= MAX_THUMBNAIL_AXIS:
            raise ValueError("max_size must be between 1 and 128 pixels")
        self.directory = Path(directory)
        self.max_bytes = int(max_bytes)
        self.max_size = int(max_size)
        self._lock = threading.RLock()
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: object) -> Path:
        """Return the traversal-safe cache path assigned to *key*."""

        digest = hashlib.sha256(str(key).encode("utf-8")).hexdigest()
        return self.directory / f"{digest}.png"

    def get(self, key: object) -> Path | None:
        """Return an existing thumbnail path without changing its age."""

        path = self.path_for(key)
        return path if path.is_file() else None

    def store(
        self,
        key: object,
        image: str | os.PathLike[str] | Image.Image | Any,
        *,
        outline_mask: Any | None = None,
    ) -> Path:
        """Prepare and cache a thumbnail, evicting the oldest files first."""

        payload = thumbnail_png(
            image,
            outline_mask=outline_mask,
            max_size=self.max_size,
        )
        if len(payload) > self.max_bytes:
            raise ValueError("encoded thumbnail exceeds the cache byte limit")
        path = self.path_for(key)
        with self._lock:
            path.write_bytes(payload)
            os.utime(path, None)
            self._evict_oldest(protect=path)
        return path

    put = store

    @property
    def total_bytes(self) -> int:
        """Current size of the PNG files owned by this cache."""

        with self._lock:
            return sum(path.stat().st_size for path in self.directory.glob("*.png"))

    def _evict_oldest(self, *, protect: Path) -> None:
        entries = sorted(
            (path for path in self.directory.glob("*.png") if path != protect),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
        )
        total = sum(path.stat().st_size for path in entries) + protect.stat().st_size
        for path in entries:
            if total <= self.max_bytes:
                break
            size = path.stat().st_size
            path.unlink()
            total -= size

    def clear(self) -> int:
        """Delete this cache's PNG entries and return the number removed."""

        with self._lock:
            paths = list(self.directory.glob("*.png"))
            for path in paths:
                path.unlink()
            return len(paths)

    def discard(self) -> int:
        """Clear the run cache and remove its directory when it is empty."""

        removed = self.clear()
        try:
            self.directory.rmdir()
        except OSError:
            pass
        return removed


encode_thumbnail = thumbnail_png

__all__ = [
    "DEFAULT_CACHE_BYTES",
    "MAX_THUMBNAIL_AXIS",
    "ThumbnailCache",
    "encode_thumbnail",
    "make_thumbnail",
    "thumbnail_png",
]
