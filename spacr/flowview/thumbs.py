"""Qt-free thumbnail preparation and bounded on-disk caching."""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

MAX_THUMBNAIL_AXIS = 128
DEFAULT_CACHE_BYTES = 50 * 1024 * 1024
_HEX_DIGITS = frozenset("0123456789abcdef")


def _load_array(image: str | os.PathLike[str] | Image.Image | Any) -> np.ndarray:
    """Load *image* into an array detached from file and Pillow storage.

    :param image: image path, Pillow image, or array-like pixel data.
    :returns: pixel array; palette images are expanded to RGBA.
    """

    if isinstance(image, (str, os.PathLike)):
        with Image.open(image) as loaded:
            return np.asarray(loaded.convert("RGBA") if loaded.mode == "P" else loaded).copy()
    if isinstance(image, Image.Image):
        converted = image.convert("RGBA") if image.mode == "P" else image
        return np.asarray(converted).copy()
    return np.asarray(image)


def _stretch_channel(channel: np.ndarray) -> np.ndarray:
    """Scale one intensity channel to unsigned 8-bit display values.

    :param channel: numeric channel; non-finite values are rendered as zero.
    :returns: 8-bit values stretched between the finite 2nd and 98th percentiles.
    """

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
    """Convert an alpha channel to unsigned 8-bit opacity.

    :param channel: numeric opacity values, either normalized or in byte scale.
    :returns: clipped 8-bit opacity, with non-finite values made transparent.
    """

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
    """Prepare a supported grayscale or multichannel array for Pillow.

    :param array: ``HxW`` or ``HxWxC`` pixels, where ``C`` is 1 through 4.
    :returns: contrast-stretched 8-bit pixels with alpha preserved separately.
    :raises ValueError: when the array does not have a supported channel layout.
    """

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
    """Locate four-connected boundaries of nonzero labels.

    :param labels: two-dimensional binary or labelled segmentation mask.
    :returns: Boolean mask selecting foreground pixels beside another label.
    """

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
    """Return a contrast-stretched thumbnail within the requested edge limit.

    ``outline_mask`` may contain binary or labelled segmentation data.  Its
    boundaries are overlaid as one-pixel neutral-white lines after nearest-
    neighbour downsampling; filled regions are never painted.

    :param image: image path, Pillow image, or array-like pixel data.
    :param outline_mask: optional two-dimensional mask matching the source size.
    :param max_size: maximum width or height, capped by ``MAX_THUMBNAIL_AXIS``.
    :returns: detached Pillow image containing the rendered thumbnail.
    :raises ValueError: when the size, image shape, or mask shape is invalid.
    """

    normalised_max_size = int(max_size)
    if not 1 <= normalised_max_size <= MAX_THUMBNAIL_AXIS:
        raise ValueError("max_size must be between 1 and 128 pixels")
    array = _load_array(image)
    if array.ndim < 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("thumbnail images must have non-empty height and width")
    stretched = _contrast_stretch(array)
    thumbnail = Image.fromarray(stretched)
    thumbnail.thumbnail(
        (normalised_max_size, normalised_max_size),
        Image.Resampling.LANCZOS,
    )

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
    """Encode :func:`make_thumbnail` as deterministic PNG bytes.

    :param image: image path, Pillow image, or array-like pixel data.
    :param outline_mask: optional two-dimensional mask matching the source size.
    :param max_size: maximum width or height of the encoded thumbnail.
    :returns: complete PNG file contents suitable for atomic publication.
    """

    output = io.BytesIO()
    make_thumbnail(
        image,
        outline_mask=outline_mask,
        max_size=max_size,
    ).save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


class ThumbnailCache:
    """A configurable oldest-first disk cache of PNG thumbnails.

    :param directory: where the thumbnails are written. Created if missing.
    :param max_bytes: the cache's size budget. When writing would exceed it,
        the oldest entries are evicted until it fits -- oldest-first rather
        than least-recently-used, because a thumbnail is cheap to regenerate
        and tracking access times costs a write on every read.
    :param max_size: the longest edge, in pixels, of a generated thumbnail.
        Capped at ``MAX_THUMBNAIL_AXIS``.
    :raises ValueError: when ``max_bytes`` is not positive, or ``max_size`` is
        outside 1..``MAX_THUMBNAIL_AXIS``.

    File publication is atomic across cache instances, so concurrent readers
    see either the old or new complete PNG. Budgeting and deletion are guarded
    only within one instance; use one mutating instance per directory.
    """

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        max_bytes: int = DEFAULT_CACHE_BYTES,
        max_size: int = MAX_THUMBNAIL_AXIS,
    ) -> None:
        """Create a cache directory with byte and pixel limits.

        :param directory: directory reserved for generated thumbnail entries.
        :param max_bytes: positive total byte budget for cache-owned PNG files.
        :param max_size: generated thumbnail's maximum width or height.
        :raises ValueError: when a normalized limit falls outside its range.
        """

        normalised_max_bytes = int(max_bytes)
        normalised_max_size = int(max_size)
        if normalised_max_bytes <= 0:
            raise ValueError("max_bytes must be greater than zero")
        if not 1 <= normalised_max_size <= MAX_THUMBNAIL_AXIS:
            raise ValueError("max_size must be between 1 and 128 pixels")
        self.directory = Path(directory)
        self.max_bytes = normalised_max_bytes
        self.max_size = normalised_max_size
        self._lock = threading.RLock()
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: object) -> Path:
        """Return the traversal-safe cache path assigned to *key*.

        :param key: cache identity, normalized with ``str`` before hashing.
        :returns: path whose filename is a lowercase SHA-256 digest plus ``.png``.

        Keys with identical string representations intentionally share a path.
        """

        digest = hashlib.sha256(str(key).encode("utf-8")).hexdigest()
        return self.directory / f"{digest}.png"

    def get(self, key: object) -> Path | None:
        """Return an existing thumbnail path without changing its age.

        :param key: cache identity accepted by :meth:`path_for`.
        :returns: cache-owned path when it exists, otherwise ``None``.
        """

        path = self.path_for(key)
        with self._lock:
            return path if path.is_file() else None

    def store(
        self,
        key: object,
        image: str | os.PathLike[str] | Image.Image | Any,
        *,
        outline_mask: Any | None = None,
    ) -> Path:
        """Prepare and atomically cache a thumbnail.

        :param key: cache identity accepted by :meth:`path_for`.
        :param image: image path, Pillow image, or array-like pixel data.
        :param outline_mask: optional two-dimensional mask matching the source.
        :returns: path of the published PNG after oldest-first eviction.
        :raises ValueError: when the encoded PNG exceeds the byte budget.
        """

        payload = thumbnail_png(
            image,
            outline_mask=outline_mask,
            max_size=self.max_size,
        )
        if len(payload) > self.max_bytes:
            raise ValueError("encoded thumbnail exceeds the cache byte limit")
        path = self.path_for(key)
        with self._lock:
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=self.directory,
                    prefix=f".{path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as handle:
                    temporary = Path(handle.name)
                    handle.write(payload)
                os.replace(temporary, path)
                temporary = None
                self._evict_oldest(protect=path)
            finally:
                if temporary is not None:
                    try:
                        temporary.unlink()
                    except OSError:
                        pass
        return path

    put = store

    @property
    def total_bytes(self) -> int:
        """Return the byte size of this cache's generated PNG entries."""

        with self._lock:
            return sum(path.stat().st_size for path in self._owned_paths())

    def _owned_paths(self) -> list[Path]:
        """Return regular PNG files in this cache's SHA-256 namespace."""

        return [
            path
            for path in self.directory.glob("*.png")
            if path.is_file()
            and len(path.stem) == 64
            and all(character in _HEX_DIGITS for character in path.stem)
        ]

    def _evict_oldest(self, *, protect: Path) -> None:
        """Remove oldest owned entries until the cache meets its byte budget.

        :param protect: newly published entry that must survive this eviction.
        """

        entries = sorted(
            (path for path in self._owned_paths() if path != protect),
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
        """Delete generated PNG entries and return the number removed.

        Foreign files, including PNGs outside the SHA-256 namespace, remain.
        """

        with self._lock:
            paths = self._owned_paths()
            for path in paths:
                path.unlink()
            return len(paths)

    def discard(self) -> int:
        """Clear the cache and remove its directory when empty.

        :returns: number of generated PNG entries removed.

        A successful directory removal makes this instance terminal; create a
        new cache before storing another thumbnail.
        """

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
