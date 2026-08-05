"""Decode object crops lazily, and keep the last few — the hover seam.

Showing the crop under the cursor is the difference between a scatter plot of
ninety thousand dots and a scatter plot you can interrogate. It is also the
easiest way to make a plot unusable: a PNG decode is a handful of milliseconds,
a mouse sweep is a hundred hover events a second, and the two multiply into a
plot that lags a quarter-second behind the cursor and feels broken.

So nothing here decodes on the paint path, and nothing decodes twice:

* :meth:`CropThumbnails.peek` answers from memory or not at all. It is what a
  hover handler calls, because the answer has to be instant or absent.
* :meth:`CropThumbnails.pixmap` decodes on a miss. A caller runs it behind a
  short debounce (or on a click), never once per mouse-move event.
* The cache is keyed on ``(path, mtime, size, px)``, following
  :mod:`spacr.crops`, so a crop re-written by a re-run is re-read rather than
  served stale from the last session's decode.

Why not ``QPixmap(path)``
-------------------------

Crop PNGs are not ordinary images. Anything spaCR wrote before the BGR fix has
the first stain in the *blue* channel, and a 16-bit single-channel crop opened
with ``convert('RGB')`` is clipped to solid white by PIL. Both are corrected by
:func:`spacr.qt.annotate_engine.load_crop_image`, which is therefore the only
door used here — a hover preview showing different colours from the annotation
grid is worse than no hover preview, because both look plausible.
"""
from __future__ import annotations

import logging
import os
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

from PySide6.QtGui import QPixmap

LOG = logging.getLogger(__name__)

__all__ = [
    "CropThumbnails",
    "resolve_crop_path",
    "DEFAULT_CAPACITY",
    "DEFAULT_SIZE",
]

#: How many decoded thumbnails to keep. Sized for a hover sweep across a
#: cluster rather than for the whole plot: at 192 px RGBA that is ~14 MB, and
#: the working set of "the points I am moving the mouse over" is small even
#: when the plot is not.
DEFAULT_CAPACITY = 96

#: The longest edge of a cached thumbnail, in pixels. Big enough to see a
#: parasite in, small enough that a hundred of them do not cost real memory.
DEFAULT_SIZE = 192


def resolve_crop_path(path: str, db_path: str = "") -> str:
    """The crop's path on *this* machine, re-anchored if the dataset moved.

    ``png_list`` stores absolute paths built at measure time, so a dataset
    copied to another disk resolves to nothing and every preview is blank.
    :func:`spacr.qt.screens.annotate._reanchor_png_path` already knows how to
    rebuild one from the ``/data/`` segment under the database's own root;
    this borrows it rather than growing a third copy of that rule, and falls
    back to the stored path when the Annotate screen is not importable (a
    headless test, a trimmed install).
    """
    text = str(path or "")
    if not text or os.path.isfile(text) or not db_path:
        return text
    try:
        from .screens.annotate import _reanchor_png_path
    except Exception:
        LOG.debug("cannot re-anchor crop paths without the Annotate screen",
                  exc_info=True)
        return text
    try:
        return _reanchor_png_path(text, str(db_path))
    except Exception:
        LOG.debug("could not re-anchor %s", text, exc_info=True)
        return text


class CropThumbnails:
    """A bounded LRU of decoded crop thumbnails.

    :param db_path: the ``measurements.db`` the crops belong to. Used to
        re-anchor moved paths and to resolve the folder's stored channel
        order; optional, because a folder of crops is a legitimate source too.
    :param size: longest edge of the thumbnails, in pixels.
    :param capacity: how many to keep.

    Not a ``QObject``, and holds no widget: a screen can own one, a test can
    drive one, and neither keeps the other alive.
    """

    def __init__(self, db_path: str = "", *, size: int = DEFAULT_SIZE,
                 capacity: int = DEFAULT_CAPACITY):
        self.db_path = str(db_path or "")
        self.size = max(16, int(size))
        self.capacity = max(1, int(capacity))
        self._cache: "OrderedDict[Tuple[Any, ...], Optional[QPixmap]]" = \
            OrderedDict()
        #: Counters, for a status line and for the tests that assert a hover
        #: sweep decoded each crop once rather than once per mouse event.
        self.hits = 0
        self.misses = 0
        self.decodes = 0
        self.failures = 0

    # -- keys ---------------------------------------------------------------
    def _key(self, path: str) -> Tuple[Any, ...]:
        """``(abspath, mtime_ns, size, px)`` — the identity of one decode.

        Stat is included so a re-run that rewrote the crop invalidates the
        entry. A crop that cannot be stat'ed still gets a key (with zeros), so
        a missing file is remembered as missing rather than re-attempted on
        every hover.
        """
        resolved = os.path.abspath(path) if path else ""
        try:
            stat = os.stat(resolved)
            stamp: Tuple[Any, ...] = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            stamp = (0, 0)
        return (resolved, stamp[0], stamp[1], self.size)

    # -- reading ------------------------------------------------------------
    def peek(self, path: str) -> Optional[QPixmap]:
        """The thumbnail if it is already decoded, else ``None``. Never blocks.

        The call a mouse-move handler makes. Returning ``None`` means "not
        yet", not "there is no crop" — ask :meth:`pixmap` for that, off the
        hover path.
        """
        if not path:
            return None
        key = self._key(str(path))
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        self.hits += 1
        return self._cache[key]

    def pixmap(self, path: str) -> Optional[QPixmap]:
        """The thumbnail, decoding it if this is the first time.

        :returns: the ``QPixmap``, or ``None`` when the crop cannot be read.
            A failure is *cached* as ``None`` rather than raised: a missing
            file under the cursor must not throw out of a mouse handler, and
            it must not be retried sixty times a second either.
        """
        if not path:
            return None
        key = self._key(str(path))
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        pixmap = self._decode(str(path))
        self._store(key, pixmap)
        return pixmap

    def _decode(self, path: str) -> Optional[QPixmap]:
        """Read one crop through the corrected reader, or ``None``."""
        self.decodes += 1
        resolved = resolve_crop_path(path, self.db_path)
        try:
            from PIL.ImageQt import ImageQt

            from .annotate_engine import load_crop_image

            image = load_crop_image(resolved, db_path=self.db_path or None)
            image.thumbnail((self.size, self.size))
            return QPixmap.fromImage(ImageQt(image).copy())
        except Exception:
            self.failures += 1
            LOG.debug("could not decode the crop %s", resolved, exc_info=True)
            return None

    def _store(self, key: Tuple[Any, ...],
               pixmap: Optional[QPixmap]) -> None:
        self._cache[key] = pixmap
        self._cache.move_to_end(key)
        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)

    # -- housekeeping -------------------------------------------------------
    def prime(self, path: str) -> Optional[QPixmap]:
        """Decode ``path`` now so a later :meth:`peek` is instant.

        The same as :meth:`pixmap`; named separately because the intent at the
        call site is different — this is what a debounce timer runs, and
        reading it as "prime the cache" rather than "get the pixmap" is what
        stops it drifting back onto the hover path.
        """
        return self.pixmap(path)

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, path: object) -> bool:
        return self._key(str(path)) in self._cache

    def clear(self) -> None:
        """Drop everything. For a new source, or a screen closing."""
        self._cache.clear()

    def describe(self) -> str:
        """One line of cache health, for a status bar."""
        return (f"{len(self._cache)}/{self.capacity} crops cached · "
                f"{self.decodes} decode(s), {self.hits} hit(s), "
                f"{self.failures} unreadable")


def crop_paths_for_keys(db_path: str, keys) -> Dict[str, str]:
    """``{object key: crop path}``, resolved ONCE for a whole plot.

    :func:`spacr.active_learning.crops_for_object_keys` scans the crop table
    per call — right for opening a subset, ruinous once per hover event. This
    is the plot-time call: resolve every plotted point up front and let hover
    be a dict lookup.

    That function keeps the caller's order but *drops* keys it cannot resolve,
    so when everything resolves the answer is a zip and costs one scan. When
    some keys miss, the returned list is a subsequence and no longer lines up,
    so the range is bisected: each half is asked separately until a half is
    either fully resolved (zip it) or a single key (resolve it or record the
    miss). That is a handful of extra scans for a handful of missing crops,
    rather than one scan per key.

    Keys with no crop are absent from the result rather than mapped to ``""``
    — "this object has no crop" and "this object's crop is at nowhere" are
    different claims and only the first one is true.
    """
    wanted = [str(k) for k in keys]
    if not wanted or not db_path:
        return {}
    from ..active_learning import crops_for_object_keys

    out: Dict[str, str] = {}

    def resolve(batch) -> None:
        if not batch:
            return
        rows = crops_for_object_keys(db_path, batch)
        if len(rows) == len(batch):
            for key, (path, _annotation) in zip(batch, rows):
                out[key] = path
            return
        if len(batch) == 1:
            if rows:
                out[batch[0]] = rows[0][0]
            return
        middle = len(batch) // 2
        resolve(batch[:middle])
        resolve(batch[middle:])

    resolve(wanted)
    return out
