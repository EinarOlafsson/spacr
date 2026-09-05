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
import sys
import time
import weakref
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


# Weak registration makes every live thumbnail cache observable to the
# process-wide budget without making the cleanup service the reason a closed
# screen stays alive.
_LIVE_CACHES: "weakref.WeakSet[CropThumbnails]" = weakref.WeakSet()


def live_thumbnail_caches():
    """A snapshot of the decoded-thumbnail caches that still have owners."""
    return tuple(_LIVE_CACHES)


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
        """Create the thumbnail cache.

        :param db_path: measurements database the crops are read from.
        :param size: longest edge of a decoded thumbnail, in pixels.
        :param capacity: how many thumbnails to keep; the least recently used
            are dropped first.
        """
        self.db_path = str(db_path or "")
        self.size = max(16, int(size))
        self.capacity = max(1, int(capacity))
        self._cache: "OrderedDict[Tuple[Any, ...], Optional[QPixmap]]" = \
            OrderedDict()
        self._last_used: Dict[Tuple[Any, ...], float] = {}
        self._bytes: Dict[Tuple[Any, ...], int] = {}
        #: Counters, for a status line and for the tests that assert a hover
        #: sweep decoded each crop once rather than once per mouse event.
        self.hits = 0
        self.misses = 0
        self.decodes = 0
        self.failures = 0
        _LIVE_CACHES.add(self)
        cleanup = sys.modules.get("spacr.qt.resource_cleanup")
        install = getattr(cleanup, "install_budget_sweep", None)
        if callable(install):
            install()

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
        self._last_used[key] = time.time()
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
            self._last_used[key] = time.time()
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
        """Put one decoded thumbnail in the cache and evict down to capacity.

        A ``None`` is cached too: a crop that could not be decoded must not be
        retried on every hover.

        :param key: the cache key.
        :param pixmap: the decoded thumbnail, or ``None`` for a failed decode.
        """
        self._cache[key] = pixmap
        self._cache.move_to_end(key)
        self._last_used[key] = time.time()
        self._bytes[key] = self._pixmap_bytes(pixmap)
        while len(self._cache) > self.capacity:
            old, _ = self._cache.popitem(last=False)
            self._last_used.pop(old, None)
            self._bytes.pop(old, None)

    @staticmethod
    def _pixmap_bytes(pixmap: Optional[QPixmap]) -> int:
        """Storage represented by a pixmap, without copying its pixels."""
        if pixmap is None or pixmap.isNull():
            return 0
        try:
            return max(0, int(pixmap.width()) * int(pixmap.height())
                       * int(pixmap.depth()) // 8)
        except (AttributeError, TypeError, ValueError):
            return 0

    def cache_budget_entries(self):
        """``(key, bytes, last use, in use)`` rows for the global sweep.

        A QLabel/QGraphicsItem that is displaying a pixmap owns its own
        implicitly-shared Qt value.  Removing this lookup entry cannot blank
        that control, so no thumbnail entry needs pinning here.
        """
        now = time.time()
        return [
            (key, int(self._bytes.get(key, self._pixmap_bytes(pixmap))),
             float(self._last_used.get(key, now)), False)
            for key, pixmap in list(self._cache.items())
        ]

    def drop_cache_budget_entry(self, key) -> bool:
        """Evict one decoded thumbnail selected by the memory policy."""
        existed = key in self._cache
        self._cache.pop(key, None)
        self._last_used.pop(key, None)
        self._bytes.pop(key, None)
        return existed

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
        """Return how many entries the cache holds."""
        return len(self._cache)

    def __contains__(self, path: object) -> bool:
        """Report whether a crop path is cached.

        :param path: the crop path; coerced with :func:`str`.
        :returns: ``True`` if it has an entry, including a cached failure.
        """
        return self._key(str(path)) in self._cache

    def clear(self) -> None:
        """Drop everything. For a new source, or a screen closing."""
        self._cache.clear()
        self._last_used.clear()
        self._bytes.clear()

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
        # Never called with an empty batch: the caller has already refused
        # an empty key list and a bisection of two or more keys cannot
        # produce an empty half.
        """Resolve one batch of keys, bisecting when some are missing.

        Never called with an empty batch: the caller refuses an empty key list
        and a bisection of two or more cannot produce an empty half.
        """
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
