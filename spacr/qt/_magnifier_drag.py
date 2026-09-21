"""What a press-and-drag with the live magnifier adds: the stroke's arithmetic.

Built on the live magnifier
(:class:`spacr.qt.screens.make_masks._LiveMagnifier`). Pressing and pulling
with the magnifier on keeps segmenting boxes along the path and applying what
they find, and the objects the cursor passes over become ONE object.

This module is that arithmetic and nothing else -- no Qt, no model, no thread
-- so every rule below is tested on plain arrays. The magnifier asks its
worker for the boxes, hands their answers in here, and gives what comes out
to the screen, which owns the mask, the undo history and the ledger.

WORDS

    path    every image pixel the cursor passed over: successive mouse
            positions joined by straight lines, so a fast pull whose events
            skip pixels still passes over what lies between them.
    frame   one box's answer: a label image in box coordinates, and the box
            ``(x0, y0, x1, y1)`` in image pixels. Frames along a drag overlap.
    piece   one object of one frame.

THE RULES

    1. A piece the path passes over is TOUCHED. Every touched piece, from
       every frame, goes into one object.
    2. A piece that has at least half of the smaller of the two in common
       with an object already gathered -- counted inside the new frame's box,
       the only place both frames looked -- is that object seen again, and
       joins it. This is what merges one object's frames: a cell longer than
       the box arrives as overlapping pieces, one per box, and leaves as one
       object. Neighbours whose outlines disagree by a few pixels stay apart.
    3. Every other object is added as an object of its own when
       ``keep_untouched`` (all objects in the zoom area) and left out when
       not (only objects touching the mouse).
    4. A pixel two objects both claim stays with the one that claimed it
       first. Nothing is moved: every object lands on the pixels its frames
       found it on.
"""
from __future__ import annotations

from typing import List, NamedTuple, Optional, Tuple

import numpy as np
from scipy import ndimage

#: Rule 2: the share of the smaller of a piece and an object already gathered
#: that the two must have in common to be one object.
_SAME_OBJECT_SHARE = 0.5

#: The longest the screen waits, in milliseconds, before showing what a drag
#: adds so far. Frames and touches arriving faster than this are shown
#: together, so a fast pull repaints the mask at most 25 times a second.
_PREVIEW_MS = 40


def _frame_step(size: int) -> int:
    """How far the cursor travels between frames for a box of side ``size``.

    A quarter of the box: every pixel of the path then lies in the middle half
    of some frame, where the box's own edge does not cut the object under it.
    """
    return max(1, int(size) // 4)


class _StrokeOutcome(NamedTuple):
    """What a stroke adds, as it stands.

    ``labels`` is an int32 label image whose top-left pixel is image pixel
    ``origin``. Value 1 is the object the path passed over when there is one;
    every other positive value is an object of its own. ``merged`` is how many
    pieces went into the object the path passed over, ``objects`` how many
    objects ``labels`` holds and ``frames`` how many frames were taken in.
    """

    labels: np.ndarray
    origin: Tuple[int, int]
    merged: int
    objects: int
    frames: int


def _line_pixels(start, end) -> np.ndarray:
    """Every pixel on the straight line from ``start`` to ``end``.

    :returns: ``(N, 2)`` int64 rows of ``(x, y)``, both ends included, one row
        per pixel along the longer axis, so the line has no gaps.
    """
    x0, y0 = (int(v) for v in start)
    x1, y1 = (int(v) for v in end)
    along = np.linspace(0.0, 1.0, max(abs(x1 - x0), abs(y1 - y0)) + 1)
    xs = np.rint(x0 + (x1 - x0) * along).astype(np.int64)
    ys = np.rint(y0 + (y1 - y0) * along).astype(np.int64)
    return np.stack([xs, ys], axis=1)


class _DragStroke:
    """One press-and-drag: its path, the frames it waits for, what it adds.

    :param shape: the image's ``(height, width)``. Points are clipped into it.
    :param start: the pixel pressed on, ``(x, y)``. The caller asks for the
        first frame there.
    :param step: how far the cursor travels, in image pixels along either
        axis, before :meth:`extend` wants another frame (see
        :func:`_frame_step`); 0 never wants one, for a stroke whose single
        frame is the whole image.
    :param keep_untouched: rule 3 -- True adds every object the frames found,
        False only the object the path passed over.
    """

    def __init__(self, shape, start, *, step: int,
                 keep_untouched: bool = True):
        """Start a stroke at ``start`` with no frame and nothing to add."""
        self.shape = (int(shape[0]), int(shape[1]))
        self.step = max(0, int(step))
        self.keep_untouched = bool(keep_untouched)
        #: Whether the button has come up.
        self.released = False
        #: Whether what the stroke adds changed since :meth:`outcome` said.
        self.dirty = False
        start = self._clip(start)
        self._last = start
        self._centre = start
        self._path: List[np.ndarray] = [np.array([start], dtype=np.int64)]
        self._waiting: set = set()
        self._frames = 0
        #: The piece that claimed each image pixel, 0 for none (rule 4).
        #: Made when the first frame arrives.
        self._owner: Optional[np.ndarray] = None
        #: Union-find parents over piece ids, which start at 1. An object is
        #: named by its oldest piece, so ids in order are objects in order.
        self._parent: List[int] = [0]
        self._touched: set = set()
        self._span: Tuple[int, int, int, int] = (0, 0, 0, 0)

    def _clip(self, point) -> Tuple[int, int]:
        """``point`` as whole pixels inside the image."""
        height, width = self.shape
        return (min(max(int(point[0]), 0), width - 1),
                min(max(int(point[1]), 0), height - 1))


    def extend(self, point) -> List[Tuple[int, int]]:
        """Follow the cursor to ``point``; return where frames are now wanted.

        The line from the last point joins the path, and pieces already taken
        in that it passes over are touched. A frame is wanted at each pixel of
        the line ``step`` or more from the last frame's centre, so a jump
        several steps long wants several and no stretch of the path goes
        unseen.
        """
        point = self._clip(point)
        if point == self._last:
            return []
        line = _line_pixels(self._last, point)[1:]
        self._last = point
        self._path.append(line)
        self._touch_owned(line)
        wanted: List[Tuple[int, int]] = []
        if self.step:
            cx, cy = self._centre
            for x, y in line.tolist():
                if max(abs(x - cx), abs(y - cy)) >= self.step:
                    cx, cy = x, y
                    wanted.append((x, y))
            self._centre = (cx, cy)
        return wanted

    def _touch_owned(self, pixels: np.ndarray) -> None:
        """Touch every piece that owns one of ``pixels`` (rule 1)."""
        if self._owner is None:
            return
        owners = np.unique(self._owner[pixels[:, 1], pixels[:, 0]])
        fresh = {int(v) for v in owners[owners > 0]} - self._touched
        if fresh:
            self._touched |= fresh
            self.dirty = True

    def release(self) -> None:
        """The button came up. Frames still on their way are still taken in."""
        self.released = True


    def expect(self, key) -> None:
        """Wait for the frame asked for under ``key``."""
        self._waiting.add(key)

    def drop(self, key) -> None:
        """Stop waiting for ``key``, whose frame could not be segmented."""
        self._waiting.discard(key)

    def waiting(self) -> int:
        """How many frames asked for have not arrived."""
        return len(self._waiting)

    def ready(self) -> bool:
        """Whether the button is up and every frame asked for has arrived."""
        return self.released and not self._waiting

    def deliver(self, key, labels, box) -> bool:
        """Take in the frame asked for under ``key``.

        :param labels: the frame's label image, shaped like its box.
        :param box: ``(x0, y0, x1, y1)`` in image pixels, ends exclusive.
        :returns: False, taking nothing in, when no frame waits under ``key``.
        """
        if key not in self._waiting:
            return False
        self._waiting.discard(key)
        self._take_in(np.asarray(labels).astype(np.int32, copy=False), box)
        return True

    def _take_in(self, labels: np.ndarray, box) -> None:
        """Apply rules 1, 2 and 4 to one frame's pieces."""
        x0, y0, x1, y1 = (int(v) for v in box[:4])
        if self._owner is None:
            self._owner = np.zeros(self.shape, dtype=np.int32)
            self._span = (x0, y0, x1, y1)
        sx0, sy0, sx1, sy1 = self._span
        self._span = (min(sx0, x0), min(sy0, y0), max(sx1, x1), max(sy1, y1))
        self._frames += 1
        window = self._owner[y0:y1, x0:x1]
        roots = self._roots()
        before = roots[window]
        area = np.bincount(before.ravel(), minlength=roots.size)
        under = self._under_path(labels, (x0, y0, x1, y1))
        for value, where in enumerate(ndimage.find_objects(labels), start=1):
            if where is None:
                continue
            body = labels[where] == value
            piece = len(self._parent)
            self._parent.append(piece)
            if value in under:
                self._touched.add(piece)
            shared = np.bincount(before[where][body], minlength=roots.size)
            size = int(np.count_nonzero(body))
            for root in np.flatnonzero(shared[1:]) + 1:
                if shared[root] >= _SAME_OBJECT_SHARE * min(size, area[root]):
                    self._join(piece, int(root))
            claim = window[where]
            claim[body & (claim == 0)] = piece
        self.dirty = True

    def _under_path(self, labels: np.ndarray, box) -> set:
        """The labels of one frame that the path passes over, inside its box."""
        x0, y0, x1, y1 = box
        path = np.concatenate(self._path)
        xs, ys = path[:, 0], path[:, 1]
        inside = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
        hits = labels[ys[inside] - y0, xs[inside] - x0]
        return {int(v) for v in np.unique(hits[hits > 0])}

    def _find(self, piece: int) -> int:
        """The oldest piece of ``piece``'s object, halving the path there."""
        parent = self._parent
        while parent[piece] != piece:
            parent[piece] = parent[parent[piece]]
            piece = parent[piece]
        return piece

    def _join(self, one: int, other: int) -> None:
        """Make two pieces' objects one object, named by the older piece."""
        first, second = self._find(one), self._find(other)
        self._parent[max(first, second)] = min(first, second)

    def _roots(self) -> np.ndarray:
        """Every piece's object as its oldest piece; index 0 is background."""
        parent = np.asarray(self._parent, dtype=np.int64)
        grand = parent[parent]
        while not np.array_equal(grand, parent):
            parent, grand = grand, grand[grand]
        return parent


    def outcome(self) -> Optional[_StrokeOutcome]:
        """What the stroke adds now, or None before any frame has arrived.

        Rule 3 is applied here rather than as frames arrive, so an object the
        path reaches late still goes in, and the answer can be asked for at any
        time: while the button is down, to show, and once more at the end.
        """
        self.dirty = False
        if self._owner is None:
            return None
        roots = self._roots()
        touched = {int(roots[piece]) for piece in self._touched}
        lookup = np.zeros(roots.size, dtype=np.int32)
        merged, next_id = 0, 1 + int(bool(touched))
        for piece in range(1, roots.size):
            root = int(roots[piece])
            if root in touched:
                lookup[piece] = 1
                merged += 1
            elif self.keep_untouched and root == piece:
                lookup[piece] = next_id
                next_id += 1
            elif self.keep_untouched:
                lookup[piece] = lookup[root]
        x0, y0, x1, y1 = self._span
        labels = lookup[self._owner[y0:y1, x0:x1]]
        objects = int(np.count_nonzero(np.unique(labels)))
        return _StrokeOutcome(labels, (x0, y0), merged, objects, self._frames)
