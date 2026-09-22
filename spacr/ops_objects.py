"""One object list for a well, sewn across window seams and numbered once.

SEGMENT, SEW, NUMBER -- the three steps between a composite and an object
table. The compose step gives a mosaic that is built a
window at a time and never materialised whole; this is what segmentation and
numbering do on top of it:

* SEGMENT ONCE, on the composite, window by window.
* SEW THE LABELS ACROSS WINDOW SEAMS. An object straddling a boundary is
  one object, matched on its well-frame centroid.
* NUMBER IN THE WELL FRAME, NOT PER WINDOW. Per-window numbering would
  give two windows an object #1 apiece.

THE SEAM PROBLEM IS ALREADY HALF-SOLVED BY THE COMPOSE STEP, and saying how
avoids
reinventing it. :func:`spacr.ops_compose.windows_over` tiles with an OVERLAP,
and its contract is that the overlap exceeds the largest object. So a nucleus
near a seam is not split between two windows -- it is seen WHOLE by at least
one of them, and usually twice. The job here is therefore deduplication, not
reconstruction: find the observations that are the same nucleus and keep the
one that saw all of it.

    THAT DISTINCTION IS THE WHOLE DESIGN. Reconstructing an object from two
    partial masks means deciding how to union pixels across a seam, which is
    fiddly and lossy. Choosing between two complete masks is neither.

WHICH LEAVES ONE FAILURE THAT MUST NOT BE SILENT: an object larger than the
overlap, which every window clips. :func:`number` refuses those rather than
emitting a fragment, because a fragment looks exactly like a small nucleus
and would be counted as one for the rest of the run.

THE IDS ARE A JOIN KEY, so they are assigned deterministically -- raster
order on the well-frame centroid -- and not by iteration order over a dict or
by whichever window happened to be segmented first. The contract is: "This
id, and the
centroid beside it, is the join key for every later phase." Two runs over the
same data must produce the same numbers or nothing downstream can be
compared.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .ops_compose import ComposeError, Window

__all__ = [
    "ObjectsError",
    "WindowObject", "PlateObject",
    "objects_in_window", "segment_windows", "sew", "number", "objects_frame",
    "unseen_records",
    "DEFAULT_CENTROID_TOLERANCE", "DEFAULT_AREA_RATIO",
]

LOG = logging.getLogger("spacr.ops_objects")

#: How far apart two complete observations of one nucleus may sit, in pixels
#: of the well frame, and still be called the same nucleus. Generous relative
#: to a centroid's precision (sub-pixel for a clean mask) and tight relative to
#: nuclear spacing, which is what makes the match unambiguous rather than
#: merely plausible.
DEFAULT_CENTROID_TOLERANCE = 3.0

#: How different two complete observations' areas may be and still be the
#: same object on area alone. Both saw the whole nucleus, but a window edge
#: cuts away context the segmenter uses, and that alone can draw one nucleus a
#: quarter smaller in one window than in the next. So a pair that fails this
#: ratio is still one object when its centroids lie within the smaller
#: observation's equivalent radius: two nuclei whose masks do not overlap
#: cannot sit that close.
DEFAULT_AREA_RATIO = 0.8

_CLIP_MARGIN = 2


class ObjectsError(ValueError):
    """An object list that cannot be trusted, with the way out in the text.

    Raised rather than returned empty for the same reason
    :class:`spacr.ops_compose.ComposeError` is: every one of these is a
    sentence an operator can act on -- "raise the window overlap above the
    largest nucleus" -- and a caller that swallowed it would carry on with a
    count that is quietly wrong.
    """


@dataclass(frozen=True)
class WindowObject:
    """One segmented object as ONE window saw it, in well-frame coordinates.

    :param window: the window it was segmented in, so a disagreement can be
        traced back to the pixels that produced it.
    :param label: its label value within that window's label image.
    :param clipped: whether its mask touches a window edge that is not also a
        canvas edge -- meaning this window did NOT see all of it. A clipped
        observation is never preferred and never emitted alone.
    :param centroid_y: row of the object's centroid, in WELL-frame pixels --
        not window-frame. Matching across a seam compares centroids from two
        different windows, so they have to be in a frame both agree on.
    :param centroid_x: column of the same centroid, same frame.
    :param area: the mask's pixel count as this window saw it. Smaller than
        the truth whenever ``clipped`` is set, which is what makes it usable
        as the tie-break between two observations of one nucleus.
    :param bbox: ``(top, left, bottom, right)`` in well-frame pixels, every
        one of them INCLUSIVE -- ``bottom`` and ``right`` are the last row
        and column the mask occupies, so its height is ``bottom - top + 1``.
        This docstring said exclusive until 2026-09-19 and the code has
        always been inclusive (`_label_extents` reduces with
        ``np.maximum``), which cost anyone measuring an extent from it one
        pixel in each direction.
    """

    window: Window
    label: int
    centroid_y: float
    centroid_x: float
    area: int
    bbox: Tuple[int, int, int, int]
    clipped: bool = False


@dataclass(frozen=True)
class PlateObject:
    """One nucleus, numbered for the whole well. The join key for sampling.

    The field names are the ``ops_objects`` columns the storage contract
    specifies, so the table is written from these without a translation
    layer.

    :param object_id: the well-wide number. Unique across the whole well,
        which is the entire reason this type exists -- per-window numbering
        gives two windows an object #1 apiece.
    :param centroid_x: column of the centroid, in well-frame pixels.
    :param centroid_y: row of the centroid, same frame.
    :param area: pixel count of the mask that was kept, which is the
        UNCLIPPED observation wherever one exists.
    :param bbox: ``(top, left, bottom, right)`` in well-frame pixels, every
        one of them inclusive, as :class:`WindowObject` carries them.
    :param window: the window whose observation was kept, as ``(row, col)``.
        Recorded so a suspect object can be traced back to the pixels it was
        segmented from.
    :param n_observations: how many windows saw this nucleus. Greater than
        one means it sat in an overlap and the observations were sewn.
    """

    object_id: int
    centroid_x: float
    centroid_y: float
    area: int
    bbox: Tuple[int, int, int, int]
    window: Tuple[int, int]
    n_observations: int = 1

    def row(self) -> Dict[str, object]:
        """One ``ops_objects`` row."""
        top, left, bottom, right = self.bbox
        return {
            "object_id": self.object_id,
            "centroid_x": self.centroid_x,
            "centroid_y": self.centroid_y,
            "area": self.area,
            "bbox_top": top, "bbox_left": left,
            "bbox_bottom": bottom, "bbox_right": right,
            "window_top": self.window[0], "window_left": self.window[1],
            "n_observations": self.n_observations,
        }


def objects_in_window(window: Window, labels: np.ndarray, *,
                      canvas: Optional[Tuple[int, int]] = None,
                      ) -> Tuple[WindowObject, ...]:
    """Turn one window's label image into well-frame observations.

    :param window: where this label image sits in the well.
    :param labels: integer label image, zero is background, as a segmenter
        returns it. Must match the window's shape.
    :param canvas: the well's ``(height, width)``. Used only to tell a window
        edge that is also the WELL edge -- where an object genuinely ends --
        from an interior seam, where a touching object is clipped. Omitted,
        every edge is treated as interior, which is the safe direction: it
        can only make this function more cautious.
    :raises ObjectsError: when ``labels`` is not the window's shape, because
        every coordinate below would be silently wrong.
    """
    labels = np.asarray(labels)
    if labels.shape != (window.height, window.width):
        raise ObjectsError(
            f"the label image is {labels.shape} but window {window.offset()} "
            f"is {(window.height, window.width)}; every centroid derived from "
            f"it would be offset by the difference")

    height, width = labels.shape
    flat = labels.reshape(-1)
    positions = np.flatnonzero(flat > 0)
    if positions.size == 0:
        return ()

    (values, top, left, bottom, right, area,
     row_sum, col_sum) = _label_extents(flat[positions], positions, width)
    canvas_height, canvas_width = canvas if canvas else (None, None)
    clipped = (((top == 0) & (window.top > 0))
               | ((left == 0) & (window.left > 0))
               | ((bottom == height - 1)
                  & (canvas_height is None
                     or window.top + height < canvas_height))
               | ((right == width - 1)
                  & (canvas_width is None
                     or window.left + width < canvas_width)))
    return tuple(
        WindowObject(
            window=window, label=int(value),
            centroid_y=centroid_y + window.top,
            centroid_x=centroid_x + window.left,
            area=count,
            bbox=(first_row + window.top, first_col + window.left,
                  last_row + window.top, last_col + window.left),
            clipped=flag)
        for value, centroid_y, centroid_x, count, first_row, first_col,
        last_row, last_col, flag in zip(
            values.tolist(), (row_sum / area).tolist(),
            (col_sum / area).tolist(), area.tolist(), top.tolist(),
            left.tolist(), bottom.tolist(), right.tolist(),
            clipped.tolist()))


def _label_extents(values: np.ndarray, positions: np.ndarray, width: int):
    """Every label's box, pixel count and coordinate sums, from one sort.

    372 PART 14-L measured the loop this replaces at labels x pixels -- one
    ``labels == value`` mask over the whole window per label, 18.6 ms a label
    in a 2,048 px window. Here the foreground pixels are sorted by label once
    (a stable sort, so each label's pixels stay in raster order) and every
    statistic is a reduction over the contiguous runs.

    THE OUTPUT MUST BE THE OLD FUNCTION'S, bit for bit, because it feeds a
    join key. So the centroid is a sum divided by a count exactly as
    ``ndarray.mean`` computes it: the coordinates are integers, every
    partial sum below 2**53 is exact in float64 whatever the summation
    order, and the one division is the same correctly rounded operation.

    :param values: the label value of each foreground pixel.
    :param positions: each foreground pixel's flat raster index, ascending.
    :param width: the label image's width, to turn an index into a column.
    :returns: per label in ascending value order -- the values, top, left,
        bottom and right (inclusive), pixel count, and the sums of the row
        and column indices as float64.
    """
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    change = np.empty(ordered.size, dtype=bool)
    change[0] = True
    np.not_equal(ordered[1:], ordered[:-1], out=change[1:])
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], ordered.size)
    positions = positions[order]
    rows = positions // width
    cols = positions - rows * width
    return (ordered[starts], rows[starts],
            np.minimum.reduceat(cols, starts), rows[ends - 1],
            np.maximum.reduceat(cols, starts), ends - starts,
            np.add.reduceat(rows.astype(np.float64), starts),
            np.add.reduceat(cols.astype(np.float64), starts))


def segment_windows(windows: Iterable[Window],
                    segment: Callable[[Window], np.ndarray], *,
                    canvas: Optional[Tuple[int, int]] = None,
                    ) -> Tuple[WindowObject, ...]:
    """Run a segmenter over each window and collect the observations.

    THE SEGMENTER IS INJECTED, exactly as ``compose_window`` takes its
    ``read_tile``. This module never imports cellpose, never chooses a model
    and never decides a diameter -- it is the geometry of doing that window
    by window, and it is testable on planted labels with no model present.

    :param windows: from :func:`spacr.ops_compose.windows_over`.
    :param segment: called with one window, returns its label image. A
        caller composes the pixels (``compose_window``) and segments them
        inside this callable, so the composite is never held whole.
    :param canvas: the well's shape, passed through to
        :func:`objects_in_window`.
    :returns: every observation from every window, unsewn and unnumbered.
    """
    found: List[WindowObject] = []
    for window in windows:
        labels = segment(window)
        if labels is None:
            continue
        found.extend(objects_in_window(window, labels, canvas=canvas))
    return tuple(found)


@dataclass(frozen=True)
class _Table:
    """A well's observations as columns, for matching without a Python loop.

    :param y: centroid rows.
    :param x: centroid columns.
    :param area: pixel counts.
    :param top: box tops, inclusive.
    :param left: box lefts, inclusive.
    :param bottom: box bottoms, inclusive, as :func:`objects_in_window`
        records them.
    :param right: box rights, inclusive.
    :param label: label values.
    :param window: a dense id per distinct window.
    :param clipped: the clipped flags.
    :param rank: each observation's place in (window top, left, height,
        width, label) order. Every tie in this module is broken on it, which
        is what makes the grouping independent of the order observations
        arrive in.
    """

    y: np.ndarray
    x: np.ndarray
    area: np.ndarray
    top: np.ndarray
    left: np.ndarray
    bottom: np.ndarray
    right: np.ndarray
    label: np.ndarray
    window: np.ndarray
    clipped: np.ndarray
    rank: np.ndarray

    @classmethod
    def of(cls, observations: Sequence[WindowObject]) -> "_Table":
        """Read the observations into columns in one pass.

        :param observations: the well's observations, in any order.
        :returns: the columns, row ``i`` being ``observations[i]``.
        """
        windows: Dict[Window, int] = {}
        rows = [(one.centroid_y, one.centroid_x, one.area,
                 one.bbox[0], one.bbox[1], one.bbox[2], one.bbox[3],
                 one.label, windows.setdefault(one.window, len(windows)),
                 1.0 if one.clipped else 0.0)
                for one in observations]
        data = np.array(rows, dtype=np.float64).reshape(len(rows), 10)
        window = data[:, 8].astype(np.int64)
        label = data[:, 7].astype(np.int64)
        geometry = np.array(
            [(one.top, one.left, one.height, one.width) for one in windows],
            dtype=np.int64).reshape(len(windows), 4)
        order = np.lexsort((label, geometry[window, 3], geometry[window, 2],
                            geometry[window, 1], geometry[window, 0]))
        rank = np.empty(len(rows), dtype=np.int64)
        rank[order] = np.arange(len(rows))
        return cls(y=data[:, 0], x=data[:, 1], area=data[:, 2],
                   top=data[:, 3], left=data[:, 4], bottom=data[:, 5],
                   right=data[:, 6], label=label, window=window,
                   clipped=data[:, 9] > 0, rank=rank)


def sew(observations: Sequence[WindowObject], *,
        tolerance: float = DEFAULT_CENTROID_TOLERANCE,
        area_ratio: float = DEFAULT_AREA_RATIO,
        ) -> Tuple[Tuple[WindowObject, ...], ...]:
    """Group observations that are the same nucleus.

    Every group is one physical object. A nucleus in a window overlap is
    grouped from two (or four, at a corner) observations; one in a window's
    interior forms a group of one.

    THE CLIPPED ONES ARE THE POINT. An observation whose mask ran into an
    interior seam did not see the whole nucleus, so its area and centroid are
    both wrong. It is kept only long enough to be matched to a complete
    observation of the same nucleus and is then dropped in favour of it. A
    clipped observation matching nothing complete means no window saw that
    nucleus whole, which :func:`number` refuses.

    Two labels of one window are never one object, because a label image does
    not give one nucleus two labels. Two complete observations from different
    windows are one nucleus when their centroids agree within ``tolerance``
    on each axis and either their areas agree to ``area_ratio`` or their
    centroids lie within the smaller one's equivalent radius -- the radius of
    a disc of its area. Pairs whose areas agree are joined first, then the
    rest closest first, and a join that would give a group two labels of one
    window is refused.

    A clipped observation is matched by containment, not by a fixed distance:
    the cut moves its centroid inward by up to the nucleus's radius, so a
    fixed tolerance misses exactly the deep clips. It joins a complete
    observation from another window when its centroid lies inside that
    observation's equivalent disc, or its bounding box lies inside that
    observation's box give or take two pixels. It joins only the closest such
    observation, so one clip can never make two nuclei one object. Clipped
    observations that match nothing complete are grouped with each other
    where their boxes overlap.

    :param observations: every window's view of every object in one well,
        already in well-frame coordinates. Order does not matter; the
        grouping is by geometry, not by arrival.
    :returns: groups, each a tuple of observations, in no particular order --
        :func:`number` imposes the order that matters.
    """
    observations = list(observations)
    if not observations:
        return ()
    table = _Table.of(observations)
    group = np.full(len(observations), -1, dtype=np.int64)
    complete = np.flatnonzero(~table.clipped)
    fragments = np.flatnonzero(table.clipped)
    if complete.size:
        group[complete] = _sew_complete(table, complete, tolerance, area_ratio)
    if complete.size and fragments.size:
        joined = _join_clips(table, fragments, complete, group)
        attached = joined >= 0
        group[fragments[attached]] = group[joined[attached]]
        fragments = fragments[~attached]
    if fragments.size:
        group[fragments] = (int(group.max()) + 1
                            + _sew_fragments(table, fragments))
    return _assemble(observations, group, table.rank)


def _widened(radius: float) -> float:
    """A search radius a hair larger than ``radius``.

    The KD-tree only proposes candidates; the exact test is repeated on the
    returned pairs with the original comparison, so widening can add work
    but never a match, and a pair sitting exactly on the boundary is never
    lost to the tree's own rounding.

    :param radius: the radius the exact test will use.
    :returns: the radius to search with.
    """
    return float(radius) * (1.0 + 1e-9) + 1e-9


def _pairs_within_tolerance(y: np.ndarray, x: np.ndarray, tolerance: float):
    """Index pairs whose centroids agree within ``tolerance`` on both axes.

    The same per-axis test the quadratic scan made, found through a
    Chebyshev (``p=inf``) KD-tree query and then re-checked exactly.

    :param y: centroid rows.
    :param x: centroid columns.
    :param tolerance: the per-axis limit, inclusive.
    :returns: two index arrays, first < second.
    """
    if y.size < 2 or not tolerance >= 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    from scipy.spatial import cKDTree

    pairs = cKDTree(np.column_stack((y, x))).query_pairs(
        _widened(tolerance), p=np.inf, output_type="ndarray")
    first, second = pairs[:, 0], pairs[:, 1]
    exact = ((np.abs(y[first] - y[second]) <= tolerance)
             & (np.abs(x[first] - x[second]) <= tolerance))
    return first[exact], second[exact]


def _levels(reach: np.ndarray) -> np.ndarray:
    """Power-of-two size classes for per-observation search radii.

    :param reach: how far each observation can reach, in pixels.
    :returns: ``k`` per observation with ``2**k >= reach`` and ``k >= 0``.
    """
    return np.ceil(np.log2(np.maximum(reach, 1.0))).astype(np.int64)


def _pairs_within_reach(y_a, x_a, reach_a, y_b=None, x_b=None, reach_b=None):
    """Index pairs whose centroids lie within the sum of their reaches.

    Each side's reach differs per observation -- a large object can match
    across a larger distance -- and one global radius would be the largest
    object's, which in a dense well means every observation against dozens.
    So the observations are split into power-of-two size classes, one
    KD-tree per class, and each pair of classes is searched at the sum of
    the two classes' radii. The exact Chebyshev test follows.

    :param y_a: centroid rows of the first set.
    :param x_a: centroid columns of the first set.
    :param reach_a: per-observation reach of the first set.
    :param y_b: centroid rows of the second set, or ``None`` to pair the
        first set with itself.
    :param x_b: centroid columns of the second set.
    :param reach_b: per-observation reach of the second set.
    :returns: two index arrays into the first and second set. Paired with
        itself, each unordered pair appears once, first < second.
    """
    from scipy.spatial import cKDTree

    same = y_b is None
    if same:
        y_b, x_b, reach_b = y_a, x_a, reach_a
    points_a = np.column_stack((y_a, x_a))
    points_b = np.column_stack((y_b, x_b))
    level_a, level_b = _levels(reach_a), _levels(reach_b)
    firsts: List[np.ndarray] = []
    seconds: List[np.ndarray] = []
    trees_b = {}
    for class_a in np.unique(level_a).tolist():
        members_a = np.flatnonzero(level_a == class_a)
        tree_a = cKDTree(points_a[members_a])
        for class_b in np.unique(level_b).tolist():
            if same and class_b < class_a:
                continue
            radius = _widened(2.0 ** class_a + 2.0 ** class_b)
            if same and class_b == class_a:
                pairs = tree_a.query_pairs(radius, p=np.inf,
                                           output_type="ndarray")
                firsts.append(members_a[pairs[:, 0]])
                seconds.append(members_a[pairs[:, 1]])
                continue
            if class_b not in trees_b:
                members_b = np.flatnonzero(level_b == class_b)
                trees_b[class_b] = (members_b, cKDTree(points_b[members_b]))
            members_b, tree_b = trees_b[class_b]
            found = tree_a.sparse_distance_matrix(
                tree_b, radius, p=np.inf, output_type="ndarray")
            firsts.append(members_a[found["i"]])
            seconds.append(members_b[found["j"]])
    first = np.concatenate(firsts) if firsts else np.empty(0, np.int64)
    second = np.concatenate(seconds) if seconds else np.empty(0, np.int64)
    if same:
        first, second = np.minimum(first, second), np.maximum(first, second)
        distinct = first != second
        first, second = first[distinct], second[distinct]
    exact = ((np.abs(y_a[first] - y_b[second])
              <= reach_a[first] + reach_b[second])
             & (np.abs(x_a[first] - x_b[second])
                <= reach_a[first] + reach_b[second]))
    return first[exact], second[exact]


def _depth(distance: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Distance as a fraction of a radius; 0 or infinity where it is zero.

    :param distance: centroid distances.
    :param radius: the radius each is measured against.
    :returns: ``distance / radius``, with a zero radius giving 0 for a zero
        distance and infinity otherwise.
    """
    safe = np.where(radius > 0, radius, 1.0)
    return np.where(radius > 0, distance / safe,
                    np.where(distance > 0, np.inf, 0.0))


def _sew_complete(table: _Table, members: np.ndarray, tolerance: float,
                  area_ratio: float) -> np.ndarray:
    """Group complete observations into nuclei. Group ids per member.

    372 PART 14-L, V11a: on well A1, 1,048 nuclei were numbered twice. Two
    windows saw each whole, centroids within 3 px (736 within 1 px), but the
    window edge cut the segmenter's context and one mask came out smaller --
    area ratio median 0.74 against 0.8. The rules, and why each is safe:

    * Two labels of one window are two objects: a label image's labels are
      disjoint. (The same window AND label is the same observation passed
      twice, which is one object.)
    * A cross-window pair within ``tolerance`` on both axes is one object
      when the area ratio passes, as before, OR when the centroid distance
      is within the smaller observation's equivalent radius
      ``sqrt(area / pi)``. Two discs that do not overlap sit at least
      ``r1 + r2`` apart, so centroids inside the smaller disc mean the masks
      overlap; that is half the spacing two touching nuclei can reach.
    * Candidate joins are processed area-ratio matches first, then by
      distance as a fraction of the smaller radius, then by rank; a join
      that would put two labels of one window in a group is skipped. Where
      no component holds such a clash, the result is exactly the connected
      components of the match graph -- the old function's single linkage.

    :param table: the well's observations as columns.
    :param members: indices of the complete observations.
    :param tolerance: the per-axis centroid limit.
    :param area_ratio: the area agreement that alone makes a match.
    :returns: a group id per member.
    """
    first, second = _pairs_within_tolerance(
        table.y[members], table.x[members], tolerance)
    a, b = members[first], members[second]
    same_window = table.window[a] == table.window[b]
    same_observation = same_window & (table.label[a] == table.label[b])
    larger = np.maximum(table.area[a], table.area[b])
    smaller = np.minimum(table.area[a], table.area[b])
    positive = larger > 0
    ratio = np.divide(smaller, larger, out=np.zeros_like(smaller),
                      where=positive)
    agree = positive & (ratio >= area_ratio)
    distance = np.hypot(table.y[a] - table.y[b], table.x[a] - table.x[b])
    radius = np.sqrt(np.maximum(smaller, 0.0) / math.pi)
    inside = positive & (distance <= radius)
    keep = same_observation | (~same_window & (agree | inside))
    tier = np.where(agree | same_observation, 0, 1)
    return _components(
        members.size, first[keep], second[keep],
        (tier[keep], _depth(distance, radius)[keep]),
        table.rank[members], table.window[members], table.label[members])


def _join_clips(table: _Table, clips: np.ndarray, complete: np.ndarray,
                group: np.ndarray) -> np.ndarray:
    """The one complete observation each clipped observation belongs to.

    372 PART 14-L, V11b: ``number(strict=True)`` refused 10,182 groups on
    well A1. Each was a clip -- 9 x 6 px, area 38 in the diagnosed case -- of
    a nucleus the neighbouring window saw whole (area 96), its centroid 3.1
    px from the whole one's against a 3.0 px tolerance (well median 3.75,
    p90 4.69). A cut moves a clip's centroid inward, by up to the radius for
    a shallow cut, so no fixed centroid tolerance is right.

    ELIGIBLE: a complete observation from another window, when EITHER the
    clip's centroid lies inside its equivalent disc (a clip is part of the
    nucleus, and a part's centroid lies inside a convex whole) OR the clip's
    box lies inside its box grown by ``_CLIP_MARGIN`` px (a part's box lies
    inside the whole's box whatever the shape; the margin absorbs the pixel
    or so two windows' masks disagree by). And not when the complete one's
    group already holds a complete observation from the clip's own window:
    that window saw this nucleus whole under another label.

    CHOSEN: the eligible observation the clip lies deepest inside --
    distance as a fraction of its equivalent radius, then rank. One only, so
    a clip between two nuclei cannot join them (the quadratic sew did: any
    member matching any observation merged the groups).

    TWO CLIPS OF ONE WINDOW MAY JOIN ONE GROUP, the one place rule (c) of
    PART 14-L is relaxed. A clip never decides what a group emits -- the
    complete observation does -- so the only cost of a wrong join is an
    ``n_observations`` one too high. The cost of refusing it is worse: a
    segmenter that split a cut nucleus into two labels at the edge would
    leave one piece unclaimed, and ``number(strict=True)`` would refuse the
    well for a harmless fragment.

    :param table: the well's observations as columns.
    :param clips: indices of the clipped observations.
    :param complete: indices of the complete observations.
    :param group: group ids, already assigned for the complete observations.
    :returns: per clip, the index of the complete observation it joins, or
        -1.
    """
    radius = np.sqrt(np.maximum(table.area[complete], 0.0) / math.pi)
    extent = np.maximum(table.bottom[complete] - table.top[complete],
                        table.right[complete] - table.left[complete])
    reach = np.maximum(radius, extent + _CLIP_MARGIN)
    at_clip, at_complete = _pairs_within_reach(
        table.y[clips], table.x[clips], np.zeros(clips.size),
        table.y[complete], table.x[complete], reach)
    k, c = clips[at_clip], complete[at_complete]
    distance = np.hypot(table.y[k] - table.y[c], table.x[k] - table.x[c])
    in_disc = distance <= radius[at_complete]
    in_box = ((table.top[k] >= table.top[c] - _CLIP_MARGIN)
              & (table.bottom[k] <= table.bottom[c] + _CLIP_MARGIN)
              & (table.left[k] >= table.left[c] - _CLIP_MARGIN)
              & (table.right[k] <= table.right[c] + _CLIP_MARGIN))
    windows = int(table.window.max()) + 1
    held = np.unique(group[complete] * windows + table.window[complete])
    seen_whole_here = np.isin(group[c] * windows + table.window[k], held)
    eligible = ((table.window[k] != table.window[c]) & (in_disc | in_box)
                & ~seen_whole_here)
    at_clip, c = at_clip[eligible], c[eligible]
    depth = _depth(distance[eligible], radius[at_complete][eligible])
    order = np.lexsort((table.rank[c], depth, at_clip))
    at_clip, c = at_clip[order], c[order]
    first = np.ones(at_clip.size, dtype=bool)
    first[1:] = at_clip[1:] != at_clip[:-1]
    joined = np.full(clips.size, -1, dtype=np.int64)
    joined[at_clip[first]] = c[first]
    return joined


def _sew_fragments(table: _Table, fragments: np.ndarray) -> np.ndarray:
    """Group the clipped observations no complete one claimed.

    Every one of these is refused or dropped by :func:`number`, so the
    grouping decides only what the refusal counts and how large it says the
    object is. Pieces of one object cut by neighbouring windows share the
    overlap band, so their boxes overlap: an object larger than the overlap
    is reported once with its whole extent rather than once per window. Two
    labels of one window are still never joined, which bounds a chain to
    the few windows that cover one point.

    :param table: the well's observations as columns.
    :param fragments: indices of the unclaimed clipped observations.
    :returns: a group id per fragment.
    """
    extent = np.maximum(table.bottom[fragments] - table.top[fragments],
                        table.right[fragments] - table.left[fragments])
    first, second = _pairs_within_reach(
        table.y[fragments], table.x[fragments], extent.astype(np.float64))
    a, b = fragments[first], fragments[second]
    overlap = ((table.top[a] <= table.bottom[b])
               & (table.top[b] <= table.bottom[a])
               & (table.left[a] <= table.right[b])
               & (table.left[b] <= table.right[a]))
    same_window = table.window[a] == table.window[b]
    same_observation = same_window & (table.label[a] == table.label[b])
    keep = overlap & (~same_window | same_observation)
    distance = np.hypot(table.y[a] - table.y[b], table.x[a] - table.x[b])
    return _components(
        fragments.size, first[keep], second[keep],
        (np.zeros(int(keep.sum()), dtype=np.int64), distance[keep]),
        table.rank[fragments], table.window[fragments],
        table.label[fragments])


def _components(count: int, first: np.ndarray, second: np.ndarray,
                costs: Tuple[np.ndarray, ...], rank: np.ndarray,
                window: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Connected components that never hold two labels of one window.

    The plain components come from a sparse graph in one call. Only a
    component in which some window appears under two labels needs more: its
    edges are replayed cheapest first through union-find, and a join that
    would bring two labels of one window together is skipped. On a real
    well those components are rare, so the Python loop is short.

    :param count: how many nodes.
    :param first: one end of each edge.
    :param second: the other end.
    :param costs: sort keys for the edges, most significant first; ties fall
        to the lower then the higher endpoint rank.
    :param rank: each node's total-order rank.
    :param window: each node's window id.
    :param label: each node's label.
    :returns: a component id per node.
    """
    if first.size == 0:
        return np.arange(count, dtype=np.int64)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    graph = coo_matrix((np.ones(first.size, dtype=bool), (first, second)),
                       shape=(count, count))
    total, component = connected_components(graph, directed=False)
    component = component.astype(np.int64)
    clashing = _clashing(component, total, window, label)
    if not clashing.any():
        return component

    replay = clashing[component[first]]
    first, second = first[replay], second[replay]
    low = np.minimum(rank[first], rank[second])
    high = np.maximum(rank[first], rank[second])
    keys = (high, low) + tuple(cost[replay] for cost in reversed(costs))
    order = np.lexsort(keys)
    nodes = np.flatnonzero(clashing[component]).tolist()
    parent = {node: node for node in nodes}
    labels_by_window = {node: {int(window[node]): int(label[node])}
                        for node in nodes}
    for a, b in zip(first[order].tolist(), second[order].tolist()):
        root_a, root_b = _root(parent, a), _root(parent, b)
        if root_a == root_b:
            continue
        if len(labels_by_window[root_a]) > len(labels_by_window[root_b]):
            root_a, root_b = root_b, root_a
        into = labels_by_window[root_b]
        if any(into.get(key, value) != value
               for key, value in labels_by_window[root_a].items()):
            continue
        parent[root_a] = root_b
        into.update(labels_by_window.pop(root_a))
    for node in nodes:
        component[node] = total + _root(parent, node)
    return component


def _root(parent: Dict[int, int], node: int) -> int:
    """Union-find's representative of ``node``, halving the path on the way.

    :param parent: the forest, node to parent; a root is its own parent.
    :param node: the node to look up.
    :returns: the root of its tree.
    """
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def _clashing(component: np.ndarray, total: int, window: np.ndarray,
              label: np.ndarray) -> np.ndarray:
    """Which components hold one window under two different labels.

    :param component: a component id per node.
    :param total: how many components.
    :param window: each node's window id.
    :param label: each node's label.
    :returns: a flag per component.
    """
    order = np.lexsort((label, window, component))
    comp, win, lab = component[order], window[order], label[order]
    distinct = np.ones(comp.size, dtype=bool)
    distinct[1:] = ((comp[1:] != comp[:-1]) | (win[1:] != win[:-1])
                    | (lab[1:] != lab[:-1]))
    comp, win = comp[distinct], win[distinct]
    flags = np.zeros(total, dtype=bool)
    repeated = (comp[1:] == comp[:-1]) & (win[1:] == win[:-1])
    flags[comp[1:][repeated]] = True
    return flags


def _assemble(observations: Sequence[WindowObject], group: np.ndarray,
              rank: np.ndarray) -> Tuple[Tuple[WindowObject, ...], ...]:
    """Turn a group id per observation into tuples, in a fixed order.

    Members are ordered by rank and groups by their first member's rank, so
    the same observations in any order give the identical tuple.

    :param observations: the observations, indexed as ``group`` is.
    :param group: a group id per observation.
    :param rank: each observation's total-order rank.
    :returns: the groups.
    """
    _, group = np.unique(group, return_inverse=True)
    order = np.lexsort((rank, group))
    ordered = group[order]
    starts = np.flatnonzero(np.concatenate(([True],
                                            ordered[1:] != ordered[:-1])))
    ends = np.append(starts[1:], ordered.size)
    members = [observations[index] for index in order.tolist()]
    groups = [tuple(members[start:end])
              for start, end in zip(starts.tolist(), ends.tolist())]
    leading = rank[order][starts]
    return tuple(groups[index]
                 for index in np.argsort(leading, kind="stable").tolist())


def _identity(one: WindowObject) -> Tuple[int, int, int, int, int]:
    """The (window top, left, height, width, label) an observation is known by.

    :param one: an observation.
    :returns: a key no two observations of a well share.
    """
    window = one.window
    return (window.top, window.left, window.height, window.width, one.label)


def _preferred(one: WindowObject):
    """Sort key choosing the observation a group keeps: largest area first.

    372 PART 14-L item 5: ties on area fell to arrival order, so the same
    well could keep a different observation -- and so a different centroid
    -- on a different run. They fall to window then label instead.

    :param one: a complete observation.
    :returns: the key; the smallest is kept.
    """
    return (-one.area,) + _identity(one)


def _extent(group: Sequence[WindowObject]) -> Tuple[int, int, float, float]:
    """The box around every observation of a group.

    :param group: observations of one object.
    :returns: height, width, and the box centre's column and row.
    """
    top = min(one.bbox[0] for one in group)
    left = min(one.bbox[1] for one in group)
    bottom = max(one.bbox[2] for one in group)
    right = max(one.bbox[3] for one in group)
    return (bottom - top + 1, right - left + 1,
            (left + right) / 2.0, (top + bottom) / 2.0)


def _spanned_side(one: WindowObject) -> int:
    """The window side an observation runs the full length of, or 0.

    :param one: an observation.
    :returns: the window's height or width when the mask reaches both ends
        of it, the larger if both; otherwise 0.
    """
    top, left, bottom, right = one.bbox
    window = one.window
    sides = [0]
    if top <= window.top and bottom >= window.bottom - 1:
        sides.append(window.height)
    if left <= window.left and right >= window.right - 1:
        sides.append(window.width)
    return max(sides)


def _unseen_report(unseen: Sequence[Sequence[WindowObject]]) -> Tuple[str, str]:
    """What to say about groups no window saw whole.

    372 PART 14-L, V11b: the old refusal always said to raise the overlap,
    and on well A1 every one of its 10,182 groups was a clip the overlap had
    nothing to do with. :func:`number` is not given the overlap, so it cannot
    tell the cases apart for every group; it states the extent it can
    measure and the remedy for each case. An object that runs the full
    length of a window side is the one case it can name: no overlap fixes
    that.

    :param unseen: the groups with no complete observation.
    :returns: the refusal message and the shorter warning logged when the
        groups are dropped.
    """
    extents = [_extent(group) for group in unseen]
    height, width, centre_x, centre_y = max(
        extents, key=lambda box: (max(box[0], box[1]), box[0], box[1]))
    largest = max(height, width)
    sides = [max(_spanned_side(one) for one in group) for group in unseen]
    spanning = [side for side in sides if side]
    lead = (f"{len(unseen)} group(s) of observations were clipped by every "
            f"window that saw them, so no window saw one whole. The largest "
            f"extends {height} x {width} px near ({centre_x:.0f}, "
            f"{centre_y:.0f}).")
    parts = [lead]
    if spanning:
        parts.append(
            f"{len(spanning)} of them run the full length of a window side "
            f"({max(spanning)} px), which no overlap can fix: the windows "
            f"have to be larger than the object.")
    if len(spanning) < len(unseen):
        parts.append(
            f"The window overlap has to exceed the largest object: if it is "
            f"{largest} px or less, raise it above {largest} px and re-run. "
            f"If it is already larger, these are fragments that matched no "
            f"window's whole view of their nucleus.")
    parts.append(
        "number(..., strict=False) drops them and logs how many; emitting "
        "them would count fragments as nuclei.")
    warning = (f"dropped {len(unseen)} group(s) of observations that no "
               f"window saw whole; the largest extends {height} x {width} px "
               f"near ({centre_x:.0f}, {centre_y:.0f})")
    return " ".join(parts), warning


def unseen_records(groups: Sequence[Sequence[WindowObject]]
                   ) -> Tuple[Dict[str, object], ...]:
    """One record per group no window saw whole, for a run report.

    WHY THIS IS NOT THE REFUSAL MESSAGE. :func:`_unseen_report` names the
    LARGEST group and the remedy, which is what an operator reading one line
    needs. It is also all that survived well A1's run: 54 groups were dropped
    and the report kept one box and a count, so the question the run raised --
    are these Cellpose fragments with no counterpart, or clips whose complete
    observation the join missed? -- could not be answered afterwards without
    segmenting the well again. These records are the cheap half of that
    answer, written while the observations are still in memory.

    The other half is :func:`spacr.ops_engine.run_ops`'s, which compares each
    box against the objects that WERE numbered: a refusal with a numbered
    object over it is a join that missed, and one with empty well frame
    around it is a fragment.

    :param groups: from :func:`sew` -- every group, not only the refused
        ones; the ones with a complete observation are skipped here.
    :returns: one dict per refused group, in the order the groups came,
        each carrying the group's box in well-frame pixels, its centre, how
        many observations it holds and from how many windows, the largest
        window side it spans (0 when it spans none), and the summed and
        largest clipped areas.
    """
    out: List[Dict[str, object]] = []
    for group in groups:
        if not len(group) or any(not one.clipped for one in group):
            continue
        top = min(one.bbox[0] for one in group)
        left = min(one.bbox[1] for one in group)
        bottom = max(one.bbox[2] for one in group)
        right = max(one.bbox[3] for one in group)
        areas = [int(one.area) for one in group]
        out.append({
            "top": int(top), "left": int(left),
            "bottom": int(bottom), "right": int(right),
            "height": int(bottom - top + 1), "width": int(right - left + 1),
            "centre_y": (top + bottom) / 2.0, "centre_x": (left + right) / 2.0,
            "observations": len(group),
            "windows": len({one.window.offset() for one in group}),
            "spanned_side": max(_spanned_side(one) for one in group),
            "area_total": sum(areas), "area_max": max(areas),
        })
    return tuple(out)


def number(groups: Sequence[Sequence[WindowObject]], *,
           strict: bool = True) -> Tuple[PlateObject, ...]:
    """One id per object, assigned in the well frame, deterministically.

    RASTER ORDER ON THE WELL-FRAME CENTROID -- top to bottom, then left to
    right -- rather than the order windows were segmented in. The id is a
    join key, so two runs over the same data have to produce the same
    numbers; ordering by anything the scheduler can vary would break that
    quietly and only in the results.

    Ids start at 1. Zero is background in every label image this came from,
    and an object numbered 0 would be invisible to any downstream mask
    comparison.

    :param groups: from :func:`sew`.
    :param strict: refuse a group with no complete observation. Turned off,
        such a group is dropped rather than numbered -- nothing with a
        truncated area is emitted -- and how many were dropped is logged as
        a warning.
    :raises ObjectsError: when a group holds only clipped observations:
        either the object is larger than the window overlap, or it is a
        fragment that matched no window's whole view of its nucleus. The
        message gives the object's extent and the remedy for each case.
    """
    chosen: List[Tuple[WindowObject, int]] = []
    unseen: List[Tuple[WindowObject, ...]] = []
    for group in groups:
        complete = [one for one in group if not one.clipped]
        if not complete:
            if len(group):
                unseen.append(tuple(group))
            continue
        chosen.append((min(complete, key=_preferred), len(group)))

    if unseen:
        refusal, warning = _unseen_report(unseen)
        if strict:
            raise ObjectsError(refusal)
        LOG.warning(warning)

    chosen.sort(key=lambda pair: (round(pair[0].centroid_y, 3),
                                  round(pair[0].centroid_x, 3))
                + _identity(pair[0]))
    return tuple(
        PlateObject(object_id=index,
                    centroid_x=one.centroid_x, centroid_y=one.centroid_y,
                    area=one.area, bbox=one.bbox,
                    window=one.window.offset(), n_observations=count)
        for index, (one, count) in enumerate(chosen, start=1))


def objects_frame(objects: Sequence[PlateObject]):
    """The ``ops_objects`` table, as a DataFrame.

    Imported locally so this module stays usable -- and testable -- without
    pandas, which is the same reason :mod:`spacr.scorecard` reaches for the
    standard library.

    :param objects: the numbered objects, one row each, in the order given.
    """
    import pandas as pd

    return pd.DataFrame([one.row() for one in objects])
