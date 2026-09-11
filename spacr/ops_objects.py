"""One object list for a well, sewn across window seams and numbered once.

SEGMENT, SEW, NUMBER -- the three steps between a composite and an object
table. The compose step gives a mosaic that is built a
window at a time and never materialised whole; this is what segmentation and
numbering do on top of it:

    B2  SEGMENT ONCE, on the composite, window by window.
    B3  SEW THE LABELS ACROSS WINDOW SEAMS. An object straddling a boundary
        is one object, matched on its well-frame centroid.
    B4  PLATE-LEVEL OBJECT NUMBERS, ASSIGNED IN THE WELL FRAME. Per-window
        numbering would give two windows an object #1 apiece.

THE SEAM PROBLEM IS ALREADY HALF-SOLVED BY B1, and saying how avoids
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
overlap, which every window clips. :func:`sew` refuses those rather than
emitting a fragment, because a fragment looks exactly like a small nucleus
and would be counted as one for the rest of the run.

THE IDS ARE A JOIN KEY, so they are assigned deterministically -- raster
order on the well-frame centroid -- and not by iteration order over a dict or
by whichever window happened to be segmented first. 372: "This id, and the
centroid beside it, is the join key for every later phase." Two runs over the
same data must produce the same numbers or nothing downstream can be
compared.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .ops_compose import ComposeError, Window

__all__ = [
    "ObjectsError",
    "WindowObject", "PlateObject",
    "objects_in_window", "segment_windows", "sew", "number", "objects_frame",
    "DEFAULT_CENTROID_TOLERANCE", "DEFAULT_AREA_RATIO",
]

LOG = logging.getLogger("spacr.ops_objects")

#: How far apart two observations of one nucleus may sit, in pixels of the
#: well frame, and still be called the same nucleus. Generous relative to a
#: centroid's precision (sub-pixel for a clean mask) and tight relative to
#: nuclear spacing, which is what makes the match unambiguous rather than
#: merely plausible.
DEFAULT_CENTROID_TOLERANCE = 3.0

#: How different two observations' areas may be and still be the same object.
#: Both saw the whole nucleus, so they should agree closely; a large
#: disagreement means one of them is clipped or they are two nuclei.
DEFAULT_AREA_RATIO = 0.8


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
    """One nucleus, numbered for the whole well. The join key for Phase C.

    The field names are the ``ops_objects`` columns Phase D specifies, so the
    table is written from these without a translation layer.
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

    found: List[WindowObject] = []
    values = np.unique(labels)
    values = values[values > 0]
    if values.size == 0:
        return ()

    height, width = labels.shape
    canvas_height, canvas_width = canvas if canvas else (None, None)
    for value in values:
        mask = labels == value
        rows, cols = np.nonzero(mask)
        top, bottom = int(rows.min()), int(rows.max())
        left, right = int(cols.min()), int(cols.max())
        # Touching an interior seam means this window cut the object short.
        touches = []
        if top == 0:
            touches.append(window.top > 0)
        if left == 0:
            touches.append(window.left > 0)
        if bottom == height - 1:
            touches.append(canvas_height is None
                           or window.top + height < canvas_height)
        if right == width - 1:
            touches.append(canvas_width is None
                           or window.left + width < canvas_width)
        found.append(WindowObject(
            window=window, label=int(value),
            centroid_y=float(rows.mean()) + window.top,
            centroid_x=float(cols.mean()) + window.left,
            area=int(mask.sum()),
            bbox=(top + window.top, left + window.left,
                  bottom + window.top, right + window.left),
            clipped=any(touches)))
    return tuple(found)


def segment_windows(windows: Iterable[Window],
                    segment: Callable[[Window], np.ndarray], *,
                    canvas: Optional[Tuple[int, int]] = None,
                    ) -> Tuple[WindowObject, ...]:
    """B2: run a segmenter over each window and collect the observations.

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


def _same_object(a: WindowObject, b: WindowObject, *,
                 tolerance: float, area_ratio: float) -> bool:
    """Whether two observations are the same nucleus seen twice.

    BOTH TESTS, not just the centroid. Two touching nuclei can have centroids
    three pixels apart; what they cannot easily have is three pixels apart
    AND near-identical areas. Requiring both makes a false match need two
    coincidences rather than one.
    """
    if abs(a.centroid_y - b.centroid_y) > tolerance:
        return False
    if abs(a.centroid_x - b.centroid_x) > tolerance:
        return False
    larger = max(a.area, b.area)
    if larger <= 0:
        return False
    return min(a.area, b.area) / larger >= area_ratio


def sew(observations: Sequence[WindowObject], *,
        tolerance: float = DEFAULT_CENTROID_TOLERANCE,
        area_ratio: float = DEFAULT_AREA_RATIO,
        ) -> Tuple[Tuple[WindowObject, ...], ...]:
    """B3: group observations that are the same nucleus.

    Every group is one physical object. A nucleus in a window overlap is
    grouped from two (or four, at a corner) observations; one in a window's
    interior forms a group of one.

    THE CLIPPED ONES ARE THE POINT. An observation whose mask ran into an
    interior seam did not see the whole nucleus, so its area and centroid are
    both wrong. It is kept only long enough to be matched to a complete
    observation of the same nucleus and is then dropped in favour of it. A
    clipped observation matching nothing complete means no window saw that
    nucleus whole, which :func:`number` refuses.

    Matching a clipped observation uses the centroid only: its area is
    truncated by definition, so requiring areas to agree would prevent
    exactly the match that rescues it.

    :returns: groups, each a tuple of observations, in no particular order --
        :func:`number` imposes the order that matters.
    """
    remaining = list(observations)
    groups: List[Tuple[WindowObject, ...]] = []
    while remaining:
        seed = remaining.pop()
        group = [seed]
        changed = True
        while changed:
            changed = False
            for other in list(remaining):
                if any(_same_object(
                        member, other, tolerance=tolerance,
                        area_ratio=(0.0 if (member.clipped or other.clipped)
                                    else area_ratio))
                       for member in group):
                    group.append(other)
                    remaining.remove(other)
                    changed = True
        groups.append(tuple(group))
    return tuple(groups)


def number(groups: Sequence[Sequence[WindowObject]], *,
           strict: bool = True) -> Tuple[PlateObject, ...]:
    """B4: one id per object, assigned in the well frame, deterministically.

    RASTER ORDER ON THE WELL-FRAME CENTROID -- top to bottom, then left to
    right -- rather than the order windows were segmented in. The id is a
    join key, so two runs over the same data have to produce the same
    numbers; ordering by anything the scheduler can vary would break that
    quietly and only in the results.

    Ids start at 1. Zero is background in every label image this came from,
    and an object numbered 0 would be invisible to any downstream mask
    comparison.

    :param groups: from :func:`sew`.
    :param strict: refuse a group with no complete observation. Turn it off
        only to inspect a bad run -- the objects it then emits are fragments
        with truncated areas.
    :raises ObjectsError: when a group holds only clipped observations, which
        means the object is larger than the window overlap.
    """
    chosen: List[Tuple[WindowObject, int]] = []
    unseen: List[WindowObject] = []
    for group in groups:
        complete = [one for one in group if not one.clipped]
        if not complete:
            unseen.append(max(group, key=lambda one: one.area))
            continue
        # The largest complete observation: all of them saw the whole object,
        # so they differ only by segmentation noise at its border.
        chosen.append((max(complete, key=lambda one: one.area), len(group)))

    if unseen and strict:
        worst = max(unseen, key=lambda one: one.area)
        raise ObjectsError(
            f"{len(unseen)} object(s) were clipped by every window that saw "
            f"them, so none saw one whole -- the largest spans at least "
            f"{worst.bbox[2] - worst.bbox[0] + 1} x "
            f"{worst.bbox[3] - worst.bbox[1] + 1} px near "
            f"({worst.centroid_x:.0f}, {worst.centroid_y:.0f}). The window "
            f"overlap has to exceed the largest object; raise it and "
            f"re-run. Emitting these would count fragments as nuclei.")

    chosen.sort(key=lambda pair: (round(pair[0].centroid_y, 3),
                                  round(pair[0].centroid_x, 3)))
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
    """
    import pandas as pd

    return pd.DataFrame([one.row() for one in objects])
