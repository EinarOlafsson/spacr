"""Gates — a shape drawn on a plot, which *is* a filter.

This is how the people who use spaCR already think about their data, because it
is how flow cytometry has worked for forty years: draw a threshold on a
histogram, draw a polygon round the cloud on a two-parameter scatter, name it,
and everything downstream is about the cells inside it. What makes it a gate
rather than a lasso is that it is a **predicate**, not a list of objects:

* it can be **re-applied to another dataset** — the next plate, the re-run, the
  replicate — and still mean the same thing;
* it can be **saved** and read back;
* it can be **sequenced** — gate on gate on gate — and the hierarchy is what
  carries the reasoning ("of the single cells, the live ones, of those the
  infected ones"), together with the percentage at each step that says whether
  the reasoning survived contact with the data.

That is exactly the distinction :mod:`spacr.selection` draws between a *filter*
and a *selection*, so a gate produces the first: :meth:`GateSet.filter_for`
returns a :class:`spacr.selection.DataFilter`, and every linked view honours it
the moment it is published. Nothing in the views needs to know what a gate is.

The clause
----------

:class:`GateClause` is duck-typed onto ``DataFilter`` the way
:class:`~spacr.selection.RangeFilter` and
:class:`~spacr.selection.CategoryFilter` are — a ``column`` attribute, a
``mask(frame)`` and a ``describe()``. Its ``column`` is ``"gate:<name>"``,
which makes ``DataFilter.add``'s replace-by-column rule do the right thing: a
re-drawn gate of the same name **replaces** its older self instead of stacking
two versions of the same shape.

A whole chain becomes ONE clause rather than one per gate, and that is
load-bearing rather than tidy: ``singlets`` might be ``area >= 100`` and its
child ``big`` might be ``area <= 500``, and as two range clauses on ``area``
the second would replace the first and silently widen the population.

Threshold and rectangle gates can *also* hand back genuine
:class:`~spacr.selection.RangeFilter` clauses (:meth:`Gate.range_filters`), for
a caller that wants the gate to appear in the Local Data Filter as ordinary
per-column controls the user can then nudge.

Geometry
--------

Point-in-polygon is the even–odd ray-casting rule, vectorised. A row whose x or
y is missing is **outside every gate** — not "unknown", not silently kept. An
object with no measurement is not an object inside the region, and letting it
through would put objects with no value into a population the user defined by
value; the same rule :class:`~spacr.selection.RangeFilter` applies to NaN.

Percentages
-----------
:meth:`GateSet.stats` reports, per gate, the count, the percentage **of its
parent** and the percentage of the whole table. Both, always: 90% of a parent
that is itself 2% of the table is 1.8% of the objects, and a hierarchy that
prints only one of the two numbers is the standard way a gating strategy
flatters itself.

No Qt in here — pure numpy and pandas, like :mod:`spacr.selection` and
:mod:`spacr.qt.widgets.graph_spec`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ...selection import DataFilter, RangeFilter

__all__ = [
    "GateError",
    "THRESHOLD", "RECTANGLE", "POLYGON", "GATE_KINDS",
    "Gate", "ThresholdGate", "RectGate", "PolygonGate",
    "gate_from_dict", "GateClause", "GateStats", "GateSet",
    "points_in_polygon",
]


class GateError(ValueError):
    """A gate that cannot mean anything, with the reason in the message.

    Raised where the gate is *built* rather than where it is applied: a gate
    with two vertices, a parent that does not exist or a name already taken
    would otherwise fail somewhere downstream, long after the drag that caused
    it.
    """


THRESHOLD = "threshold"
RECTANGLE = "rectangle"
POLYGON = "polygon"
ELLIPSE = "ellipse"

#: Every shape a gate can be, in the order the tool buttons list them.
GATE_KINDS: Tuple[str, ...] = (THRESHOLD, RECTANGLE, POLYGON, ELLIPSE)


def _clean_name(name: str) -> str:
    text = str(name).strip()
    if not text:
        raise GateError(
            "a gate needs a name. The name is what makes it re-appliable and "
            "what the hierarchy is read by; an unnamed region is a lasso")
    return text


def _ordered(low: Optional[float], high: Optional[float]
             ) -> Tuple[Optional[float], Optional[float]]:
    """``(low, high)``, swapped if the user dragged right to left."""
    if low is None or high is None:
        return low, high
    lo, hi = float(low), float(high)
    return (lo, hi) if lo <= hi else (hi, lo)


def _shift_bound(value: Optional[float], delta: float) -> Optional[float]:
    """Move one bound, leaving an open end open.

    ``None`` means "unbounded on this side", and adding to it would turn an
    open gate into a closed one the user never drew.
    """
    return None if value is None else float(value) + float(delta)


def _scale_bound(value: Optional[float], anchor: Optional[float],
                 factor: float) -> Optional[float]:
    """Scale one bound about ``anchor``. An open end stays open."""
    if value is None or anchor is None:
        return value
    return float(anchor) + (float(value) - float(anchor)) * float(factor)


def _check_factor(factor: float) -> None:
    """A resize factor has to be positive.

    Zero collapses the gate to a point and a negative one turns it inside
    out; both would be accepted silently by the arithmetic and would leave
    the user with a gate selecting nothing, or the complement of what they
    drew.
    """
    if float(factor) <= 0:
        raise GateError(
            f"a resize factor must be positive; got {factor!r}. Zero "
            f"collapses the gate and a negative value turns it inside out.")


def _numeric(frame: pd.DataFrame, column: str, what: str) -> np.ndarray:
    if column not in frame.columns:
        raise GateError(
            f"{what} names column {column!r}, which this table does not have. "
            f"A gate drawn on one dataset only re-applies to a table that "
            f"carries the same measurements")
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def points_in_polygon(x: np.ndarray, y: np.ndarray,
                      vertices: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Even–odd ray casting, vectorised over every point at once.

    :param vertices: the polygon, closed implicitly — the last vertex joins the
        first, so a caller does not have to repeat it (and repeating it is
        harmless).
    :returns: a boolean array. A point with a non-finite coordinate is
        **outside**, always.
    """
    px = np.asarray([float(v[0]) for v in vertices], dtype=float)
    py = np.asarray([float(v[1]) for v in vertices], dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    inside = np.zeros(x.shape, dtype=bool)
    count = len(px)
    previous = count - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        for current in range(count):
            xi, yi = px[current], py[current]
            xj, yj = px[previous], py[previous]
            # A horizontal edge never straddles the ray, so the division it
            # would divide by zero for is masked out before it is used.
            straddles = (yi > y) != (yj > y)
            crossing = (xj - xi) * (y - yi) / (yj - yi) + xi
            inside ^= straddles & (x < crossing)
            previous = current
    return inside & np.isfinite(x) & np.isfinite(y)


# ---------------------------------------------------------------------------
# The three shapes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Gate:
    """Base of the three gate shapes. Frozen: a gate is a value.

    :param name: unique within a :class:`GateSet`, and the thing the hierarchy
        and the filter clause are read by.
    :param parent: the gate this one is drawn *inside*, by name, or ``None``
        for a root gate. Sequential gating is this field and nothing else.
    """

    name: str
    parent: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _clean_name(self.name))
        parent = self.parent
        object.__setattr__(self, "parent",
                           str(parent).strip() if parent else None)
        if self.parent == self.name:
            raise GateError(
                f"gate {self.name!r} is its own parent; a gate is drawn inside "
                f"another one, not inside itself")

    @property
    def kind(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    @property
    def columns(self) -> Tuple[str, ...]:  # pragma: no cover - overridden
        raise NotImplementedError

    def mask(self, frame: pd.DataFrame) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def range_filters(self) -> Tuple[RangeFilter, ...]:
        """This gate as ordinary per-column range clauses, where it is one.

        Empty for a polygon, which is not a conjunction of ranges and must not
        pretend to be — a bounding box would quietly include the corners.
        """
        return ()

    def describe(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError

    def with_parent(self, parent: Optional[str]) -> "Gate":
        return replace(self, parent=parent)

    def rename(self, name: str) -> "Gate":
        return replace(self, name=name)

    # -- editing after the fact --------------------------------------------
    #
    # A gate you cannot adjust is a gate you redraw from scratch, which is
    # the single biggest gap in this editor. Both operations return a NEW
    # gate rather than mutating: these are frozen dataclasses, the GateSet
    # holds them by name, and an in-place edit would change a gate that
    # something else is already holding a reference to.
    #
    # Both are defined on the base so a caller can move ANY gate without
    # knowing which kind it has -- which is what the canvas drag handler
    # needs, since the user just clicks a shape.

    def translated(self, dx: float, dy: float) -> "Gate":
        """Return this gate moved by ``(dx, dy)`` in DATA units.

        Data units, not pixels: a gate is a statement about measurements,
        and moving it by pixels would mean it drifted whenever the axes
        rescaled.

        :param dx: shift along the gate's x measurement.
        :param dy: shift along its y measurement. Ignored by a one-column
            gate, which has no y.
        """
        raise NotImplementedError

    def scaled(self, factor: float, *,
               about: Optional[Tuple[float, float]] = None) -> "Gate":
        """Return this gate grown or shrunk about a fixed point.

        :param factor: >1 grows, <1 shrinks.
        :param about: the point held fixed; the gate's own centre by
            default, which is what "pull to expand" means when the user has
            not grabbed a particular edge.
        :raises GateError: a non-positive factor, which would invert or
            collapse the shape rather than resize it.
        """
        raise NotImplementedError

    def centre(self) -> Tuple[Optional[float], Optional[float]]:
        """The gate's middle in data units, for the default resize anchor.

        ``None`` on an axis the gate does not bound -- an open-ended
        threshold has no centre along its own column, and pretending it does
        would move it somewhere arbitrary on the first drag.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class ThresholdGate(Gate):
    """A cut on one column — the line dragged across a histogram.

    ``None`` on a bound means unbounded on that side, which is what a threshold
    dragged to the edge should mean rather than "exclude everything". At least
    one bound is required: a gate with neither is the whole population, and
    naming that is a way to lose track of it.
    """

    column: str = ""
    low: Optional[float] = None
    high: Optional[float] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not str(self.column).strip():
            raise GateError(
                f"threshold gate {self.name!r} has no column; a cut has to be "
                f"a cut on something")
        object.__setattr__(self, "column", str(self.column).strip())
        low, high = _ordered(self.low, self.high)
        object.__setattr__(self, "low", None if low is None else float(low))
        object.__setattr__(self, "high", None if high is None else float(high))
        if self.low is None and self.high is None:
            raise GateError(
                f"threshold gate {self.name!r} has neither a low nor a high "
                f"bound, so it selects everything. Drag a line, or delete it")

    @property
    def kind(self) -> str:
        return THRESHOLD

    @property
    def columns(self) -> Tuple[str, ...]:
        return (self.column,)

    def mask(self, frame: pd.DataFrame) -> np.ndarray:
        values = _numeric(frame, self.column, f"gate {self.name!r}")
        keep = np.isfinite(values)
        if self.low is not None:
            keep &= values >= self.low
        if self.high is not None:
            keep &= values <= self.high
        return keep

    def range_filters(self) -> Tuple[RangeFilter, ...]:
        return (RangeFilter(self.column, low=self.low, high=self.high),)

    def describe(self) -> str:
        if self.low is None:
            return f"{self.column} ≤ {self.high:g}"
        if self.high is None:
            return f"{self.column} ≥ {self.low:g}"
        return f"{self.low:g} ≤ {self.column} ≤ {self.high:g}"

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": THRESHOLD, "name": self.name, "parent": self.parent,
                "column": self.column, "low": self.low, "high": self.high}

    def translated(self, dx: float, dy: float) -> "ThresholdGate":
        """``dy`` is ignored: a threshold is a cut on ONE column, so it has
        no second axis to move along."""
        return replace(self, low=_shift_bound(self.low, dx),
                       high=_shift_bound(self.high, dx))

    def centre(self) -> Tuple[Optional[float], Optional[float]]:
        if self.low is None or self.high is None:
            # Open-ended, so there is no middle. Reported rather than
            # invented: a made-up centre would send the first resize
            # somewhere arbitrary.
            return None, None
        return (float(self.low) + float(self.high)) / 2.0, None

    def scaled(self, factor: float, *,
               about: Optional[Tuple[float, float]] = None) -> "ThresholdGate":
        _check_factor(factor)
        anchor = about[0] if about is not None else self.centre()[0]
        if anchor is None:
            # Nothing to scale about, and nothing sensible to do. Returned
            # unchanged rather than raising: the user dragged, and a
            # half-open gate simply has no width to grow.
            return self
        return replace(self, low=_scale_bound(self.low, anchor, factor),
                       high=_scale_bound(self.high, anchor, factor))


@dataclass(frozen=True)
class RectGate(Gate):
    """A rectangle on a two-parameter scatter — the quadrant gate.

    Kept as its own shape rather than as a four-vertex polygon so it can hand
    back real :class:`~spacr.selection.RangeFilter` clauses, which a rectangle
    genuinely is and a polygon genuinely is not.
    """

    x_column: str = ""
    y_column: str = ""
    x_low: Optional[float] = None
    x_high: Optional[float] = None
    y_low: Optional[float] = None
    y_high: Optional[float] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("x_column", "y_column"):
            if not str(getattr(self, name)).strip():
                raise GateError(
                    f"rectangle gate {self.name!r} has no {name}; a rectangle "
                    f"is drawn on two measurements")
            object.__setattr__(self, name, str(getattr(self, name)).strip())
        if self.x_column == self.y_column:
            raise GateError(
                f"rectangle gate {self.name!r} is drawn on {self.x_column!r} "
                f"against itself; every point would be on the diagonal")
        for lo_name, hi_name in (("x_low", "x_high"), ("y_low", "y_high")):
            low, high = _ordered(getattr(self, lo_name), getattr(self, hi_name))
            object.__setattr__(self, lo_name,
                               None if low is None else float(low))
            object.__setattr__(self, hi_name,
                               None if high is None else float(high))
        if all(getattr(self, n) is None
               for n in ("x_low", "x_high", "y_low", "y_high")):
            raise GateError(
                f"rectangle gate {self.name!r} has no bounds at all, so it "
                f"selects everything")

    @property
    def kind(self) -> str:
        return RECTANGLE

    @property
    def columns(self) -> Tuple[str, ...]:
        return (self.x_column, self.y_column)

    def mask(self, frame: pd.DataFrame) -> np.ndarray:
        what = f"gate {self.name!r}"
        x = _numeric(frame, self.x_column, what)
        y = _numeric(frame, self.y_column, what)
        keep = np.isfinite(x) & np.isfinite(y)
        for values, low, high in ((x, self.x_low, self.x_high),
                                  (y, self.y_low, self.y_high)):
            if low is not None:
                keep &= values >= low
            if high is not None:
                keep &= values <= high
        return keep

    def range_filters(self) -> Tuple[RangeFilter, ...]:
        return (RangeFilter(self.x_column, low=self.x_low, high=self.x_high),
                RangeFilter(self.y_column, low=self.y_low, high=self.y_high))

    def describe(self) -> str:
        def side(column, low, high):
            if low is None:
                return f"{column} ≤ {high:g}"
            if high is None:
                return f"{column} ≥ {low:g}"
            return f"{low:g} ≤ {column} ≤ {high:g}"
        return (f"{side(self.x_column, self.x_low, self.x_high)} and "
                f"{side(self.y_column, self.y_low, self.y_high)}")

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": RECTANGLE, "name": self.name, "parent": self.parent,
                "x_column": self.x_column, "y_column": self.y_column,
                "x_low": self.x_low, "x_high": self.x_high,
                "y_low": self.y_low, "y_high": self.y_high}

    def translated(self, dx: float, dy: float) -> "RectGate":
        return replace(self,
                       x_low=_shift_bound(self.x_low, dx),
                       x_high=_shift_bound(self.x_high, dx),
                       y_low=_shift_bound(self.y_low, dy),
                       y_high=_shift_bound(self.y_high, dy))

    def centre(self) -> Tuple[Optional[float], Optional[float]]:
        def middle(low, high):
            if low is None or high is None:
                return None
            return (float(low) + float(high)) / 2.0
        return middle(self.x_low, self.x_high), middle(self.y_low, self.y_high)

    def scaled(self, factor: float, *,
               about: Optional[Tuple[float, float]] = None) -> "RectGate":
        _check_factor(factor)
        own = self.centre()
        ax = about[0] if about is not None else own[0]
        ay = about[1] if about is not None else own[1]
        return replace(
            self,
            x_low=_scale_bound(self.x_low, ax, factor),
            x_high=_scale_bound(self.x_high, ax, factor),
            y_low=_scale_bound(self.y_low, ay, factor),
            y_high=_scale_bound(self.y_high, ay, factor))


@dataclass(frozen=True)
class PolygonGate(Gate):
    """A closed region on a two-parameter scatter.

    The shape a real population needs: cell clouds are not rectangles, and
    approximating one with a bounding box is how the corner debris gets counted
    as cells.
    """

    x_column: str = ""
    y_column: str = ""
    vertices: Tuple[Tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("x_column", "y_column"):
            if not str(getattr(self, name)).strip():
                raise GateError(
                    f"polygon gate {self.name!r} has no {name}; a polygon is "
                    f"drawn on two measurements")
            object.__setattr__(self, name, str(getattr(self, name)).strip())
        if self.x_column == self.y_column:
            raise GateError(
                f"polygon gate {self.name!r} is drawn on {self.x_column!r} "
                f"against itself")
        points = tuple((float(a), float(b)) for a, b in self.vertices)
        if len(points) > 1 and points[0] == points[-1]:
            # A closing vertex is accepted and dropped: the polygon closes
            # itself, and keeping the duplicate would make an edge of length 0.
            points = points[:-1]
        if len(points) < 3:
            raise GateError(
                f"polygon gate {self.name!r} has {len(points)} vertices; a "
                f"region needs at least three. Click three points, then close "
                f"the shape")
        # The shoelace area. Zero means the vertices are collinear (or all in
        # one place), which is a *line*: it would select nothing, and a gate
        # that selects nothing because of a slipped click is worth catching at
        # the click rather than three screens later.
        xs = np.array([p[0] for p in points], dtype=float)
        ys = np.array([p[1] for p in points], dtype=float)
        area = 0.5 * abs(float(np.dot(xs, np.roll(ys, -1))
                               - np.dot(ys, np.roll(xs, -1))))
        if not area > 0:
            raise GateError(
                f"polygon gate {self.name!r} has no area — its vertices are "
                f"all on one line, so nothing could ever be inside it")
        object.__setattr__(self, "vertices", points)

    @property
    def kind(self) -> str:
        return POLYGON

    @property
    def columns(self) -> Tuple[str, ...]:
        return (self.x_column, self.y_column)

    def mask(self, frame: pd.DataFrame) -> np.ndarray:
        what = f"gate {self.name!r}"
        x = _numeric(frame, self.x_column, what)
        y = _numeric(frame, self.y_column, what)
        return points_in_polygon(x, y, self.vertices)

    def bounds(self) -> Tuple[float, float, float, float]:
        """``(x_low, x_high, y_low, y_high)`` — for drawing, never for masking."""
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        return min(xs), max(xs), min(ys), max(ys)

    def describe(self) -> str:
        return (f"{len(self.vertices)}-sided region on {self.x_column} × "
                f"{self.y_column}")

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": POLYGON, "name": self.name, "parent": self.parent,
                "x_column": self.x_column, "y_column": self.y_column,
                "vertices": [list(v) for v in self.vertices]}

    def translated(self, dx: float, dy: float) -> "PolygonGate":
        return replace(self, vertices=tuple(
            (float(x) + dx, float(y) + dy) for x, y in self.vertices))

    def centre(self) -> Tuple[Optional[float], Optional[float]]:
        """The vertex centroid.

        Not the area centroid: for dragging, the vertex mean is stable,
        cheap, and is what the user sees as the middle of the shape. The
        area centroid of a strongly concave polygon can sit outside it,
        which makes a resize look like it moved.
        """
        xs = [float(x) for x, _ in self.vertices]
        ys = [float(y) for _, y in self.vertices]
        return sum(xs) / len(xs), sum(ys) / len(ys)

    def scaled(self, factor: float, *,
               about: Optional[Tuple[float, float]] = None) -> "PolygonGate":
        _check_factor(factor)
        anchor = about if about is not None else self.centre()
        ax, ay = float(anchor[0]), float(anchor[1])
        return replace(self, vertices=tuple(
            (ax + (float(x) - ax) * factor, ay + (float(y) - ay) * factor)
            for x, y in self.vertices))

    def with_vertex(self, index: int, x: float, y: float) -> "PolygonGate":
        """Move ONE vertex -- the per-vertex drag handle.

        :raises GateError: an index outside the polygon, which would
            otherwise silently move a different corner than the one grabbed.
        """
        if not (-len(self.vertices) <= index < len(self.vertices)):
            raise GateError(
                f"polygon {self.name!r} has {len(self.vertices)} vertices; "
                f"there is no vertex {index}")
        points = list(self.vertices)
        points[index] = (float(x), float(y))
        return replace(self, vertices=tuple(points))


@dataclass(frozen=True)
class EllipseGate(Gate):
    """An oval region — the shape a real population usually is.

    A cloud of cells is round-ish and a rectangle around it always takes
    corner debris with it. An ellipse is the cheapest shape that does not,
    and unlike a polygon it is defined by four numbers, so it can be dragged
    out in one gesture and resized without touching vertices.

    A circle is an ellipse with equal radii; there is no separate kind,
    because a "circle" that cannot be squashed is a shape the user has to
    delete and redraw the moment the axes are not comparable — and on a
    scatter of two different measurements they never are.
    """

    x_column: str = ""
    y_column: str = ""
    x_centre: float = 0.0
    y_centre: float = 0.0
    x_radius: float = 0.0
    y_radius: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("x_column", "y_column"):
            if not str(getattr(self, name)).strip():
                raise GateError(
                    f"ellipse gate {self.name!r} has no {name}; an ellipse is "
                    f"drawn on two measurements")
            object.__setattr__(self, name, str(getattr(self, name)).strip())
        if self.x_column == self.y_column:
            raise GateError(
                f"ellipse gate {self.name!r} is drawn on {self.x_column!r} "
                f"against itself")
        for name in ("x_radius", "y_radius"):
            value = float(getattr(self, name))
            if value <= 0:
                raise GateError(
                    f"ellipse gate {self.name!r} has {name}={value!r}; a "
                    f"radius of zero or less selects nothing")
            object.__setattr__(self, name, value)
        for name in ("x_centre", "y_centre"):
            object.__setattr__(self, name, float(getattr(self, name)))

    @property
    def kind(self) -> str:
        return ELLIPSE

    @property
    def columns(self) -> Tuple[str, ...]:
        return (self.x_column, self.y_column)

    def mask(self, frame: pd.DataFrame) -> np.ndarray:
        x = _numeric(frame, self.x_column, f"ellipse gate {self.name!r}")
        y = _numeric(frame, self.y_column, f"ellipse gate {self.name!r}")
        # Normalised radius: <= 1 is inside. Written this way rather than as
        # a distance so the two axes keep their own scales -- the whole point
        # of an ellipse over a circle on a two-measurement scatter.
        dx = (x - self.x_centre) / self.x_radius
        dy = (y - self.y_centre) / self.y_radius
        with np.errstate(invalid="ignore"):
            inside = (dx * dx + dy * dy) <= 1.0
        return np.nan_to_num(inside, nan=False).astype(bool)

    def describe(self) -> str:
        return (f"{self.x_column}/{self.y_column} within "
                f"({self.x_centre:g}±{self.x_radius:g}, "
                f"{self.y_centre:g}±{self.y_radius:g})")

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": ELLIPSE, "name": self.name, "parent": self.parent,
                "x_column": self.x_column, "y_column": self.y_column,
                "x_centre": self.x_centre, "y_centre": self.y_centre,
                "x_radius": self.x_radius, "y_radius": self.y_radius}

    def translated(self, dx: float, dy: float) -> "EllipseGate":
        return replace(self, x_centre=self.x_centre + float(dx),
                       y_centre=self.y_centre + float(dy))

    def centre(self) -> Tuple[Optional[float], Optional[float]]:
        return self.x_centre, self.y_centre

    def scaled(self, factor: float, *,
               about: Optional[Tuple[float, float]] = None) -> "EllipseGate":
        _check_factor(factor)
        f = float(factor)
        if about is None:
            # Grow in place: the centre is fixed and only the radii change.
            return replace(self, x_radius=self.x_radius * f,
                           y_radius=self.y_radius * f)
        ax, ay = float(about[0]), float(about[1])
        return replace(
            self,
            x_centre=ax + (self.x_centre - ax) * f,
            y_centre=ay + (self.y_centre - ay) * f,
            x_radius=self.x_radius * f, y_radius=self.y_radius * f)

    @classmethod
    def from_drag(cls, name: str, x_column: str, y_column: str,
                  x0: float, y0: float, x1: float, y1: float,
                  *, parent: Optional[str] = None) -> "EllipseGate":
        """Build the ellipse INSCRIBED in the dragged box.

        Inscribed rather than circumscribed, so the shape ends where the
        pointer did. A user who drags a box expects the shape to touch the
        corner they released at, not to extend past it.
        """
        return cls(name=name, parent=parent,
                   x_column=x_column, y_column=y_column,
                   x_centre=(float(x0) + float(x1)) / 2.0,
                   y_centre=(float(y0) + float(y1)) / 2.0,
                   x_radius=abs(float(x1) - float(x0)) / 2.0 or 1e-12,
                   y_radius=abs(float(y1) - float(y0)) / 2.0 or 1e-12)


_GATE_CLASSES = {THRESHOLD: ThresholdGate, RECTANGLE: RectGate,
                 POLYGON: PolygonGate, ELLIPSE: EllipseGate}



# ---------------------------------------------------------------------------
# Density clustering
#
# DBSCAN, not k-means: a scatter of cells has dense populations of unequal
# size sitting in sparse debris, which is exactly the shape DBSCAN was made
# for and exactly the shape k-means is bad at. It also does not need to be
# told how many populations there are, which is the number a user opening
# this dialog does not yet know.
#
# Clusters become REAL GATES rather than a separate kind of selection. A
# cluster is then editable, nestable, serialisable and usable as a
# DataFilter clause -- everything a hand-drawn gate can do -- because it IS
# one. A parallel "cluster selection" concept would have needed all of that
# rebuilt beside it.
# ---------------------------------------------------------------------------


class ClusterError(GateError):
    """Clustering cannot run, or produced nothing worth gating."""


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Return the hull vertices of ``points``, counter-clockwise.

    Andrew's monotone chain, written out rather than pulled from scipy: this
    module's contract is that importing it is cheap and Qt-free, and a
    scipy.spatial import at gate-drawing time costs more than thirty lines.

    :param points: ``(n, 2)`` array.
    :returns: ``(m, 2)`` array of hull vertices.
    """
    pts = np.unique(points, axis=0)
    if len(pts) <= 2:
        return pts
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def _half(sequence):
        out: List[np.ndarray] = []
        for point in sequence:
            while len(out) >= 2:
                (x1, y1), (x2, y2) = out[-2], out[-1]
                # Cross product of the last edge with the candidate edge.
                cross = ((x2 - x1) * (point[1] - y1)
                         - (y2 - y1) * (point[0] - x1))
                if cross > 0:
                    break
                out.pop()
            out.append(point)
        return out

    lower = _half(pts)
    upper = _half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def cluster_gates(frame: pd.DataFrame, x_column: str, y_column: str, *,
                  eps: float = 0.5, min_samples: int = 10,
                  scale: bool = True, max_clusters: int = 20,
                  name_prefix: str = "cluster",
                  parent: Optional[str] = None) -> List["PolygonGate"]:
    """Find dense populations with DBSCAN and return one gate per cluster.

    :param frame: the measurement table.
    :param x_column: the scatter's x measurement.
    :param y_column: the scatter's y measurement.
    :param eps: DBSCAN neighbourhood radius. In SCALED units when
        ``scale`` is true, which is what makes one default work across
        measurements whose ranges differ by orders of magnitude.
    :param min_samples: points needed to seed a cluster.
    :param scale: standardise both axes before clustering. On by default
        because ``cell_area`` runs to thousands and ``eccentricity`` to one,
        and unscaled DBSCAN on that pair clusters on area alone.
    :param max_clusters: refuse beyond this many. Two hundred gates is not a
        result, it is a wrongly-tuned eps, and drawing them all makes the
        editor unusable while the user works out why.
    :param name_prefix: gate names are ``<prefix> 1``, ``<prefix> 2``, ...
    :param parent: parent gate name, so clusters can be found inside a gate.
    :returns: one :class:`PolygonGate` per cluster, largest first. Empty when
        DBSCAN finds only noise.
    :raises ClusterError: a missing column, no usable rows, bad parameters,
        or more clusters than ``max_clusters``.
    """
    try:
        from sklearn.cluster import DBSCAN
    except Exception as exc:                       # pragma: no cover
        raise ClusterError(
            f"clustering needs scikit-learn ({exc})") from exc

    for column in (x_column, y_column):
        if column not in frame.columns:
            raise ClusterError(f"{column!r} is not a column of this table")
    if x_column == y_column:
        raise ClusterError("clustering needs two different measurements")
    if eps <= 0:
        raise ClusterError(f"eps must be positive, got {eps!r}")
    if int(min_samples) < 2:
        raise ClusterError(
            f"min_samples must be at least 2, got {min_samples!r}")

    data = frame[[x_column, y_column]].apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < int(min_samples):
        raise ClusterError(
            f"only {len(data)} rows have both {x_column!r} and {y_column!r}; "
            f"fewer than min_samples={min_samples}")

    raw = data.to_numpy(dtype=float)
    spread = raw.std(axis=0)
    # A constant axis is refused rather than worked around. Every cluster on
    # it is a straight line, every hull is collinear and has no area, and the
    # honest result would be an empty list -- which reads as "clustering is
    # broken" rather than "this measurement is the same for every object".
    flat = [column for column, sd in zip((x_column, y_column), spread)
            if sd == 0]
    if flat:
        raise ClusterError(
            f"{flat[0]!r} is the same value for every object here, so there "
            f"is nothing to cluster along it. Pick a measurement that varies.")
    if scale:
        work = (raw - raw.mean(axis=0)) / spread
    else:
        work = raw

    labels = DBSCAN(eps=float(eps),
                    min_samples=int(min_samples)).fit_predict(work)
    found = [lab for lab in np.unique(labels) if lab != -1]
    if not found:
        return []
    if len(found) > int(max_clusters):
        raise ClusterError(
            f"DBSCAN found {len(found)} clusters, more than max_clusters="
            f"{max_clusters}. Raise eps to merge them, or raise "
            f"max_clusters if this is really what you meant.")

    # Largest first, so the populations that matter are drawn and named
    # before the specks.
    found.sort(key=lambda lab: int((labels == lab).sum()), reverse=True)

    gates: List[PolygonGate] = []
    for index, label in enumerate(found, start=1):
        hull = _convex_hull(raw[labels == label])
        if len(hull) < 3:
            # A collinear cluster has no area. Skipped rather than widened
            # into a fake polygon, which would select rows outside it.
            continue
        gates.append(PolygonGate(
            name=f"{name_prefix} {index}",
            parent=parent,
            x_column=x_column,
            y_column=y_column,
            vertices=tuple((float(px), float(py)) for px, py in hull),
        ))
    return gates


def gate_from_dict(payload: Mapping[str, Any]) -> Gate:
    """Rebuild one gate from :meth:`Gate.to_dict`.

    :raises GateError: on an unknown or missing ``kind``, naming what was
        found — a gate file written by a newer build must fail with a sentence
        rather than a ``KeyError``.
    """
    data = dict(payload)
    kind = str(data.pop("kind", "")).strip()
    if kind not in _GATE_CLASSES:
        raise GateError(
            f"unknown gate kind {kind!r}; this build understands "
            f"{', '.join(GATE_KINDS)}")
    cls = _GATE_CLASSES[kind]
    if kind == POLYGON and "vertices" in data:
        data["vertices"] = tuple(tuple(v) for v in data["vertices"])
    fields = {f for f in cls.__dataclass_fields__}
    unknown = set(data) - fields
    if unknown:
        raise GateError(
            f"{kind} gate {data.get('name')!r} carries "
            f"{', '.join(sorted(unknown))}, which this build does not "
            f"understand; it was probably written by a newer spaCR")
    return cls(**data)


# ---------------------------------------------------------------------------
# The clause every linked view honours
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateClause:
    """A whole gate chain as one :class:`~spacr.selection.DataFilter` clause.

    Duck-typed onto ``DataFilter`` exactly as
    :class:`~spacr.selection.RangeFilter` is — ``column``, ``mask``,
    ``describe`` — so no view has to learn what a gate is.

    One clause for the whole chain rather than one per gate: two gates on the
    same column (``area ≥ 100`` and its child ``area ≤ 500``) would otherwise
    replace each other under ``DataFilter.add``'s replace-by-column rule and
    silently widen the population.
    """

    gates: Tuple[Gate, ...]

    def __post_init__(self) -> None:
        if not self.gates:
            raise GateError("a gate clause needs at least one gate")
        object.__setattr__(self, "gates", tuple(self.gates))

    @property
    def column(self) -> str:
        """``"gate:<leaf name>"`` — the key ``DataFilter.add`` replaces on.

        Not a real column, and deliberately not one: a re-drawn gate of the
        same name replaces its older self, and two different gates never
        collide.
        """
        return f"gate:{self.gates[-1].name}"

    @property
    def name(self) -> str:
        return self.gates[-1].name

    def mask(self, frame: pd.DataFrame) -> np.ndarray:
        keep = np.ones(len(frame), dtype=bool)
        for gate in self.gates:
            keep &= gate.mask(frame)
        return keep

    def describe(self) -> str:
        chain = " ⊂ ".join(g.name for g in reversed(self.gates))
        return f"gate {chain} ({self.gates[-1].describe()})"


# ---------------------------------------------------------------------------
# The hierarchy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateStats:
    """One row of the gating hierarchy: how many survived, and out of what.

    :param n_parent: the population this gate was drawn *inside*. For a root
        gate, the whole table.
    :param of_parent: the fraction of that population, in ``[0, 1]``.
    :param of_total: the fraction of the whole table. Both are reported
        because 90% of a parent that is 2% of the table is 1.8% of the objects.
    """

    name: str
    depth: int
    n_total: int
    n_parent: int
    n_in: int

    @property
    def of_parent(self) -> float:
        return (self.n_in / self.n_parent) if self.n_parent else float("nan")

    @property
    def of_total(self) -> float:
        return (self.n_in / self.n_total) if self.n_total else float("nan")

    def describe(self) -> str:
        indent = "    " * self.depth
        parent = ("—" if not self.n_parent
                  else f"{100.0 * self.of_parent:.1f}% of parent")
        total = ("" if not self.n_total
                 else f", {100.0 * self.of_total:.1f}% of all")
        return f"{indent}{self.name}: {self.n_in:,} ({parent}{total})"


@dataclass
class GateSet:
    """An ordered, named collection of gates, with parents.

    Mutable — it is a thing the user edits — but every gate in it is frozen and
    it round-trips through :meth:`to_json`. Order is definition order, which is
    also a valid topological order because a parent has to exist before a child
    can name it.
    """

    gates: List[Gate] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Re-added one at a time through `add`, so a set built from a list —
        # or read back from a file — gets the same parent and cycle checks a
        # set built by clicking does.
        incoming = list(self.gates)
        self.gates = []
        for gate in incoming:
            self.add(gate)

    # -- editing -----------------------------------------------------------
    def add(self, gate: Gate) -> "GateSet":
        """Add ``gate``, replacing any gate of the same name.

        Replacing rather than appending is what makes re-drawing a gate an
        edit: the children keep pointing at the name, so adjusting a threshold
        moves everything below it rather than orphaning it.

        :raises GateError: if the parent does not exist, or if the gate would
            close a cycle.
        """
        if gate.parent and gate.parent not in self.names:
            raise GateError(
                f"gate {gate.name!r} is drawn inside {gate.parent!r}, which "
                f"does not exist. The gates are: "
                f"{', '.join(self.names) or '(none)'}")
        before = list(self.gates)
        self.gates = [g for g in self.gates if g.name != gate.name] + [gate]
        try:
            self.path(gate.name)
        except GateError:
            # Put the set back exactly as it was. Re-drawing a gate into a
            # cycle must not also delete the gate it was replacing.
            self.gates = before
            raise
        return self

    def remove(self, name: str, *, cascade: bool = True) -> "GateSet":
        """Drop ``name``.

        :param cascade: also drop everything gated inside it. On by default:
            a child whose parent is gone is a gate on a population that no
            longer exists, and silently re-rooting it would change what it
            means without saying so.
        """
        name = str(name)
        if name not in self.names:
            return self
        doomed = {name}
        if cascade:
            changed = True
            while changed:
                changed = False
                for gate in self.gates:
                    if gate.parent in doomed and gate.name not in doomed:
                        doomed.add(gate.name)
                        changed = True
        else:
            for gate in self.gates:
                if gate.parent == name:
                    raise GateError(
                        f"gate {name!r} has {gate.name!r} inside it. Remove "
                        f"that first, or remove with cascade")
        self.gates = [g for g in self.gates if g.name not in doomed]
        return self

    def clear(self) -> "GateSet":
        self.gates = []
        return self

    def get(self, name: str) -> Gate:
        for gate in self.gates:
            if gate.name == str(name):
                return gate
        raise GateError(
            f"there is no gate called {name!r}; the gates are "
            f"{', '.join(self.names) or '(none)'}")

    # -- reading -----------------------------------------------------------
    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(g.name for g in self.gates)

    @property
    def is_empty(self) -> bool:
        return not self.gates

    def __len__(self) -> int:
        return len(self.gates)

    def __contains__(self, name: object) -> bool:
        return str(name) in self.names

    def children(self, name: Optional[str]) -> Tuple[Gate, ...]:
        """The gates drawn directly inside ``name`` (``None`` for the roots)."""
        return tuple(g for g in self.gates
                     if g.parent == (str(name) if name else None))

    def path(self, name: str) -> Tuple[Gate, ...]:
        """The chain from the outermost gate down to ``name``, inclusive.

        :raises GateError: on a cycle, naming the gates in it — a hierarchy
            that loops would otherwise hang whatever walked it.
        """
        gate = self.get(name)
        chain: List[Gate] = [gate]
        seen = {gate.name}
        while chain[0].parent:
            parent = chain[0].parent
            if parent in seen:
                raise GateError(
                    f"the gates {', '.join(sorted(seen))} are drawn inside "
                    f"each other in a loop; a gate hierarchy is a tree")
            seen.add(parent)
            chain.insert(0, self.get(parent))
        return tuple(chain)

    def depth(self, name: str) -> int:
        return len(self.path(name)) - 1

    def order(self) -> Tuple[Gate, ...]:
        """Every gate, parents before children, siblings in definition order.

        The order the hierarchy is drawn and the percentages are read in.
        """
        out: List[Gate] = []

        def walk(parent: Optional[str]) -> None:
            for gate in self.children(parent):
                out.append(gate)
                walk(gate.name)

        walk(None)
        return tuple(out)

    # -- applying ----------------------------------------------------------
    def mask(self, frame: pd.DataFrame, name: str) -> np.ndarray:
        """The rows of ``frame`` inside ``name`` **and every gate above it**.

        :raises GateError: naming the missing column if ``frame`` does not
            carry what the chain needs — a gate re-applied to a table without
            the measurement is a mistake worth an exception, not a silently
            empty population.
        """
        keep = np.ones(len(frame), dtype=bool)
        for gate in self.path(name):
            keep &= gate.mask(frame)
        return keep

    def population(self, frame: pd.DataFrame, name: str) -> pd.DataFrame:
        """``frame`` narrowed to the gate and its ancestors."""
        return frame.loc[self.mask(frame, name)]

    def clause_for(self, name: str) -> GateClause:
        """The whole chain as one filter clause."""
        return GateClause(self.path(name))

    def filter_for(self, name: str,
                   base: Optional[DataFilter] = None) -> DataFilter:
        """A :class:`~spacr.selection.DataFilter` carrying this gate.

        :param base: an existing filter to add the clause to. The gate is added
            rather than replacing what is there, so a gate and the Local Data
            Filter's own clauses compose — which is what "the gate becomes a
            filter every linked view honours" has to mean in a screen that also
            has a filter panel.
        """
        data_filter = base if base is not None else DataFilter()
        return data_filter.add(self.clause_for(name))

    def stats(self, frame: pd.DataFrame) -> Tuple[GateStats, ...]:
        """Count and percentages for every gate, parents first."""
        total = int(len(frame))
        counts: Dict[str, int] = {}
        out: List[GateStats] = []
        for gate in self.order():
            n_in = int(self.mask(frame, gate.name).sum())
            counts[gate.name] = n_in
            n_parent = counts.get(gate.parent, total) if gate.parent else total
            out.append(GateStats(name=gate.name, depth=self.depth(gate.name),
                                 n_total=total, n_parent=n_parent, n_in=n_in))
        return tuple(out)

    def report(self, frame: pd.DataFrame) -> str:
        """The hierarchy as text, one gate per line, indented by depth."""
        rows = self.stats(frame)
        if not rows:
            return "no gates"
        head = f"{len(frame):,} objects"
        return "\n".join([head] + [row.describe() for row in rows])

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {"gates": [g.to_dict() for g in self.gates]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GateSet":
        rows = dict(payload).get("gates") or []
        return cls([gate_from_dict(row) for row in rows])

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "GateSet":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise GateError(f"this is not a gate file: {exc}") from None
        return cls.from_dict(payload)

    def save(self, path: str) -> str:
        """Write the gates to ``path`` as JSON. Returns the path."""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.to_json())
        return path

    @classmethod
    def load(cls, path: str) -> "GateSet":
        """Read a gate file written by :meth:`save`."""
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_json(handle.read())

    def describe(self) -> str:
        if not self.gates:
            return "no gates"
        return " · ".join(f"{g.name}: {g.describe()}" for g in self.order())
