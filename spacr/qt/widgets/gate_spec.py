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
    "THRESHOLD", "RECTANGLE", "POLYGON", "CYLINDER", "PRISM", "COMPOSITE",
    "COMPOSITE_OPS", "GATE_KINDS", "CylinderGate", "PrismGate",
    "CompositeGate", "BoxGate", "EllipseGate", "PolygonGate", "RectGate",
    "ThresholdGate", "Gate",
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
#: Click a point and let the data decide the shape. Not a drag: the user
#: picks a seed and the gate grows to fit the population around it.
WAND = "wand"
#: A box in three measurements -- what a gate is in the volume.
BOX = "box"
#: An oval drawn on one plane of the volume and extended along the third.
CYLINDER = "cylinder"
#: A polygon drawn on one plane of the volume and extended along the third.
PRISM = "prism"
#: Other gates combined: union, intersection or difference.
COMPOSITE = "composite"

#: How a composite gate combines its operands.
COMPOSITE_OPS: Tuple[str, ...] = ("union", "intersect", "subtract")

#: Every shape a gate can be, in the order the tool buttons list them.
GATE_KINDS: Tuple[str, ...] = (THRESHOLD, RECTANGLE, POLYGON, ELLIPSE,
                              WAND, BOX, CYLINDER, PRISM, COMPOSITE)


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


def _midpoint(low: Optional[float], high: Optional[float]) -> Optional[float]:
    """The centre of one axis of a gate, or ``None`` if it has no centre.

    An open end has no midpoint: a gate bounded only below extends to
    infinity, so averaging its one finite bound with nothing would report a
    centre that sits on the edge of the region rather than in it. Both bounds
    are required, for the same reason :func:`_shift_bound` refuses to close an
    open end.

    Called by :meth:`GateSpec.centre`, which referenced it before it was
    written -- every call raised ``NameError``.
    """
    if low is None or high is None:
        return None
    return (float(low) + float(high)) / 2.0


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

#: The visible axis limits: ``(x_low, x_high, y_low, y_high)``. Handles need
#: them only to place anchors on sides a gate leaves unbounded.
View = Tuple[float, float, float, float]


@dataclass(frozen=True)
class Handle:
    """One draggable anchor on a gate.

    ``role`` is what the anchor MEANS, not where it is: "x_low", "vertex:3",
    "x_low,y_high". Position is derived from the gate and the view and is
    therefore never stored -- a handle that remembered a coordinate would go
    stale the moment the gate moved.
    """

    x: float
    y: float
    role: str
    #: Corner handles change two bounds at once; side handles change one.
    #: The canvas draws them differently so the user can tell before pulling.
    corner: bool = False


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

    def thresholds(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """``{column: (low, high)}`` for every threshold this gate can take.

        NOT :meth:`PolygonGate.bounds`, which is that shape's bounding box
        and predates this -- hence the different name rather than an
        override that would have changed what a caller of the old one got
        back.

        Instruction 52 point 4's read side: "the user should also be able to
        set thresholds for each individual gate for the measurements they are
        defined by". Derived from :meth:`range_filters`, so a gate whose shape
        is not a conjunction of ranges offers only the bounds it really has --
        a cylinder offers its NORMAL and not its oval, which is exactly the
        axis the user needs in order to bound its height.
        """
        return {clause.column: (clause.low, clause.high)
                for clause in self.range_filters()}

    def with_threshold(self, column: str, low: Optional[float],
                    high: Optional[float]) -> "Gate":
        """Return this gate with ``column`` bounded to ``low``..``high``.

        :raises GateError: when this gate has no bound on ``column``. A gate
            that silently ignored the edit would leave the panel showing a
            number the gate does not honour.
        """
        raise GateError(
            f"{self.kind} gate {self.name!r} has no bound on {column!r}; it "
            f"can be given "
            + (", ".join(self.thresholds()) or "no thresholds at all"))

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

    # -- anchor points ----------------------------------------------------
    # Resizing is "pull a corner or a side", so every kind has to be able to
    # say where its corners and sides ARE, and what it becomes when one is
    # dragged. Both live here rather than in the canvas because they are
    # geometry -- no axes, no pixels, no Qt -- and because a canvas that
    # special-cased four gate kinds inside a mouse handler is how the drag
    # code became unreadable the first time.

    def handles(self, view: "View") -> Tuple["Handle", ...]:
        """The draggable anchor points, in data units.

        :param view: the visible axis limits, used ONLY to place handles on
            sides the gate does not bound. An unbounded side is at infinity
            and cannot be drawn or grabbed there; putting its handle at the
            edge of the view lets the user pull a bound onto a gate that
            never had one.
        :returns: the anchors. Empty when the gate has nothing to pull.
        """
        return ()

    def with_handle(self, role: str, x: float, y: float) -> "Gate":
        """Return this gate with the ``role`` anchor moved to ``(x, y)``.

        :param role: a role from :meth:`handles`.
        :param x: the new position along the gate's x measurement.
        :param y: along its y measurement.
        :returns: a new gate, or ``self`` when the drag would collapse the
            shape. Refusing beats raising here: this runs on mouse-release,
            and a gate that snaps back is a clear "that is too small" while
            a traceback out of an event handler is not.
        :raises GateError: a role this gate does not have, which is a bug in
            the caller rather than something the user did.
        """
        raise GateError(f"{type(self).__name__} has no handle {role!r}")


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

    def with_threshold(self, column: str, low: Optional[float],
                    high: Optional[float]) -> "ThresholdGate":
        if column != self.column:
            return super().with_threshold(column, low, high)
        low, high = _ordered(low, high)
        return replace(self, low=low, high=high)

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

    def handles(self, view: "View") -> Tuple["Handle", ...]:
        """One anchor per bound, at the middle of the view's height.

        An unbounded side gets no handle. A threshold with no upper bound is
        open to infinity, and an anchor at the edge of the view would look
        like a bound the gate does not have -- the user would drag it and
        discover they had just invented one.
        """
        _x0, _x1, y0, y1 = view
        mid = (float(y0) + float(y1)) / 2.0
        out = []
        if self.low is not None:
            out.append(Handle(float(self.low), mid, "low"))
        if self.high is not None:
            out.append(Handle(float(self.high), mid, "high"))
        return tuple(out)

    def with_handle(self, role: str, x: float, y: float) -> "ThresholdGate":
        if role not in ("low", "high"):
            raise GateError(f"threshold gate has no handle {role!r}")
        low, high = self.low, self.high
        if role == "low":
            low = float(x)
        else:
            high = float(x)
        if low is not None and high is not None and low > high:
            # Dragged past the other bound. Swapping beats refusing: the
            # user's intent is unambiguous and a gate that will not invert
            # feels stuck at exactly the moment they are trying to fix it.
            low, high = high, low
        return replace(self, low=low, high=high)

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

    def thresholds(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """Both sides, whether or not they are currently set."""
        return {self.x_column: (self.x_low, self.x_high),
                self.y_column: (self.y_low, self.y_high)}

    def with_threshold(self, column: str, low: Optional[float],
                    high: Optional[float]) -> "RectGate":
        field_of = {self.x_column: ("x_low", "x_high"),
                    self.y_column: ("y_low", "y_high")}
        if column not in field_of:
            return super().with_threshold(column, low, high)
        low, high = _ordered(low, high)
        low_name, high_name = field_of[column]
        return replace(self, **{low_name: low, high_name: high})

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

    def bounds_in(self, view: "View") -> Tuple[float, float, float, float]:
        """The corners as drawn: unbounded sides fall back to the view edge.

        A rectangle open on one side really does extend forever, so it is
        drawn to the edge of the axes. That edge is where its handle goes,
        and pulling it there gives the gate a bound it did not have -- which
        is the only way to close an open side without redrawing the gate.
        """
        vx0, vx1, vy0, vy1 = (float(v) for v in view)
        x0 = vx0 if self.x_low is None else float(self.x_low)
        x1 = vx1 if self.x_high is None else float(self.x_high)
        y0 = vy0 if self.y_low is None else float(self.y_low)
        y1 = vy1 if self.y_high is None else float(self.y_high)
        return x0, x1, y0, y1

    def handles(self, view: "View") -> Tuple["Handle", ...]:
        """Four corners and four side midpoints."""
        x0, x1, y0, y1 = self.bounds_in(view)
        xm, ym = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        return (
            Handle(x0, y0, "x_low,y_low", corner=True),
            Handle(x1, y0, "x_high,y_low", corner=True),
            Handle(x0, y1, "x_low,y_high", corner=True),
            Handle(x1, y1, "x_high,y_high", corner=True),
            Handle(x0, ym, "x_low"),
            Handle(x1, ym, "x_high"),
            Handle(xm, y0, "y_low"),
            Handle(xm, y1, "y_high"),
        )

    def with_handle(self, role: str, x: float, y: float) -> "RectGate":
        parts = [p for p in str(role).split(",") if p]
        if not parts or any(p not in ("x_low", "x_high", "y_low", "y_high")
                            for p in parts):
            raise GateError(f"rectangle gate has no handle {role!r}")
        values = dict(x_low=self.x_low, x_high=self.x_high,
                      y_low=self.y_low, y_high=self.y_high)
        for part in parts:
            values[part] = float(x) if part.startswith("x_") else float(y)
        for lo, hi in (("x_low", "x_high"), ("y_low", "y_high")):
            a, b = values[lo], values[hi]
            if a is not None and b is not None and a > b:
                # Pulled through the opposite side. The user has turned the
                # rectangle inside out, which they clearly meant; keeping it
                # a rectangle is the only correction needed.
                values[lo], values[hi] = b, a
        return replace(self, **values)

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

    def handles(self, view: "View") -> Tuple["Handle", ...]:
        """One anchor per vertex. A polygon has no sides to pull that are not
        already two vertices, so there are no side handles."""
        return tuple(Handle(float(vx), float(vy), f"vertex:{i}", corner=True)
                     for i, (vx, vy) in enumerate(self.vertices))

    def with_handle(self, role: str, x: float, y: float) -> "PolygonGate":
        if not str(role).startswith("vertex:"):
            raise GateError(f"polygon gate has no handle {role!r}")
        try:
            index = int(str(role).split(":", 1)[1])
        except ValueError:
            raise GateError(f"polygon gate has no handle {role!r}") from None
        return self.with_vertex(index, float(x), float(y))

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

    def handles(self, view: "View") -> Tuple["Handle", ...]:
        """Four on the axes of the oval, four on its bounding box.

        The axis handles change one radius, the corners change both. Corners
        are placed on the bounding box rather than on the curve because that
        is where the user reaches for them -- the curve at 45 degrees is
        inside the box and feels like a miss.
        """
        cx, cy = float(self.x_centre), float(self.y_centre)
        rx, ry = float(self.x_radius), float(self.y_radius)
        return (
            Handle(cx - rx, cy, "x_radius"),
            Handle(cx + rx, cy, "x_radius"),
            Handle(cx, cy - ry, "y_radius"),
            Handle(cx, cy + ry, "y_radius"),
            Handle(cx - rx, cy - ry, "x_radius,y_radius", corner=True),
            Handle(cx + rx, cy - ry, "x_radius,y_radius", corner=True),
            Handle(cx - rx, cy + ry, "x_radius,y_radius", corner=True),
            Handle(cx + rx, cy + ry, "x_radius,y_radius", corner=True),
        )

    def with_handle(self, role: str, x: float, y: float) -> "EllipseGate":
        parts = [p for p in str(role).split(",") if p]
        if not parts or any(p not in ("x_radius", "y_radius") for p in parts):
            raise GateError(f"ellipse gate has no handle {role!r}")
        rx, ry = float(self.x_radius), float(self.y_radius)
        if "x_radius" in parts:
            rx = abs(float(x) - float(self.x_centre))
        if "y_radius" in parts:
            ry = abs(float(y) - float(self.y_centre))
        if rx <= 0 or ry <= 0:
            # A zero radius is not an ellipse and EllipseGate refuses one.
            # Handing back the gate unchanged makes the handle stop at the
            # centre instead of the drag raising into a mouse handler.
            return self
        return replace(self, x_radius=rx, y_radius=ry)

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


@dataclass(frozen=True)
class BoxGate(Gate):
    """A rectangular region in THREE measurements.

    What a gate is in the volume. It is not a polygon with depth bolted on:
    a shape dragged on a rotated projection has no well-defined extent along
    the axis pointing at the viewer, so any attempt to read one off invents a
    number. Three ranges say exactly what is meant and read the same from
    every angle.

    A box whose z range is unbounded is a RECTANGLE extended through the
    volume, which is what a 2D gate already is when seen in 3D -- so the two
    agree rather than being different answers to the same question.
    """

    x_column: str = ""
    y_column: str = ""
    z_column: str = ""
    x_low: Optional[float] = None
    x_high: Optional[float] = None
    y_low: Optional[float] = None
    y_high: Optional[float] = None
    z_low: Optional[float] = None
    z_high: Optional[float] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("x_column", "y_column", "z_column"):
            if not str(getattr(self, name)).strip():
                raise GateError(
                    f"box gate {self.name!r} has no {name}; a box is drawn on "
                    f"three measurements")
            object.__setattr__(self, name, str(getattr(self, name)).strip())
        for low, high in (("x_low", "x_high"), ("y_low", "y_high"),
                          ("z_low", "z_high")):
            a, b = getattr(self, low), getattr(self, high)
            if a is not None and b is not None and a > b:
                object.__setattr__(self, low, b)
                object.__setattr__(self, high, a)

    @property
    def columns(self) -> Tuple[str, ...]:
        return (self.x_column, self.y_column, self.z_column)

    def mask(self, frame: pd.DataFrame) -> np.ndarray:
        what = f"gate {self.name!r}"
        keep = np.ones(len(frame), dtype=bool)
        for column, low, high in (
                (self.x_column, self.x_low, self.x_high),
                (self.y_column, self.y_low, self.y_high),
                (self.z_column, self.z_low, self.z_high)):
            values = _numeric(frame, column, what)
            keep &= np.isfinite(values)
            if low is not None:
                keep &= values >= low
            if high is not None:
                keep &= values <= high
        return keep

    def range_filters(self) -> Tuple[RangeFilter, ...]:
        out = []
        for column, low, high in (
                (self.x_column, self.x_low, self.x_high),
                (self.y_column, self.y_low, self.y_high),
                (self.z_column, self.z_low, self.z_high)):
            if low is not None or high is not None:
                out.append(RangeFilter(column=column, low=low, high=high))
        return tuple(out)

    def describe(self) -> str:
        def side(column, low, high):
            if low is None and high is None:
                return f"any {column}"
            if low is None:
                return f"{column} ≤ {high:g}"
            if high is None:
                return f"{column} ≥ {low:g}"
            return f"{low:g} ≤ {column} ≤ {high:g}"
        return " and ".join(
            side(c, lo, hi) for c, lo, hi in (
                (self.x_column, self.x_low, self.x_high),
                (self.y_column, self.y_low, self.y_high),
                (self.z_column, self.z_low, self.z_high)))

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": BOX, "name": self.name, "parent": self.parent,
                "x_column": self.x_column, "y_column": self.y_column,
                "z_column": self.z_column,
                "x_low": self.x_low, "x_high": self.x_high,
                "y_low": self.y_low, "y_high": self.y_high,
                "z_low": self.z_low, "z_high": self.z_high}

    def translated(self, dx: float, dy: float) -> "BoxGate":
        return replace(self,
                       x_low=_shift_bound(self.x_low, dx),
                       x_high=_shift_bound(self.x_high, dx),
                       y_low=_shift_bound(self.y_low, dy),
                       y_high=_shift_bound(self.y_high, dy))

    def centre(self) -> Tuple[Optional[float], Optional[float]]:
        return (_midpoint(self.x_low, self.x_high),
                _midpoint(self.y_low, self.y_high))

    def scaled(self, factor: float, *,
               about: Optional[Tuple[float, float]] = None) -> "BoxGate":
        _check_factor(factor)
        cx, cy = about if about is not None else self.centre()
        return replace(self,
                       x_low=_scale_bound(self.x_low, factor, cx),
                       x_high=_scale_bound(self.x_high, factor, cx),
                       y_low=_scale_bound(self.y_low, factor, cy),
                       y_high=_scale_bound(self.y_high, factor, cy))

    def thresholds(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """All three sides, whether or not they are currently set."""
        return {self.x_column: (self.x_low, self.x_high),
                self.y_column: (self.y_low, self.y_high),
                self.z_column: (self.z_low, self.z_high)}

    def with_threshold(self, column: str, low: Optional[float],
                    high: Optional[float]) -> "BoxGate":
        """Any of the three sides, by name."""
        field_of = {self.x_column: ("x_low", "x_high"),
                    self.y_column: ("y_low", "y_high"),
                    self.z_column: ("z_low", "z_high")}
        if column not in field_of:
            return super().with_threshold(column, low, high)
        low, high = _ordered(low, high)
        low_name, high_name = field_of[column]
        return replace(self, **{low_name: low, high_name: high})

    def to_rect(self) -> "RectGate":
        """The box seen from the front: its x and y, ignoring depth.

        What the 2D editor shows and edits. Its handles, its drag and its
        outline then all work unchanged, and the depth the 2D view cannot
        express is simply left alone rather than silently reset.
        """
        return RectGate(name=self.name, parent=self.parent,
                        x_column=self.x_column, y_column=self.y_column,
                        x_low=self.x_low, x_high=self.x_high,
                        y_low=self.y_low, y_high=self.y_high)

    @classmethod
    def from_limits(cls, name: str, columns: Sequence[str],
                    limits: Sequence[Tuple[float, float]], *,
                    parent: Optional[str] = None) -> "BoxGate":
        """A box enclosing what is currently in view.

        How a gate is made in the volume: frame a population by spinning and
        zooming, then keep what you framed. The view is already the gesture --
        asking the user to also drag a shape on a rotated projection would be
        asking them to aim at something that is not flat.
        """
        if len(columns) < 3 or len(limits) < 3:
            raise GateError(
                "a box gate needs three measurements and three ranges")
        (x0, x1), (y0, y1), (z0, z1) = limits[:3]
        return cls(name=name, parent=parent,
                   x_column=columns[0], y_column=columns[1],
                   z_column=columns[2],
                   x_low=float(x0), x_high=float(x1),
                   y_low=float(y0), y_high=float(y1),
                   z_low=float(z0), z_high=float(z1))


# Registered after the class rather than in the literal above: BoxGate is
# defined further down the file, beside the volume it belongs to, and a
# forward reference in the dict would be a NameError at import.
_GATE_CLASSES[BOX] = BoxGate


@dataclass(frozen=True)
class CylinderGate(Gate):
    """An oval drawn on one plane of the volume, extended along the third.

    Instruction 52's cylinder. The user draws in 2D on the plane they chose
    -- which is the only place a drag has a well-defined meaning -- and the
    shape is extended along the axis pointing out of it.

    THE PLANE IS NAMED BY ITS COLUMNS, not by a string. ``u_column`` and
    ``v_column`` are the two the oval is drawn on and ``axis_column`` is the
    normal; that says which of XY / XZ / YZ was the anchor without a second
    field that could disagree with the columns. It also reads the same from
    every camera angle, which is the principle :class:`BoxGate` states: a
    shape dragged on a rotated projection has no well-defined extent along
    the axis pointing at the viewer, so any attempt to read one off invents
    a number.

    AN UNBOUNDED AXIS IS THE DEFAULT AND THAT IS A DECISION, not an
    accident. A cylinder with no bound along its normal means exactly what
    the 2D ellipse on that plane already meant, so drawing one in 3D and
    drawing one in 2D agree; narrowing it is then an explicit act rather
    than something the user has to undo.
    """

    u_column: str = ""
    v_column: str = ""
    axis_column: str = ""
    u_centre: float = 0.0
    v_centre: float = 0.0
    u_radius: float = 0.0
    v_radius: float = 0.0
    axis_low: Optional[float] = None
    axis_high: Optional[float] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("u_column", "v_column", "axis_column"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise GateError(
                    f"cylinder gate {self.name!r} has no {name}; a cylinder "
                    f"is drawn on a plane and extended along the third "
                    f"measurement")
            object.__setattr__(self, name, value)
        if len({self.u_column, self.v_column, self.axis_column}) != 3:
            raise GateError(
                f"cylinder gate {self.name!r} names the same measurement "
                f"twice; the plane and its normal have to be three different "
                f"measurements")
        for name in ("u_radius", "v_radius"):
            if float(getattr(self, name)) < 0:
                object.__setattr__(self, name, abs(float(getattr(self, name))))
        low, high = _ordered(self.axis_low, self.axis_high)
        object.__setattr__(self, "axis_low", low)
        object.__setattr__(self, "axis_high", high)

    @property
    def kind(self) -> str:
        return CYLINDER

    @property
    def columns(self) -> Tuple[str, ...]:
        return (self.u_column, self.v_column, self.axis_column)

    def mask(self, frame: pd.DataFrame) -> np.ndarray:
        what = f"gate {self.name!r}"
        u = _numeric(frame, self.u_column, what)
        v = _numeric(frame, self.v_column, what)
        axis = _numeric(frame, self.axis_column, what)
        keep = np.isfinite(u) & np.isfinite(v) & np.isfinite(axis)
        if not self.u_radius or not self.v_radius:
            # A zero radius is an empty gate, not a division by zero.
            return np.zeros(len(frame), dtype=bool)
        with np.errstate(invalid="ignore"):
            inside = (((u - self.u_centre) / self.u_radius) ** 2
                      + ((v - self.v_centre) / self.v_radius) ** 2) <= 1.0
        keep &= inside
        if self.axis_low is not None:
            keep &= axis >= self.axis_low
        if self.axis_high is not None:
            keep &= axis <= self.axis_high
        return keep

    def range_filters(self) -> Tuple[RangeFilter, ...]:
        """Only the normal, and only when it is bounded.

        The oval is not a conjunction of ranges -- the same reason
        :class:`PolygonGate` returns nothing -- so offering a bounding box
        for it would quietly include the corners. The axis bound IS a range
        and is exact, so it is given.
        """
        if self.axis_low is None and self.axis_high is None:
            return ()
        return (RangeFilter(column=self.axis_column, low=self.axis_low,
                            high=self.axis_high),)

    def describe(self) -> str:
        oval = (f"oval on {self.u_column}/{self.v_column} at "
                f"({self.u_centre:g}, {self.v_centre:g}) "
                f"± ({self.u_radius:g}, {self.v_radius:g})")
        if self.axis_low is None and self.axis_high is None:
            return f"{oval}, any {self.axis_column}"
        if self.axis_low is None:
            return f"{oval}, {self.axis_column} ≤ {self.axis_high:g}"
        if self.axis_high is None:
            return f"{oval}, {self.axis_column} ≥ {self.axis_low:g}"
        return (f"{oval}, {self.axis_low:g} ≤ {self.axis_column} "
                f"≤ {self.axis_high:g}")

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": CYLINDER, "name": self.name, "parent": self.parent,
                "u_column": self.u_column, "v_column": self.v_column,
                "axis_column": self.axis_column,
                "u_centre": self.u_centre, "v_centre": self.v_centre,
                "u_radius": self.u_radius, "v_radius": self.v_radius,
                "axis_low": self.axis_low, "axis_high": self.axis_high}

    def translated(self, dx: float, dy: float) -> "CylinderGate":
        return replace(self, u_centre=self.u_centre + float(dx),
                       v_centre=self.v_centre + float(dy))

    def centre(self) -> Tuple[Optional[float], Optional[float]]:
        return (self.u_centre, self.v_centre)

    def scaled(self, factor: float, *,
               about: Optional[Tuple[float, float]] = None) -> "CylinderGate":
        _check_factor(factor)
        cu, cv = about if about is not None else self.centre()
        return replace(
            self,
            u_centre=cu + (self.u_centre - cu) * factor,
            v_centre=cv + (self.v_centre - cv) * factor,
            u_radius=self.u_radius * factor,
            v_radius=self.v_radius * factor)

    def thresholds(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """The normal, ALWAYS -- even when it is currently unbounded.

        `range_filters` answers "what does this gate filter on"; an unbounded
        normal filters on nothing and is rightly absent there. This answers
        "what can the user set", and an axis they cannot see in the panel is
        an axis they cannot bound -- which is the whole of point 4.
        """
        return {self.axis_column: (self.axis_low, self.axis_high)}

    def with_threshold(self, column: str, low: Optional[float],
                    high: Optional[float]) -> "Gate":
        """Bound the NORMAL. The oval/polygon is not a range and is not one.

        This is how the user bounds the cylinder's height, which is what point 4
        of instruction 52 asks for.
        """
        if column != self.axis_column:
            return super().with_threshold(column, low, high)
        low, high = _ordered(low, high)
        return replace(self, axis_low=low, axis_high=high)

    def to_ellipse(self) -> "EllipseGate":
        """The cylinder seen down its own axis: the oval that was drawn.

        The same service :meth:`BoxGate.to_rect` provides -- the 2D editor's
        handles, drag and outline all work on this unchanged, and the extent
        along the normal that a 2D view cannot express is left alone rather
        than silently reset.
        """
        return EllipseGate(name=self.name, parent=self.parent,
                           x_column=self.u_column, y_column=self.v_column,
                           x_centre=self.u_centre, y_centre=self.v_centre,
                           x_radius=self.u_radius, y_radius=self.v_radius)

    @classmethod
    def from_ellipse(cls, ellipse: "EllipseGate", axis_column: str, *,
                     axis_low: Optional[float] = None,
                     axis_high: Optional[float] = None) -> "CylinderGate":
        """Extrude a drawn oval along ``axis_column``.

        This is what "translated to 3 dims when the gate is generated"
        means: the drawing stays 2D and reuses the existing geometry, and
        the third dimension is added at the end.
        """
        return cls(name=ellipse.name, parent=ellipse.parent,
                   u_column=ellipse.x_column, v_column=ellipse.y_column,
                   axis_column=axis_column,
                   u_centre=ellipse.x_centre, v_centre=ellipse.y_centre,
                   u_radius=ellipse.x_radius, v_radius=ellipse.y_radius,
                   axis_low=axis_low, axis_high=axis_high)


@dataclass(frozen=True)
class PrismGate(Gate):
    """A polygon drawn on one plane of the volume, extended along the third.

    Instruction 52's prism, and the sibling of :class:`CylinderGate` in
    every respect -- the plane is named by its columns, the normal is
    unbounded by default so it agrees with the 2D polygon, and the drawing
    stays 2D.
    """

    u_column: str = ""
    v_column: str = ""
    axis_column: str = ""
    vertices: Tuple[Tuple[float, float], ...] = ()
    axis_low: Optional[float] = None
    axis_high: Optional[float] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("u_column", "v_column", "axis_column"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise GateError(
                    f"prism gate {self.name!r} has no {name}; a prism is "
                    f"drawn on a plane and extended along the third "
                    f"measurement")
            object.__setattr__(self, name, value)
        if len({self.u_column, self.v_column, self.axis_column}) != 3:
            raise GateError(
                f"prism gate {self.name!r} names the same measurement twice; "
                f"the plane and its normal have to be three different "
                f"measurements")
        vertices = tuple((float(a), float(b)) for a, b in self.vertices)
        if len(vertices) < 3:
            raise GateError(
                f"prism gate {self.name!r} has {len(vertices)} vertices; a "
                f"polygon needs at least three")
        object.__setattr__(self, "vertices", vertices)
        low, high = _ordered(self.axis_low, self.axis_high)
        object.__setattr__(self, "axis_low", low)
        object.__setattr__(self, "axis_high", high)

    @property
    def kind(self) -> str:
        return PRISM

    @property
    def columns(self) -> Tuple[str, ...]:
        return (self.u_column, self.v_column, self.axis_column)

    def mask(self, frame: pd.DataFrame) -> np.ndarray:
        what = f"gate {self.name!r}"
        u = _numeric(frame, self.u_column, what)
        v = _numeric(frame, self.v_column, what)
        axis = _numeric(frame, self.axis_column, what)
        keep = points_in_polygon(u, v, self.vertices)
        keep &= np.isfinite(axis)
        if self.axis_low is not None:
            keep &= axis >= self.axis_low
        if self.axis_high is not None:
            keep &= axis <= self.axis_high
        return keep

    def range_filters(self) -> Tuple[RangeFilter, ...]:
        """The normal only -- see :meth:`CylinderGate.range_filters`."""
        if self.axis_low is None and self.axis_high is None:
            return ()
        return (RangeFilter(column=self.axis_column, low=self.axis_low,
                            high=self.axis_high),)

    def describe(self) -> str:
        shape = (f"{len(self.vertices)}-sided polygon on "
                 f"{self.u_column}/{self.v_column}")
        if self.axis_low is None and self.axis_high is None:
            return f"{shape}, any {self.axis_column}"
        if self.axis_low is None:
            return f"{shape}, {self.axis_column} ≤ {self.axis_high:g}"
        if self.axis_high is None:
            return f"{shape}, {self.axis_column} ≥ {self.axis_low:g}"
        return (f"{shape}, {self.axis_low:g} ≤ {self.axis_column} "
                f"≤ {self.axis_high:g}")

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": PRISM, "name": self.name, "parent": self.parent,
                "u_column": self.u_column, "v_column": self.v_column,
                "axis_column": self.axis_column,
                "vertices": [list(v) for v in self.vertices],
                "axis_low": self.axis_low, "axis_high": self.axis_high}

    def translated(self, dx: float, dy: float) -> "PrismGate":
        return replace(self, vertices=tuple(
            (u + float(dx), v + float(dy)) for u, v in self.vertices))

    def centre(self) -> Tuple[Optional[float], Optional[float]]:
        array = np.asarray(self.vertices, dtype=float)
        return (float(array[:, 0].mean()), float(array[:, 1].mean()))

    def scaled(self, factor: float, *,
               about: Optional[Tuple[float, float]] = None) -> "PrismGate":
        _check_factor(factor)
        cu, cv = about if about is not None else self.centre()
        return replace(self, vertices=tuple(
            (cu + (u - cu) * factor, cv + (v - cv) * factor)
            for u, v in self.vertices))

    def thresholds(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """The normal, always -- see :meth:`CylinderGate.bounds`."""
        return {self.axis_column: (self.axis_low, self.axis_high)}

    def with_threshold(self, column: str, low: Optional[float],
                    high: Optional[float]) -> "Gate":
        """Bound the NORMAL. The oval/polygon is not a range and is not one.

        This is how the user bounds the prism's height, which is what point 4
        of instruction 52 asks for.
        """
        if column != self.axis_column:
            return super().with_threshold(column, low, high)
        low, high = _ordered(low, high)
        return replace(self, axis_low=low, axis_high=high)

    def to_polygon(self) -> "PolygonGate":
        """The prism seen down its own axis -- see
        :meth:`CylinderGate.to_ellipse`."""
        return PolygonGate(name=self.name, parent=self.parent,
                           x_column=self.u_column, y_column=self.v_column,
                           vertices=self.vertices)

    @classmethod
    def from_polygon(cls, polygon: "PolygonGate", axis_column: str, *,
                     axis_low: Optional[float] = None,
                     axis_high: Optional[float] = None) -> "PrismGate":
        """Extrude a drawn polygon along ``axis_column``."""
        return cls(name=polygon.name, parent=polygon.parent,
                   u_column=polygon.x_column, v_column=polygon.y_column,
                   axis_column=axis_column, vertices=polygon.vertices,
                   axis_low=axis_low, axis_high=axis_high)


_GATE_CLASSES[CYLINDER] = CylinderGate
_GATE_CLASSES[PRISM] = PrismGate


@dataclass(frozen=True)
class CompositeGate(Gate):
    """Other gates, combined. Instruction 52's point 5.

        "if the user draws another gate on the same 3d graph they should be
         able to set the new gate as being its own gate, subtracting or
         add[ing] from/to the other gates in view"

    WHY THIS IS THE FEATURE AND NOT A CONVENIENCE. "The bright, small, round
    ones" is three measurements at once, and answering it today means gating
    twice and intersecting the results by hand -- an intersection that exists
    only in whatever the user did next. A gate that IS "this cylinder minus
    that box" is a statement someone else can re-run.

    OPERANDS ARE NAMES, NOT COPIES, and that is the whole design decision.
    Holding copies would make ``mask`` self-contained, which is tidier -- and
    would mean that adjusting one of the gates being combined left the
    composite showing the old shape, silently, for as long as nobody looked.
    One source of truth costs a resolver; two sources cost a wrong answer.

    So :meth:`mask` cannot work alone and says so rather than guessing.
    :meth:`GateSet.mask` passes the set in.

    NOT THE SAME THING AS ``parent``. A parent is SEQUENTIAL gating -- draw
    inside what you already kept -- which is always an intersection and
    always a tree. This is set algebra between siblings, which is neither.
    Both exist because both are asked for.
    """

    operation: str = "union"
    operands: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        operation = str(self.operation).strip().lower()
        if operation not in COMPOSITE_OPS:
            raise GateError(
                f"composite gate {self.name!r} has operation "
                f"{self.operation!r}; it must be one of "
                f"{', '.join(COMPOSITE_OPS)}")
        object.__setattr__(self, "operation", operation)
        operands = tuple(str(o).strip() for o in self.operands if str(o).strip())
        if len(operands) < 2:
            raise GateError(
                f"composite gate {self.name!r} combines {len(operands)} "
                f"gate(s); combining needs at least two")
        if self.name in operands:
            raise GateError(
                f"composite gate {self.name!r} combines itself")
        if len(set(operands)) != len(operands):
            raise GateError(
                f"composite gate {self.name!r} names the same gate twice; "
                f"a gate unioned with itself is itself, and subtracted from "
                f"itself is empty -- neither is likely to be meant")
        object.__setattr__(self, "operands", operands)

    @property
    def kind(self) -> str:
        return COMPOSITE

    @property
    def columns(self) -> Tuple[str, ...]:
        """Empty: the columns are its operands', and only the set knows them.

        A composite that guessed would be guessing about gates it cannot
        see. :meth:`GateSet.columns_for` answers this properly.
        """
        return ()

    def mask(self, frame: pd.DataFrame) -> np.ndarray:
        raise GateError(
            f"composite gate {self.name!r} combines "
            f"{', '.join(self.operands)} and cannot be evaluated on its own; "
            f"ask the gate set for it")

    def mask_with(self, frame: pd.DataFrame,
                  lookup: Mapping[str, "Gate"]) -> np.ndarray:
        """Evaluate against a name -> gate mapping.

        :raises GateError: naming any operand the mapping does not hold. A
            composite whose operand was deleted must not quietly become the
            union of what is left.
        """
        missing = [o for o in self.operands if o not in lookup]
        if missing:
            raise GateError(
                f"composite gate {self.name!r} names "
                f"{', '.join(missing)}, which no longer exist")
        masks = [np.asarray(lookup[o], dtype=bool) if isinstance(
                     lookup[o], np.ndarray) else lookup[o].mask(frame)
                 for o in self.operands]
        if self.operation == "union":
            out = masks[0].copy()
            for extra in masks[1:]:
                out |= extra
            return out
        if self.operation == "intersect":
            out = masks[0].copy()
            for extra in masks[1:]:
                out &= extra
            return out
        # subtract: the FIRST operand minus every other. Order matters, and
        # it is the order the user listed them in -- A minus B is not B minus
        # A, and a set that sorted its operands would silently change which.
        out = masks[0].copy()
        for extra in masks[1:]:
            out &= ~extra
        return out

    def describe(self) -> str:
        joiner = {"union": " or ", "intersect": " and ",
                  "subtract": " minus "}[self.operation]
        return joiner.join(self.operands)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": COMPOSITE, "name": self.name, "parent": self.parent,
                "operation": self.operation, "operands": list(self.operands)}

    def translated(self, dx: float, dy: float) -> "CompositeGate":
        """Unchanged: moving a composite would have to move its operands,
        and those are other gates with their own users."""
        return self

    def scaled(self, factor: float, *,
               about: Optional[Tuple[float, float]] = None) -> "CompositeGate":
        _check_factor(factor)
        return self

    def centre(self) -> Tuple[Optional[float], Optional[float]]:
        return (None, None)


_GATE_CLASSES[COMPOSITE] = CompositeGate


class WandError(GateError):
    """A wand click that cannot become a gate, with the reason."""


def wand_select(frame: pd.DataFrame, x_column: str, y_column: str,
                x: float, y: float, *, tolerance: float = 0.05,
                max_radius: float = 0.35, scale: bool = True) -> np.ndarray:
    """Grow a selection outward from a clicked point.

    The watershed gesture: click inside a population and the gate finds its
    edge. Starting from the object nearest the click, the selection repeatedly
    takes every unselected object within ``tolerance`` of something already
    selected, and stops when nothing new is close enough. Two limits keep it
    from swallowing the plot:

    ``tolerance``
        how far apart two objects can be and still count as neighbours. This
        is what makes it a WATERSHED rather than a circle -- the selection
        flows along a dense ridge and stops at a gap, so an elongated or
        bent population comes out whole and the sparse space around it does
        not.
    ``max_radius``
        how far from the CLICK the selection may reach at all. Without it a
        single chain of objects bridging two populations merges them, which
        on a real scatter happens more often than not.

    Both are in SCALED units by default -- each axis mapped onto 0..1 across
    the data -- so one pair of defaults works on measurements whose ranges
    differ by orders of magnitude, and so "distance" means the same in x as
    in y. Without that, a tolerance is a distance in whichever measurement
    has the larger numbers and the other axis is effectively ignored.

    :param frame: the measurement table.
    :param x: the clicked x, in DATA units.
    :param y: the clicked y, in data units.
    :returns: a boolean mask over ``frame``.
    :raises WandError: a column that is missing, or a click with no finite
        object anywhere near it.
    """
    what = "wand"
    xs = np.asarray(_numeric(frame, x_column, what), dtype=float)
    ys = np.asarray(_numeric(frame, y_column, what), dtype=float)
    finite = np.isfinite(xs) & np.isfinite(ys)
    if not finite.any():
        raise WandError(
            f"no object has both {x_column} and {y_column}, so there is "
            f"nothing to grow a gate from")

    if float(tolerance) <= 0:
        raise WandError("the neighbour tolerance must be greater than zero")
    if float(max_radius) <= 0:
        raise WandError("the maximum distance from the click must be "
                        "greater than zero")

    px, py = float(x), float(y)
    if scale:
        # Map each axis onto 0..1 across the DATA, not the view: a gate is a
        # statement about measurements, and scaling by the visible window
        # would make the same click give a different gate at a different zoom.
        sx, sy = _unit_scale(xs[finite]), _unit_scale(ys[finite])
        ux, uy = sx(xs), sy(ys)
        upx, upy = float(sx(np.array([px]))[0]), float(sy(np.array([py]))[0])
    else:
        ux, uy, upx, upy = xs, ys, px, py

    index = np.flatnonzero(finite)
    points = np.column_stack([ux[index], uy[index]])
    from_click = np.hypot(points[:, 0] - upx, points[:, 1] - upy)

    reachable = from_click <= float(max_radius)
    if not reachable.any():
        raise WandError(
            "no object is within the maximum distance of that click; click "
            "closer to a population, or raise the maximum distance")

    # The seed is the nearest object to the click, NOT the click itself: the
    # user points at a cloud, and a click landing in a gap between two of its
    # objects must still start inside the cloud.
    seed = int(np.argmin(from_click))

    candidates = np.flatnonzero(reachable)
    local = points[candidates]
    seed_local = int(np.flatnonzero(candidates == seed)[0])

    selected = np.zeros(len(local), dtype=bool)
    selected[seed_local] = True
    frontier = [seed_local]
    tol = float(tolerance)
    while frontier:
        current = local[frontier]
        frontier = []
        # Distance from every unselected candidate to the newest selections.
        remaining = np.flatnonzero(~selected)
        if remaining.size == 0:
            break
        gaps = np.hypot(
            local[remaining][:, None, 0] - current[None, :, 0],
            local[remaining][:, None, 1] - current[None, :, 1])
        grown = remaining[np.any(gaps <= tol, axis=1)]
        if grown.size:
            selected[grown] = True
            frontier = list(grown)

    mask = np.zeros(len(frame), dtype=bool)
    mask[index[candidates[selected]]] = True
    return mask


def _unit_scale(values: np.ndarray):
    """A function mapping ``values`` onto 0..1, and anything else with them.

    A degenerate axis -- every object at the same value -- maps to zero
    rather than dividing by nothing, so a wand click on a constant
    measurement selects by the other axis instead of raising.
    """
    low = float(np.min(values))
    high = float(np.max(values))
    span = high - low
    if not np.isfinite(span) or span <= 0:
        return lambda v: np.zeros_like(np.asarray(v, dtype=float))
    return lambda v: (np.asarray(v, dtype=float) - low) / span


def wand_gate(frame: pd.DataFrame, x_column: str, y_column: str,
              x: float, y: float, *, name: str = "(unnamed)",
              tolerance: float = 0.05, max_radius: float = 0.35,
              scale: bool = True,
              parent: Optional[str] = None) -> "PolygonGate":
    """Grow a selection from a click and fit a polygon around it.

    A POLYGON, not the selection itself, because a gate has to be a shape:
    re-applied to another table it must select that table's objects, and a
    list of row numbers cannot. Fitting the hull is what turns "these
    objects" into "this region", which is the difference between a lasso and
    a gate.

    :returns: the fitted gate, unnamed unless ``name`` is given.
    :raises WandError: too few objects to make a polygon out of.
    """
    mask = wand_select(frame, x_column, y_column, x, y,
                       tolerance=tolerance, max_radius=max_radius, scale=scale)
    chosen = frame.loc[mask, [x_column, y_column]].to_numpy(dtype=float)
    if len(chosen) < 3:
        raise WandError(
            f"only {len(chosen)} object(s) grew from that click, and a "
            f"polygon needs three; raise the neighbour tolerance")
    hull = _convex_hull(chosen)
    if len(hull) < 3:
        raise WandError(
            "the objects that grew from that click are in a straight line, "
            "which has no area to gate")
    return PolygonGate(name=name, parent=parent,
                       x_column=x_column, y_column=y_column,
                       vertices=tuple((float(a), float(b)) for a, b in hull))


def _cluster_matrix(frame: pd.DataFrame, x_column: str, y_column: str, *,
                    eps: float, min_samples: int, scale: bool):
    """Validate the request and return ``(raw, work)`` for DBSCAN.

    Shared by :func:`cluster_gates` and :func:`cluster_walk_candidates` so a
    Walk searches the SAME preparation the run it is tuning will use. Scaling
    is the reason this matters: ``eps`` means a different distance either
    side of that transform, so a search that scaled differently from the
    final run would recommend a number that then behaved unlike the preview.

    ``raw`` is in the measurements' own units, for hulls that must be drawn
    on the real axes; ``work`` is what DBSCAN sees.

    :raises ClusterError: a missing column, one column twice, a parameter out
        of range, too few usable rows, or a measurement that never varies.
    """
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
    work = (raw - raw.mean(axis=0)) / spread if scale else raw
    return raw, work


@dataclass(frozen=True)
class ClusterCandidate:
    """One point in a clustering Walk: what this ``eps`` actually produced.

    Kept as data rather than rendered straight into a table so the search is
    testable without a GUI, and so the caller decides what "best" means --
    which it must, because the scores disagree on purpose. Silhouette is
    blind to how much was discarded, so a run that calls 90% of the plate
    noise and tightly clusters the rest scores beautifully and is useless.
    """

    #: The neighbourhood radius tried, in the units DBSCAN saw.
    eps: float
    #: Clusters found, excluding the noise label.
    clusters: int
    #: Share of usable rows DBSCAN labelled noise, 0.0 to 1.0.
    noise_fraction: float
    #: Mean silhouette over the clustered points, or None when fewer than two
    #: clusters were found and the score is undefined.
    silhouette: Optional[float]


def cluster_walk_candidates(frame: pd.DataFrame, x_column: str,
                            y_column: str, *, eps: float = 0.5,
                            min_samples: int = 10, scale: bool = True,
                            steps: int = 12, span: float = 3.0,
                            method: str = "dbscan"
                            ) -> List[ClusterCandidate]:
    """Try a range of ``eps`` around the chosen one and score each result.

    This is what "Walk" means elsewhere in spaCR -- try the space and show
    the candidates -- applied to the one parameter that decides how many
    populations DBSCAN finds. ``min_samples`` is deliberately held fixed:
    sweeping both turns one readable list into a grid, and eps is the
    parameter users actually get wrong.

    The sweep is GEOMETRIC, from ``eps / span`` to ``eps * span``, because
    eps is a distance and its useful range spans orders of magnitude -- an
    arithmetic sweep from 0.1 to 3.0 spends most of its steps in a region
    where every one gives the same single blob.

    :param eps: the centre of the sweep, normally the user's current value.
    :param steps: how many radii to try, at least 2.
    :param span: multiplicative half-width, so 3.0 tries a ninefold range.
    :returns: one :class:`ClusterCandidate` per radius, ordered by ``eps``
        ascending. Radii that produce no cluster are INCLUDED, with
        ``clusters=0``, because "nothing below here works" is the most
        useful part of the answer.
    :raises ClusterError: anything :func:`_cluster_matrix` refuses, plus a
        ``steps`` or ``span`` that cannot describe a sweep.
    """
    try:
        from sklearn.metrics import silhouette_score
    except Exception as exc:                       # pragma: no cover
        raise ClusterError(f"clustering needs scikit-learn ({exc})") from exc

    if int(steps) < 2:
        raise ClusterError(f"a walk needs at least 2 steps, got {steps!r}")
    if span <= 1.0:
        raise ClusterError(
            f"span must be greater than 1 to describe a range, got {span!r}")

    raw, work = _cluster_matrix(frame, x_column, y_column,
                                eps=eps, min_samples=min_samples, scale=scale)

    lo, hi = float(eps) / float(span), float(eps) * float(span)
    steps = int(steps)
    radii = [lo * (hi / lo) ** (i / (steps - 1)) for i in range(steps)]

    candidates: List[ClusterCandidate] = []
    for radius in radii:
        labels = _fit_labels(work, method=method, eps=radius,
                             min_samples=min_samples)
        found = [lab for lab in np.unique(labels) if lab != -1]
        noise = float((labels == -1).sum()) / float(len(labels))
        score = None
        if len(found) >= 2:
            keep = labels != -1
            # Silhouette is defined on the CLUSTERED points only. Including
            # noise as if it were one more cluster would reward runs that
            # discard the awkward objects, which is the opposite of useful.
            try:
                score = float(silhouette_score(work[keep], labels[keep]))
            except Exception:
                score = None
        candidates.append(ClusterCandidate(
            eps=float(radius), clusters=len(found),
            noise_fraction=noise, silhouette=score))
    return candidates


def best_cluster_candidate(candidates: Sequence[ClusterCandidate], *,
                           max_noise: float = 0.5
                           ) -> Optional[ClusterCandidate]:
    """Pick the candidate to recommend, or None when none is defensible.

    Highest silhouette among those that found at least two clusters and did
    not discard more than ``max_noise`` of the objects. The noise ceiling is
    the important half: silhouette alone is maximised by a radius that keeps
    a handful of tight points and calls everything else debris, which scores
    near 1.0 and answers a question nobody asked.

    Ties break toward the LARGER radius, which merges rather than splits --
    the conservative direction when two settings score alike.
    """
    usable = [c for c in candidates
              if c.silhouette is not None and c.clusters >= 2
              and c.noise_fraction <= max_noise]
    if not usable:
        return None
    return max(usable, key=lambda c: (c.silhouette, c.eps))


#: Clustering algorithms this build actually runs. Defined HERE, beside
#: `_fit_labels` which implements them, so the picker cannot offer a method
#: the code does not have -- which is exactly what went wrong: Gate Settings
#: listed "kmeans" and `cluster_gates` called DBSCAN regardless, so choosing
#: k-means returned DBSCAN's answer under another name.
CLUSTER_METHODS: Tuple[str, ...] = ("dbscan", "hdbscan")


def _fit_labels(work, *, method: str, eps: float, min_samples: int):
    """Cluster ``work`` with the named algorithm and return labels.

    Both supported methods are DENSITY-BASED and label sparse points ``-1``,
    which is what lets the rest of the clustering path treat them alike: the
    noise fraction means the same thing, the hulls are built the same way,
    and a Walk can score either.

    :param method: ``"dbscan"`` or ``"hdbscan"``.
    :raises ClusterError: on an unknown method, naming what is available.
    """
    name = str(method or "dbscan").strip().lower()
    if name == "dbscan":
        from sklearn.cluster import DBSCAN
        return DBSCAN(eps=float(eps),
                      min_samples=int(min_samples)).fit_predict(work)
    if name == "hdbscan":
        try:
            from sklearn.cluster import HDBSCAN
        except ImportError as exc:                 # pragma: no cover
            raise ClusterError(
                "HDBSCAN needs scikit-learn 1.3 or newer; choose DBSCAN or "
                f"upgrade scikit-learn ({exc})") from exc
        # `eps` becomes the floor below which HDBSCAN stops splitting, so the
        # control keeps the meaning it has for DBSCAN -- larger merges. Zero
        # (its own default) would make the setting inert, which is the defect
        # this whole change is about.
        return HDBSCAN(min_cluster_size=max(2, int(min_samples)), copy=False,
                       cluster_selection_epsilon=float(eps)).fit_predict(work)
    raise ClusterError(
        f"unknown clustering method {method!r}; this build has "
        f"{', '.join(CLUSTER_METHODS)}")


def cluster_gates(frame: pd.DataFrame, x_column: str, y_column: str, *,
                  eps: float = 0.5, min_samples: int = 10,
                  scale: bool = True, max_clusters: int = 20,
                  name_prefix: str = "cluster", method: str = "dbscan",
                  parent: Optional[str] = None) -> List["PolygonGate"]:
    """Find dense populations and return one gate per cluster.

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
    :param method: ``"dbscan"`` or ``"hdbscan"``. Gate Settings offered this
        choice while this function always ran DBSCAN, so picking another
        algorithm silently returned DBSCAN's answer.
    :param parent: parent gate name, so clusters can be found inside a gate.
    :returns: one :class:`PolygonGate` per cluster, largest first. Empty when
        only noise is found.
    :raises ClusterError: a missing column, no usable rows, bad parameters,
        an unknown method, or more clusters than ``max_clusters``.
    """
    raw, work = _cluster_matrix(frame, x_column, y_column,
                                eps=eps, min_samples=min_samples, scale=scale)

    labels = _fit_labels(work, method=method, eps=eps,
                         min_samples=min_samples)
    found = [lab for lab in np.unique(labels) if lab != -1]
    if not found:
        return []
    if len(found) > int(max_clusters):
        raise ClusterError(
            f"{method} found {len(found)} clusters, more than max_clusters="
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
    if kind == PRISM and "vertices" in data:
        data["vertices"] = tuple(tuple(v) for v in data["vertices"])
    if kind == COMPOSITE and "operands" in data:
        data["operands"] = tuple(data["operands"])
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
        return self._mask_chain(frame, name, ())

    def _mask_chain(self, frame: pd.DataFrame, name: str,
                    seen: Tuple[str, ...]) -> np.ndarray:
        """:meth:`mask`, carrying the composites already being evaluated."""
        keep = np.ones(len(frame), dtype=bool)
        for gate in self.path(name):
            keep &= self._mask_of(frame, gate, seen)
        return keep

    def _mask_of(self, frame: pd.DataFrame, gate: Gate,
                 seen: Tuple[str, ...]) -> np.ndarray:
        """One gate's own mask, resolving a composite's operands.

        :param seen: the composites already being evaluated, so a cycle is a
            sentence naming the loop rather than a RecursionError.
        """
        if not isinstance(gate, CompositeGate):
            return gate.mask(frame)
        if gate.name in seen:
            raise GateError(
                "composite gates form a loop: "
                + " -> ".join(seen + (gate.name,)))
        lookup = {}
        for operand in gate.operands:
            if operand not in self:
                raise GateError(
                    f"composite gate {gate.name!r} names {operand!r}, which "
                    f"no longer exists")
            # Each operand carries its OWN ancestors, because a gate drawn
            # inside another means the pair, and combining it as though it
            # were the shape alone would include rows its parent excluded.
            lookup[operand] = self._mask_chain(
                frame, operand, seen + (gate.name,))
        return gate.mask_with(frame, lookup)

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
