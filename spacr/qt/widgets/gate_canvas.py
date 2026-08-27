"""Axis-specific cutoff and transform controls for the Gate Editor.

:func:`axis_at` identifies the axis associated with a pointer position from
its bounding box. :class:`AxisCutoffs` stores display limits by measurement,
and :func:`axis_menu_items` represents the corresponding menu as data so it
can be tested without opening a graphical popup.

Cutoffs affect only the displayed range; they do not filter rows or change
gate membership. Limits are keyed by measurement rather than by the current
X/Y assignment, so they follow a measurement when axes are exchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterator, Optional, Sequence, Tuple

from .gate_settings import AXIS_SCALES

__all__ = [
    "AXIS_NAMES", "AxisCutoff", "AxisCutoffs", "AxisMenuItem",
    "axis_at", "axis_menu_items", "apply_cutoffs", "parse_cutoff",
]

#: What each axis is called in a menu title.
AXIS_NAMES: Dict[str, str] = {"x": "X axis", "y": "Y axis"}

#: Scales that draw nothing at all where the measurement reaches zero or
#: below, so the menu greys them there instead of accepting a click that
#: cannot take effect.
POSITIVE_ONLY: Tuple[str, ...] = ("log", "logit")


class CutoffError(ValueError):
    """A cutoff that cannot be applied, with the reason in its message."""


@dataclass(frozen=True)
class AxisCutoff:
    """The lowest and highest value an axis shows. ``None`` means "the data".

    Either end may be left open: cutting the bottom off a long tail while
    letting the top follow the data is the common case, and forcing both ends
    would make the user invent a number for the end they did not care about.

    :raises CutoffError: when the low end is not below the high end. Equal
        ends give an axis with no extent, which matplotlib draws as a blank
        panel rather than as an error.
    """

    low: Optional[float] = None
    high: Optional[float] = None

    def __post_init__(self) -> None:
        for name in ("low", "high"):
            value = getattr(self, name)
            if value is None:
                continue
            object.__setattr__(self, name, float(value))
        if (self.low is not None and self.high is not None
                and not self.low < self.high):
            raise CutoffError(
                f"the low cutoff ({self.low:g}) must be below the high one "
                f"({self.high:g}); an axis whose ends meet has no extent and "
                f"draws as a blank panel")

    @property
    def is_set(self) -> bool:
        """Whether either end has been pinned."""
        return self.low is not None or self.high is not None

    def limits(self, low: float, high: float) -> Tuple[float, float]:
        """``(low, high)`` with the unpinned ends filled from the data."""
        return (low if self.low is None else self.low,
                high if self.high is None else self.high)

    def describe(self) -> str:
        """The cutoff as a user reads it: ``10 – 500``, ``≥ 10``, ``≤ 500``."""
        if self.low is not None and self.high is not None:
            return f"{self.low:g} – {self.high:g}"
        if self.low is not None:
            return f"≥ {self.low:g}"
        if self.high is not None:
            return f"≤ {self.high:g}"
        return "none"


class AxisCutoffs:
    """The cutoffs a session has set, keyed by measurement.

    Empty cutoffs are removed rather than stored as ``(None, None)``. A
    measurement is therefore present only when at least one limit is active.
    """

    def __init__(self, initial: Optional[Dict[str, AxisCutoff]] = None):
        self._by_column: Dict[str, AxisCutoff] = dict(initial or {})

    def __len__(self) -> int:
        return len(self._by_column)

    def __contains__(self, column: object) -> bool:
        return str(column) in self._by_column

    def __iter__(self) -> Iterator[str]:
        return iter(self._by_column)

    def columns(self) -> Tuple[str, ...]:
        """Every measurement that carries a cutoff, in the order they were set."""
        return tuple(self._by_column)

    def get(self, column: Optional[str]) -> AxisCutoff:
        """The cutoff for ``column``, or an empty one. Never ``None``.

        Callers ask this on every render, so returning an empty cutoff rather
        than ``None`` keeps the ``if cutoff is None`` branch out of the
        drawing path.
        """
        if not column:
            return AxisCutoff()
        return self._by_column.get(str(column), AxisCutoff())

    def set(self, column: str, low: Optional[float] = None,
            high: Optional[float] = None) -> AxisCutoff:
        """Pin ``column`` between ``low`` and ``high``. Returns what was stored.

        Setting both ends to ``None`` clears the column rather than storing an
        empty cutoff, so a cleared measurement stops reporting as cut off.
        """
        cutoff = AxisCutoff(low, high)
        name = str(column)
        if not cutoff.is_set:
            self._by_column.pop(name, None)
            return cutoff
        self._by_column[name] = cutoff
        return cutoff

    def clear(self, column: str) -> bool:
        """Forget ``column``'s cutoff. Returns whether there was one."""
        return self._by_column.pop(str(column), None) is not None

    def clear_all(self) -> int:
        """Forget every cutoff. Returns how many were dropped."""
        count = len(self._by_column)
        self._by_column.clear()
        return count


def parse_cutoff(text: str) -> Optional[float]:
    """A number typed into a cutoff box, or ``None`` for "leave this end".

    Blank means the data decides that end. A blank box is the only way to say
    "cut the bottom off and let the top follow the data", so it is a value
    rather than an error.

    :raises CutoffError: for text that is neither blank nor a number, naming
        what was typed -- a silent fall back to "the data decides" would look
        exactly like the cutoff having been applied and done nothing.
    """
    stripped = str(text).strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        raise CutoffError(
            f"{stripped!r} is not a number; leave the box empty to let the "
            f"data decide that end") from None


def axis_at(point: Sequence[float],
            bbox: Sequence[float]) -> Optional[str]:
    """Which axis a click at ``point`` landed on: ``"x"``, ``"y"`` or ``None``.

    :param point: ``(x, y)`` in the figure's display coordinates, which have
        their origin at the BOTTOM left -- the convention every matplotlib
        bounding box uses, so the two never need converting between.
    :param bbox: the plotting rectangle as ``(x0, y0, x1, y1)``.
    :returns: ``None`` inside the rectangle, where the plot's own menu
        belongs, and for the margins that belong to neither axis.

    The strip BELOW the rectangle is the x axis and the strip to its LEFT is
    the y axis -- that is where the ticks and the axis label are drawn, so it
    is where a user aiming at "the axis" clicks. In the corner where the two
    strips overlap the further overshoot wins: well to the left and barely
    below is the y axis, and the other way round is the x axis.
    """
    x, y = float(point[0]), float(point[1])
    x0, y0, x1, y1 = (float(value) for value in bbox)
    if x0 <= x <= x1 and y < y0:
        return "x"
    if y0 <= y <= y1 and x < x0:
        return "y"
    if x < x0 and y < y0:
        return "y" if (x0 - x) > (y0 - y) else "x"
    return None


@dataclass(frozen=True)
class AxisMenuItem:
    """One row of an axis menu, as data.

    :param label: what the row says, or ``None`` for a separator.
    :param callback: what clicking it does. ``None`` for a row that is only
        there to be read.
    :param checked: ``True``/``False`` for a row that shows a tick, ``None``
        for one that is not checkable.
    :param enabled: whether it can be clicked.
    :param why: why not, when it cannot. A greyed row with no reason is a
        dead end that reads as a bug.
    """

    label: Optional[str]
    callback: Optional[Callable[[], None]] = None
    checked: Optional[bool] = None
    enabled: bool = True
    why: str = ""


def axis_menu_items(axis: str, column: Optional[str], *,
                    scale: str = "linear",
                    cutoff: Optional[AxisCutoff] = None,
                    positive: bool = True,
                    on_scale: Optional[Callable[[str], None]] = None,
                    on_cutoffs: Optional[Callable[[], None]] = None,
                    on_clear: Optional[Callable[[], None]] = None):
    """The menu behind a right-click on one axis, as a list of rows.

    :param axis: ``"x"`` or ``"y"``.
    :param column: the measurement on that axis, or ``None`` when the axis is
        empty -- in which case there is nothing to lay out or cut off, and
        every row says so rather than being missing.
    :param scale: the scale in force, ticked in the list.
    :param cutoff: what is pinned now, shown on the clearing row so the menu
        says what it would undo.
    :param positive: whether the measurement stays above zero. The scales in
        :data:`POSITIVE_ONLY` draw an empty panel where it does not, so they
        are greyed with that reason instead of accepting a click that cannot
        take effect.
    :param on_scale: called with the chosen scale.
    :param on_cutoffs: called to ask for new cutoffs.
    :param on_clear: called to drop the cutoffs on this axis.
    """
    title = AXIS_NAMES.get(axis, axis)
    cutoff = cutoff or AxisCutoff()
    if not column:
        return [AxisMenuItem(f"{title}: nothing plotted", enabled=False,
                             why="Choose a measurement for this axis first.")]

    rows = [AxisMenuItem(f"{title}: {column}", enabled=False),
            AxisMenuItem(None)]
    for name in AXIS_SCALES:
        blocked = (not positive) and name in POSITIVE_ONLY
        rows.append(AxisMenuItem(
            name,
            callback=(None if blocked or on_scale is None
                      else (lambda value=name: on_scale(value))),
            checked=(name == scale),
            enabled=not blocked,
            why=("" if not blocked else
                 f"{column} reaches zero or below, and a {name} axis over it "
                 f"draws nothing at all.")))
    rows.append(AxisMenuItem(None))
    rows.append(AxisMenuItem("Set cutoffs…", callback=on_cutoffs))
    rows.append(AxisMenuItem(
        f"Clear cutoffs ({cutoff.describe()})" if cutoff.is_set
        else "Clear cutoffs",
        callback=(on_clear if cutoff.is_set else None),
        enabled=cutoff.is_set,
        why="" if cutoff.is_set else f"No cutoffs are set on {column}."))
    return rows


def apply_cutoffs(axes, columns: Sequence[Optional[str]],
                  cutoffs: AxisCutoffs) -> Tuple[str, ...]:
    """Narrow ``axes`` to the cutoffs set for the measurements it draws.

    :param axes: a matplotlib ``Axes``.
    :param columns: ``(x_column, y_column)`` -- what is on each axis now.
    :param cutoffs: the session's cutoffs, keyed by measurement.
    :returns: the axes that were narrowed, e.g. ``("x",)``.

    The unpinned end of a one-sided cutoff is taken from the limits the data
    already produced, so cutting the bottom off leaves the top where the
    scatter put it rather than collapsing it onto the cut.
    """
    narrowed = []
    getters = {"x": (axes.get_xlim, axes.set_xlim),
               "y": (axes.get_ylim, axes.set_ylim)}
    for axis, column in zip(("x", "y"), tuple(columns) + (None, None)):
        cutoff = cutoffs.get(column)
        if not cutoff.is_set:
            continue
        get, set_ = getters[axis]
        low, high = get()
        set_(*cutoff.limits(float(low), float(high)))
        narrowed.append(axis)
    return tuple(narrowed)
