"""Local data filter — narrow every open view at once.

JMP's Local Data Filter, which is the feature its users reach for most: a panel
of live controls that subsets every plot, table and image grid simultaneously,
so *"does this hit survive if I drop the low-count wells?"* is a second's work
rather than a re-run.

It writes into :func:`spacr.qt.linked_selection.linked_selection`, so the panel
knows nothing about the views and the views know nothing about the panel.

Choosing what to offer
----------------------

A spaCR measurement table has hundreds of columns, so offering all of them in
one list is the same as offering none. The panel classifies them instead:

* **categorical** — few enough distinct values to tick
  (:data:`MAX_CATEGORY_VALUES`), which is what plate, row, column, gene and
  class look like;
* **numeric** — anything pandas reads as a number, offered as a range;
* **skipped** — high-cardinality text, which is neither tickable nor
  rangeable, and object keys, which identify rows rather than describe them.

The classification is a *suggestion*: the picker lists everything it can filter,
and the user chooses. Nothing is filtered until they do.

Cost
----

Every clause change re-evaluates the filter over the whole frame, so the panel
**debounces**. A dragged spinbox emits per keystroke, and re-filtering a
million rows per keystroke would make the control unusable — which is why
:class:`spacr.selection.DataFilter` also replaces rather than appends a clause
on the same column.
"""
from __future__ import annotations

import weakref
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from ..linked_selection import linked_selection
from ...selection import CategoryFilter, DataFilter, RangeFilter
from ..theme import SPACING
from .toggle import Toggle

__all__ = ["DataFilterPanel", "MAX_CATEGORY_VALUES", "classify_columns"]


#: Above this many distinct values a column is a range, not a tick list. Fifty
#: is about where a scrollable checkbox list stops being faster than typing a
#: number — a 384-well plate's columns fit, its wells do not.
MAX_CATEGORY_VALUES = 50

#: Re-filter this long after the last edit. Long enough to swallow a burst of
#: spinbox keystrokes, short enough not to feel laggy.
DEBOUNCE_MS = 200

#: Columns that identify a row rather than describe it. Filtering on them is
#: what the SELECTION is for, and offering them here invites building a
#: selection out of predicates, which does not survive a re-run.
_IDENTITY_COLUMNS = frozenset({"object_label", "prcfo", "prcf", "prc"})


#: ``id(frame) -> (weakref to frame, shape, kinds)``. See
#: :func:`classify_columns`. Entries whose frame has been collected are
#: pruned on the next miss.
_KINDS_CACHE: Dict[int, tuple] = {}

#: How many frames' classifications to remember. Loading one table calls
#: :func:`classify_columns` seven times through four callers, and two frames
#: (raw and filtered) are live at once, so four is generous.
_KINDS_CACHE_MAX = 4


def classify_columns(frame: pd.DataFrame) -> Dict[str, str]:
    """Sort ``frame``'s columns into ``'category'``, ``'range'`` or ``'skip'``.

    Pure and Qt-free so the rule can be tested directly — the panel is only a
    rendering of it.

    **Memoised per frame object.** The classification runs
    ``Series.nunique()`` over every column, which on a 200 000-row x
    48-column measurement table costs 230 ms. Loading one table used to pay
    that *four* times inside a single ``set_frame`` — this function, plus
    :func:`spacr.qt.widgets.graph_spec.column_kinds`, plus
    ``GraphSpec.kinds_for``, plus ``plottable_columns``, each re-deriving the
    same answer from the same object — 0.9 s of the ~1.9 s the GUI thread
    spent delivering a freshly loaded frame.

    The key is the frame's *identity*, not its contents: a subset has fewer
    rows and can genuinely classify differently, so re-deriving for a filtered
    frame is correct rather than wasteful. ``id()`` alone would be unsound
    because CPython reuses addresses, so the entry also holds a ``weakref``
    and the hit is confirmed with ``is`` — a collected frame cannot produce a
    false hit, because its weakref resolves to ``None``. The shape is checked
    too, since a frame can be mutated in place.

    :param frame: any measurement-shaped frame.
    :returns: column name → kind. A fresh dict on every call, because callers
        (``GraphSpec.kinds_for``) update it in place.
    """
    key = id(frame)
    entry = _KINDS_CACHE.get(key)
    if entry is not None:
        ref, shape, cached = entry
        if ref() is frame and shape == frame.shape:
            return dict(cached)
    kinds = _classify_columns_uncached(frame)
    if len(_KINDS_CACHE) >= _KINDS_CACHE_MAX:
        for dead in [k for k, (r, _s, _c) in _KINDS_CACHE.items()
                     if r() is None]:
            _KINDS_CACHE.pop(dead, None)
        while len(_KINDS_CACHE) >= _KINDS_CACHE_MAX:
            _KINDS_CACHE.pop(next(iter(_KINDS_CACHE)), None)
    try:
        _KINDS_CACHE[key] = (weakref.ref(frame), frame.shape, kinds)
    except TypeError:
        # Not weak-referenceable. Skip the cache rather than risk an
        # id-reuse false hit.
        pass
    return dict(kinds)


def _classify_columns_uncached(frame: pd.DataFrame) -> Dict[str, str]:
    """The rule itself, so the cache is one wrapper that can be tested
    against the thing it wraps."""
    kinds: Dict[str, str] = {}
    for name in frame.columns:
        if name in _IDENTITY_COLUMNS:
            kinds[name] = "skip"
            continue
        series = frame[name]
        if pd.api.types.is_numeric_dtype(series) and not \
                pd.api.types.is_bool_dtype(series):
            # A numeric column with a handful of values (a class label, a
            # plate number) is far more useful as ticks than as a range.
            distinct = series.nunique(dropna=True)
            kinds[name] = ("category" if distinct <= 12 else "range")
            continue
        distinct = series.nunique(dropna=True)
        kinds[name] = ("category" if distinct <= MAX_CATEGORY_VALUES
                       else "skip")
    return kinds


class _ClauseRow(QFrame):
    """One active clause, with its own controls and a remove button."""

    changed = Signal()
    removed = Signal(str)

    def __init__(self, column: str, parent=None):
        super().__init__(parent)
        self.column = column
        self.setObjectName("FilterClauseRow")
        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(0, 0, 0, SPACING["xs"])
        self._outer.setSpacing(SPACING["xs"])

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        label = QLabel(column)
        label.setObjectName("FilterClauseTitle")
        head.addWidget(label, 1)
        drop = QPushButton("×")
        drop.setObjectName("FilterClauseRemove")
        drop.setFixedWidth(22)
        drop.setToolTip(f"Stop filtering on {column}")
        drop.clicked.connect(lambda: self.removed.emit(self.column))
        head.addWidget(drop)
        self._outer.addLayout(head)

    def clause(self):  # pragma: no cover - overridden
        raise NotImplementedError


class _RangeRow(_ClauseRow):
    """Low/high bounds for a numeric column."""

    def __init__(self, column: str, series: pd.Series, parent=None):
        super().__init__(column, parent)
        values = pd.to_numeric(series, errors="coerce")
        lo = float(np.nanmin(values)) if values.notna().any() else 0.0
        hi = float(np.nanmax(values)) if values.notna().any() else 1.0
        if hi <= lo:
            # A constant column would give a spinbox with no travel; widen it
            # so the control is still usable rather than inert.
            hi = lo + 1.0

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self._low = QDoubleSpinBox()
        self._high = QDoubleSpinBox()
        for box, value in ((self._low, lo), (self._high, hi)):
            box.setDecimals(3)
            # The bounds are widened well past the data so a user can type a
            # cut-off outside the observed range — which is exactly what you do
            # when you want "everything above what this plate happens to show".
            box.setRange(lo - abs(lo) - 1e6, hi + abs(hi) + 1e6)
            box.setValue(value)
            box.valueChanged.connect(lambda _v: self.changed.emit())
        row.addWidget(QLabel("min"))
        row.addWidget(self._low, 1)
        row.addWidget(QLabel("max"))
        row.addWidget(self._high, 1)
        self._outer.addLayout(row)

    def state(self) -> dict:
        """Enough to rebuild this row. Column included so a saved set can
        say which column it was for -- the panel is rebuilt from the frame,
        and a column that has gone away has to be reported, not guessed."""
        return {"kind": "range", "column": self.column,
                "low": float(self._low.value()),
                "high": float(self._high.value())}

    def restore(self, state: dict) -> None:
        """Put the saved bounds back, clamped to what the CURRENT data allows.

        Clamped rather than trusted: a filter set saved against one plate and
        loaded against another would otherwise silently select nothing, and
        an empty selection looks identical to a filter that did not load.
        """
        self._low.setValue(max(self._low.minimum(),
                               min(float(state.get("low", self._low.value())),
                                   self._low.maximum())))
        self._high.setValue(max(self._high.minimum(),
                                min(float(state.get("high", self._high.value())),
                                    self._high.maximum())))

    def clause(self) -> RangeFilter:
        return RangeFilter(self.column,
                           low=self._low.value(), high=self._high.value())


class _CategoryRow(_ClauseRow):
    """A tick per distinct value."""

    def __init__(self, column: str, series: pd.Series, parent=None):
        super().__init__(column, parent)
        self._boxes: List[Toggle] = []
        values = sorted({str(v) for v in series.dropna().unique()})

        holder = QWidget()
        col = QVBoxLayout(holder)
        col.setContentsMargins(SPACING["sm"], 0, 0, 0)
        col.setSpacing(0)
        for value in values:
            box = Toggle(value)
            box.setChecked(True)
            box.stateChanged.connect(lambda _s: self.changed.emit())
            self._boxes.append(box)
            col.addWidget(box)

        if len(values) > 8:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setMaximumHeight(160)
            scroll.setWidget(holder)
            self._outer.addWidget(scroll)
        else:
            self._outer.addWidget(holder)

    def state(self) -> dict:
        return {"kind": "category", "column": self.column,
                "chosen": [b.text() for b in self._boxes if b.isChecked()]}

    def restore(self, state: dict) -> None:
        """Tick the saved values that still exist in this frame.

        A value that has gone is skipped rather than recreated: the box list
        comes from the data, and a checkbox for a category with no rows is a
        control that can only ever select nothing.
        """
        chosen = set(state.get("chosen", ()))
        for box in self._boxes:
            box.setChecked(box.text() in chosen)

    def clause(self) -> CategoryFilter:
        return CategoryFilter(
            self.column,
            tuple(b.text() for b in self._boxes if b.isChecked()))


class DataFilterPanel(QWidget):
    """The panel. Call :meth:`set_frame`, then let the user drive.

    Emits :attr:`filter_changed` after publishing, for a host that wants to
    update a count label without subscribing to the shared model itself.
    """

    filter_changed = Signal()

    def __init__(self, parent=None, *, link=None):
        super().__init__(parent)
        self.setObjectName("DataFilterPanel")
        # Injectable so a test can drive a private instance rather than the
        # process-wide one, which every other open view is also listening to.
        self._link = link if link is not None else linked_selection()
        self._frame: Optional[pd.DataFrame] = None
        self._rows: Dict[str, _ClauseRow] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["sm"], SPACING["sm"],
                                 SPACING["sm"], SPACING["sm"])
        outer.setSpacing(SPACING["xs"])

        picker = QHBoxLayout()
        picker.setContentsMargins(0, 0, 0, 0)
        self._picker = QComboBox()
        self._picker.setObjectName("FilterColumnPicker")
        self._picker.setToolTip("Add a filter on this column")
        picker.addWidget(self._picker, 1)
        add = QPushButton("Add")
        add.setObjectName("FilterAddButton")
        add.clicked.connect(self._add_selected)
        picker.addWidget(add)
        outer.addLayout(picker)

        self._clauses = QVBoxLayout()
        self._clauses.setContentsMargins(0, 0, 0, 0)
        self._clauses.setSpacing(0)
        outer.addLayout(self._clauses)

        outer.addStretch(1)

        self._summary = QLabel("no filter")
        self._summary.setObjectName("FilterSummary")
        self._summary.setWordWrap(True)
        outer.addWidget(self._summary)

        self._clear = QPushButton("Clear all")
        self._clear.setObjectName("FilterClearButton")
        self._clear.clicked.connect(self.clear)
        outer.addWidget(self._clear)

        # One shared debounce: a burst of edits across several clauses still
        # costs one re-filter.
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(DEBOUNCE_MS)
        self._debounce.timeout.connect(self._publish)

    # -- population ----------------------------------------------------
    def set_frame(self, frame: pd.DataFrame) -> None:
        """Point the panel at a table and offer its filterable columns.

        Existing clauses are dropped rather than carried over: a clause naming
        a column the new frame does not have would raise on the next apply,
        and silently keeping only the ones that still resolve would narrow by
        less than the panel claims to.
        """
        self._frame = frame
        self.clear()
        kinds = classify_columns(frame)
        self._kinds = kinds
        self._picker.clear()
        for name in sorted(k for k, v in kinds.items() if v != "skip"):
            self._picker.addItem(name)

    def available_columns(self) -> List[str]:
        """The columns the picker is offering, in order."""
        return [self._picker.itemText(i) for i in range(self._picker.count())]

    # -- clauses -------------------------------------------------------
    def add_column(self, column: str) -> None:
        """Add a clause row for ``column``. Adding twice is a no-op."""
        if self._frame is None or column in self._rows:
            return
        kind = getattr(self, "_kinds", {}).get(column)
        if kind == "range":
            row: _ClauseRow = _RangeRow(column, self._frame[column])
        elif kind == "category":
            row = _CategoryRow(column, self._frame[column])
        else:
            return
        row.changed.connect(self._schedule)
        row.removed.connect(self.remove_column)
        self._rows[column] = row
        self._clauses.addWidget(row)
        self._schedule()

    def remove_column(self, column: str) -> None:
        row = self._rows.pop(column, None)
        if row is None:
            return
        self._clauses.removeWidget(row)
        row.setParent(None)
        row.deleteLater()
        self._schedule()

    def clear(self) -> None:
        """Drop every clause and publish an empty filter."""
        for column in list(self._rows):
            row = self._rows.pop(column)
            self._clauses.removeWidget(row)
            row.setParent(None)
            row.deleteLater()
        self._schedule()

    def _add_selected(self) -> None:
        text = self._picker.currentText()
        if text:
            self.add_column(text)

    # -- publishing ----------------------------------------------------
    def _schedule(self) -> None:
        self._debounce.start()

    # -- saving a filter set -------------------------------------------
    # Gates have had Save/Load since the beginning and filters had not, so a
    # filter set -- which is as much of an analysis decision as a gate -- had
    # to be rebuilt by hand every session.

    def state(self) -> dict:
        """The whole panel, as plain data.

        Versioned because the row kinds will grow. A reader that meets a
        version it does not know refuses rather than guessing, since a
        half-applied filter set silently selects the wrong rows.
        """
        # `_rows` is a dict and dicts keep insertion order, which IS the
        # order the user added the filters in and the order they read on
        # screen. Restoring in the same order matters for a category filter
        # whose box list is built from what earlier filters left.
        rows = [row.state() for row in self._rows.values()
                if hasattr(row, "state")]
        return {"version": 1, "filters": rows}

    def restore(self, state: dict) -> List[str]:
        """Apply a saved set to the CURRENT frame.

        :returns: the columns that could not be restored, so the caller can
            say so. A filter set saved against one table and loaded against
            another is a normal thing to do -- what must not happen is it
            appearing to work while quietly filtering on nothing.
        """
        if int(state.get("version", 0)) != 1:
            raise ValueError(
                f"unknown filter-set version {state.get('version')!r}; "
                f"this build understands version 1")

        self.clear()
        missing = []
        available = set(self.available_columns())
        for entry in state.get("filters", ()):
            column = str(entry.get("column", ""))
            if column not in available:
                missing.append(column)
                continue
            self.add_column(column)
            row = self._rows.get(column)
            if row is not None and hasattr(row, "restore"):
                row.restore(entry)
        self._publish()
        return missing

    def save(self, path: str) -> str:
        """Write the filter set to ``path`` as JSON."""
        import json

        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.state(), handle, indent=2)
        return path

    def load(self, path: str) -> List[str]:
        """Read a filter set from ``path``. Returns the missing columns."""
        import json

        with open(path, encoding="utf-8") as handle:
            return self.restore(json.load(handle))

    def current_filter(self) -> DataFilter:
        """The filter the controls currently describe."""
        data_filter = DataFilter()
        for row in self._rows.values():
            data_filter.add(row.clause())
        return data_filter

    def _publish(self) -> None:
        data_filter = self.current_filter()
        self._summary.setText(data_filter.describe())
        self._link.set_filter(data_filter)
        self.filter_changed.emit()

    def flush(self) -> None:
        """Publish immediately instead of waiting out the debounce.

        For tests, and for a host that needs the filter applied before it does
        something else — closing the panel, or running an export.
        """
        self._debounce.stop()
        self._publish()
