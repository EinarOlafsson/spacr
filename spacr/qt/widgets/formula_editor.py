"""The column formula editor — type an expression, get a column.

The chrome over :mod:`spacr.qt.widgets.formula`. Everything about *what an
expression means* is in there; this module is a name box, an expression box, a
list of what has been defined and a preview of the first few values.

Three things it insists on
--------------------------

**It validates while you type, not when you commit.** The expression is parsed
on every keystroke (a parse of a forty-character string is microseconds), so
the error message for ``area / perimter`` appears next to the typo rather than
after an Add that appears to do nothing. The Add button is disabled until the
formula parses *and* resolves against the loaded table.

**It shows the values, not just the syntax.** A formula can be perfectly valid
and produce a column that is 90% infinities, which no amount of syntax checking
catches. The preview evaluates the expression over the head of the table and
prints :attr:`~spacr.qt.widgets.formula.ColumnResult.notice` — "3 of 4 rows
have a finite value · 1 became NaN or infinite in the calculation" — which is
the sentence that stops a bad ratio column reaching a chart.

**It hands back a frame, not a mutation.** :meth:`FormulaPanel.computed_frame`
returns the loaded table *plus* the computed columns, a copy every time. The
loaded frame is never grown, so removing a formula removes its column, and two
screens showing the same table do not accumulate each other's columns.

Where the columns go
--------------------

Nowhere, by itself. The panel emits :attr:`FormulaPanel.formulas_changed` and
the host re-hands the computed frame to whatever it is driving — the trellis,
the gate editor, the feature explorer, the Local Data Filter. That is the whole
integration: a computed column is an ordinary column from the moment it exists,
classified by the same :func:`~spacr.qt.widgets.data_filter_panel.classify_columns`
rule as a measured one, so it appears in the column well, in the filter picker
and in the export without any of them being told about formulas.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import pandas as pd
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QPushButton, QVBoxLayout, QWidget,
)

from ..theme import RADIUS, SPACING, register_widget_qss
from .formula import (
    FUNCTION_HELP, ColumnFormula, ColumnResult, FormulaError, FormulaSet,
    compute,
)

LOG = logging.getLogger("spacr.qt.formula_editor")

__all__ = ["FormulaPanel", "FormulaDialog", "PREVIEW_ROWS"]

#: Rows the live preview evaluates over. Enough to see whether a ratio is
#: sensible, small enough that the parse-per-keystroke stays free on a
#: million-row table. The committed column is computed over all of it.
PREVIEW_ROWS = 2_000

#: Validation is re-run this long after the last keystroke.
DEBOUNCE_MS = 120


class FormulaPanel(QWidget):
    """Define computed columns for one table.

    :param frame: the table, or ``None`` until :meth:`set_frame`.

    Emits :attr:`formulas_changed` whenever the set changes — the host's cue
    to re-read :meth:`computed_frame`.
    """

    formulas_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("FormulaPanel")
        self._frame: Optional[pd.DataFrame] = None
        self._formulas = FormulaSet()
        self._results: List[ColumnResult] = []
        self._computed: Optional[pd.DataFrame] = None
        #: Why the defined formulas do not apply to the loaded table, if they
        #: do not. Kept rather than only printed once: it has to survive the
        #: validator running on an empty box, which is the state right after a
        #: table change, and a panel that silently drops a column the user
        #: defined is the failure this whole file is against.
        self._apply_error = ""

        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["sm"], SPACING["sm"],
                                 SPACING["sm"], SPACING["sm"])
        outer.setSpacing(SPACING["xs"])

        title = QLabel("Computed columns", self)
        title.setObjectName("FormulaTitle")
        outer.addWidget(title)

        entry = QHBoxLayout()
        entry.setContentsMargins(0, 0, 0, 0)
        entry.setSpacing(SPACING["xs"])
        self._name = QLineEdit(self)
        self._name.setObjectName("FormulaName")
        self._name.setPlaceholderText("new column")
        self._name.setMaximumWidth(160)
        self._name.setToolTip(
            "The new column's name. Letters, digits and underscores, so it "
            "can be used in another formula without quoting.")
        entry.addWidget(self._name)
        entry.addWidget(QLabel("=", self))
        self._expression = QLineEdit(self)
        self._expression.setObjectName("FormulaExpression")
        self._expression.setPlaceholderText("area / perimeter ** 2")
        self._expression.setToolTip(
            "An arithmetic expression over the table's numeric columns.\n"
            "+ - * / // % **, comparisons, and / or / not, and the functions "
            "listed below.")
        entry.addWidget(self._expression, 1)
        outer.addLayout(entry)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(SPACING["xs"])
        self._replace = QCheckBox("replace an existing column", self)
        self._replace.setToolTip(
            "Off by default: shadowing a measured column silently would make "
            "every earlier chart of it unreproducible.")
        controls.addWidget(self._replace)
        controls.addStretch(1)
        self._add = QPushButton("Add column", self)
        self._add.setObjectName("PrimaryButton")
        self._add.setEnabled(False)
        self._add.clicked.connect(self.commit)
        controls.addWidget(self._add)
        outer.addLayout(controls)

        self._status = QLabel("", self)
        self._status.setObjectName("FormulaStatus")
        self._status.setWordWrap(True)
        self._status.setProperty("state", "idle")
        outer.addWidget(self._status)

        self._list = QListWidget(self)
        self._list.setObjectName("FormulaList")
        self._list.setToolTip(
            "The columns defined so far, in the order they are computed — "
            "each one can use the ones above it. Delete removes the selected "
            "formula.")
        self._list.setMaximumHeight(140)
        outer.addWidget(self._list, 1)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        remove = QPushButton("Remove", self)
        remove.clicked.connect(self.remove_selected)
        row.addWidget(remove)
        clear = QPushButton("Clear all", self)
        clear.clicked.connect(self.clear)
        row.addWidget(clear)
        row.addStretch(1)
        outer.addLayout(row)

        self._help = QLabel(self._help_text(), self)
        self._help.setObjectName("FormulaHelp")
        self._help.setWordWrap(True)
        outer.addWidget(self._help)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(DEBOUNCE_MS)
        self._debounce.timeout.connect(self._validate)
        self._expression.textChanged.connect(self._schedule)
        self._name.textChanged.connect(self._schedule)
        self._replace.toggled.connect(self._schedule)
        self._expression.returnPressed.connect(self.commit)

    # -- the table -------------------------------------------------------
    def set_frame(self, frame: Optional[pd.DataFrame]) -> None:
        """Point the panel at a table.

        Existing formulas are **kept** and re-applied. That is the opposite of
        what the Local Data Filter does with its clauses, on purpose: a filter
        clause naming a missing column narrows by less than it claims and is
        dangerous to keep, while a formula naming a missing column simply
        fails, says which column, and is exactly what the user wants back when
        they reload the same table.
        """
        self._frame = frame
        self._recompute()
        self._validate()

    def frame(self) -> Optional[pd.DataFrame]:
        """The table as loaded, without the computed columns."""
        return self._frame

    def computed_frame(self) -> Optional[pd.DataFrame]:
        """The table **plus** the computed columns; a copy, never the original.

        Returns the loaded frame unchanged when nothing is defined, so a host
        can call this unconditionally.
        """
        if self._frame is None:
            return None
        return self._computed if self._computed is not None else self._frame

    # -- the formulas ----------------------------------------------------
    def formulas(self) -> FormulaSet:
        """The defined formulas. The panel's own object — copy before editing."""
        return self._formulas

    def set_formulas(self, formulas: Sequence[ColumnFormula]) -> None:
        """Replace the whole set — for restoring a saved analysis."""
        self._formulas = FormulaSet(list(formulas))
        self._recompute()
        self._refresh_list()
        self.formulas_changed.emit()

    def add_formula(self, formula: ColumnFormula) -> bool:
        """Add ``formula``, or report why it cannot be computed here.

        :returns: True when it was added.
        """
        candidate = FormulaSet(list(self._formulas.formulas)).add(formula)
        if self._frame is not None:
            try:
                candidate.apply(self._frame)
            except FormulaError as exc:
                self._say(str(exc), "error")
                return False
        self._formulas = candidate
        self._recompute()
        self._refresh_list()
        self.formulas_changed.emit()
        return True

    def commit(self) -> bool:
        """Add the formula currently in the two boxes."""
        try:
            formula = self._current_formula()
        except FormulaError as exc:
            self._say(str(exc), "error")
            return False
        if formula is None:
            return False
        if not self.add_formula(formula):
            return False
        self._name.clear()
        self._expression.clear()
        self._replace.setChecked(False)
        self._say(f"added {formula.name}", "ok")
        return True

    def remove_selected(self) -> None:
        item = self._list.currentItem()
        if item is None:
            return
        self.remove(item.data(Qt.UserRole))

    def remove(self, name: str) -> None:
        """Drop the formula called ``name`` and its column."""
        if name not in self._formulas.names:
            return
        self._formulas.remove(name)
        self._recompute()
        self._refresh_list()
        self.formulas_changed.emit()

    def clear(self) -> None:
        if self._formulas.is_empty:
            return
        self._formulas.clear()
        self._recompute()
        self._refresh_list()
        self.formulas_changed.emit()

    def results(self) -> List[ColumnResult]:
        """One :class:`~spacr.qt.widgets.formula.ColumnResult` per column."""
        return list(self._results)

    def status(self) -> str:
        """The line under the boxes — the validation message or the notice."""
        return self._status.text()

    # -- internals -------------------------------------------------------
    def _current_formula(self) -> Optional[ColumnFormula]:
        name = self._name.text().strip()
        expression = self._expression.text().strip()
        if not name or not expression:
            return None
        return ColumnFormula(name, expression,
                             replace=self._replace.isChecked())

    def _schedule(self) -> None:
        self._add.setEnabled(False)
        self._debounce.start()

    def _validate(self) -> None:
        """Parse and evaluate the pending formula over the head of the table."""
        self._debounce.stop()
        name = self._name.text().strip()
        expression = self._expression.text().strip()
        if not expression:
            self._add.setEnabled(False)
            if self._apply_error:
                self._say(self._apply_error, "error")
            else:
                self._say("" if not name else
                          "type an expression, then press Enter", "idle")
            return
        try:
            formula = ColumnFormula(name or "preview", expression,
                                    replace=self._replace.isChecked())
        except FormulaError as exc:
            self._add.setEnabled(False)
            self._say(str(exc), "error")
            return
        if self._frame is None:
            self._add.setEnabled(bool(name))
            self._say("parses — load a table to see the values", "ok")
            return
        base = self.computed_frame()
        head = base.head(PREVIEW_ROWS)
        try:
            _preview, results = compute(head, [formula])
        except FormulaError as exc:
            self._add.setEnabled(False)
            self._say(str(exc), "error")
            return
        self._add.setEnabled(bool(name))
        note = results[0].notice
        if len(base) > len(head):
            note += f" (previewed on the first {len(head):,} rows)"
        if formula.uses_whole_table():
            note += (" · uses the whole table, so this column changes if the "
                     "table does")
        self._say(note, "ok" if name else "idle")

    def _recompute(self) -> None:
        """Apply every formula to the loaded frame, or report the first failure.

        A failure leaves :meth:`computed_frame` returning the loaded table
        unchanged — usable, minus the columns that could not be computed — and
        the reason on the status line, where it stays until the formula is
        fixed or dropped.
        """
        self._computed = None
        self._results = []
        self._apply_error = ""
        if self._frame is None:
            return
        try:
            self._computed, self._results = self._formulas.apply(self._frame)
        except FormulaError as exc:
            LOG.info("computed columns do not apply to this table: %s", exc)
            self._apply_error = str(exc)
            self._say(self._apply_error, "error")

    def _refresh_list(self) -> None:
        self._list.clear()
        notices = {r.formula.name: r.notice for r in self._results}
        for formula in self._formulas.formulas:
            item = QListWidgetItem(formula.describe())
            item.setData(Qt.UserRole, formula.name)
            item.setToolTip(notices.get(formula.name, formula.expression))
            self._list.addItem(item)

    def _say(self, text: str, state: str) -> None:
        self._status.setText(text)
        self._status.setProperty("state", state)
        self._status.style().unpolish(self._status)
        self._status.style().polish(self._status)

    @staticmethod
    def _help_text() -> str:
        picks = ("log", "sqrt", "abs", "clip", "where", "minimum", "maximum",
                 "zscore", "rank", "mean", "median", "std", "quantile",
                 "count", "min", "max")
        return ("Operators: + - * / // % **, < <= > >= == !=, and / or / not. "
                "Functions: " + ", ".join(picks) + ". "
                "min() and max() are one number for the whole table; "
                "minimum() and maximum() compare per object. "
                "Backtick a name with spaces: `cell area`.")


class FormulaDialog(QDialog):
    """:class:`FormulaPanel` in a window, for a screen with no room for it.

    Non-modal, so the chart behind it redraws as columns are added — which is
    the point of adding them.
    """

    def __init__(self, parent=None, *, panel: Optional[FormulaPanel] = None):
        super().__init__(parent)
        self.setObjectName("FormulaDialog")
        self.setWindowTitle("Computed columns")
        self.setModal(False)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.panel = panel if panel is not None else FormulaPanel(self)
        outer.addWidget(self.panel)
        close = QPushButton("Close", self)
        close.clicked.connect(self.accept)
        outer.addWidget(close)
        self.resize(560, 460)


def _formula_qss(palette, _opacity) -> str:
    return f"""
    QLabel#FormulaTitle {{
        color: {palette['fg']};
        font-weight: 600;
    }}
    QLineEdit#FormulaName, QLineEdit#FormulaExpression {{
        border: 1px solid {palette['border']};
        border-radius: {RADIUS['sm']}px;
        padding: 4px 6px;
        background: {palette['surface_alt']};
        color: {palette['fg']};
    }}
    QLabel#FormulaStatus[state="error"] {{ color: {palette['error']}; }}
    QLabel#FormulaStatus[state="ok"] {{ color: {palette['success']}; }}
    QLabel#FormulaStatus[state="idle"] {{ color: {palette['fg_muted']}; }}
    QLabel#FormulaHelp {{
        color: {palette['fg_muted']};
        font-size: 11px;
    }}
    QListWidget#FormulaList {{
        border: 1px solid {palette['border_soft']};
        border-radius: {RADIUS['sm']}px;
        background: {palette['surface']};
        color: {palette['fg']};
    }}
    """


try:
    register_widget_qss("FormulaPanel", _formula_qss)
except ValueError:  # pragma: no cover - a re-import in one process
    pass
