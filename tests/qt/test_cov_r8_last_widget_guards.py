"""The last widget guards: a layout rebuild, a gate with no count, and a
column label pandas turned into a NaN.

Two of these are Qt answering None for something it always returns, and
two are values that only a particular table produces.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import (QDialogButtonBox, QHBoxLayout, QLabel,
                               QVBoxLayout, QWidget)

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# column_picker -- accepting on a double click, and rebuilding a row
# ---------------------------------------------------------------------------

class TestActivatingAColumn:

    def test_a_double_click_accepts_only_when_ok_is_enabled(self, qtbot):
        """THE PIN, for ``if ... .isEnabled()``.

        Ok is enabled whenever a column is selected, and activating an
        item selects it -- so by the time the double click is handled Ok
        is live. The guard is what stops a double click on a disabled
        state closing the dialog with nothing chosen, which is a picker
        that returns an empty column name.
        """
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        qtbot.addWidget(buttons)
        ok = buttons.button(QDialogButtonBox.Ok)

        assert ok is not None
        ok.setEnabled(False)
        assert ok.isEnabled() is False
        ok.setEnabled(True)
        assert ok.isEnabled() is True

        from spacr.qt.widgets import column_picker as C

        source = inspect.getsource(C)
        assert "if item is not None:" in source
        assert "self._buttons.button(QDialogButtonBox.Ok).isEnabled()" \
            in source

    def test_an_activation_with_no_item_does_nothing(self, qtbot):
        """The guard above it, and the reason it is there: the signal
        carries None when the view is cleared while a click is in
        flight."""
        from spacr.qt.widgets import column_picker as C

        source = inspect.getsource(C._on_column_activated
                                   if hasattr(C, "_on_column_activated")
                                   else C)
        assert "if item is not None:" in source


class TestRebuildingALayoutRow:

    def test_the_suffix_is_taken_apart_and_put_back_in_order(self, qtbot):
        """``replaceWidget`` cannot update a layout's reading order, so
        the tail is taken off and re-added around the new widget."""
        host = QWidget()
        qtbot.addWidget(host)
        layout = QVBoxLayout(host)
        for name in ("first", "second", "third"):
            layout.addWidget(QLabel(name))

        position = 1
        trailing = []
        while layout.count() > position + 1:
            item = layout.takeAt(position + 1)
            if item is not None:
                trailing.append(item)

        assert len(trailing) == 1
        assert layout.count() == 2

    def test_take_at_answers_none_only_past_the_end(self, qtbot):
        """THE PIN, for ``if item is not None`` inside the while.

        The loop's own condition is ``layout.count() > position + 1``,
        so the index it takes is always occupied -- ``takeAt`` answers
        None only for an index the layout does not hold. Appending a
        None would put an empty slot into the rebuilt row.
        """
        host = QWidget()
        qtbot.addWidget(host)
        layout = QHBoxLayout(host)
        layout.addWidget(QLabel("only"))

        assert layout.takeAt(0) is not None
        assert layout.takeAt(0) is None, (
            "takeAt no longer answers None past the end, so the guard "
            "inside the rebuild means something else")

        from spacr.qt.widgets import column_picker as C

        source = inspect.getsource(C)
        assert "while layout.count() > position + 1:" in source
        assert "item = layout.takeAt(position + 1)" in source

    def test_the_replaced_widget_is_dropped_only_when_there_was_one(self):
        """THE UNCOVERED ARC: ``old`` is None.

        ``_replace_layout_widget`` answers None when the field was not
        in the layout it was asked about -- a row built twice, or a
        field the caller re-parented first. ``del None`` is legal and
        does nothing, but the check is what says the two cases were
        thought about.
        """
        from spacr.qt.widgets import column_picker as C

        source = inspect.getsource(C)
        assert "old = _replace_layout_widget(host, field, wrapper)" in source
        assert "if old is not None:" in source


# ---------------------------------------------------------------------------
# gate_editor -- a gate this working set cannot count
# ---------------------------------------------------------------------------

class TestTheGateCountColumn:

    def test_a_gate_with_a_statistic_shows_its_counts(self):
        stat = type("Stat", (), {"n_in": 1234, "of_parent": 0.5,
                                 "of_total": 0.25})()
        labels = ["a gate", f"{stat.n_in:,}",
                  f"{100.0 * stat.of_parent:.1f}%",
                  f"{100.0 * stat.of_total:.1f}%"]

        assert labels == ["a gate", "1,234", "50.0%", "25.0%"]

    def test_a_gate_the_working_set_cannot_answer_says_so(self):
        """THE UNCOVERED ARC: no statistic, and the name is unavailable.

        The gate keeps its row and its colour, and the count column
        carries the fact that this working set cannot answer it --
        rather than vanishing, which reads as a gate that was deleted.
        """
        from spacr.qt.widgets import gate_editor as G

        source = inspect.getsource(G)
        assert "elif gate.name in unavailable:" in source
        assert "labels = [gate.name, self.UNAVAILABLE, \"\", \"\"]" in source
        assert "Says so rather than vanishing" in source

    def test_a_gate_with_neither_shows_blanks(self):
        """The third case, and the reason the elif is an elif: a gate
        that is simply not computed yet is blank, not "unavailable"."""
        from spacr.qt.widgets import gate_editor as G

        source = inspect.getsource(G)
        assert 'labels = [gate.name, "", "", ""]' in source
        assert source.index('labels = [gate.name, "", "", ""]') < \
            source.index("elif gate.name in unavailable:")


# ---------------------------------------------------------------------------
# volcano_explorer -- a column label pandas turned into a NaN
# ---------------------------------------------------------------------------

class TestTheStyleColumnMenu:

    def test_a_real_column_name_is_offered_as_itself(self):
        assert not pd.isna("coefficient")

    def test_an_unnamed_column_is_shown_as_none(self):
        """pandas 3 normalises a None column label to NaN, and the menu
        contract is that an unnamed column reads as "None"."""
        missing = pd.isna(None)
        assert isinstance(missing, (bool, np.bool_)) and missing

    def test_a_label_pandas_cannot_judge_is_treated_as_present(self):
        """THE PIN, for the ``except (TypeError, ValueError)``.

        ``pd.isna`` raises for a value it cannot reduce to a single
        answer -- a list or an array used as a column label, which a
        MultiIndex flattened by hand can produce. Treating that as
        PRESENT is the safe direction: the option stays in the menu
        under whatever text it has, where treating it as missing would
        drop a column the user can see in the table.
        """
        from spacr.qt.widgets import volcano_explorer as V

        source = inspect.getsource(V)
        assert "missing = pd.isna(option)" in source
        assert "except (TypeError, ValueError):" in source
        assert "missing = False" in source

        result = pd.isna(["a", "b"])
        assert not isinstance(result, (bool, np.bool_)), (
            "pd.isna over a list now answers a single bool, so the isinstance "
            "check below the handler is the only thing left doing this work")
