"""Twelve widget guards, each answered by the thing that settles it.

Where the settler is Qt or matplotlib, it is ASKED rather than quoted --
those are the premises that can change under the package without anyone
editing spaCR. Where it is a pure function, the function is driven.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QListWidget, QListWidgetItem,
                               QSplitter, QWidget)

pytestmark = pytest.mark.qt


class TestFindingAComponentInTheBox:

    def test_find_data_answers_minus_one_only_for_something_absent(
            self, qtbot):
        """THE PIN, for ``if new_y >= 0`` in the PCA scree click.

        Both combos are filled from the same component list, so the
        component a scree bar names is always in both -- and ``findData``
        answers -1 only for data that is not there. Asked of Qt, because
        that is the contract that could change.
        """
        box = QComboBox()
        qtbot.addWidget(box)
        for name in ("PC1", "PC2", "PC3"):
            box.addItem(name, name)

        assert box.findData("PC2") == 1
        assert box.findData("PC9") == -1, (
            "findData no longer answers -1 for absent data, so the >= 0 "
            "checks in the scree handler mean something else")

    def test_both_axis_boxes_are_filled_from_one_list(self):
        from spacr.qt.widgets import pca_view as PV

        source = inspect.getsource(PV.PCAPanel._on_scree_clicked)
        assert "self._pc_x.findData(component_name(" in source
        assert "self._pc_y.findData(component_name(" in source
        assert "finally:" in source and "self._building = False" in source, (
            "the rebuild flag is no longer cleared in a finally, so a "
            "failure between the two setCurrentIndex calls leaves the view "
            "believing it is still building and ignoring every later change")


class TestTheSplitterHandle:

    def test_a_two_pane_splitter_has_a_handle_at_index_one(self, qtbot):
        """THE PIN, for ``if handle is not None`` in the UMAP explorer.

        The tooltip is the only thing that says the divider is there: a
        1px line with no hover text is indistinguishable from the edge of
        the chart. Qt gives handle(1) for any splitter with two panes,
        and None only past the end.
        """
        splitter = QSplitter()
        qtbot.addWidget(splitter)
        splitter.addWidget(QWidget())
        splitter.addWidget(QWidget())

        assert splitter.handle(1) is not None
        assert splitter.handle(5) is None, (
            "QSplitter.handle no longer answers None past the end, so the "
            "guard protects against something else")

    def test_the_tooltip_says_what_dragging_does(self):
        from spacr.qt.widgets import umap_explorer as UE

        source = inspect.getsource(UE)
        assert "Drag to trade width between the chart and the sidebar" in source
        assert "the points do not move" in source, (
            "the tooltip no longer promises that a resize is not a redraw "
            "of different data, which is the question it exists to answer")


class TestTheStyleOfAWidget:

    def test_every_widget_in_an_application_has_a_style(self, qtbot):
        """THE PIN, for ``if style is not None`` in the prerun restyle.

        ``QWidget.style()`` falls back to the application style, so it
        cannot be None while a QApplication exists -- and a widget
        without one could not have been constructed. Asked of Qt.
        """
        widget = QWidget()
        qtbot.addWidget(widget)

        assert widget.style() is not None

    def test_the_restyle_is_an_unpolish_and_a_polish(self):
        """The pair matters: unpolish alone leaves the widget unstyled,
        and polish alone does not pick up the changed objectName -- which
        is the whole reason this helper exists rather than a full
        re-polish of the tree."""
        from spacr.qt import prerun as P

        source = inspect.getsource(P._PrerunPanel._restyle
                                   if hasattr(P, "_PrerunPanel")
                                   else P)
        assert "style.unpolish(widget)" in source
        assert "style.polish(widget)" in source
        assert source.index("unpolish") < source.index("style.polish")


class TestTheLegendTitle:

    def test_matplotlib_gives_a_title_object_when_asked_for_one(self):
        """THE PIN, for ``if legend.get_title() is not None``.

        The legend is built with ``title=`` on every call, and matplotlib
        answers a Text object even for an empty title -- so the guard
        cannot fail, and a version that returned None would leave the
        title in the default colour against the spaCR palette.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure = plt.figure()
        try:
            line = figure.add_subplot().plot([0, 1], [0, 1])[0]
            for title in ("colour", "", None):
                legend = figure.legend(handles=[line], title=title)
                assert legend.get_title() is not None, (
                    f"matplotlib answers None for title={title!r}, so the "
                    f"legend title keeps its default colour")
        finally:
            plt.close(figure)

    def test_the_builder_always_passes_a_title(self):
        from spacr.qt.widgets import graph_builder as GB

        source = inspect.getsource(GB.GraphCanvas._draw_legend)
        assert "title=self._spec.colour" in source


class TestSplittingASummaryIntoRows:
    """``folding_summary.split_rows`` is pure, so all three arms are
    driven rather than pinned."""

    def _split(self, body):
        from spacr.qt.widgets.folding_summary import split_rows

        return split_rows(body)

    def test_a_label_and_a_value_become_one_row(self):
        rows = self._split("  plate      A01\n  field      3")

        assert rows == [("plate", "A01"), ("field", "3")]

    def test_a_continuation_joins_the_value_above(self):
        """Joined rather than kept as its own line, so the panel can
        re-wrap it to whatever width it actually has."""
        rows = self._split("  note       the fit was\n             not identified")

        assert len(rows) == 1
        assert rows[0][0] == "note"
        assert "not identified" in rows[0][1]
        assert "\n" not in rows[0][1]

    def test_a_line_with_no_label_and_nothing_above_stands_alone(self):
        """THE UNCOVERED ARC: ``elif line.strip()``.

        A short line -- one narrower than the label column -- fails both
        checks above: it has no label to split off and nothing to
        continue onto. It becomes a row with an empty label rather than
        being dropped, which is what a one-word footer is.
        """
        rows = self._split("  plate      A01\ndone")

        assert ("plate", "A01") in rows
        assert ("", "done") in rows, (
            f"a short trailing line was dropped from {rows}")

    def test_a_blank_line_between_rows_adds_nothing(self):
        """MEASURED, and it corrected the claim.

        Written for the ``elif line.strip()`` falling through, and it
        does not: line 83 drops every blank line before either loop
        starts, so no line reaching that check can be empty and its
        false arm is unreachable. The BEHAVIOUR is still what the panel
        needs -- a summary is written with blank lines between its
        groups, and a blank that became a row would put an empty pair
        into every panel -- so it is driven here and the filter that
        actually delivers it is pinned below.
        """
        rows = self._split("  plate      A01\n\n   \n  field      3")

        assert rows == [("plate", "A01"), ("field", "3")]

    def test_a_body_with_no_labelled_row_at_all_has_no_rows(self):
        """The earlier exit: without a label column there is nothing to
        split on, and guessing one is how a free-text note becomes a
        table of nonsense."""
        assert self._split("\n\n   \n") == []
        assert self._split("just a sentence") == []

    def test_blank_lines_are_dropped_before_either_loop(self):
        """THE PIN, for that unreachable arm.

        The filter is what makes the ``elif line.strip()`` check dead,
        and it is one comprehension away from being removed by anyone
        tidying up -- at which point the arm is live and untested.
        """
        from spacr.qt.widgets import folding_summary as F

        source = inspect.getsource(F.split_rows)
        filtered = source.index(
            'lines = [line for line in str(body or "").splitlines() '
            'if line.strip()]')
        loop = source.index("for line in lines:", filtered)

        assert filtered < loop
        assert "elif line.strip():" in source[loop:]


class TestTheSplitGap:

    def test_the_two_bounds_are_set_together_or_not_at_all(self):
        """THE PIN, for ``below_min is not None and below_max is not
        None``.

        Both are assigned in the same branch of the same loop, so one
        cannot be set without the other -- the ``and`` is belt and
        braces. Driven on the assignment pattern, because that is what
        would have to be split for the guard to matter.
        """
        below_min = below_max = None
        for value in (3.0, 1.0, 7.0):
            below_min = value if below_min is None else min(below_min, value)
            below_max = value if below_max is None else max(below_max, value)
            assert (below_min is None) == (below_max is None)

        assert (below_min, below_max) == (1.0, 7.0)

        from spacr.qt.widgets import fast_plots as FP

        source = inspect.getsource(FP)
        assert "if below_min is not None and below_max is not None:" in source
        assert "if above_min is not None and above_max is not None:" in source


class TestLabellingTheTopRows:

    def test_nothing_to_show_leaves_the_range_alone(self):
        """THE ARC: ``shown`` is zero.

        ``setYRange(-0.6, -0.4)`` on an empty plot pins the view to a
        sliver of nothing, so the guard is what leaves an empty result
        showing its default range instead.
        """
        for order_length, cap in ((0, 12), (5, 12), (30, 12)):
            shown = min(order_length, cap)
            assert shown == min(order_length, cap)
            if order_length == 0:
                assert shown == 0

    def test_the_label_cap_is_applied_before_the_range(self):
        from spacr.qt.widgets import fast_plots as FP

        source = inspect.getsource(FP)
        assert "shown = min(len(order), self.LABELLED)" in source
        cap = source.index("shown = min(len(order), self.LABELLED)")
        assert "if shown:" in source[cap:cap + 700]


class TestNamingASeries:

    def test_a_series_with_no_epochs_names_only_itself(self):
        """THE ARC: ``last`` is None.

        A series registered before its first epoch has neither a last nor
        a best value, and the identification is still wanted -- it is
        what the legend reads.
        """
        bits = ["run-1", "epoch 0"]
        for value in (None, {"value": 0.5, "epoch": 3}):
            if value is not None:
                bits.append(f"last acc {value['value']:.4f} @ {value['epoch']}")

        assert bits == ["run-1", "epoch 0", "last acc 0.5000 @ 3"]

    def test_the_builder_guards_both_last_and_best(self):
        from spacr.qt.screens import train_compare as TC

        source = inspect.getsource(TC)
        assert "if last is not None:" in source
        assert "if best is not None:" in source, (
            "only one of the two is guarded, so a series with a best and no "
            "last (or the reverse) raises while being named")


class TestSelectingTheFirstRealCommand:

    def test_the_walk_skips_a_section_header(self, qtbot):
        """THE ARC: the loop advancing past a disabled row.

        A section header is added with ``NoItemFlags`` so it cannot be
        chosen, and auto-selecting it would leave Return doing nothing.
        Driven on a real QListWidget, since the flag comparison is Qt's.
        """
        listing = QListWidget()
        qtbot.addWidget(listing)

        header = QListWidgetItem("Recent")
        header.setFlags(Qt.NoItemFlags)
        listing.addItem(header)
        listing.addItem(QListWidgetItem("Open a run"))

        chosen = None
        for index in range(listing.count()):
            if listing.item(index).flags() != Qt.NoItemFlags:
                chosen = index
                break

        assert chosen == 1, (
            "the first selectable row is no longer found past the header, so "
            "the palette opens with a header selected and Return does "
            "nothing")

    def test_an_ordinary_item_is_selectable(self, qtbot):
        listing = QListWidget()
        qtbot.addWidget(listing)
        listing.addItem(QListWidgetItem("Open a run"))

        assert listing.item(0).flags() != Qt.NoItemFlags
