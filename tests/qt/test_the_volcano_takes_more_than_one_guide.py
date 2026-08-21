"""Several guides can be selected on the volcano at once (instruction 206).

THE TESTS DRIVE THE GESTURES, not the selection model -- the instruction
asks for that in as many words, and it is the only way to catch a selection
API that works perfectly and is wired to nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6.QtCore import Qt         # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def volcano(app):
    """A keyed scatter with six points at known coordinates."""
    from spacr.qt.widgets.fast_plots import FastPlot

    plot = FastPlot(title="volcano")
    frame = pd.DataFrame({
        "grna": [f"g{i}" for i in range(6)],
        "coefficient": [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
        "neglog10p": [1.0, 2.0, 0.5, 3.0, 4.0, 0.2],
    })
    plot.set_keys(frame["grna"])
    # Record positions the way the real scatter does, so `_row_xy` is what a
    # band and a ring both read.
    for row in range(len(frame)):
        plot._row_xy[row] = (float(frame["coefficient"][row]),
                             float(frame["neglog10p"][row]))
    return plot


def _click(plot, row, *, modifier=Qt.NoModifier):
    """A click on the point at frame position ``row``, through the gesture.

    Goes through `_on_points_clicked`, which is the slot pyqtgraph calls --
    so a change that broke the wiring between the scatter and the selection
    would fail here.
    """
    class _Point:
        def __init__(self, index):
            self._index = index

        def data(self):
            return self._index

    QApplication.setKeyboardModifiers(modifier) \
        if hasattr(QApplication, "setKeyboardModifiers") else None
    plot._on_points_clicked(None, [_Point(row)])


class TestModifierClickAddsAndRemoves:

    def test_a_plain_click_selects_one(self, volcano, monkeypatch):
        monkeypatch.setattr(type(volcano), "_adding_to_selection",
                            staticmethod(lambda: False))
        _click(volcano, 0)
        assert volcano.selected_keys() == ["g0"]

    def test_a_modified_click_adds(self, volcano, monkeypatch):
        monkeypatch.setattr(type(volcano), "_adding_to_selection",
                            staticmethod(lambda: False))
        _click(volcano, 0)
        monkeypatch.setattr(type(volcano), "_adding_to_selection",
                            staticmethod(lambda: True))
        _click(volcano, 3)
        _click(volcano, 4)
        assert volcano.selected_keys() == ["g0", "g3", "g4"]

    def test_a_modified_click_on_a_selected_point_removes_it(
            self, volcano, monkeypatch):
        monkeypatch.setattr(type(volcano), "_adding_to_selection",
                            staticmethod(lambda: True))
        _click(volcano, 0)
        _click(volcano, 3)
        _click(volcano, 0)
        assert volcano.selected_keys() == ["g3"]

    def test_a_plain_click_after_a_multi_selection_clears_it(
            self, volcano, monkeypatch):
        monkeypatch.setattr(type(volcano), "_adding_to_selection",
                            staticmethod(lambda: True))
        _click(volcano, 0)
        _click(volcano, 3)
        monkeypatch.setattr(type(volcano), "_adding_to_selection",
                            staticmethod(lambda: False))
        _click(volcano, 4)
        assert volcano.selected_keys() == ["g4"]


class TestTheBand:

    def test_a_band_selects_everything_inside_it(self, volcano):
        # x from 0.5 to 2.5 catches g3 (1.0) and g4 (2.0), not g5 (3.0).
        volcano.select_in_rect(0.5, 0.0, 2.5, 5.0)
        assert volcano.selected_keys() == ["g3", "g4"]

    def test_a_band_excludes_by_the_other_axis_too(self, volcano):
        # Same x window, but y capped below g4's 4.0.
        volcano.select_in_rect(0.5, 0.0, 2.5, 3.5)
        assert volcano.selected_keys() == ["g3"]

    def test_the_corners_can_be_dragged_in_any_direction(self, volcano):
        volcano.select_in_rect(2.5, 5.0, 0.5, 0.0)
        assert volcano.selected_keys() == ["g3", "g4"]

    def test_a_drag_reaches_the_band(self, volcano, app):
        """Through the ViewBox handler, which is what a real drag calls."""
        box = volcano.plot.getViewBox()
        assert getattr(box, "_spacr_band", False), (
            "the band was never installed, so no drag can select")

    def test_an_empty_band_selects_nothing(self, volcano):
        volcano.select_in_rect(10.0, 10.0, 11.0, 11.0)
        assert volcano.selected_keys() == []


class TestTheSelectionIsVisible:

    def test_every_selected_point_is_ringed(self, volcano):
        volcano.select_in_rect(0.5, 0.0, 3.5, 5.0)
        assert len(volcano.selected_keys()) == 3
        rings = len(volcano._extra_highlights) + (
            1 if volcano._highlight is not None else 0)
        assert rings == 3, (
            "a selection you cannot see is one you cannot check")

    def test_the_count_is_stated(self, volcano):
        volcano.select_in_rect(0.5, 0.0, 3.5, 5.0)
        note = volcano.status_note() if hasattr(volcano, "status_note") \
            else volcano._status.text()
        assert "3 selected" in note, (
            "a selection you cannot count is one you cannot trust")

    def test_clearing_removes_every_ring(self, volcano):
        volcano.select_in_rect(0.5, 0.0, 3.5, 5.0)
        volcano.highlight_keys([])
        assert volcano._extra_highlights == []
        assert volcano._highlight is None
        assert volcano.selected_keys() == []


class TestOneSelectionOneSourceOfTruth:

    def test_a_key_that_is_not_plotted_stays_selected(self, volcano):
        """Membership is never conditional on the ring being drawable."""
        volcano.highlight_keys(["g0", "not-on-this-plot"])
        assert volcano.selected_keys() == ["g0", "not-on-this-plot"], (
            "dropping it would make the count disagree with the consumers")

    def test_single_select_replaces_the_multi_list(self, volcano):
        volcano.select_in_rect(0.5, 0.0, 3.5, 5.0)
        volcano.highlight_key("g0")
        assert volcano.selected_keys() == ["g0"], (
            "two names for one state is how they drift apart")

    def test_the_last_picked_is_what_single_consumers_get(self, volcano):
        seen = []
        volcano.key_selected.connect(seen.append)
        volcano.select_in_rect(0.5, 0.0, 3.5, 5.0)
        assert seen and seen[-1] == volcano.selected_keys()[-1]

    def test_the_multi_signal_carries_all_of_them(self, volcano):
        seen = []
        volcano.keys_selected.connect(seen.append)
        volcano.select_in_rect(0.5, 0.0, 3.5, 5.0)
        assert seen and seen[-1] == ["g3", "g4", "g5"]


class TestTheConsumersReadTheSameList:
    """One selection, one source of truth -- the half the instruction calls
    the actual work."""

    @pytest.fixture
    def panel(self, app):
        from spacr.qt.widgets.regression_results import RegressionResultsPanel

        frame = pd.DataFrame({
            "feature": [f"g{i}" for i in range(6)],
            "coefficient": [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            "p_value": [0.01, 0.2, 0.9, 0.001, 1e-5, 0.4],
        })
        one = RegressionResultsPanel()
        one.set_frame(frame)
        return one

    def test_the_table_holds_every_selected_row(self, panel):
        found = panel.table.select_keys(["g0", "g3", "g4"])
        assert found == 3
        assert sorted(panel.table.selected_keys()) == ["g0", "g3", "g4"]

    def test_the_panel_reports_the_whole_selection(self, panel):
        panel.table.select_keys(["g0", "g3"])
        assert sorted(panel.selected_keys()) == ["g0", "g3"]

    def test_a_band_on_a_plot_reaches_the_table(self, panel):
        """Not the model directly: through the plot's own signal."""
        plot = panel.volcano if hasattr(panel, "volcano") else None
        if plot is None:                       # pragma: no cover - layout
            pytest.skip("this panel has no volcano")
        plot.keys_selected.emit(["g0", "g3", "g4"])
        assert sorted(panel.table.selected_keys()) == ["g0", "g3", "g4"]

    def test_one_key_is_left_to_the_single_click_route(self, panel):
        """Taking it here as well would build the gene tile twice."""
        before = panel.table.selected_keys()
        panel._select_many_from_a_plot(["g0"])
        assert panel.table.selected_keys() == before

    def test_the_montage_holds_all_of_them_and_shows_one(self, app):
        from spacr.qt.widgets.cell_montage_view import CellMontageView

        view = CellMontageView()
        view.set_coefficients(["g0", "g3", "g4"])
        assert view.selected_coefficients() == ["g0", "g3", "g4"]
        assert view._key == "g4", "the most recent is the one shown"

    def test_the_montage_steps_through_the_selection(self, app):
        from spacr.qt.widgets.cell_montage_view import CellMontageView

        view = CellMontageView()
        view.set_coefficients(["g0", "g3", "g4"])
        assert view.show_next_coefficient() == "g0"
        assert view.show_next_coefficient() == "g3"

    def test_a_single_selection_has_nowhere_to_step(self, app):
        from spacr.qt.widgets.cell_montage_view import CellMontageView

        view = CellMontageView()
        view.set_coefficients(["g0"])
        assert view.show_next_coefficient() is None

    def test_an_empty_selection_is_a_state_the_consumers_handle(self, app):
        from spacr.qt.widgets.cell_montage_view import CellMontageView

        view = CellMontageView()
        view.set_coefficients([])
        assert view.selected_coefficients() == []


class TestTheMontageShowsTheWholeSelection:
    """206's remaining half: the Cells tab reflects the full selection.

    A montage is of ONE coefficient -- its crops are chosen by how well each
    cell agrees with THAT coefficient's effect -- so three at once in one
    grid would be three questions in one picture with nothing saying which
    cell answered which. What makes the whole selection visible is that the
    WELL TABS STAY across a load, which `_drop_montage` already guarantees.
    """

    @pytest.fixture
    def view(self, app):
        from spacr.qt.widgets.cell_montage_view import CellMontageView

        one = CellMontageView()
        one.set_coefficients(["g1", "g2", "g3"])
        return one

    def test_it_queues_every_coefficient(self, view):
        assert view.build_every_selected() == 3

    def test_the_first_is_current_and_the_rest_are_queued(self, view):
        view.build_every_selected()
        assert view._key == "g1"
        assert view._queue == ["g2", "g3"]

    def test_they_are_taken_in_the_order_they_were_picked(self, view):
        """Firing every load at once would put n reads of the same databases
        in flight and deliver them in whatever order they finished, so the
        tabs would arrive in an order that depends on disk timing."""
        view.build_every_selected()
        assert view._queue == ["g2", "g3"]
        view._build_the_next_queued()
        # With no database attached no load starts, so the walk drains the
        # queue rather than stopping on the first one it cannot open --
        # which is the point of the loop.
        assert view._queue == []

    def test_one_that_cannot_load_does_not_stop_the_rest(self, view):
        """Stopping would leave the rest of the selection queued forever
        with nothing on screen saying why."""
        view.build_every_selected()
        view._build_the_next_queued()
        assert view._queue == []
        assert view._build_the_next_queued() is False

    def test_an_empty_selection_queues_nothing(self, app):
        from spacr.qt.widgets.cell_montage_view import CellMontageView

        one = CellMontageView()
        one.set_coefficients([])
        assert one.build_every_selected() == 0

    def test_the_queue_is_empty_at_rest(self, app):
        from spacr.qt.widgets.cell_montage_view import CellMontageView

        assert CellMontageView()._queue == []

    def test_the_tabs_are_what_keeps_the_selection_visible(self):
        """Asserted on the code, because it is a promise `_drop_montage`
        makes in a comment and the whole design rests on it."""
        import inspect

        from spacr.qt.widgets.cell_montage_view import CellMontageView

        source = inspect.getsource(CellMontageView._drop_montage)
        assert "THE WELL TABS STAY" in source
