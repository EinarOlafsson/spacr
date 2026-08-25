"""A grouped plot keeps its data, so it can be redrawn, exported and refused.

The widget holds the rows it drew rather than only the graphics, which is
what makes "show this another way" and "export the numbers behind it"
possible. The edges are all about not drawing something that would mislead:
an empty table draws nothing rather than an empty frame that looks like a
result, a redraw with no data at all is refused, and a line joins its points
in x order because a line through unsorted points is a scribble.
"""
from __future__ import annotations

import builtins

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets.grouped_plot import GroupedPlot, PlotSpec


def _frame():
    return pd.DataFrame({"well": ["a", "a", "b", "b"],
                         "area": [1.0, 2.0, 3.0, 4.0]})


# -- what a spec can group ---------------------------------------------------

def test_a_spec_whose_value_column_is_gone_groups_nothing():
    """A saved figure reopened against another table must draw nothing."""
    spec = PlotSpec(frame=_frame(), value="perimeter", group="well")
    assert spec.groups() == {}


def test_a_spec_with_no_frame_at_all_groups_nothing():
    """A spec built before its data arrived is not an error."""
    assert PlotSpec(frame=None, value="area").groups() == {}


def test_a_spec_with_no_group_column_is_one_series_named_by_its_value():
    """An ungrouped column is still a distribution worth drawing."""
    groups = PlotSpec(frame=_frame(), value="area").groups()
    assert list(groups) == ["area"]
    np.testing.assert_array_equal(groups["area"], [1.0, 2.0, 3.0, 4.0])


def test_a_group_column_the_frame_does_not_have_is_one_series():
    """A stale group column must not silently produce an empty picture."""
    groups = PlotSpec(frame=_frame(), value="area", group="plate").groups()
    assert list(groups) == ["area"]


# -- the widget --------------------------------------------------------------

@pytest.fixture
def a_plot(qtbot):
    """Build a :class:`GroupedPlot` that dies with the test that asked for it.

    Every plot below is a parentless top-level widget, and a top-level widget
    that is never closed cannot be freed at all -- pyqtgraph gives each one a
    context menu with ten submenus under it, all of them top-level too, and
    the connections holding them run through Qt's C++ side where Python's
    collector cannot follow. Built bare, the five tests here left 75 windows
    standing for the rest of the process; built through this, they leave
    none. ``qtbot`` closes and deletes what it is handed at teardown.
    """
    def build(*args, **kwargs):
        widget = GroupedPlot(*args, **kwargs)
        qtbot.addWidget(widget)
        return widget

    return build


def test_a_plot_holding_no_data_refuses_to_be_redrawn(a_plot):
    """"Show as a box" needs rows; drawing nothing would look like a result."""
    plot = a_plot()
    with pytest.raises(ValueError) as excinfo:
        plot.show_as("box")
    assert "no data" in str(excinfo.value)


def test_a_plot_holding_no_data_offers_no_comparison(a_plot):
    """A statistical comparison of nothing is not a comparison."""
    assert a_plot().comparison_groups() is None


def test_a_plot_holding_no_data_has_no_table_to_export(a_plot):
    """A plot drawn from bare arrays honestly has no source rows, and the
    column-mapping controls grey out on exactly that answer."""
    assert a_plot().frame() is None


def test_a_plot_holding_a_spec_exports_the_rows_it_drew(a_plot):
    """The exported data.csv has to be the rows behind the picture."""
    frame = _frame()
    plot = a_plot(PlotSpec(frame=frame, value="area", group="well"))
    assert plot.frame() is frame


def test_a_line_joins_its_points_in_x_order(a_plot):
    """A line through unsorted points draws a scribble, not a series."""
    frame = pd.DataFrame({"dose": [3.0, 1.0, 2.0],
                          "response": [30.0, 10.0, 20.0]})
    plot = a_plot()
    drawn = plot.show_spec(PlotSpec(frame=frame, value="response",
                                    group="dose", kind="line"))
    assert drawn == 1
    curves = [item for item in plot.plot.listDataItems()
              if item.xData is not None]
    assert curves, "no curve was drawn"
    xs = list(curves[0].xData)
    assert xs == sorted(xs), xs


def test_a_group_keeps_its_ink_when_the_house_style_cannot_be_loaded(
        qapp, monkeypatch):
    """Losing the style module must not leave the figure with no ink."""
    real_import = builtins.__import__

    def _blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if "figures.style" in name or (
                level and fromlist and "ROLES" in fromlist
                and name.endswith("style")):
            raise ImportError("style unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    first = GroupedPlot._ink_for("a", 0, "a", "")
    second = GroupedPlot._ink_for("b", 1, "a", "")
    assert first is not None and second is not None
    assert first != second
