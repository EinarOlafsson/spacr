"""A plot opens on the view "Reset view" gives, not on the last run's window.

Reported on 2026-08-19: "the volcano plot starts with a wide x scal and i
allways have to press reset view, it should start the way it is with reset
vieew!"

The cause was one line's ORDER. `pyqtgraph`'s `ViewBox.autoRange()` ends in
`setRange(..., disableAutoRange=True)`, so calling it turns auto-ranging OFF.
`auto_range_axes` -- whose whole job is to give the axes back to the data,
and which the results panel calls BETWEEN runs, before the new table is
drawn -- called it last and so left the axes frozen on the previous run's
points. Every subsequent run opened inside the wrong window, and pressing
"Reset view" (autorange over the points that are there NOW) put it right.
"""
import numpy as np
import pandas as pd
import pytest


def _frame(scale, n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "coefficient": rng.normal(0, scale, n),
        "p_value": rng.uniform(0, 1, n),
        "feature": [f"g{i}" for i in range(n)],
    })


@pytest.fixture()
def volcano(qtbot):
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    widget = VolcanoPlot()
    qtbot.addWidget(widget)
    widget.resize(1000, 700)
    widget.show()
    return widget


def _x(widget):
    return tuple(round(float(v), 6) for v in widget.plot.viewRange()[0])


def test_auto_range_axes_leaves_auto_ranging_on(volcano, qtbot):
    """THE BUG, at its smallest. `autoRange()` switches it off; order fixes it."""
    volcano.set_results(_frame(4.0))
    volcano.auto_range_axes()
    assert all(volcano.plot.getViewBox().autoRangeEnabled())


def test_a_new_run_does_not_open_in_the_last_run_s_window(volcano, qtbot):
    """Measured +-13 then +-0.6: twenty times too wide, and nothing said so."""
    volcano.set_results(_frame(4.0))
    qtbot.wait(10)
    wide = _x(volcano)

    volcano.auto_range_axes()          # what the panel does between runs
    volcano.set_results(_frame(0.2, seed=1))
    qtbot.wait(10)
    opening = _x(volcano)

    assert opening != wide, "the new run opened in the old run's window"
    volcano.plot.autoRange()
    qtbot.wait(10)
    assert opening == pytest.approx(_x(volcano)), (
        "the opening view is not the view Reset view gives")


def test_reset_view_does_not_freeze_the_next_draw(volcano, qtbot):
    """The same trap on the BUTTON: after it, the next redraw was stale.

    Both reset actions call `auto_range_axes` for this reason.
    """
    volcano.set_results(_frame(4.0))
    volcano.auto_range_axes()
    qtbot.wait(10)

    volcano.set_results(_frame(0.2, seed=2))
    qtbot.wait(10)
    opening = _x(volcano)
    volcano.plot.autoRange()
    qtbot.wait(10)
    assert opening == pytest.approx(_x(volcano))


def test_a_typed_limit_still_wins(volcano, qtbot):
    """Auto-ranging left ON must not overrule a limit the user TYPED."""
    volcano.set_results(_frame(1.0))
    volcano.set_axis_limits(x=(-2.0, 2.0))
    qtbot.wait(10)
    assert _x(volcano) == pytest.approx((-2.0, 2.0), abs=1e-6)

    volcano.set_results(_frame(1.0, seed=3))
    qtbot.wait(10)
    assert _x(volcano) == pytest.approx((-2.0, 2.0), abs=1e-6), (
        "a redraw threw away the limits the user typed")


def test_auto_range_axes_is_the_way_out_of_a_typed_limit(volcano, qtbot):
    """It clears the pin -- that is its documented job -- and follows again."""
    volcano.set_results(_frame(1.0))
    volcano.set_axis_limits(x=(-2.0, 2.0))
    volcano.auto_range_axes()
    qtbot.wait(10)
    assert _x(volcano) != pytest.approx((-2.0, 2.0), abs=1e-6)
    assert all(volcano.plot.getViewBox().autoRangeEnabled())


def test_both_reset_actions_go_through_auto_range_axes():
    """Pinned on the source: a bare `plot.autoRange` re-introduces the bug."""
    import pathlib
    text = pathlib.Path("spacr/qt/widgets/fast_plots.py").read_text()
    assert "self.plot.autoRange)" not in text, (
        "a Reset view action still calls plot.autoRange directly")
    assert "lambda: self.plot.autoRange()" not in text
