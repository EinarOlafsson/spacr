"""Right-clicking an axis lays that measurement out and cuts it down.

The gesture people reach for on a cytometry scatter is a right-click on the
AXIS, not a trip to a settings window: an axis is where the ticks and the
label already are, so it is where a question about that measurement gets
asked. Two things are behind it -- the scale the axis is drawn on, and how
much of the measurement is worth showing.

The invariant these tests exist to hold is that neither one moves a single
object: a cutoff and a transform change the VIEW, and a gate keeps exactly
the objects it already held. A cutoff that filtered would make a population
depend on how far the plot happened to be cut down, which is the one thing a
gate must never do.
"""

import numpy as np
import pandas as pd
import pytest

from PySide6.QtCore import QPoint

from spacr.qt.screens.gate_editor import GateEditorScreen
from spacr.qt.widgets.gate_canvas import (
    AxisCutoff, AxisCutoffs, CutoffError, apply_cutoffs, axis_at,
    axis_menu_items, parse_cutoff,
)
from spacr.qt.widgets.gate_spec import GateSet, RectGate

BOX = (50.0, 40.0, 500.0, 400.0)


class TestWhichAxisWasClicked:
    """The strip below the plot is the x axis; the strip left of it is y."""

    def test_inside_the_plot_is_neither(self):
        assert axis_at((300.0, 200.0), BOX) is None

    def test_below_the_plot_is_the_x_axis(self):
        assert axis_at((300.0, 15.0), BOX) == "x"

    def test_left_of_the_plot_is_the_y_axis(self):
        assert axis_at((20.0, 200.0), BOX) == "y"

    def test_above_and_right_belong_to_neither(self):
        """The title strip and the right margin carry no axis, so a click
        there gets the plot's own menu rather than an axis nobody aimed at."""
        assert axis_at((300.0, 450.0), BOX) is None
        assert axis_at((600.0, 200.0), BOX) is None
        assert axis_at((600.0, 15.0), BOX) is None

    def test_the_corner_goes_to_whichever_it_is_further_into(self):
        """Well to the left and barely below is the y axis, and the other
        way round is the x axis -- otherwise the corner would always answer
        the same way and one of the two axes would be unreachable there."""
        assert axis_at((5.0, 35.0), BOX) == "y"
        assert axis_at((45.0, 5.0), BOX) == "x"


class TestACutoffIsAPair:

    def test_an_empty_cutoff_is_not_set(self):
        assert not AxisCutoff().is_set

    def test_a_reversed_pair_is_refused_with_the_reason(self):
        """Ends that meet or cross give an axis with no extent, which draws
        as a blank panel -- indistinguishable from the plot having broken."""
        with pytest.raises(CutoffError) as caught:
            AxisCutoff(200.0, 100.0)
        assert "below" in str(caught.value)
        with pytest.raises(CutoffError):
            AxisCutoff(100.0, 100.0)

    def test_one_open_end_takes_the_data_s(self):
        assert AxisCutoff(low=10.0).limits(0.0, 90.0) == (10.0, 90.0)
        assert AxisCutoff(high=10.0).limits(0.0, 90.0) == (0.0, 10.0)

    def test_a_blank_box_means_let_the_data_decide(self):
        assert parse_cutoff("") is None
        assert parse_cutoff("  ") is None
        assert parse_cutoff(" 12.5 ") == 12.5

    def test_text_that_is_not_a_number_says_so(self):
        """Falling back to "the data decides" would look exactly like the
        cutoff having applied and done nothing."""
        with pytest.raises(CutoffError) as caught:
            parse_cutoff("ten")
        assert "ten" in str(caught.value)


class TestCutoffsAreKeptPerMeasurement:

    def test_clearing_both_ends_forgets_the_column(self):
        cutoffs = AxisCutoffs()
        cutoffs.set("area", 1.0, 2.0)
        assert "area" in cutoffs
        cutoffs.set("area", None, None)
        assert "area" not in cutoffs and len(cutoffs) == 0

    def test_an_unset_column_answers_with_an_empty_cutoff(self):
        assert not AxisCutoffs().get("area").is_set
        assert not AxisCutoffs().get(None).is_set

    def test_the_cutoff_follows_the_measurement_to_the_other_axis(self):
        """Keyed by column rather than by slot: swapping the axes must not
        apply the intensity cutoff to area."""
        cutoffs = AxisCutoffs()
        cutoffs.set("area", 10.0, 20.0)
        assert cutoffs.get("area").limits(0.0, 99.0) == (10.0, 20.0)
        assert cutoffs.get("intensity").limits(0.0, 99.0) == (0.0, 99.0)


class TestTheMenuOffered:

    def test_it_names_the_axis_and_the_measurement(self):
        labels = [item.label for item in axis_menu_items("y", "area")]
        assert labels[0] == "Y axis: area"

    def test_the_scale_in_force_is_ticked_and_only_that_one(self):
        ticked = [item.label for item in axis_menu_items("x", "area",
                                                         scale="symlog")
                  if item.checked]
        assert ticked == ["symlog"]

    def test_a_measurement_reaching_zero_greys_log_and_says_why(self):
        """A log axis over data that reaches zero draws nothing at all, so
        the row is greyed with the reason rather than accepting a click that
        cannot take effect."""
        rows = {item.label: item for item in
                axis_menu_items("x", "area", positive=False)}
        assert not rows["log"].enabled
        assert "zero" in rows["log"].why
        assert rows["symlog"].enabled

    def test_clearing_is_off_until_something_is_cut_and_then_says_what(self):
        rows = {item.label: item for item in axis_menu_items("x", "area")}
        assert not rows["Clear cutoffs"].enabled
        assert "No cutoffs" in rows["Clear cutoffs"].why
        rows = [item for item in
                axis_menu_items("x", "area", cutoff=AxisCutoff(10.0, 20.0))
                if item.label and item.label.startswith("Clear")]
        assert rows[0].enabled and "10 – 20" in rows[0].label

    def test_an_empty_axis_offers_one_row_that_says_so(self):
        rows = axis_menu_items("x", None)
        assert len(rows) == 1 and not rows[0].enabled
        assert "measurement" in rows[0].why


def test_apply_cutoffs_narrows_only_the_axis_that_carries_one():
    """The unpinned axis keeps the limits the data produced."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    axes = Figure().add_subplot(111)
    axes.set_xlim(0.0, 100.0)
    axes.set_ylim(0.0, 100.0)
    cutoffs = AxisCutoffs()
    cutoffs.set("area", 10.0, 20.0)
    assert apply_cutoffs(axes, ("area", "intensity"), cutoffs) == ("x",)
    assert axes.get_xlim() == (10.0, 20.0)
    assert axes.get_ylim() == (0.0, 100.0)


# ---------------------------------------------------------------------------
# The gesture on the running screen
# ---------------------------------------------------------------------------

@pytest.fixture
def screen(qt_theme_applied, qtbot):
    widget = GateEditorScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.resize(900, 620)
    widget.show()
    frame = pd.DataFrame({
        "area": np.linspace(1.0, 500.0, 300),
        "intensity": np.linspace(0.5, 50.0, 300),
    })
    widget.set_frame(frame, label="fixture")
    widget._x.setCurrentText("area")
    widget._y.setCurrentText("intensity")
    qt_theme_applied.processEvents()
    widget._frame_for_test = frame
    return widget


def _panel(screen):
    return screen.gates.canvas.figure().get_axes()[0]


def _widget_point(screen, display_x, display_y):
    """A figure display point as the canvas widget's own coordinates."""
    figure = screen.gates.canvas.figure()
    surface = figure.canvas
    ratio = float(getattr(surface, "device_pixel_ratio", 1) or 1)
    return surface.mapTo(screen.gates.canvas,
                         QPoint(int(display_x / ratio),
                                int((figure.bbox.height - display_y) / ratio)))


class TestTheGestureReachesTheAxis:

    def test_a_click_below_the_plot_is_read_as_the_x_axis(self, screen):
        box = _panel(screen).bbox
        point = _widget_point(screen, (box.x0 + box.x1) / 2, box.y0 - 10)
        assert screen.axis_under(point) == "x"

    def test_a_click_left_of_the_plot_is_read_as_the_y_axis(self, screen):
        box = _panel(screen).bbox
        point = _widget_point(screen, max(1.0, box.x0 - 10),
                              (box.y0 + box.y1) / 2)
        assert screen.axis_under(point) == "y"

    def test_a_click_on_the_plot_still_gets_the_plot_menu(self, screen):
        box = _panel(screen).bbox
        point = _widget_point(screen, (box.x0 + box.x1) / 2,
                              (box.y0 + box.y1) / 2)
        assert screen.axis_under(point) is None

    def test_the_menu_describes_the_measurement_actually_on_that_axis(
            self, screen):
        assert screen.axis_menu_items("x")[0].label == "X axis: area"
        assert screen.axis_menu_items("y")[0].label == "Y axis: intensity"


class TestTheScaleIsOneSettingWithTwoRoutes:

    def test_choosing_a_scale_from_the_menu_writes_the_setting(self, screen,
                                                               qt_theme_applied):
        rows = {item.label: item for item in screen.axis_menu_items("x")}
        rows["symlog"].callback()
        qt_theme_applied.processEvents()
        assert screen.settings().x_scale == "symlog"
        assert screen.axis_menu_items("x")[2].label == "linear"
        assert {item.label for item in screen.axis_menu_items("x")
                if item.checked} == {"symlog"}

    def test_the_plot_is_actually_laid_out_on_the_chosen_scale(
            self, screen, qt_theme_applied):
        rows = {item.label: item for item in screen.axis_menu_items("x")}
        rows["log"].callback()
        qt_theme_applied.processEvents()
        assert _panel(screen).get_xscale() == "log"

    def test_a_retired_log_flag_cannot_override_the_menu(self, screen,
                                                         qt_theme_applied):
        """`log_x` is the older spelling of the same choice and wins while
        the scale is linear, so choosing linear has to retire it too or the
        axis snaps straight back to log."""
        screen.apply_settings(screen.settings().replaced(log_x=True))
        qt_theme_applied.processEvents()
        rows = {item.label: item for item in screen.axis_menu_items("x")}
        rows["linear"].callback()
        qt_theme_applied.processEvents()
        assert screen.settings().scale_for("x") == "linear"
        assert _panel(screen).get_xscale() == "linear"


class TestCutoffsChangeTheViewAndNothingElse:

    def test_a_cutoff_narrows_the_drawn_axis(self, screen, qt_theme_applied):
        before = _panel(screen).get_xlim()
        screen.set_axis_cutoffs("x", 100.0, 200.0)
        qt_theme_applied.processEvents()
        assert before != (100.0, 200.0)
        assert _panel(screen).get_xlim() == (100.0, 200.0)

    def test_one_open_end_leaves_the_other_where_the_data_put_it(
            self, screen, qt_theme_applied):
        top = _panel(screen).get_xlim()[1]
        screen.set_axis_cutoffs("x", 100.0, None)
        qt_theme_applied.processEvents()
        assert _panel(screen).get_xlim() == (100.0, top)

    def test_the_gate_keeps_every_object_it_held(self, screen,
                                                 qt_theme_applied):
        """The whole reason a cutoff is a view: a population that changed
        with the cutoff would depend on how far the plot was cut down."""
        frame = screen._frame_for_test
        gates = GateSet()
        gates.add(RectGate(name="all", x_column="area", y_column="intensity",
                           x_low=0.0, x_high=600.0,
                           y_low=0.0, y_high=60.0))
        screen.gates.set_gates(gates)
        qt_theme_applied.processEvents()
        before = gates.stats(frame)[0].n_in
        screen.set_axis_cutoffs("x", 100.0, 200.0)
        qt_theme_applied.processEvents()
        assert gates.stats(frame)[0].n_in == before == len(frame)

    def test_the_cutoff_survives_the_next_render(self, screen,
                                                 qt_theme_applied):
        """A render recomputes the limits from the data and happens on every
        gate edit, so a cutoff that were applied once would be gone by the
        next click."""
        screen.set_axis_cutoffs("x", 100.0, 200.0)
        qt_theme_applied.processEvents()
        screen.gates.canvas.render_now()
        qt_theme_applied.processEvents()
        assert _panel(screen).get_xlim() == (100.0, 200.0)

    def test_clearing_gives_the_axis_back_to_the_data(self, screen,
                                                      qt_theme_applied):
        before = _panel(screen).get_xlim()
        screen.set_axis_cutoffs("x", 100.0, 200.0)
        qt_theme_applied.processEvents()
        assert screen.clear_axis_cutoffs("x")
        qt_theme_applied.processEvents()
        assert _panel(screen).get_xlim() == before
        assert not screen.clear_axis_cutoffs("x")

    def test_a_reversed_pair_is_refused_rather_than_drawn(self, screen):
        with pytest.raises(CutoffError):
            screen.set_axis_cutoffs("x", 200.0, 100.0)

    def test_the_cut_measurement_keeps_its_cutoff_when_it_moves_axes(
            self, screen, qt_theme_applied):
        screen.set_axis_cutoffs("x", 100.0, 200.0)
        qt_theme_applied.processEvents()
        screen._x.setCurrentText("intensity")
        screen._y.setCurrentText("area")
        qt_theme_applied.processEvents()
        assert _panel(screen).get_ylim() == (100.0, 200.0)
        assert screen.axis_menu_items("y")[-1].label.startswith(
            "Clear cutoffs (100 – 200)")


class TestTheSettingsWindowCannotShowAScaleNothingUses:

    def test_an_open_window_is_refreshed_when_the_menu_changes_a_scale(
            self, screen, qt_theme_applied):
        screen.open_settings()
        first = screen._settings_dialog
        assert first is not None
        rows = {item.label: item for item in screen.axis_menu_items("x")}
        rows["log"].callback()
        qt_theme_applied.processEvents()
        assert screen._settings_dialog is not None
        assert screen._settings_dialog.settings().x_scale == "log"

    def test_a_window_that_can_be_told_is_told_rather_than_rebuilt(
            self, screen, qt_theme_applied):
        """A control that can show a value chosen elsewhere keeps its tab and
        scroll position, which is the difference between adjusting a setting
        and hunting for it."""
        screen.open_settings()
        dialog = screen._settings_dialog
        seen = []
        dialog.set_scale = lambda axis, scale: seen.append((axis, scale))
        rows = {item.label: item for item in screen.axis_menu_items("y")}
        rows["symlog"].callback()
        qt_theme_applied.processEvents()
        assert seen == [("y", "symlog")]
        assert screen._settings_dialog is dialog
