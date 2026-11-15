"""Figure settings: reading a chart back out of its artists, and the guards.

``derive_replot_recipe`` exists because most spaCR figures arrive with no
record of the data behind them, so everything here is asserted against a real
matplotlib figure -- bars, points and lines that were actually drawn.
"""
from __future__ import annotations


import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from matplotlib.figure import Figure  # noqa: E402

from spacr.qt.widgets import figure_settings as fs  # noqa: E402

pytestmark = pytest.mark.qt


def _bar_figure():
    figure = Figure()
    axes = figure.add_subplot(111)
    axes.bar(["a", "b", "c"], [3.0, 5.0, 4.0])
    return figure, axes


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

def test_every_shape_matplotlib_stores_a_colour_in_becomes_a_hex():
    assert fs._as_hex("red") == "#ff0000"
    assert fs._as_hex((1.0, 0.0, 0.0, 1.0)) == "#ff0000"
    assert fs._as_hex(np.array([[1.0, 0.0, 0.0, 1.0]])) == "#ff0000"
    assert fs._as_hex([(1.0, 0.0, 0.0, 1.0)]) == "#ff0000"
    assert fs._as_hex([[0.0, 0.0, 1.0, 1.0]]) == "#0000ff"


def test_a_colour_nothing_can_read_falls_back():
    assert fs._as_hex(object(), fallback="#123456") == "#123456"


# ---------------------------------------------------------------------------
# Tick labels
# ---------------------------------------------------------------------------

def test_the_tick_labels_are_read_off_the_axes():
    figure, axes = _bar_figure()
    figure.canvas.draw()
    labels = fs._tick_labels(axes)
    assert set(labels.values()) == {"a", "b", "c"}


def test_an_axes_that_cannot_be_asked_has_no_tick_labels():
    class Awkward:
        def get_xticks(self):
            raise RuntimeError("no renderer")

    assert fs._tick_labels(Awkward()) == {}


def test_a_position_with_no_tick_label_is_named_by_its_number():
    labels = {0.0: "a", 1.0: "b"}
    assert fs._named(labels, 0.0) == "a"
    assert fs._named(labels, 1.2) == "b"      # within half a tick
    assert fs._named(labels, 7.0) == "7"      # nothing near it
    assert fs._named({}, 2.5) == "2.5"


# ---------------------------------------------------------------------------
# Naming a figure
# ---------------------------------------------------------------------------

def test_a_figure_is_named_by_its_suptitle_then_by_its_axes():
    figure = Figure()
    axes = figure.add_subplot(111)
    assert fs._figure_title(figure) == ""
    axes.set_title("per-plate rate")
    assert fs._figure_title(figure) == "per-plate rate"
    figure.suptitle("the whole screen")
    assert fs._figure_title(figure) == "the whole screen"


def test_something_that_is_not_a_figure_has_no_title():
    assert fs._figure_title(object()) == ""


# ---------------------------------------------------------------------------
# Style vocabulary
# ---------------------------------------------------------------------------

def test_a_setting_is_labelled_by_its_own_name():
    assert fs.style_setting_label("grid_colour") == "Grid colour"
    assert fs.style_setting_label("aspect") == "Lock axis scales"


def test_a_colour_and_a_transparent_ground_are_told_apart():
    assert fs._looks_like_a_colour("#ff0000")
    assert not fs._looks_like_a_colour("red")
    assert not fs._looks_like_a_colour(3)
    assert fs._is_transparent_ground("none")
    assert fs._is_transparent_ground(" TRANSPARENT ")
    assert fs._is_transparent_ground("")
    assert not fs._is_transparent_ground("#000000")


def test_a_free_form_setting_offers_no_choices():
    assert fs.style_choices_for("not_a_setting") == ()
    assert isinstance(fs.style_choices_for("palette"), tuple)


def test_a_default_is_told_from_an_edit_within_a_spinboxs_precision():
    assert fs._same_setting(True, True)
    assert not fs._same_setting(False, True)
    assert fs._same_setting(0.6000000000000001, 0.6)
    assert not fs._same_setting(0.7, 0.6)
    assert fs._same_setting("deep", "deep")
    assert not fs._same_setting("muted", "deep")


# ---------------------------------------------------------------------------
# The preferences panel
# ---------------------------------------------------------------------------

def test_reset_puts_every_control_back_to_the_package_default(qtbot):
    panel = fs.FigureStylePreferences()
    qtbot.addWidget(panel)
    before = panel.values()
    for name, (getter, setter, default) in panel._general_controls.items():
        if isinstance(default, bool):
            setter(not default)
        elif isinstance(default, (int, float)):
            setter(default + 1)
    changed, _per_graph = panel.values()
    assert changed
    panel.reset()
    assert panel.values() == before
