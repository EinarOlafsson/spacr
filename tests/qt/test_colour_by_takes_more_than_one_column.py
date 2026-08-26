"""Colour by takes a selection of columns, not only one (instruction 222).

LAYERED, NOT COMBINED. Two columns of three levels COMBINED is a nine-entry
legend and three is twenty-seven, which instruction 122 measured at 40 ms of
every redraw -- and which nobody can decode anyway. Hue, then shape, then
opacity, always in that order, so the same choice always gives the same
picture.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def panel(qtbot):
    """A results panel that is destroyed with the test that asked for it.

    Handed to ``qtbot`` rather than merely built, because this panel is a
    parentless top-level widget carrying a pyqtgraph plot per tab, and every
    one of those plots builds its own context menu with ten submenus under
    it. Nothing but this fixture ever refers to the panel, and a top-level
    widget that is never closed cannot be freed at all: the running plot
    timers hold bound methods of the panel through connections Qt owns, so
    the cycle passes through C++ where Python's collector cannot see it.
    Left to itself one test of this file retains about 160 windows, and the
    file retained 1,760 of them -- for the rest of the process, slowing
    every later restyle in it, because a palette change visits every live
    widget. ``qtbot`` closes and deletes what it is given at teardown, which
    is what makes that number nought.
    """
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    one = RegressionResultsPanel()
    qtbot.addWidget(one)
    one.set_frame(pd.DataFrame({
        "feature": [f"g{i}" for i in range(9)],
        "coefficient": [-2, -1, 0, 1, 2, 3, 0.5, -0.5, 1.5],
        "p_value": [.01, .2, .9, .001, 1e-5, .4, .03, .6, .02],
        "condition": ["nc", "pc", "x"] * 3,
        "qc": ["pass", "fail", "pass"] * 3,
        "plate": ["p1", "p1", "p2"] * 3,
    }))
    return one


def _pick(combo, value):
    combo.setCurrentIndex(combo.findData(value))


class TestMoreThanOne:

    def test_one_column_is_still_one_channel(self, panel):
        assert panel.colour_channels() == ["condition"]

    def test_a_second_column_is_a_second_channel(self, panel):
        _pick(panel._colour_by_2, "qc")
        assert panel.colour_channels() == ["condition", "qc"]

    def test_a_third_is_a_third(self, panel):
        _pick(panel._colour_by_2, "qc")
        _pick(panel._colour_by_3, "plate")
        assert panel.colour_channels() == ["condition", "qc", "plate"]

    def test_all_three_still_draw_every_point(self, panel):
        _pick(panel._colour_by_2, "qc")
        _pick(panel._colour_by_3, "plate")
        assert len(panel.volcano._row_xy) == 9

    def test_the_control_is_reachable(self, panel):
        """It was hidden when it duplicated the volcano's own menu. It now
        offers what that menu cannot, and a feature nobody can reach is not
        a feature."""
        assert not panel._colour_by.isHidden()
        assert not panel._colour_by_2.isHidden()
        assert not panel._colour_by_3.isHidden()


class TestTheOrderIsFixed:
    """"the layering has a fixed order so the same choice always produces
    the same picture"."""

    def test_hue_is_always_the_first(self, panel):
        _pick(panel._colour_by_2, "qc")
        assert panel.colour_channels()[0] == "condition"

    def test_swapping_the_boxes_swaps_the_encodings(self, panel):
        _pick(panel._colour_by, "qc")
        _pick(panel._colour_by_2, "condition")
        assert panel.colour_channels() == ["qc", "condition"]


class TestTheChannelsRefuseWhatWouldMislead:

    def test_no_first_channel_means_no_others(self, panel):
        """A shape meaning one thing beside a colour meaning the q-value is
        two claims on one dot with nothing saying which."""
        _pick(panel._colour_by_2, "qc")
        _pick(panel._colour_by, None)
        assert panel.colour_channels() == []

    def test_the_same_column_twice_is_one_channel(self, panel):
        """Drawing the same fact twice tells the reader nothing new."""
        _pick(panel._colour_by_2, "condition")
        assert panel.colour_channels() == ["condition"]

    def test_the_third_cannot_repeat_the_second(self, panel):
        _pick(panel._colour_by_2, "qc")
        _pick(panel._colour_by_3, "qc")
        assert panel.colour_channels() == ["condition", "qc"]

    def test_there_is_no_fourth_box(self, panel):
        assert not hasattr(panel, "_colour_by_4"), (
            "past three encodings a point carries more than a reader can "
            "decode; refusing beats drawing something uninterpretable")


class TestTheEncoders:

    def test_shapes_are_distinguishable_and_few(self):
        from spacr.qt.widgets.fast_plots import FastPlot

        assert len(FastPlot.SHAPES) == 4
        assert len(set(FastPlot.SHAPES)) == 4

    def test_each_level_gets_a_shape(self):
        from spacr.qt.widgets.fast_plots import FastPlot

        symbols, shapes = FastPlot._categorical_symbols(
            pd.Series(["a", "b", "a", "c"]))
        assert symbols == [shapes["a"], shapes["b"], shapes["a"],
                           shapes["c"]]
        assert len(set(shapes.values())) == 3

    def test_shapes_repeat_visibly_past_the_palette(self):
        """Two levels sharing a shape is in the legend rather than a silent
        collision."""
        from spacr.qt.widgets.fast_plots import FastPlot

        _, shapes = FastPlot._categorical_symbols(
            pd.Series(list("abcdef")))
        assert shapes["a"] == shapes["e"]

    def test_opacity_never_deletes_a_point(self, app):
        """A fully transparent point is an absent point, and a channel that
        can delete a datapoint is not an encoding."""
        import pyqtgraph as pg

        from spacr.qt.widgets.fast_plots import FastPlot

        base = [pg.mkBrush("#ff0000")] * 4
        out = FastPlot._categorical_opacity(
            base, pd.Series(["a", "b", "c", "d"]), 4)
        alphas = [b.color().alpha() for b in out]
        assert min(alphas) > 0
        assert max(alphas) == 255


class TestTheLegendSaysWhichChannelCarriesWhich:
    """"The legend says which channel carries which column."

    The three encodings sit on one point, so a key naming only the LEVEL --
    "fail", "p2" -- leaves the reader to guess whether they are reading the
    colour, the shape or the fade.
    """

    def _drawn(self, panel):
        return panel.volcano._legend_entries()

    def test_the_shape_column_is_named_in_the_key(self, panel):
        _pick(panel._colour_by_2, "qc")
        labels = [label for label, _ in self._drawn(panel)]

        assert any(label.startswith("qc (shape): ") for label in labels)
        assert "qc (shape): fail" in labels
        assert "qc (shape): pass" in labels

    def test_the_shape_in_the_key_is_the_shape_on_the_point(self, panel):
        """A key drawn with a different symbol describes another picture."""
        from spacr.qt.widgets.fast_plots import FastPlot

        _pick(panel._colour_by_2, "qc")
        _, shapes = FastPlot._categorical_symbols(pd.Series(["fail", "pass"]))
        drawn = {label: style for label, style in self._drawn(panel)}

        assert drawn["qc (shape): fail"]["symbol"] == shapes["fail"]

    def test_the_opacity_column_is_named_too(self, panel):
        _pick(panel._colour_by_2, "qc")
        _pick(panel._colour_by_3, "plate")
        labels = [label for label, _ in self._drawn(panel)]

        assert "plate (opacity): p1" in labels
        assert "plate (opacity): p2" in labels

    def test_the_fade_in_the_key_is_the_fade_on_the_point(self, panel):
        """And never zero, for the same reason the encoder never fades to
        nothing."""
        _pick(panel._colour_by_2, "qc")
        _pick(panel._colour_by_3, "plate")
        drawn = {label: style for label, style in self._drawn(panel)}
        alphas = [drawn[f"plate (opacity): {level}"]["brush"].color().alpha()
                  for level in ("p1", "p2")]

        assert min(alphas) > 0
        assert max(alphas) == 255
        assert alphas[0] != alphas[1]

    def test_one_column_leaves_the_key_exactly_as_it_was(self, panel):
        """The common case must not change: hue only, and no channel
        suffixes on it."""
        labels = [label for label, _ in self._drawn(panel)]

        assert labels
        assert not any("(shape)" in label or "(opacity)" in label
                       for label in labels)

    def test_the_legend_can_be_switched_on_for_the_extra_channels(self, panel):
        """A key that cannot be reached is not a key. The checkbox counts
        every channel, not only the hue."""
        _pick(panel._colour_by_2, "qc")

        assert panel.volcano._legend_box.isEnabled()
        assert str(len(self._drawn(panel))) in panel.volcano._legend_box.text()

    def test_switching_it_on_puts_every_entry_on_the_plot(self, panel):
        _pick(panel._colour_by_2, "qc")
        panel.volcano._legend_box.setChecked(True)
        legend = panel.volcano.plot.plotItem.legend

        assert legend is not None
        drawn = {str(label.text) for _, label in legend.items}
        assert "qc (shape): fail" in drawn
