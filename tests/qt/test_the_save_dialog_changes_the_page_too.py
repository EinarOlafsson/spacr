"""Instruction 186 D and E, from a live session on the Cells tab.

D.  "in save styled the i cannot change the graphs background color, only the
     lines" -- and "size and resolution are grayed out".
E.  "Colour by lets me color by a single location, all should be an option."
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pg = pytest.importorskip("pyqtgraph")

import numpy as np
import pandas as pd

from spacr.localisation import ALL as ALL_COMPARTMENTS


class TestTheChosenBackgroundReachesThePage:
    """Two things set the page colour and the static one was winning.

    `SaveFigureDialog.save` passed its background to `export_styled`, which
    restyled the live scene with it -- and then `_write_export` overwrote the
    exporter's page with `_export_ground()`, a STATIC method reading the
    global saved-figure look that knew nothing about the dialog. So the ink
    changed and the paper never did.
    """

    @pytest.fixture
    def plot(self, qtbot):
        from spacr.qt.widgets.fast_plots import FastPlot

        widget = FastPlot()
        qtbot.addWidget(widget)
        return widget

    def test_a_chosen_ground_wins_over_the_global_look(self, plot):
        with plot._dressed_for_the_file(background="#FFFFFF"):
            ground = plot._export_ground()

        assert ground.name().lower() == "#ffffff", (
            "the colour the user picked has to be the colour of the page")

    def test_black_is_honoured_too(self, plot):
        with plot._dressed_for_the_file(background="#000000"):
            assert plot._export_ground().name().lower() == "#000000"

    def test_transparent_is_not_a_colour_and_does_not_become_one(self, plot):
        """The dialog's "transparent" is the empty string, which must leave
        the look in charge exactly as it was before this change."""
        outside = plot._export_ground()
        with plot._dressed_for_the_file(background=""):
            inside = plot._export_ground()

        assert inside.rgba() == outside.rgba()

    def test_the_choice_is_put_back_when_the_render_ends(self, plot):
        before = plot._export_ground().rgba()
        with plot._dressed_for_the_file(background="#FFFFFF"):
            pass

        assert plot._export_ground().rgba() == before, (
            "a save must not leave the next one writing onto its page")

    def test_it_is_restored_even_when_the_export_raises(self, plot):
        before = plot._export_ground().rgba()
        with pytest.raises(RuntimeError):
            with plot._dressed_for_the_file(background="#FFFFFF"):
                raise RuntimeError("the export failed")

        assert plot._export_ground().rgba() == before


class TestTheGreyedControlsSayWhy:
    """106 in a dialog: the reason has to be where it is read.

    Both reasons were already written -- to `setToolTip` on the DISABLED
    widget, which is the exact failure 106 exists for.
    """

    def test_the_size_reason_is_in_the_label_not_only_the_tooltip(self):
        from spacr.qt.widgets.save_figure_dialog import (_SIZE_REASON,
                                                         _reason_label)

        label = _reason_label("size", _SIZE_REASON)

        assert "size" in label.text()
        assert "right-click menu" in label.text(), (
            "a user must not have to hover a greyed box to learn why")
        assert label.toolTip() == _SIZE_REASON

    def test_the_resolution_reason_says_vector_has_none(self):
        from spacr.qt.widgets.save_figure_dialog import (_RESOLUTION_REASON,
                                                         _reason_label)

        label = _reason_label("resolution", _RESOLUTION_REASON)

        assert "vector" in label.text()

    def test_a_live_control_gets_a_plain_label(self):
        from spacr.qt.widgets.save_figure_dialog import _reason_label

        label = _reason_label("size")

        assert label.text() == "size" and label.isEnabled()


class TestColourByEveryLocalisation:

    def test_the_sentinel_cannot_collide_with_a_real_compartment(self):
        from spacr import localisation

        assert ALL_COMPARTMENTS not in localisation.table().values()
        assert not ALL_COMPARTMENTS.isprintable() or "\x00" in ALL_COMPARTMENTS

    def test_every_compartment_gets_its_own_colour(self, qtbot):
        from spacr.qt.widgets.fast_plots import FastPlot

        widget = FastPlot()
        qtbot.addWidget(widget)
        brushes, legend = widget._categorical_brushes(
            pd.Series(["nucleus", "nucleus", "apicoplast", "elsewhere"]))

        assert len(brushes) == 4
        assert len(legend) == 3, "one entry per distinct value"
        assert any("nucleus (2)" in name for name in legend), (
            "the count belongs beside the label")

    def test_the_counts_come_from_the_codes(self, qtbot):
        """Not a value_counts on the frame -- that is the 45 ms this path
        exists to avoid."""
        from spacr.qt.widgets.fast_plots import FastPlot

        widget = FastPlot()
        qtbot.addWidget(widget)
        _brushes, legend = widget._categorical_brushes(
            pd.Series(["a"] * 100 + ["b"] * 5))

        assert any("(100)" in name for name in legend)
        assert any("(5)" in name for name in legend)
