"""The five column headings say what their field controls (instruction 202).

"for the measurements table tab, there needs to be a tooltip when the mouse
is hovered over measurement, and level and plot and show and compare that
explains what each field related to them controlls."

EACH TOOLTIP EXPLAINS THE FIELD, not the heading. "Level" is a word the user
can already read; what they cannot read off the screen is that it decides
whether a datapoint is a cell, a well or a plate.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt              # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from spacr.gene_measurement_compare import HEADING_HELP  # noqa: E402

THE_FIVE = ("measurement", "level", "plot", "show", "compare")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class TestAllFiveHaveHelp:

    @pytest.mark.parametrize("field", THE_FIVE)
    def test_each_one_is_there(self, field):
        assert field in HEADING_HELP

    @pytest.mark.parametrize("field", THE_FIVE)
    def test_each_one_says_something(self, field):
        """So a new column cannot be added without one."""
        assert len(HEADING_HELP[field].strip()) > 60

    def test_there_are_no_others_left_undocumented(self):
        assert set(HEADING_HELP) == set(THE_FIVE)


class TestTheyExplainTheFieldNotTheWord:

    def test_level_says_what_a_datapoint_becomes(self):
        text = HEADING_HELP["level"].lower()
        assert "datapoint" in text or "one point" in text
        for word in ("cell", "well", "plate"):
            assert word in text

    def test_level_warns_about_the_replicate(self):
        """A screen randomises at the well, and testing across cells returns
        p < 1e-10 on noise."""
        assert "randomis" in HEADING_HELP["level"].lower()

    def test_show_says_the_statistics_do_not_follow_it(self):
        text = HEADING_HELP["show"].lower()
        assert "statistic" in text and "whole" in text

    def test_compare_says_the_p_value_depends_on_it(self):
        assert "p-value" in HEADING_HELP["compare"].lower()

    def test_plot_says_it_changes_nothing_but_the_picture(self):
        text = HEADING_HELP["plot"].lower()
        assert "changes nothing" in text or "nothing about the values" in text

    def test_measurement_says_it_is_offered_from_the_data(self):
        assert "offered" in HEADING_HELP["measurement"].lower()


class TestTheyAreOnTheHeadings:

    @pytest.fixture
    def panel(self, app):
        import pandas as pd

        from spacr.qt.widgets.measurement_compare_dialog import (
            MeasurementComparePanel)

        objects = pd.DataFrame({
            "prcfo": ["p1_r1_c1_f1_o1", "p1_r1_c2_f1_o1"],
            "area": [10.0, 20.0], "gene": ["a", "b"],
        })
        return MeasurementComparePanel(objects, {"a": ["a"], "b": ["b"]})

    @pytest.mark.parametrize("field", THE_FIVE)
    def test_the_heading_carries_its_help(self, panel, field):
        label = panel.headings.get(field)
        assert label is not None, f"no heading for {field}"
        assert label.toolTip() == HEADING_HELP[field]

    @pytest.mark.parametrize("field", THE_FIVE)
    def test_the_heading_looks_like_help(self, panel, field):
        assert panel.headings[field].cursor().shape() == Qt.WhatsThisCursor

    def test_the_help_is_keyed_by_field_not_by_label_text(self, panel):
        """A tooltip looked up by the heading's TEXT stops working the moment
        somebody renames the heading -- and stops working silently, because
        the label still draws and the help is simply gone."""
        label = panel.headings["level"]
        before = label.toolTip()
        label.setText("Aggregation")
        assert label.toolTip() == before
        assert panel.headings["level"] is label

    def test_the_tooltip_does_not_time_out(self, panel):
        """Same behaviour instruction 208 asks for elsewhere: help that
        disappears when you reach for it cannot be read to the end."""
        assert panel.headings["level"].toolTipDuration() == -1
