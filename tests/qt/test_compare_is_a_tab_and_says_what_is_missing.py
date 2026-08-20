"""Instruction 186 B and C, from a live session on the Cells tab.

B.  "when the Compare a measurement pops up it is infrom to the other tabs.
     this should have its own tab after summary."
B2. "it should be possible to show only one class."
B3. "there should be text somwhere telling the user to check the show all in
     well to have something to compare to if they have top by score or
     attribiuted chosen."
C.  "the measurements sub tabs should not snap to the middle when closed but
     the top."
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

import numpy as np
import pandas as pd


def _objects(n=40, one_group=False):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "plateID": ["p1"] * n,
        "rowID": [f"r{1 + i % 4}" for i in range(n)],
        "columnID": [f"c{1 + i % 5}" for i in range(n)],
        "cell_area": rng.uniform(100.0, 900.0, n),
        "pathogen_area": rng.uniform(10.0, 90.0, n),
    })


def _panel(qtbot, objects, groups, settings=None):
    from spacr.qt.widgets.measurement_compare_dialog import (
        MeasurementComparePanel)

    panel = MeasurementComparePanel(objects, groups, settings=settings or {})
    qtbot.addWidget(panel)
    return panel


class TestShowingOneClass:

    def test_every_class_is_the_default(self, qtbot):
        objects = _objects()
        panel = _panel(qtbot, objects,
                       {"picked": list(range(0, 20))})

        assert panel.only.currentData() in ("", None)
        assert panel.only.count() >= 3, "every class, plus each class"

    def test_choosing_one_draws_only_that_one(self, qtbot):
        objects = _objects()
        panel = _panel(qtbot, objects, {"picked": list(range(0, 20))})
        names = [panel.only.itemData(i) for i in range(panel.only.count())]
        target = next(n for n in names if n)

        panel.only.setCurrentIndex(names.index(target))

        assert panel.only.currentData() == target
        assert target in panel.report.toPlainText()

    def test_the_statistics_still_describe_the_whole_comparison(self, qtbot):
        """A test computed on one of two groups is not a comparison, and a
        panel that quietly re-ran it on the visible half would report a
        different question than the one on screen."""
        objects = _objects()
        panel = _panel(qtbot, objects, {"picked": list(range(0, 20))})
        whole = panel._comparison.statistics
        names = [panel.only.itemData(i) for i in range(panel.only.count())]

        panel.only.setCurrentIndex(names.index(next(n for n in names if n)))

        assert panel._comparison.statistics == whole
        assert "whole comparison" in panel.report.toPlainText()

    def test_the_filter_is_dead_when_there_is_only_one_class(self, qtbot):
        objects = _objects()
        panel = _panel(qtbot, objects, {"picked": list(range(len(objects)))})

        assert not panel.only.isEnabled()


class TestSayingThereIsNothingToCompareAgainst:
    """The sharp one: with show_all_in_well off the montage holds ONLY the
    picked cells, so "picked vs the rest" has no rest."""

    def test_it_names_the_setting_that_fixes_it(self, qtbot):
        objects = _objects()
        panel = _panel(qtbot, objects,
                       {"picked": list(range(len(objects)))},
                       settings={"cell_picking": "attributed",
                                 "show_all_in_well": False})

        said = panel.nothing_to_compare_against()
        assert "show all in well" in said
        assert "attributed" in said, "name the picker they actually chose"
        assert said in panel.report.toPlainText(), (
            "it has to reach the panel, not just be computable")

    def test_it_names_the_rank_picker_too(self, qtbot):
        objects = _objects()
        panel = _panel(qtbot, objects,
                       {"picked": list(range(len(objects)))},
                       settings={"cell_picking": "rank",
                                 "show_all_in_well": False})

        assert "rank" in panel.nothing_to_compare_against()

    def test_two_classes_need_no_warning(self, qtbot):
        objects = _objects()
        panel = _panel(qtbot, objects, {"picked": list(range(0, 20))},
                       settings={"cell_picking": "rank",
                                 "show_all_in_well": False})

        assert panel.nothing_to_compare_against() == ""

    def test_with_show_all_on_it_does_not_blame_the_setting(self, qtbot):
        """Blaming a setting that is already on would send the user to change
        something that is not the problem."""
        objects = _objects()
        panel = _panel(qtbot, objects,
                       {"picked": list(range(len(objects)))},
                       settings={"cell_picking": "rank",
                                 "show_all_in_well": True})

        said = panel.nothing_to_compare_against()
        assert said and "show all in well" not in said


class TestCompareIsATab:

    def test_the_tab_is_named_for_the_button_that_opens_it(self, qtbot):
        from spacr.qt.widgets.cell_montage_view import CellMontageView

        view = CellMontageView()
        qtbot.addWidget(view)
        labels = [view._tabs.tabText(i) for i in range(view._tabs.count())]

        assert "Graph" not in labels, (
            "one thing under two names reads as two things")


class TestFoldsCollapseUpward:
    """C: a folded section's height goes to the bottom, not the middle."""

    @pytest.fixture
    def panel(self, qtbot):
        from spacr.qt.widgets.measurement_scan_panel import (
            MeasurementScanPanel)

        widget = MeasurementScanPanel()
        qtbot.addWidget(widget)
        return widget

    def test_there_is_a_filler_and_it_is_last(self, panel):
        last = panel._sections.widget(panel._sections.count() - 1)

        assert last is panel._filler
        assert last.minimumHeight() == 0, (
            "it exists to give up all of its height")

    def test_folding_everything_puts_the_space_at_the_bottom(self, panel):
        """The behaviour, not the stretch factors -- QSplitter exposes no
        getter for those, and the height is what the user actually sees.

        With every section folded, the headers must stack at the top and the
        leftover height must be BELOW them. Before the filler existed the
        splitter spread that height around the children and the column of
        headers floated in the middle, which is what was reported.
        """
        from PySide6.QtWidgets import QApplication

        panel.resize(600, 900)
        panel.show()
        for section in panel.sections():
            panel.set_section_expanded(section.title(), False)
        QApplication.processEvents()

        sizes = panel._sections.sizes()
        filler = sizes[panel._sections.indexOf(panel._filler)]
        sections = sum(sizes) - filler

        assert filler > sections, (
            f"the folded headers take {sections}px and the gap beneath them "
            f"{filler}px -- the space has to be under the headers, not "
            f"around them")

    def test_a_section_added_later_still_folds_upward(self, panel):
        from PySide6.QtWidgets import QLabel

        panel.add_section(QLabel("added after the filler"))
        last = panel._sections.widget(panel._sections.count() - 1)

        assert last is panel._filler

    def test_sections_does_not_report_the_filler(self, panel):
        assert panel._filler not in panel.sections()
        assert len(panel.sections()) == panel._sections.count() - 1

    def test_a_layout_stored_before_the_filler_still_restores(self, panel):
        """Dropping it would throw away every arrangement a user has."""
        sizes = [120] * len(panel.sections())

        assert panel._apply_section_layout({"folded": [], "sizes": sizes})
