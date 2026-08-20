"""Two small reports from a live session, 2026-08-20.

  "in the summary section there is a giant save button in the background that
   can only be pressed on the side of the summay text, presumably because the
   text is in front and blocking"

  "measurment sections should all start closed"
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


class TestTheActionRowSurvivesARebuild:
    """`_clear` deleted what it found and only found WIDGETS.

    The action row was a bare QHBoxLayout: taken out of the body layout and
    dropped, while Copy and Save stayed children of `_body` with nothing
    laying them out -- still visible, still clickable, stuck at whatever
    geometry they last had, and painted UNDER the sections added afterwards.
    A button in the background, reachable only where no text covered it.
    """

    @pytest.fixture
    def summary(self, qtbot):
        from spacr.qt.widgets.folding_summary import FoldingSummaryView

        view = FoldingSummaryView()
        qtbot.addWidget(view)
        view.resize(600, 500)
        return view

    SUMMARY = "=== THE ANSWER ===\nsomething\n\n=== DETAIL ===\nmore\n"

    def test_the_row_is_still_in_the_layout_after_a_summary(self, summary):
        summary.setPlainText(self.SUMMARY)

        assert summary._layout.indexOf(summary._actions) >= 0, (
            "the buttons are parented to the body with no layout managing "
            "them -- which is how they end up behind the text")

    def test_it_survives_being_rebuilt_repeatedly(self, summary):
        for text in (self.SUMMARY, "=== THE ANSWER ===\nother\n", "",
                     self.SUMMARY):
            summary.setPlainText(text)

        assert summary._layout.indexOf(summary._actions) >= 0

    def test_the_buttons_are_above_the_sections_not_behind_them(self, summary):
        from PySide6.QtWidgets import QApplication

        summary.show()
        summary.setPlainText(self.SUMMARY)
        QApplication.processEvents()

        assert summary._layout.indexOf(summary._actions) == 0
        top = summary.save_button.mapTo(summary._body,
                                        summary.save_button.rect().topLeft())
        for section in summary._sections:
            assert top.y() <= section.geometry().top(), (
                "a section starts above the action row, so the row is "
                "underneath it")

    def test_the_buttons_still_do_their_jobs(self, summary, tmp_path):
        summary.setPlainText(self.SUMMARY)

        written = summary.save_to_file(str(tmp_path / "summary.txt"))

        assert written and (tmp_path / "summary.txt").read_text().strip()

    def test_nothing_is_stranded_when_there_is_no_summary(self, summary):
        """An unfoldable summary takes the early return in `_rebuild`, which
        is the path that used to strand them just as thoroughly."""
        summary.setPlainText("a statsmodels table with no spaCR headings")

        assert summary._layout.indexOf(summary._actions) >= 0


class TestTheMeasurementSectionsStartClosed:

    def test_none_of_them_opens_by_itself(self, qtbot):
        from spacr.qt.widgets.measurement_scan_panel import (
            MeasurementScanPanel)

        panel = MeasurementScanPanel()
        qtbot.addWidget(panel)

        opened = [s.title() for s in panel.sections() if s.is_expanded()]
        assert not opened, f"these opened on their own: {opened}"

    def test_the_constant_says_so_rather_than_naming_a_section(self):
        from spacr.qt.widgets.measurement_scan_panel import (
            MeasurementScanPanel)

        assert MeasurementScanPanel.OPENS_EXPANDED == ""
