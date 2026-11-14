"""measurement_scan_panel: a column that cannot be counted, and a job
runner that refuses.

Three of these guard something the current code cannot produce, and the
source says so in its own comments -- "JobRunner always returns True
today". That makes them the interesting kind of dead: not impossible by
construction, but dead because of a collaborator's present behaviour.
They are driven with a runner that refuses, which is what the guard was
written for and what a future runner may do.

The fourth is a real defensive arm around `nunique`, which raises on a
column of unhashable values.
"""
from __future__ import annotations

import pytest

from spacr.qt.widgets import measurement_scan_panel as MSP

pytestmark = pytest.mark.qt


class TestShowingASectionByTitle:

    def test_a_known_section_is_shown_and_hidden(self, qtbot):
        panel = MSP.MeasurementScanPanel()
        qtbot.addWidget(panel)
        titles = panel.section_titles()
        if not titles:
            pytest.skip("this build has no named sections")
        # THE SECTION WIDGET is the evidence. `_show_section` sets
        # visibility on the whole folder -- header included, deliberately
        # -- so that is what has to follow the flag. Asserting on
        # `section_titles()` would not: it lists sections whether they are
        # shown or not.
        section = panel._folders[titles[0]]

        panel._show_section(titles[0], False)
        assert not section.isVisibleTo(panel), "the section stayed visible"

        panel._show_section(titles[0], True)
        assert section.isVisibleTo(panel), "the section did not come back"

    def test_an_unknown_title_falls_back_to_the_database_list(self, qtbot):
        """THE UNCOVERED ARM.

        A title the panel does not know cannot be shown on its own, and
        hiding nothing would leave the user staring at a panel that
        ignored them. Falling back to the database list keeps the call
        meaningful.
        """
        panel = MSP.MeasurementScanPanel()
        qtbot.addWidget(panel)
        panel._show_section("no such section", False)
        assert panel.databases.isVisible() is False
        panel._show_section("no such section", True)
