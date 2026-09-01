"""Two teardown guards and one lookup that falls back.

Instruction 288.

``DataManagerScreen.closeEvent`` stops every worker thread it started. A
thread whose C++ half has already gone raises RuntimeError, and a close
handler that let that out would leave the screen half-closed -- the
failure this file's neighbours keep fixing.

``_show_section`` falls back to the database list when the title names no
section. The fold and the "is there anything to show" question would
otherwise answer each other: hiding a panel whose header stayed behind
leaves a header that opens onto nothing.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


# ---------------------------------------------------------------------------
# closeEvent with a thread that is already gone
# ---------------------------------------------------------------------------

def test_a_thread_already_gone_does_not_stop_the_close(qtbot):
    """THE ARM."""
    from spacr.qt.screens.data_manager import DataManagerScreen

    screen = DataManagerScreen()
    qtbot.addWidget(screen)

    asked = []

    class _Gone:
        def quit(self):
            asked.append("quit")
            raise RuntimeError("Internal C++ object already deleted.")

        def wait(self, _ms):
            asked.append("wait")

    screen._jobs = [(_Gone(), object())]

    screen.close()                      # must not raise

    assert asked == ["quit"], f"the thread was not asked to stop: {asked}"
    assert screen._jobs == [], "the job list survived the close"


def test_a_live_thread_is_stopped_and_waited_for(qtbot):
    """So the arm above is about the failure, not about a close that
    skips its threads. Destroying a live QThread is a process-fatal Qt
    error, so this is the part that must keep working."""
    from spacr.qt.screens.data_manager import DataManagerScreen

    screen = DataManagerScreen()
    qtbot.addWidget(screen)

    asked = []

    class _Live:
        def quit(self):
            asked.append("quit")

        def wait(self, ms):
            asked.append(("wait", ms))
            return True

    screen._jobs = [(_Live(), object())]

    screen.close()

    assert asked == ["quit", ("wait", 2000)]
    assert screen._jobs == []


# ---------------------------------------------------------------------------
# _show_section with a title that names nothing
# ---------------------------------------------------------------------------

def test_an_unknown_section_title_falls_back_to_the_database_list(qtbot):
    """THE ARM. Every caller passes a title from `section_titles()`, so
    this is reached only if the two ever disagree -- which is exactly
    when a header would be left opening onto nothing."""
    from spacr.qt.widgets import measurement_scan_panel as MSP

    panel = MSP.MeasurementScanPanel()
    qtbot.addWidget(panel)

    # isHidden(), not isVisibleTo(). The panel itself is never shown in
    # this test, and isVisibleTo() folds in every ancestor's visibility --
    # so it answers False after setVisible(True) and says nothing about
    # the call under test. isHidden() reflects the explicit call.
    panel._show_section("no such section", False)
    assert panel.databases.isHidden()

    panel._show_section("no such section", True)
    assert not panel.databases.isHidden()


def test_a_known_title_moves_its_own_section_instead(qtbot):
    """The other arm, so the fallback is visibly a fallback."""
    from spacr.qt.widgets import measurement_scan_panel as MSP

    panel = MSP.MeasurementScanPanel()
    qtbot.addWidget(panel)

    titles = panel.section_titles()
    if not titles:
        pytest.skip("this build has no named sections")

    section = panel._folders[titles[0]]
    databases_before = panel.databases.isHidden()

    panel._show_section(titles[0], False)

    assert section.isHidden()
    assert panel.databases.isHidden() == databases_before, (
        "hiding a named section moved the database list as well")
