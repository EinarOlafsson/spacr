"""The Project Browser's optional conveniences, and the ones that must not fail.

Remembering a folder, opening a folder chooser and seeding the search list
from what the user last used are all conveniences. None of them may cost the
scan the user actually asked for, and the detail pane has to name a
registered result whose file has gone missing rather than listing only the
results that are merely out of date.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QFileDialog

from spacr.projects import ProjectSummary, StaleArtifact
from spacr.qt import path_probe
from spacr.qt.screens import project_browser as module
from spacr.qt.screens.project_browser import (
    ProjectBrowserScreen, make_project_browser_screen)

pytestmark = pytest.mark.qt


@pytest.fixture
def screen(qtbot, tmp_path):
    """A browser pointed at ``tmp_path``, running its scan inline."""
    widget = ProjectBrowserScreen(threaded=False, roots=(str(tmp_path),))
    qtbot.addWidget(widget)
    return widget


def test_a_folder_that_cannot_be_remembered_is_still_searched(
        qtbot, tmp_path, monkeypatch):
    """Recording a recent folder is a convenience, and it may fail.

    A read-only preferences file must not stop the browser adding the folder
    and scanning it, because scanning is what the user asked for.
    """
    import spacr.qt.prefs as prefs

    def _explode(*args, **kwargs):
        raise OSError("preferences are read-only")

    monkeypatch.setattr(prefs, "push_recent_source", _explode)
    widget = ProjectBrowserScreen(threaded=False)
    qtbot.addWidget(widget)

    assert widget.add_root(str(tmp_path), scan=False) is True
    assert widget.roots() == (os.path.abspath(str(tmp_path)),)


def test_choosing_a_folder_starts_where_the_last_search_did(
        qtbot, screen, tmp_path, monkeypatch):
    """The chooser opens on the folder already being searched.

    Starting from the home directory each time makes the second folder as
    expensive to reach as the first, on machines whose data lives many levels
    down a mount.

    The wait is not incidental. `_start_directory` asks
    :mod:`spacr.qt.path_probe` rather than ``os.path.isdir`` -- Qt stats and
    lists the start directory before it draws the dialog, and a remembered
    ``/nas_mnt`` root would freeze the click that opened the chooser -- and
    the probe answers *no* for a folder it has not been asked about yet. The
    screen's own `add_root` queued that question when the root was added; this
    waits for the answer so the assertion below is about the offer and not
    about a race.
    """
    seen = {}

    def _chooser(parent, title, start):
        seen["start"] = start
        return str(tmp_path / "another")

    (tmp_path / "another").mkdir()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", _chooser)
    searched = os.path.abspath(str(tmp_path))
    path_probe.isdir(searched)
    qtbot.waitUntil(
        lambda: path_probe.known(searched, want_dir=True) is not None,
        timeout=5000)

    screen.choose_root()

    assert seen["start"] == searched
    assert os.path.abspath(str(tmp_path / "another")) in screen.roots()


def test_the_chooser_never_opens_on_a_folder_nobody_has_probed(
        screen, tmp_path, monkeypatch):
    """The pessimistic half of the same rule.

    A search root the probe cache has never heard of is NOT handed to the
    dialog, because handing it over is what stats it. The chooser falls back
    to the home directory, and the next click gets the remembered folder back.
    """
    seen = {}

    def _chooser(parent, title, start):
        seen["start"] = start
        return ""

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", _chooser)
    unseen = os.path.abspath(str(tmp_path / "never-asked-about"))
    (tmp_path / "never-asked-about").mkdir()
    screen._roots.append(unseen)
    for root in screen.roots():
        path_probe.forget(root)

    screen.choose_root()

    assert seen["start"] != unseen
    assert seen["start"] == os.path.expanduser("~")


def test_a_cancelled_chooser_adds_nothing(screen, monkeypatch):
    """Cancelling the dialog leaves the search folders exactly as they were.

    An empty return is what cancel looks like; treating it as a path would
    add the current working directory to the search list.
    """
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        lambda *args: "")
    before = screen.roots()

    screen.choose_root()

    assert screen.roots() == before


def test_the_listed_summaries_are_the_ones_the_table_shows(screen, tmp_path):
    """The summaries accessor is what a caller reads instead of the table.

    Re-deriving the list from the widget's cells would lose every field the
    table does not have a column for, which is most of them.
    """
    summaries = screen.summaries()

    assert len(summaries) == screen._table.rowCount()
    assert all(hasattr(item, "root") for item in summaries)


def test_nothing_selected_names_no_project(screen):
    """With no row selected the answer is no root, not the first one.

    Falling back to row zero would make a double-click with nothing selected
    open whichever project happened to sort first.
    """
    screen._table.clearSelection()

    assert screen.selected_root() == ""
    assert screen.show_detail("") == ""


def test_a_registered_result_whose_file_is_gone_is_listed(screen, tmp_path):
    """A missing file is an availability problem and is named as one.

    It is kept apart from staleness on purpose: a result that is merely out
    of date can be recomputed from what is there, and one whose file has gone
    cannot. Listing only the stale ones would leave the gap invisible.
    """
    root = os.path.abspath(str(tmp_path / "plate1"))
    gone = StaleArtifact(artifact_id="a1", kind="measurements",
                         module="measure", role="db",
                         path=os.path.join(root, "measurements.db"),
                         missing=True)
    outdated = StaleArtifact(artifact_id="a2", kind="masks", module="mask",
                             role="masks", path=os.path.join(root, "masks"),
                             reasons=("the source images are newer",),
                             causes=("upstream-newer",))
    screen._summaries = (ProjectSummary(root=root, name="plate1", known=True,
                                        stale=(outdated,), missing=(gone,)),)

    drawn = screen.show_detail(root)

    assert "Out of date" in drawn
    assert "the source images are newer" in drawn
    assert "gone from" in drawn
    assert "measurements.db" in drawn


def test_a_browser_whose_recent_folders_cannot_be_read_still_opens(
        qtbot, monkeypatch):
    """Seeding the search list is a convenience the screen can do without.

    An unreadable preferences file must give an empty browser, not a screen
    that cannot be opened at all.
    """
    import spacr.qt.prefs as prefs

    def _explode(*args, **kwargs):
        raise OSError("preferences are unreadable")

    monkeypatch.setattr(prefs, "get_recent_sources", _explode)

    widget = make_project_browser_screen()
    qtbot.addWidget(widget)

    assert isinstance(widget, ProjectBrowserScreen)
    assert widget.roots() == ()
