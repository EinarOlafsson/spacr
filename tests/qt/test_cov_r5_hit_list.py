"""Edges of the Hit List screen: its bare factory and two guarded writes.

Three things this file pins, all of them places where the screen has to
keep working when something around it is missing or has not happened yet:

* the bare factory -- the constructor for a caller with no run to point the
  screen at -- builds the screen AND makes the one outgoing connection,
  because a hit list that cannot ask the workbench to investigate a hit is
  a table with a dead button on it;
* replacing the annotation files before a folder has been chosen records
  them and rebuilds nothing, since there is nothing to rebuild;
* and the summary strip -- the only place the screen tells the user why it
  has no hits -- still reports when there is no style to repolish it with.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens import hit_list as screen_module            # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture()
def screen(qtbot):
    """A hit list pointed at nothing, running inline."""
    widget = screen_module.HitListScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


# ---------------------------------------------------------------------------
# The bare factory
# ---------------------------------------------------------------------------

class _Workbench:
    """A host that offers the one slot the hit list wants to reach."""

    def __init__(self) -> None:
        self.investigated: list = []

    def _on_investigate_hit_requested(self, row) -> None:
        self.investigated.append(row)


def test_the_bare_factory_builds_the_screen_and_wires_its_one_signal(qtbot):
    """``make_hit_list_screen`` is the constructor plus the connection.

    Regression builds its own hit list seeded with the run on screen and
    connects it itself; this is the path for everybody else, and the
    connection is the whole reason the factory exists rather than callers
    using the class. So the test is that the signal ARRIVES, not that a
    flag was set.
    """
    host = _Workbench()
    screen = screen_module.make_hit_list_screen("hit_list", host)
    qtbot.addWidget(screen)

    assert isinstance(screen, screen_module.HitListScreen)
    assert screen.hits() is None, "the bare screen came up holding a run"

    screen.investigate_requested.emit({"gene": "ATG7"})

    assert host.investigated == [{"gene": "ATG7"}]
    # And the connection is recorded, so a second wiring pass adds no
    # duplicate — the same emit would otherwise arrive twice.
    assert screen_module.connect_investigation(screen, host) is False
    screen.investigate_requested.emit({"gene": "ATG5"})
    assert host.investigated == [{"gene": "ATG7"}, {"gene": "ATG5"}]


def test_the_factory_still_builds_a_screen_for_a_host_that_cannot_take_it(
        qtbot):
    """No host, no connection -- but still a working screen.

    ``spacr.qt.app`` calls screen factories with a host that may be
    anything, including ``None``. A factory that only worked for a host
    with the slot would leave the module unopenable from the registry.
    """
    screen = screen_module.make_hit_list_screen("hit_list", None)
    qtbot.addWidget(screen)

    assert isinstance(screen, screen_module.HitListScreen)
    assert getattr(screen, "_investigation_connected", False) is False
    # Nothing is listening, and emitting is still safe: the button on an
    # unhosted screen must not raise out of a click.
    screen.investigate_requested.emit({"gene": "ATG7"})

    # Give it a host afterwards and the same screen connects, so the
    # absence above is the missing host and not a broken signal.
    host = _Workbench()
    assert screen_module.connect_investigation(screen, host) is True
    screen.investigate_requested.emit({"gene": "ATG7"})
    assert host.investigated == [{"gene": "ATG7"}]


# ---------------------------------------------------------------------------
# Annotation files before there is a folder
# ---------------------------------------------------------------------------

def test_naming_annotation_files_before_a_folder_rebuilds_nothing(screen):
    """The files are recorded; the rebuild waits for a folder.

    ``set_metadata_files`` is what the drop handler and the file picker
    both call, and either can happen before the results folder has been
    chosen. Rebuilding then would read an empty path, and the screen would
    replace its opening instruction with a failure the user did not cause.
    """
    rebuilt: list = []
    # The rebuild is what is being observed, so it is the collaborator that
    # is stubbed; `set_metadata_files` itself is the code under test.
    screen.load_folder = lambda folder: rebuilt.append(folder)

    screen.set_metadata_files(["/data/plate1.csv", "/data/plate2.csv"])

    assert screen.metadata_files() == ["/data/plate1.csv", "/data/plate2.csv"]
    assert rebuilt == [], "the screen rebuilt with no folder to read"

    # With a folder in the box the same call rebuilds against it, so the
    # silence above is the missing folder rather than a lost call.
    screen._folder_edit.setText("/runs/plate/results")

    screen.set_metadata_files(["/data/plate3.csv"])

    assert screen.metadata_files() == ["/data/plate3.csv"]
    assert rebuilt == ["/runs/plate/results"]


# ---------------------------------------------------------------------------
# The summary strip with no style
# ---------------------------------------------------------------------------

def test_the_summary_still_reports_when_there_is_no_style(screen,
                                                          monkeypatch):
    """The text and the problem flag land even with no style to repolish.

    The summary strip is the only channel this screen has for saying why
    it is showing nothing. A missing style must cost the colour, never the
    sentence -- silence exactly when the build failed is the worst possible
    moment for the screen to say nothing.
    """
    monkeypatch.setattr(type(screen._summary), "style",
                        lambda self: None, raising=False)

    screen._on_job_failed("results_gene.csv has no coefficient column")

    assert screen.last_error == "results_gene.csv has no coefficient column"
    assert "no coefficient column" in screen._summary.text()
    assert screen._summary.property("problem") == "true"

    # Still a working strip: the next clean summary clears the flag.
    monkeypatch.undo()
    screen._set_summary("5 hits.", problem=False)
    assert screen._summary.text() == "5 hits."
    assert screen._summary.property("problem") == "false"
