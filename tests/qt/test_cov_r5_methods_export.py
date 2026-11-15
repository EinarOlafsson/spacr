"""Edges of the Methods & Results screen: the export, the copy, the factory.

The screen's whole promise is that the prose it shows is made of the run and
that the strip under the tabs says so. These tests drive the three places
where something that promise depends on is absent:

* an export with sections but no digest -- the appendix is what carries the
  provenance, and a file with no appendix must still be a file;
* a copy with no clipboard to put the sections on, which must not tell the
  user it copied;
* the provenance strip with no style to repolish, which must still report;

plus the bare factory, the constructor for the caller that has no run to
point the screen at.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens import methods_export as screen_module      # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    """Keep a shared registry override from answering for a tmp project."""
    from spacr import artifacts

    monkeypatch.delenv(artifacts.ARTIFACTS_DB_ENV, raising=False)


@pytest.fixture()
def screen(qtbot):
    """The screen pointed at nothing, running inline."""
    widget = screen_module.MethodsExportScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


# ---------------------------------------------------------------------------
# The bare factory
# ---------------------------------------------------------------------------

def test_the_bare_factory_builds_a_screen_with_nothing_filled_in(qtbot):
    """The constructor for a caller with no run to point it at.

    Regression seeds its own with the project and results folder that
    screen is already reading; this is what everybody else gets, and it has
    to come up empty rather than carrying whatever the last caller passed.
    """
    screen = screen_module.make_methods_export_screen()
    qtbot.addWidget(screen)

    assert isinstance(screen, screen_module.MethodsExportScreen)
    assert set(screen.sources()) == {"project", "run_dir", "results", "model"}
    assert all(value == "" for value in screen.sources().values())
    assert screen.digest() is None
    assert screen.text() == ""
    # And the source fields are live, so "empty" is the seeding and not a
    # screen that cannot hold a path.
    screen._fields["project"].setText("/runs/plate1")
    assert screen.sources()["project"] == "/runs/plate1"


# ---------------------------------------------------------------------------
# Exporting
# ---------------------------------------------------------------------------

def test_exporting_sections_with_no_digest_writes_them_without_an_appendix(
        screen, tmp_path):
    """Prose the screen is showing is exported whether or not a digest is.

    The digest travels WITH the prose as an appendix, because a methods
    section whose provenance lives in another file has no provenance. But
    a screen can be showing sections with no digest behind them -- a draft
    restored into the panes, a digest that was never built -- and refusing
    to write then would lose the user's text to protect an appendix that
    does not exist.
    """
    screen._methods_view.setPlainText("## Methods\nWe segmented 96 fields.")
    screen._results_view.setPlainText("## Results\nThe effect was 48.3.")
    assert screen.digest() is None

    target = Path(screen.export(str(tmp_path / "no-digest.md")))

    written = target.read_text(encoding="utf-8")
    assert "We segmented 96 fields." in written
    assert "The effect was 48.3." in written
    assert "Appendix: run digest" not in written
    assert "```json" not in written
    assert "with the run digest as an appendix" in screen._provenance.text()

    # With a digest the same call appends it, so the missing appendix above
    # is the missing digest and not a dropped branch.
    screen._digest = {"run": {"n_settings": 4}, "caveats": []}

    with_digest = Path(screen.export(str(tmp_path / "with-digest.md")))

    appended = with_digest.read_text(encoding="utf-8")
    assert "Appendix: run digest" in appended
    assert '"n_settings": 4' in appended
    assert appended.startswith("## Methods")


# ---------------------------------------------------------------------------
# Copying
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The provenance strip with no style
# ---------------------------------------------------------------------------
