"""Hit List's modal slots and the failure half of its ready-callback.

Same shape as the Methods Export tails, and for the same reason: every
one of these lines carries an inert `# pragma: no cover` marker. This
project sets `exclude_lines =` to an empty list, so the markers exclude
nothing -- the lines were always in the denominator, and they were
simply untested.

The three export slots are one-liners, and a one-liner is exactly where
a wrong constant hides: an "Export as HTML" that quietly writes CSV is
invisible in review and obvious to a user. So they are asserted on the
format and the filter they pass, not merely on having been called.
"""
from __future__ import annotations

import pytest

from spacr.qt.screens import hit_list as screen_module

pytestmark = pytest.mark.qt


@pytest.fixture()
def screen(qtbot):
    widget = screen_module.HitListScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


class TestWhenTheListCouldNotBeBuilt:

    def test_a_missing_hit_list_is_reported_in_the_summary(self, screen):
        screen._on_hits_ready(None)
        assert screen._summary.text() == "The hit list could not be built."
        assert screen._summary.property("problem") == "true"

    def test_a_missing_hit_list_emits_nothing(self, screen):
        """`hits_loaded` must not fire with nothing to load.

        Anything downstream connected to that signal would otherwise be
        handed None as though it were a result.
        """
        seen = []
        screen.hits_loaded.connect(seen.append)
        screen._on_hits_ready(None)
        assert seen == []

    def test_a_missing_hit_list_is_still_recorded_as_the_current_one(
            self, screen):
        """`self._all` is assigned BEFORE the guard, and stays None.

        Leaving a previous list in place would let the filters re-run
        over stale rows under a message saying the build failed.
        """
        screen._on_hits_ready(None)
        assert screen._all is None


class TestTheExportSlots:
    """Each names its own format, caption and filter."""

    @pytest.mark.parametrize("slot,fmt,filters", [
        ("_on_export_csv", "csv", "CSV (*.csv)"),
        ("_on_export_markdown", "markdown", "Markdown (*.md)"),
        ("_on_export_html", "html", "HTML (*.html)"),
    ])
    def test_the_slot_asks_for_its_own_format(self, screen, monkeypatch,
                                              slot, fmt, filters):
        asked = []
        monkeypatch.setattr(screen, "_ask_and_export",
                            lambda *a: asked.append(a))
        getattr(screen, slot)()
        assert asked == [(fmt, "Export hit list", filters)]

    def test_the_three_slots_do_not_share_a_format(self, screen,
                                                   monkeypatch):
        """Guards against a copy-paste that exports CSV three times."""
        asked = []
        monkeypatch.setattr(screen, "_ask_and_export",
                            lambda *a: asked.append(a[0]))
        screen._on_export_csv()
        screen._on_export_markdown()
        screen._on_export_html()
        assert asked == ["csv", "markdown", "html"]
        assert len(set(asked)) == 3


class TestTheTwoBrowseSlots:
    """Driven with the dialog replaced, so nothing blocks."""

    def test_choosing_a_folder_loads_it(self, screen, monkeypatch):
        from PySide6.QtWidgets import QFileDialog

        loaded = []
        monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                            staticmethod(lambda *a, **k: "/tmp/results"))
        monkeypatch.setattr(screen, "load_folder", loaded.append)
        screen._on_browse()
        assert loaded == ["/tmp/results"]

    def test_cancelling_the_folder_dialog_loads_nothing(self, screen,
                                                        monkeypatch):
        """Cancel returns "", and loading "" would clear the screen."""
        from PySide6.QtWidgets import QFileDialog

        loaded = []
        monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                            staticmethod(lambda *a, **k: ""))
        monkeypatch.setattr(screen, "load_folder", loaded.append)
        screen._on_browse()
        assert loaded == []

    def test_choosing_metadata_files_rebuilds_with_them(self, screen,
                                                        monkeypatch):
        from PySide6.QtWidgets import QFileDialog

        chosen = []
        monkeypatch.setattr(
            QFileDialog, "getOpenFileNames",
            staticmethod(lambda *a, **k: (["/tmp/a.csv", "/tmp/b.csv"], "")))
        monkeypatch.setattr(screen, "set_metadata_files", chosen.append)
        screen._on_pick_metadata()
        assert chosen == [["/tmp/a.csv", "/tmp/b.csv"]]

    def test_cancelling_the_metadata_dialog_changes_nothing(self, screen,
                                                            monkeypatch):
        """An empty list must not be read as "the user chose no files".

        Rebuilding with no metadata would silently drop the gene names
        from a list the user had already annotated.
        """
        from PySide6.QtWidgets import QFileDialog

        chosen = []
        monkeypatch.setattr(QFileDialog, "getOpenFileNames",
                            staticmethod(lambda *a, **k: ([], "")))
        monkeypatch.setattr(screen, "set_metadata_files", chosen.append)
        screen._on_pick_metadata()
        assert chosen == []
