"""The four paths in Methods Export that carry an inert `no cover` pragma.

This project's `.coveragerc` sets `exclude_lines =` to an EMPTY list on
purpose, so that the coverage figure counts everything and cannot be
raised by annotating code out of the denominator. The 102 `# pragma: no
cover` markers still in the source are therefore inert: they document an
author's intent and they exclude nothing.

Two of these four are marked `pragma: no cover - modal`, and the reason
is real -- they open a file dialog, and a test that opened one would
hang. That is an argument for replacing the dialog, not for leaving the
branch untested: what happens to the FIELD after a path comes back, and
what happens when the user presses Cancel, is behaviour a user notices.

The other two are the failure halves of the digest and draft callbacks.
They are what the screen says when the background job comes back with
nothing, and they had never been run.
"""
from __future__ import annotations

import pytest

from spacr.qt.screens import methods_export as screen_module

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    """Keep a shared registry override from answering for a tmp project."""
    from spacr import artifacts

    monkeypatch.delenv(artifacts.ARTIFACTS_DB_ENV, raising=False)


@pytest.fixture()
def screen(qtbot):
    widget = screen_module.MethodsExportScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


def _provenance(screen):
    return screen._provenance.text(), screen._provenance.property("problem")


class TestWhenTheBackgroundJobComesBackWithNothing:
    """The failure halves of the two ready-callbacks."""

    def test_a_digest_that_could_not_be_built_is_said_so(self, screen):
        screen._on_digest_ready(None)
        text, problem = _provenance(screen)
        assert text == "The digest could not be built."
        assert problem == "true", "the strip did not go into its problem state"

    def test_an_empty_digest_counts_as_no_digest(self, screen):
        """`if not digest` -- an empty dict is a failure, not a result.

        A digest with nothing in it would otherwise be rendered as `{}`
        and read as a successful build of an empty run.
        """
        screen._on_digest_ready({})
        text, problem = _provenance(screen)
        assert text == "The digest could not be built."
        assert problem == "true"

    def test_a_failed_digest_leaves_the_views_alone(self, screen):
        """It returns before touching them, and that is worth asserting.

        Half-filling the digest view on a failure would show the previous
        run's numbers under a message saying this run failed.
        """
        screen._digest_view.setPlainText("from an earlier run")
        screen._on_digest_ready(None)
        assert screen._digest_view.toPlainText() == "from an earlier run"

    def test_a_draft_that_could_not_be_produced_is_said_so(self, screen):
        screen._on_draft_ready(None)
        text, problem = _provenance(screen)
        assert text == "The draft could not be produced."
        assert problem == "true"

    def test_a_failed_draft_leaves_the_last_draft_in_place(self, screen):
        """`self._draft` is only replaced by a draft that exists."""
        sentinel = object()
        screen._draft = sentinel
        screen._on_draft_ready(None)
        assert screen._draft is sentinel


class TestTheTwoModalSlots:
    """Driven with the dialog replaced, which is why they can be tested."""

    def test_choosing_a_folder_fills_the_field(self, screen, monkeypatch):
        from PySide6.QtWidgets import QFileDialog

        monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                            staticmethod(lambda *a, **k: "/tmp/chosen-folder"))
        screen._on_browse("project", True)
        assert screen._fields["project"].text() == "/tmp/chosen-folder"

    def test_choosing_a_file_fills_the_field(self, screen, monkeypatch):
        from PySide6.QtWidgets import QFileDialog

        monkeypatch.setattr(QFileDialog, "getOpenFileName",
                            staticmethod(lambda *a, **k: ("/tmp/model.pth", "")))
        screen._on_browse("model", False)
        assert screen._fields["model"].text() == "/tmp/model.pth"

    def test_cancelling_the_folder_dialog_keeps_what_was_there(
            self, screen, monkeypatch):
        """Cancel returns "", and "" must not erase the user's path.

        This is the half a user actually notices: opening the browser to
        look, changing their mind, and finding the field emptied.
        """
        from PySide6.QtWidgets import QFileDialog

        screen._fields["project"].setText("/keep/this")
        monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                            staticmethod(lambda *a, **k: ""))
        screen._on_browse("project", True)
        assert screen._fields["project"].text() == "/keep/this"

    def test_cancelling_the_file_dialog_keeps_what_was_there(
            self, screen, monkeypatch):
        from PySide6.QtWidgets import QFileDialog

        screen._fields["model"].setText("/keep/this.pth")
        monkeypatch.setattr(QFileDialog, "getOpenFileName",
                            staticmethod(lambda *a, **k: ("", "")))
        screen._on_browse("model", False)
        assert screen._fields["model"].text() == "/keep/this.pth"

    def test_exporting_writes_to_the_path_that_came_back(self, screen,
                                                         monkeypatch):
        from PySide6.QtWidgets import QFileDialog

        wrote = []
        monkeypatch.setattr(QFileDialog, "getSaveFileName",
                            staticmethod(lambda *a, **k: ("/tmp/out.md", "")))
        monkeypatch.setattr(screen, "export", lambda path: wrote.append(path))
        screen._on_export()
        assert wrote == ["/tmp/out.md"]

    def test_cancelling_the_export_writes_nothing(self, screen, monkeypatch):
        """A cancelled Save As must not write a file anywhere."""
        from PySide6.QtWidgets import QFileDialog

        wrote = []
        monkeypatch.setattr(QFileDialog, "getSaveFileName",
                            staticmethod(lambda *a, **k: ("", "")))
        monkeypatch.setattr(screen, "export", lambda path: wrote.append(path))
        screen._on_export()
        assert wrote == []
