"""`_PathRow`'s Browse button: a modal picker, driven with the modal replaced.

The row carries a text field and a Browse button, and it is used for
both a folder and a file. Its handler is marked `# pragma: no cover -
modal native picker`, which excludes nothing here -- and the half that
matters is not the dialog but what happens to the FIELD afterwards.

Pressing Cancel returns an empty string, and an empty string must not
erase a path the user has already typed. That is the failure someone
actually notices.
"""
from __future__ import annotations

import pytest

from PySide6.QtWidgets import QFileDialog

from spacr.qt.screens.model_explanation import _PathRow

pytestmark = pytest.mark.qt


class TestChoosingAFolder:

    def test_a_chosen_folder_fills_the_field(self, qtbot, monkeypatch):
        row = _PathRow(folder=True)
        qtbot.addWidget(row)
        monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                            staticmethod(lambda *a, **k: "/tmp/chosen"))
        row._browse()
        assert row.text() == "/tmp/chosen"

    def test_cancelling_keeps_what_was_there(self, qtbot, monkeypatch):
        row = _PathRow(folder=True)
        qtbot.addWidget(row)
        row.setText("/keep/this")
        monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                            staticmethod(lambda *a, **k: ""))
        row._browse()
        assert row.text() == "/keep/this"


class TestChoosingAFile:

    def test_a_chosen_file_fills_the_field(self, qtbot, monkeypatch):
        row = _PathRow(folder=False)
        qtbot.addWidget(row)
        monkeypatch.setattr(
            QFileDialog, "getOpenFileName",
            staticmethod(lambda *a, **k: ("/tmp/table.csv", "")))
        row._browse()
        assert row.text() == "/tmp/table.csv"

    def test_cancelling_keeps_what_was_there(self, qtbot, monkeypatch):
        row = _PathRow(folder=False)
        qtbot.addWidget(row)
        row.setText("/keep/this.csv")
        monkeypatch.setattr(QFileDialog, "getOpenFileName",
                            staticmethod(lambda *a, **k: ("", "")))
        row._browse()
        assert row.text() == "/keep/this.csv"

    def test_a_folder_row_never_opens_the_file_picker(self, qtbot,
                                                      monkeypatch):
        """The two are chosen by the `folder` flag, and mixing them up
        would offer a file chooser where a directory is required."""
        opened = []
        monkeypatch.setattr(
            QFileDialog, "getOpenFileName",
            staticmethod(lambda *a, **k: opened.append("file") or ("", "")))
        monkeypatch.setattr(
            QFileDialog, "getExistingDirectory",
            staticmethod(lambda *a, **k: opened.append("folder") or ""))

        folder_row = _PathRow(folder=True)
        qtbot.addWidget(folder_row)
        folder_row._browse()

        file_row = _PathRow(folder=False)
        qtbot.addWidget(file_row)
        file_row._browse()

        assert opened == ["folder", "file"]


class TestTheFieldItself:

    def test_setting_none_clears_rather_than_writing_the_word(self, qtbot):
        """`str(value or "")` -- None must not become "None" in a path."""
        row = _PathRow()
        qtbot.addWidget(row)
        row.setText("/something")
        row.setText(None)
        assert row.text() == ""

    def test_a_non_string_is_coerced(self, qtbot):
        from pathlib import Path

        row = _PathRow()
        qtbot.addWidget(row)
        row.setText(Path("/tmp/x.csv"))
        assert row.text() == "/tmp/x.csv"
