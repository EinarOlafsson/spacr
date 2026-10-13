"""A gate file's status line when the analysis locks cannot be read.

The Gate Editor appends what each analysis lock says about a saved gate
file; a journal that cannot be read must cost those notes, never the line.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens import gate_editor  # noqa: E402


def test_unreadable_locks_leave_the_status_line_as_it_was(monkeypatch,
                                                          tmp_path):
    from spacr import run_journal

    def unreadable(path):
        raise OSError("the journal is on a disconnected drive")

    monkeypatch.setattr(run_journal, "_gate_file_lock_notes", unreadable)
    text = gate_editor._with_lock_notes("Saved gates.json",
                                        str(tmp_path / "gates.json"))
    assert text == "Saved gates.json"


def test_lock_notes_follow_the_status_line(monkeypatch, tmp_path):
    from spacr import run_journal

    monkeypatch.setattr(run_journal, "_gate_file_lock_notes",
                        lambda path: ["changes a locked gate"])
    assert gate_editor._with_lock_notes("Saved", str(tmp_path)) == (
        "Saved; changes a locked gate")
