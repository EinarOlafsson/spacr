"""The real dialogs are constructed when no chooser is injected.

Instruction 288. ``ask_for_a_folder`` and ``ask_for_a_database_column``
take ``chooser`` and ``pick`` "injected for tests", and every existing
test injects them -- so the branches that build the REAL Qt dialogs when
they are absent were the only untested lines, marked
``# pragma: no cover``. That is the production path: nothing in the
application injects anything.

Driving it does not mean opening a dialog. The Qt statics are stubbed, so
the default chooser is DEFINED and CALLED exactly as it would be, and
returns a cancel.

``somebody_is_there()`` returns False under pytest by design -- it reads
``PYTEST_CURRENT_TEST`` -- and it is checked before the chooser is built,
so it has to be stubbed too or none of this is reachable.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt import ask_for_the_path as A


@pytest.fixture(autouse=True)
def _forget_earlier_answers():
    """The module remembers answers under their key for the whole run."""
    A._ANSWERED.clear()
    yield
    A._ANSWERED.clear()


@pytest.fixture
def somebody(monkeypatch):
    """Pretend a person is there to answer."""
    monkeypatch.setattr(A, "somebody_is_there", lambda: True)


# ---------------------------------------------------------------------------
# ask_for_a_folder
# ---------------------------------------------------------------------------

def test_the_default_folder_dialog_is_built_and_used(somebody, monkeypatch):
    """THE ARM. No chooser injected, so the real one is constructed."""
    from PySide6.QtWidgets import QFileDialog

    calls = []

    def _cancelled(parent, title, start=""):
        calls.append(title)
        return ""                            # the user pressed cancel

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", _cancelled)

    path, why = A.ask_for_a_folder(
        "images", tried="the configured folder is empty", what="Images")

    assert calls, "the default folder dialog was never called"
    assert path is None
    assert "cancelled" in why


def test_an_injected_chooser_is_preferred_over_the_default(somebody,
                                                           monkeypatch):
    """So the arm above is reached because none was given, not always."""
    from PySide6.QtWidgets import QFileDialog

    def _must_not_run(*_args, **_kwargs):
        raise AssertionError("the default dialog was built anyway")

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", _must_not_run)

    path, why = A.ask_for_a_folder(
        "images", tried="nothing there", what="Images",
        chooser=lambda _title, start="": "")

    assert path is None and "cancelled" in why


def test_nobody_there_stops_before_any_dialog_is_built(monkeypatch):
    """The check that guards the arm, and the reason it is FIRST.

    Getting it wrong does not show a dialog to nobody -- it BLOCKS, and a
    blocked batch run looks like a hang.
    """
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(A, "somebody_is_there", lambda: False)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        lambda *a, **k: pytest.fail("a dialog was opened"))

    path, why = A.ask_for_a_folder(
        "images", tried="nothing there", what="Images")

    assert path is None
    assert "nobody to ask" in why


def test_pytest_is_why_somebody_is_there_says_no():
    """PREMISE. If this stopped being true the stubbing above would be
    unnecessary -- and every dialog in the suite would block."""
    import os

    assert os.environ.get("PYTEST_CURRENT_TEST")
    assert A.somebody_is_there() is False


# ---------------------------------------------------------------------------
# ask_for_a_database_column
# ---------------------------------------------------------------------------

def test_the_default_database_dialogs_are_built_and_used(somebody,
                                                         monkeypatch):
    """THE OTHER TWO ARMS: a file dialog and a list dialog."""
    from PySide6.QtWidgets import QFileDialog, QInputDialog

    opened = []

    def _file(parent, title, start="", filters=""):
        opened.append(title)
        return "", ""                        # cancelled

    monkeypatch.setattr(QFileDialog, "getOpenFileName", _file)
    monkeypatch.setattr(
        QInputDialog, "getItem",
        lambda *a, **k: pytest.fail("the list dialog ran before the file one"))

    answer, why = A.ask_for_a_database_column(
        "coords", tried="no coordinate column found")

    assert opened, "the default file dialog was never called"
    assert answer is None
    assert why


def test_the_list_dialog_is_built_when_a_database_is_chosen(somebody,
                                                            monkeypatch,
                                                            tmp_path):
    """The `pick is None` arm, which needs the file dialog to succeed."""
    import sqlite3

    from PySide6.QtWidgets import QFileDialog, QInputDialog

    database = tmp_path / "one.db"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE object (x REAL, y REAL)")
        db.execute("INSERT INTO object VALUES (1.0, 2.0)")

    # THE FILE DIALOG MUST EVENTUALLY CANCEL. Backing out of the table
    # list returns to the database chooser by design -- "back out to the
    # database rather than abandoning the form" -- so a stub that hands
    # back the same file forever is an infinite loop, not a test.
    files = [(str(database), ""), ("", "")]
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        lambda *a, **k: files.pop(0) if files else ("", ""))
    asked = []

    def _item(parent, title, prompt, options, current=0, editable=False):
        asked.append(list(options))
        return "", False                     # the user cancelled the list

    monkeypatch.setattr(QInputDialog, "getItem", _item)

    answer, why = A.ask_for_a_database_column(
        "coords", tried="no coordinate column found")

    assert asked, "the default list dialog was never built"
    assert "object" in asked[0], f"the tables offered were {asked[0]}"
    assert answer is None
    assert "cancelled" in why


def test_cancelling_the_table_returns_to_the_database(somebody, monkeypatch,
                                                      tmp_path):
    """The behaviour that made the test above hang, pinned deliberately.

    Backing out of the table list does NOT abandon the form -- it asks
    for the database again, so a mistaken file can be corrected without
    starting over. Worth a test precisely because it is the kind of loop
    that reads as a bug until you know it is the design.
    """
    import sqlite3

    from PySide6.QtWidgets import QFileDialog, QInputDialog

    database = tmp_path / "one.db"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE object (x REAL)")

    seen = []

    def _file(*_args, **_kwargs):
        seen.append(True)
        return (str(database), "") if len(seen) == 1 else ("", "")

    monkeypatch.setattr(QFileDialog, "getOpenFileName", _file)
    monkeypatch.setattr(QInputDialog, "getItem",
                        lambda *a, **k: ("", False))

    A.ask_for_a_database_column("coords2", tried="nothing found")

    assert len(seen) == 2, (
        "cancelling the table did not bring the database chooser back")
