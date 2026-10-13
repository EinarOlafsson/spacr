"""The fallback prompts: who they will ask, and what they refuse to accept.

The dialogs in :mod:`spacr.qt.ask_for_the_path` are reached only after the
ordinary resolution has already failed, so the parts nobody normally sees are
the parts that matter: the headless check that decides whether a dialog may
be shown at all, the real Qt dialogs the module falls back to when a caller
injects nothing, and the validators that reject a chosen path in the form
rather than one step later.
"""
from __future__ import annotations

import builtins
import os
import sqlite3

import pytest

pytest.importorskip("PySide6")

from spacr.qt import ask_for_the_path as ASK          # noqa: E402


@pytest.fixture(autouse=True)
def _forget():
    ASK.forget()
    yield
    ASK.forget()


@pytest.fixture
def person(monkeypatch):
    """Somebody is in front of the screen, for the length of one test."""
    monkeypatch.setattr(ASK, "somebody_is_there", lambda: True)


def _step_out_from_under_pytest(monkeypatch):
    """Drop the marker that short-circuits the headless check first.

    Called from the test body rather than a fixture: pytest re-exports
    ``PYTEST_CURRENT_TEST`` at the start of every phase, so a fixture that
    deletes it during setup has it back before the assertions run.
    """
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("SPACR_NO_PROMPTS", raising=False)


# --- whether there is anybody to ask --------------------------------------


def test_nobody_is_there_when_prompts_are_switched_off(monkeypatch, qapp):
    """``SPACR_NO_PROMPTS`` is how a batch run says "never block on me"."""
    _step_out_from_under_pytest(monkeypatch)
    monkeypatch.setenv("SPACR_NO_PROMPTS", "1")

    assert ASK.somebody_is_there() is False


def test_nobody_is_there_without_a_qt_binding(monkeypatch, qapp):
    """A console install with no PySide6 gets the error, not a dialog."""
    _step_out_from_under_pytest(monkeypatch)
    real_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "PySide6.QtWidgets":
            raise ImportError("no module named 'PySide6.QtWidgets'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded)

    assert ASK.somebody_is_there() is False


def test_nobody_is_there_before_a_qapplication_exists(monkeypatch, qapp):
    """A dialog needs an application; without one it would abort, not ask."""
    _step_out_from_under_pytest(monkeypatch)
    from PySide6.QtWidgets import QApplication

    monkeypatch.setattr(QApplication, "instance", staticmethod(lambda: None))
    monkeypatch.setenv("QT_QPA_PLATFORM", "xcb")

    assert ASK.somebody_is_there() is False


def test_nobody_is_there_on_an_offscreen_platform(monkeypatch, qapp):
    """Offscreen is a render farm or a test, and neither can answer."""
    _step_out_from_under_pytest(monkeypatch)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    assert ASK.somebody_is_there() is False


def test_somebody_is_there_with_a_running_application_on_a_real_platform(
        monkeypatch, qapp):
    """The one arrangement that earns a dialog: a person could answer it."""
    _step_out_from_under_pytest(monkeypatch)
    monkeypatch.setenv("QT_QPA_PLATFORM", "xcb")

    assert ASK.somebody_is_there() is True


def test_a_folder_prompt_says_there_is_nobody_rather_than_waiting(tmp_path):
    """Headless, the fallback resolves to the error it was asked to avoid."""
    path, why = ASK.ask_for_a_folder(
        "images", tried="/nowhere/images does not exist", what="Image folder")

    assert path is None
    assert "nobody to ask" in why


# --- the real Qt dialogs the module falls back to -------------------------


class _RecordingFileDialog:
    """Stands in for ``QFileDialog``, recording what it was asked."""

    calls: list = []
    directory = ""
    open_name = ""

    @classmethod
    def getExistingDirectory(cls, parent, title, start):
        cls.calls.append(("directory", parent, title, start))
        return cls.directory

    @classmethod
    def getOpenFileName(cls, parent, title, start, name_filter):
        cls.calls.append(("file", parent, title, start, name_filter))
        return cls.open_name, name_filter


@pytest.fixture
def file_dialog(monkeypatch):
    """Replace the Qt file dialog so the default chooser can be driven."""
    import PySide6.QtWidgets

    recorder = type("_Recorder", (_RecordingFileDialog,), {"calls": []})
    monkeypatch.setattr(PySide6.QtWidgets, "QFileDialog", recorder)
    return recorder


def test_the_default_folder_chooser_is_a_real_directory_dialog(
        person, file_dialog, tmp_path):
    """With no chooser injected, the folder prompt opens QFileDialog."""
    (tmp_path / "a.tif").write_bytes(b"")
    file_dialog.directory = str(tmp_path)
    parent = object()

    path, why = ASK.ask_for_a_folder(
        "images", tried="settings pointed at /nowhere", what="Image folder",
        validate=ASK.a_folder_holding(".tif"), parent=parent)

    assert path == str(tmp_path)
    assert "chosen just now" in why
    kind, seen_parent, title, start = file_dialog.calls[0]
    assert kind == "directory"
    assert seen_parent is parent
    assert "Image folder" in title and "settings pointed at /nowhere" in title
    assert start == ""


def test_a_cancelled_default_folder_dialog_stops_the_run(person, file_dialog):
    """An empty string back from QFileDialog is a cancel, not a folder."""
    file_dialog.directory = ""

    path, why = ASK.ask_for_a_folder(
        "images", tried="settings pointed at /nowhere", what="Image folder")

    assert path is None
    assert "cancelled" in why


class _RecordingInputDialog:
    """Stands in for ``QInputDialog.getItem``."""

    calls: list = []
    answers: list = []

    @classmethod
    def getItem(cls, parent, title, prompt, options, index, editable):
        cls.calls.append((parent, title, prompt, list(options), index,
                          editable))
        return cls.answers.pop(0)


@pytest.fixture
def input_dialog(monkeypatch):
    """Replace the Qt list dialog so the default picker can be driven."""
    import PySide6.QtWidgets

    recorder = type("_Recorder", (_RecordingInputDialog,),
                    {"calls": [], "answers": []})
    monkeypatch.setattr(PySide6.QtWidgets, "QInputDialog", recorder)
    return recorder


@pytest.fixture
def a_database(tmp_path):
    path = str(tmp_path / "measurements.db")
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE cell (object_id INT, centroid_x REAL)")
    return path


def test_the_default_database_form_uses_qt_file_and_list_dialogs(
        person, file_dialog, input_dialog, a_database):
    """Nothing injected: a file dialog, then two non-editable list dialogs."""
    file_dialog.open_name = a_database
    input_dialog.answers = [("cell", True), ("centroid_x", True)]
    parent = object()

    answer, why = ASK.ask_for_a_database_column(
        "coords", tried="no centroid column was configured", parent=parent)

    assert answer == (a_database, "cell", "centroid_x")
    assert "centroid_x" in why and "cell" in why
    kind, seen_parent, title, start, name_filter = file_dialog.calls[0]
    assert kind == "file"
    assert seen_parent is parent
    assert "no centroid column was configured" in title
    assert start == ""
    assert "*.db" in name_filter
    table_call, column_call = input_dialog.calls
    assert table_call[0] is parent
    assert table_call[3] == ["cell"]
    assert table_call[5] is False, "the table name must not be typed by hand"
    assert column_call[3] == ["object_id", "centroid_x"]


def _choosers_returning(paths):
    """A chooser handing back ``paths`` in order, one per question."""
    remaining = list(paths)

    def chooser(title, start=""):
        return remaining.pop(0)

    return chooser


def test_a_cancelled_default_table_dialog_backs_out_to_the_database(
        person, input_dialog, a_database):
    """Refusing the table list returns to the file dialog, not to the caller."""
    input_dialog.answers = [("", False)]

    answer, why = ASK.ask_for_a_database_column(
        "coords", tried="no centroid column was configured",
        chooser=_choosers_returning([a_database, ""]))

    assert answer is None
    assert "cancelled" in why
    assert len(input_dialog.calls) == 1


def test_a_cancelled_default_column_dialog_backs_out_to_the_database(
        person, file_dialog, input_dialog, a_database):
    """Refusing the column list also re-asks rather than abandoning the form."""
    input_dialog.answers = [("cell", True), ("", False)]

    answer, why = ASK.ask_for_a_database_column(
        "coords", tried="no centroid column was configured",
        chooser=_choosers_returning([a_database, ""]))

    assert answer is None
    assert "cancelled" in why
    assert len(input_dialog.calls) == 2


# --- what the folder validator refuses ------------------------------------


def test_a_folder_that_cannot_be_read_says_so_rather_than_raising(tmp_path):
    """An unreadable folder is a complaint the form shows, not an OSError."""
    unreadable = tmp_path / "locked"
    unreadable.mkdir()
    unreadable.chmod(0o000)
    try:
        if os.access(str(unreadable), os.R_OK):
            pytest.skip("this process can read a mode-000 directory")
        complaint = ASK.a_folder_holding(".tif")(str(unreadable))
    finally:
        unreadable.chmod(0o700)

    assert complaint is not None
    assert str(unreadable) in complaint
    assert "cannot be read" in complaint


def test_any_file_will_do_when_no_suffix_is_named(tmp_path):
    """``a_folder_holding()`` asks only that the folder not be empty."""
    (tmp_path / "notes.txt").write_text("anything")

    assert ASK.a_folder_holding()(str(tmp_path)) is None


def test_an_empty_folder_is_refused_when_no_suffix_is_named(tmp_path):
    """The empty folder is the same failure one step later, so it is named."""
    empty = tmp_path / "empty"
    empty.mkdir()

    complaint = ASK.a_folder_holding()(str(empty))

    assert complaint == f"{empty} is empty"


# --- what the database form refuses ---------------------------------------


def test_a_remembered_folder_is_not_mistaken_for_a_database_triple(
        person, tmp_path, monkeypatch):
    """Answers share one store, so a one-part answer must not be unpacked."""
    (tmp_path / "a.tif").write_bytes(b"")
    folder, _why = ASK.ask_for_a_folder(
        "coords", tried="nothing", what="Image folder",
        chooser=lambda title, start="": str(tmp_path))
    assert folder == str(tmp_path)

    monkeypatch.setattr(ASK, "somebody_is_there", lambda: False)
    answer, why = ASK.ask_for_a_database_column("coords", tried="no column")

    assert answer is None
    assert "nobody to ask" in why


def test_a_table_that_disappears_mid_form_is_reported_not_crashed(
        person, tmp_path):
    """A table dropped between listing and asking sends the form back."""
    path = str(tmp_path / "measurements.db")
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE cell (object_id INT)")

    complaints = []

    def chooser(title, start=""):
        complaints.append(title)
        return path if len(complaints) == 1 else ""

    def pick(title, prompt, options):
        with sqlite3.connect(path) as db:
            db.execute("DROP TABLE cell")
        return "cell"

    answer, why = ASK.ask_for_a_database_column(
        "coords", tried="no centroid column was configured",
        chooser=chooser, pick=pick)

    assert answer is None
    assert "cancelled" in why
    assert "cell has no columns" in complaints[1]


def test_a_quoted_table_name_still_lists_its_columns(tmp_path):
    """A quote in a table name must be escaped, not end the identifier."""
    path = str(tmp_path / "odd.db")
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE "cell""s" (object_id INT, area REAL)')

    assert ASK.tables_in(path) == ['cell"s']
    assert ASK.columns_in(path, 'cell"s') == ["object_id", "area"]
