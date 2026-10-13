"""Ask the user for a folder or database column that path resolution missed.

These dialogs are fallbacks. The ordinary resolution runs first every time,
and a dialog is shown only after it has failed, so a run whose paths resolve
normally never shows one.

Without a dialog, each of these cases stops with a message naming the
missing item, and that item is information the user has and the program
does not. A dialog asks for it instead, at the cost of one prompt rather
than one aborted run.

Every prompt follows these rules, which this module enforces for its
callers:

* it states what was tried first, so a typo in a setting can be told apart
  from a missing folder;
* it validates the answer before accepting it -- a chosen folder with
  nothing in it would fail one step later;
* it asks once per run, not once per image, per well or per plate;
* it returns a sentence naming the value that will be used, for the caller
  to write back and report;
* it can be cancelled: cancelling stops the run with the original error;
* it is never shown without a user present. In a script, a test or a batch
  run :func:`somebody_is_there` returns False and the original error is
  returned instead of a dialog that would block the run. This check runs
  before anything else.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, Optional, Tuple

#: Answers already given, keyed by what was asked for. One run, one question.
_ANSWERED: Dict[str, str] = {}


def forget() -> None:
    """Drop every remembered answer. For tests, and for a new run."""
    _ANSWERED.clear()


def remembered(key: str) -> Optional[str]:
    """What was already answered for ``key``, if anything.

    :param key: the key an ``ask_for_...`` call remembered its answer under.
        Returns ``None`` when nothing was answered for it in this run.
    """
    return _ANSWERED.get(key)


def somebody_is_there() -> bool:
    """Whether there is a person who could answer a dialog.

    False under pytest, with no display, or before a QApplication exists.
    Checked before anything else, because a modal dialog with no one to
    answer it blocks the run indefinitely.
    """
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    if os.environ.get("SPACR_NO_PROMPTS"):
        return False
    try:
        from PySide6.QtWidgets import QApplication
    except Exception:                                    # noqa: BLE001
        return False
    app = QApplication.instance()
    if app is None:
        return False
    if os.environ.get("QT_QPA_PLATFORM", "").startswith("offscreen"):
        return False
    return True


def ask_for_a_folder(
    key: str,
    *,
    tried: str,
    what: str,
    validate: Optional[Callable[[str], Optional[str]]] = None,
    parent=None,
    chooser: Optional[Callable[..., str]] = None,
) -> Tuple[Optional[str], str]:
    """Ask for a folder after the usual resolution has failed.

    :param key: what is being asked for. The answer is remembered under it,
        so the second image of a run does not ask again.
    :param tried: what was tried and did not work, said before the chooser
        so the user knows why they are being asked.
    :param what: a short name for the thing wanted, for the dialog title.
    :param validate: given a chosen path, returns None when it is usable or
        a sentence saying why it is not. The dialog stays open on a refusal.
    :param chooser: injected for tests. Defaults to a real folder dialog.
    :returns: ``(path, why)``. `path` is None when nobody answered, when the
        user cancelled, or when there is nobody there -- and `why` always
        says which, because those are three different situations.
    """
    already = _ANSWERED.get(key)
    if already:
        return already, f"{what}: using {already}, chosen earlier in this run"

    if not somebody_is_there():
        return None, (f"{what}: {tried} -- and there is nobody to ask, so "
                      f"this run stops.")

    if chooser is None:
        from PySide6.QtWidgets import QFileDialog

        def chooser(title, start=""):
            """Open the folder dialog. Injected so the flow can be tested."""
            return QFileDialog.getExistingDirectory(parent, title, start)

    while True:
        chosen = chooser(f"{what} — {tried}")
        if not chosen:
            return None, (f"{what}: cancelled, so this run stops with the "
                          f"error it would have given anyway.")
        complaint = validate(chosen) if validate else None
        if complaint is None:
            _ANSWERED[key] = chosen
            return chosen, f"{what}: using {chosen}, chosen just now"
        tried = complaint


def a_folder_holding(*suffixes: str) -> Callable[[str], Optional[str]]:
    """A validator: the folder must hold at least one file with a suffix.

    A chosen folder without such a file would fail one step later, so it is
    rejected before it is accepted.
    """
    wanted = tuple(s.lower() for s in suffixes)

    def check(path: str) -> Optional[str]:
        """Say what is wrong with a path, or ``None`` when it will do.

        Returns the reason as a sentence rather than a bool, so the caller
        can show why the path was refused.
        """
        if not os.path.isdir(path):
            return f"{path} is not a folder"
        try:
            names = os.listdir(path)
        except OSError as error:
            return f"{path} cannot be read ({error.strerror})"
        if not wanted:
            return None if names else f"{path} is empty"
        if any(n.lower().endswith(wanted) for n in names):
            return None
        return (f"{path} holds no {' or '.join(wanted)} file. "
                f"Choose the folder that does.")

    return check


def tables_in(database: str) -> list:
    """Every table in a SQLite file, or an empty list if it is not one.

    Never raises: a path the user chose may be any file, and the form
    reports an empty list as "holds no tables" instead of handling an
    exception.

    :param database: path to the file to read, opened read-only as SQLite. Any
        failure to open or query it yields an empty list.
    """
    from ..database_concurrency import connect

    try:
        with connect(database, readonly=True) as db:
            rows = db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "ORDER BY name").fetchall()
    except Exception:                                    # noqa: BLE001
        return []
    return [str(row[0]) for row in rows]


def columns_in(database: str, table: str) -> list:
    """Every column of one table, in the order the table declares them.

    ``PRAGMA`` takes no bound parameters, so the table name is quoted here.
    SQLite escapes a quote inside a quoted identifier by doubling it, and a
    table really can be named ``cell"s`` -- unescaped, the identifier ends
    early and the whole table reads as having no columns at all.

    :param database: path to the SQLite file, opened read-only; any failure
        yields an empty list.
    :param table: name of the table whose columns are listed; embedded double
        quotes are escaped before it is quoted into the ``PRAGMA``.
    """
    from ..database_concurrency import connect

    quoted = str(table).replace('"', '""')
    try:
        with connect(database, readonly=True) as db:
            rows = db.execute(f'PRAGMA table_info("{quoted}")').fetchall()
    except Exception:                                    # noqa: BLE001
        return []
    return [str(row[1]) for row in rows]


def ask_for_a_database_column(
    key: str,
    *,
    tried: str,
    what: str = "Coordinate column",
    parent=None,
    chooser: Optional[Callable[..., str]] = None,
    pick: Optional[Callable[..., Optional[str]]] = None,
) -> Tuple[Optional[Tuple[str, str, str]], str]:
    """Ask for a database, a table within it, and a column within that.

    The coordinate stream needs all three. They are asked one at a time so
    that the tables offered are the ones the chosen database holds, and the
    columns offered are the ones that table holds; the user never has to
    type a name the program can read.

    :param key: what is being asked for; the answer is remembered under it.
    :param tried: what was tried and did not work, shown in each title.
    :param chooser: injected for tests; defaults to a real file dialog.
    :param pick: injected for tests; defaults to a real list dialog. Called
        ``(title, prompt, options)`` and returns the choice or None.
    :returns: ``((database, table, column), why)``, or ``(None, why)``.
        ``why`` names the value used, or the reason for ``None``: no user
        present to answer, or the file dialog cancelled. A database with no
        tables, a table with no columns or a cancelled list choice asks
        again rather than returning.

    Backing out of a later step returns to the earlier one rather than
    abandoning the whole form, so choosing the wrong database costs one
    click instead of the run.
    """
    already = _ANSWERED.get(key)
    if already:
        parts = already.split("\x1f")
        if len(parts) == 3:
            return (parts[0], parts[1], parts[2]), (
                f"{what}: using {parts[2]} in {parts[1]}, chosen earlier "
                f"in this run")

    if not somebody_is_there():
        return None, (f"{what}: {tried} -- and there is nobody to ask, so "
                      f"this run stops.")

    if chooser is None:
        from PySide6.QtWidgets import QFileDialog

        def chooser(title, start=""):
            """Open the database file dialog. Injected for testing."""
            path, _filter = QFileDialog.getOpenFileName(
                parent, title, start, "Databases (*.db *.sqlite);;All (*)")
            return path

    if pick is None:
        from PySide6.QtWidgets import QInputDialog

        def pick(title, prompt, options):
            """Ask which column, from the ones the database actually has."""
            choice, accepted = QInputDialog.getItem(
                parent, title, prompt, list(options), 0, False)
            return choice if accepted else None

    complaint = tried
    while True:
        database = chooser(f"{what} — {complaint}")
        if not database:
            return None, (f"{what}: cancelled, so this run stops with the "
                          f"error it would have given anyway.")
        tables = tables_in(database)
        if not tables:
            complaint = (f"{os.path.basename(database)} holds no tables. "
                         f"Choose the database that does.")
            continue

        table = pick(what, f"Table in {os.path.basename(database)}", tables)
        if table is None:
            complaint = tried
            continue

        columns = columns_in(database, table)
        if not columns:
            complaint = f"{table} has no columns. Choose another database."
            continue

        column = pick(what, f"Column in {table}", columns)
        if column is None:
            complaint = tried
            continue

        _ANSWERED[key] = "\x1f".join((database, table, column))
        return (database, table, column), (
            f"{what}: using {column} in {table}, from "
            f"{os.path.basename(database)}, chosen just now")
