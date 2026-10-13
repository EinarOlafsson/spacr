"""Every database connection must wait for a lock rather than fail on it.

Reported as issue #15: Measure sometimes hangs at completion, with

    pandas.errors.DatabaseError: Execution failed on sql
    'SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name=?'
    : database is locked

A bare ``sqlite3.connect`` takes SQLite's 5-second busy default. Measure
writes from many worker processes at once, so 5 seconds is routinely
exceeded there and the reader fails instead of waiting.

``database_concurrency.connect`` is the module written for exactly this: a
30-second busy timeout, ``query_only`` for readers, and one connection per
thread. This test requires every connection in the package to go through it,
or to pass an explicit timeout of its own.
"""

import ast
import pathlib

import pytest

import spacr


PACKAGE = pathlib.Path(spacr.__file__).parent

#: The module that DEFINES the policy is allowed to call sqlite3 directly.
EXEMPT_MODULES = {"database_concurrency.py"}


def _bare_connects():
    """Every ``sqlite3.connect(...)`` with no explicit ``timeout=``."""
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if path.name in EXEMPT_MODULES:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:                      # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "attr", None)
            if name != "connect":
                continue
            root = getattr(func, "value", None)
            if getattr(root, "id", "") not in {"sqlite3", "_sqlite3"}:
                continue
            if any(kw.arg == "timeout" for kw in node.keywords):
                continue
            offenders.append(f"{path.relative_to(PACKAGE)}:{node.lineno}")
    return offenders


def test_no_connection_relies_on_sqlites_five_second_default():
    offenders = _bare_connects()
    assert not offenders, (
        "these open SQLite without a busy timeout, so they fail with "
        "'database is locked' instead of waiting for Measure's writers:\n  "
        + "\n  ".join(offenders)
        + "\n\nUse spacr.database_concurrency.connect, or pass timeout=."
    )


def test_the_scan_finds_the_module_it_is_guarding():
    """A scan matching nothing would pass the test above."""
    files = list(PACKAGE.rglob("*.py"))
    assert len(files) > 50
    assert (PACKAGE / "database_concurrency.py").is_file()


def test_the_helper_defaults_to_a_long_timeout():
    """30s is the number the guard above is protecting."""
    import inspect

    from spacr.database_concurrency import connect

    default = inspect.signature(connect).parameters["timeout"].default
    assert default >= 30.0, f"the busy timeout dropped to {default}"


def test_readers_get_query_only(tmp_path):
    """A reader that can write is a reader that can block a writer."""
    import sqlite3

    from spacr.database_concurrency import connect

    path = tmp_path / "m.db"
    sqlite3.connect(str(path)).execute("CREATE TABLE t (x INTEGER)")

    con = connect(str(path), readonly=True)
    with pytest.raises(sqlite3.OperationalError):
        con.execute("INSERT INTO t VALUES (1)")
