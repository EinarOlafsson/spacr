"""``python -m spacr.cli_database``: the module-level entry point.

``spacr-db-audit`` is normally the installed console script, but the module is
also runnable directly, and that path goes through the
``if __name__ == "__main__"`` guard rather than through a call to
:func:`main`. The guard is what turns the audit's verdict into a process exit
status, so these tests execute the module under the name ``__main__`` against
a real SQLite file and read that status back.
"""
from __future__ import annotations

import json
import runpy
import sqlite3
import sys
import warnings

import pytest


def _run_as_main(argv):
    """Execute ``spacr.cli_database`` under the name ``__main__``.

    :param argv: the command-line words after the program name.
    :returns: the :class:`SystemExit` the guard raised.
    """
    saved_argv = sys.argv
    saved_module = sys.modules.pop("spacr.cli_database", None)
    sys.argv = ["spacr-db-audit", *argv]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with pytest.raises(SystemExit) as excinfo:
                runpy.run_module("spacr.cli_database", run_name="__main__")
        return excinfo.value
    finally:
        sys.argv = saved_argv
        if saved_module is not None:
            sys.modules["spacr.cli_database"] = saved_module


@pytest.fixture
def healthy_db(tmp_path):
    """A small, valid database that passes ``PRAGMA quick_check``."""
    path = tmp_path / "measurements.db"
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute(
        "CREATE TABLE object (rowid_ INTEGER PRIMARY KEY, plate TEXT)")
    connection.executemany(
        "INSERT INTO object (plate) VALUES (?)",
        [(f"plate{i}",) for i in range(10)])
    connection.commit()
    connection.close()
    return path


def test_running_the_module_on_a_healthy_database_exits_zero(
        healthy_db, capsys):
    """A database that passes every requested check is exit 0."""
    exit_exc = _run_as_main([str(healthy_db), "--quick-check", "--json"])

    assert exit_exc.code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["database"]["quick_check"] == "ok"
    assert payload["database"]["path"].endswith("measurements.db")


def test_running_the_module_exits_two_with_nothing_to_do_and_one_on_a_verdict(
        tmp_path, capsys):
    """Neither a database nor ``--probe`` is a usage error, not a pass.

    Argparse raises that exit itself, from inside :func:`main`, so it says
    nothing about the guard.  The run after it does: a probe that refuses to
    start is a verdict :func:`main` RETURNS, and only the guard can turn a
    returned number into the status the shell reads.
    """
    usage = _run_as_main([])

    assert usage.code == 2
    assert "provide DATABASE, --probe, or both" in capsys.readouterr().err

    occupied = tmp_path / "scratch.db"
    occupied.write_bytes(b"")
    refused = _run_as_main(["--probe", "--scratch", str(occupied)])

    assert refused.code == 1
    assert "ERROR: FileExistsError:" in capsys.readouterr().err
    assert occupied.read_bytes() == b"", (
        "a probe that refused the path did not write to it either")


def test_running_the_module_on_a_missing_database_exits_one(tmp_path, capsys):
    """A path that is not a database fails the audit: exit 1, named error."""
    missing = tmp_path / "absent.db"

    exit_exc = _run_as_main([str(missing)])

    assert exit_exc.code == 1
    assert "ERROR:" in capsys.readouterr().err
