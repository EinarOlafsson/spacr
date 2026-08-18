"""GitHub issue #108: `FileNotFoundError: ~<DB>` from `classify_merged`.

A macOS user's settings carried a `~` path. `ensure_database_schema` handed it
to `migrate_database`, which is documented NOT to tilde-expand, so the path was
resolved under the working directory and the open failed four frames down with
a message that reads as "your database is missing" -- about a database that was
there the whole time.

The fix expands at the ENTRY POINT and leaves the low-level contract strict, so
these tests pin both halves: expansion where a user path arrives, and no
expansion where the docstring promises none.
"""

import os
import sqlite3

import pytest

import spacr

assert "/codex/repo/spacr/" in spacr.__file__, spacr.__file__


def _a_database(path):
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE IF NOT EXISTS cell (rowid_ INTEGER)")
    return path


def test_a_home_relative_path_opens(tmp_path, monkeypatch):
    """`~/x/measurements.db` opens when the home-relative file exists.

    THE REPORTED CRASH. Nothing about the database was wrong.
    """
    from spacr.database_schema import ensure_database_schema

    home = tmp_path / "home"
    (home / "exp").mkdir(parents=True)
    _a_database(home / "exp" / "measurements.db")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))   # Windows spelling
    # Somewhere the file is NOT, so a failure to expand cannot accidentally pass.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    report = ensure_database_schema("~/exp/measurements.db")
    assert os.path.isabs(report.path)
    assert str(home) in report.path, report.path


def test_the_low_level_call_still_refuses_a_tilde(tmp_path, monkeypatch):
    """`migrate_database` keeps its documented contract.

    Its docstring promises no expansion, and a caller that has already resolved
    a path must not have it resolved twice. Changing that quietly would make the
    docstring wrong, which is worse than the bug it fixed.
    """
    from spacr.database_schema import migrate_database

    home = tmp_path / "home"
    (home / "exp").mkdir(parents=True)
    _a_database(home / "exp" / "measurements.db")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError):
        migrate_database("~/exp/measurements.db")


def test_the_refusal_names_the_tilde(tmp_path, monkeypatch):
    """And it says WHY, instead of reading as "your database is missing".

    The reported message was `FileNotFoundError: ~<DB>` and nothing in it told
    the user their path had never been resolved.
    """
    from spacr.database_schema import migrate_database

    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError) as caught:
        migrate_database("~/nowhere/measurements.db")
    message = str(caught.value)
    assert "never expanded" in message, message
    assert "working directory" in message, message
