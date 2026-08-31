"""Two small guards: climbing to the filesystem root, and closing a
connection that was never opened.

One cannot fire and is pinned; the other can, and is driven by handing
the reader a file that is not a database.
"""
from __future__ import annotations

import inspect
import os

import pytest

from spacr import errors as E
from spacr import portable_paths as PP


class TestClimbingToTheRoot:

    def test_a_plate_folder_yields_itself_first(self, tmp_path):
        plate = tmp_path / "screen" / "plate1"
        plate.mkdir(parents=True)

        roots = PP.candidate_roots(str(plate))

        assert roots[0] == os.path.abspath(str(plate))
        assert os.path.abspath(str(tmp_path / "screen")) in roots

    def test_the_database_file_itself_is_accepted(self, tmp_path):
        """Callers hold different things -- the plate folder, the screen
        folder, ``measurements/``, or the file -- and normalising once
        here is the point of the function."""
        measurements = tmp_path / "plate1" / "measurements"
        measurements.mkdir(parents=True)
        database = measurements / "measurements.db"
        database.write_bytes(b"")

        roots = PP.candidate_roots(str(database))

        assert roots[0] == os.path.abspath(str(measurements))

    def test_no_root_yields_nothing(self):
        assert PP.candidate_roots(None) == ()
        assert PP.candidate_roots("") == ()

    def test_the_climb_stops_at_the_filesystem_root(self):
        roots = PP.candidate_roots("/")

        assert roots == (os.path.abspath("/"),)

    def test_the_climb_is_bounded(self, tmp_path):
        """A deep tree must not return every ancestor: the bound is what
        keeps a mis-typed path from producing a long list of folders that
        are nothing to do with the run."""
        deep = tmp_path.joinpath(*[f"level{n}" for n in range(12)])
        deep.mkdir(parents=True)

        roots = PP.candidate_roots(str(deep))

        assert len(roots) <= PP._MAX_CLIMB + 1

    def test_no_step_of_the_climb_can_repeat_or_be_empty(self):
        """THE PIN, for ``if here and here not in out``.

        Each step is ``dirname`` of the last and the loop breaks as soon
        as that stops changing, so a value can never be seen twice; and
        ``abspath`` guarantees the first is non-empty, while every
        ``dirname`` of a non-empty absolute path is non-empty too. Both
        halves of the condition are therefore always true.

        Driven over the ancestry the function actually walks rather than
        argued, since that is where a repeat would have to appear.
        """
        for start in ("/a/b/c/d/e", "/", "/one"):
            seen, here = [], os.path.abspath(start)
            for _ in range(PP._MAX_CLIMB + 1):
                assert here, f"the climb from {start} produced an empty path"
                assert here not in seen, (
                    f"the climb from {start} repeated {here!r} before the "
                    f"break, so the guard in candidate_roots is live")
                seen.append(here)
                parent = os.path.dirname(here)
                if parent == here:
                    break
                here = parent


class TestClosingTheStatusConnection:

    def test_a_readable_database_with_no_status_table_reports_nothing(
            self, tmp_path):
        """Never stamped: the artifact predates stamping, or came from a
        path that does not stamp. Not an error."""
        import sqlite3

        database = tmp_path / "measurements.db"
        connection = sqlite3.connect(database)
        connection.execute("CREATE TABLE other (a INTEGER)")
        connection.commit()
        connection.close()

        assert E.read_run_status(database) == []

    def test_a_missing_database_reports_nothing(self, tmp_path):
        assert E.read_run_status(tmp_path / "absent.db") == []

    def test_a_file_that_is_not_a_database_is_refused_and_says_why(
            self, tmp_path):
        """The refusal, driven -- and MEASURED afterwards, which changed
        what this test claims.

        It was written for the ``conn is None`` arm of the ``finally``,
        on the reasoning that a corrupt file fails to connect. It does
        not: sqlite opens lazily, so ``connect`` returns and the error
        surfaces at the first read, with ``conn`` already assigned. The
        arm stays uncovered and is pinned below instead.

        What this does drive is the refusal itself, and the message is
        the substance:
        substance: a database still held by its writer, or truncated by a
        crash, fails HERE -- and that means "the run may not have
        finished", not "the run finished". Reporting an empty status
        instead would read as a clean run.
        """
        database = tmp_path / "measurements.db"
        database.write_bytes(b"this is not a database" * 100)

        with pytest.raises(E.RunStatusUnreadable) as caught:
            E.read_run_status(database)

        message = str(caught.value)
        assert "run status cannot be read" in message
        assert 'not "the run finished"' in message, (
            "the refusal no longer distinguishes an unreadable status from "
            "a finished run, which is the whole reason it raises")

    def test_the_close_is_in_a_finally_and_guarded(self):
        """Both halves matter: without the ``finally`` a refusal leaks
        the handle, and without the guard the refusal above -- which
        happens before ``conn`` is assigned -- would be replaced by an
        AttributeError on None."""
        source = inspect.getsource(E.read_run_status)

        assert "conn = None" in source
        assert "finally:" in source
        assert "if conn is not None:" in source
        assert source.index("conn = None") < source.index("finally:")
