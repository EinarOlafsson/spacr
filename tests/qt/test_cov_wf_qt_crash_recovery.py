"""The crash-loop bookkeeping when the disk underneath it misbehaves.

``spacr.qt.crash_recovery`` is the only thing standing between a user with a
bad GL driver and an application that never opens. It runs before the window
exists, so nothing it does may raise: a home directory that cannot be read, a
counter file someone replaced with a folder, a marker left over from a
half-finished uninstall must all end with spaCR still starting.

The companion file ``test_spacr_recovers_from_a_crash_loop.py`` covers the
happy arithmetic with ``_folder`` stubbed out. These tests deliberately let
the real ``_folder`` run, and then break the filesystem under it.
"""
from __future__ import annotations

import os

import pytest

from spacr import logging_util
from spacr.qt import crash_recovery


@pytest.fixture
def logs(tmp_path, monkeypatch):
    """Point the real ``_folder()`` at a temp dir, not the user's home.

    ``_folder`` is left unpatched on purpose -- its fallback and its
    ``makedirs`` are part of what is under test -- so the redirection happens
    one level down, at ``log_dir``, and ``HOME`` is moved as well so that the
    fallback branch cannot touch a real ``~/.spacr``.
    """
    monkeypatch.setattr(logging_util, "log_dir", lambda: str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return tmp_path


def _counter_file(folder):
    return os.path.join(str(folder), crash_recovery._COUNTER)


# ---------------------------------------------------------------------------
# reading a counter somebody else has written
# ---------------------------------------------------------------------------


def test_a_negative_count_is_clamped_on_the_way_in_and_out(logs):
    """A negative total would take extra crashes to climb back out of.

    Nothing writes a negative number today, but the file is user-visible and
    ``-5`` there would mean seven real crashes before the backdrop dropped.
    Both ends clamp, so the count means what it says.
    """
    crash_recovery._write_counter(-4)
    with open(_counter_file(logs)) as handle:
        assert handle.read() == "0"

    with open(_counter_file(logs), "w") as handle:
        handle.write("-5")
    assert crash_recovery._read_counter() == 0

    crash_recovery._write_counter(3)
    assert crash_recovery._read_counter() == 3


# ---------------------------------------------------------------------------
# the whole cycle, over the real folder
# ---------------------------------------------------------------------------


def test_dropping_the_backdrop_is_two_environment_variables(logs, monkeypatch):
    """Both switches must go: the backdrop and the GL canvas inside it.

    The crash log's native frames are in the GL path, which the backdrop asks
    for. Turning the backdrop off while leaving ``SPACR_NO_GL`` unset would
    leave any other GL surface free to abort the process again.
    """
    monkeypatch.setenv("SPACR_NO_GL", "0")
    monkeypatch.setenv("SPACR_NO_BACKDROP", "0")

    crash_recovery.take_the_backdrop_out_of_this_launch()

    assert os.environ["SPACR_NO_GL"] == "1"
    assert os.environ["SPACR_NO_BACKDROP"] == "1"
    # It is a decision about this process only -- nothing was written to the
    # log folder, which is where every persistent piece of state here lives.
    assert sorted(os.listdir(str(logs))) == []
    # And the folder is genuinely being watched: the very next launch marker
    # does show up there, so the empty listing above is a real absence.
    crash_recovery.note_that_a_launch_began()
    assert sorted(os.listdir(str(logs))) == [crash_recovery._MARKER]
