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

import logging
import os

import pytest

from spacr import logging_util
from spacr.qt import crash_recovery

LOGGER = "spacr.qt.crash_recovery"


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


def _marker_file(folder):
    return os.path.join(str(folder), crash_recovery._MARKER)


# ---------------------------------------------------------------------------
# finding somewhere to keep the markers
# ---------------------------------------------------------------------------

def test_the_markers_live_beside_the_logs_and_the_folder_is_made(tmp_path,
                                                                monkeypatch):
    """A first-ever launch has no log folder yet; recovery must not need one.

    ``log_dir()`` is normally created by the logging setup, but crash
    recovery runs before that on the very first start. If ``_folder`` assumed
    the directory existed, the first launch on a fresh machine would fail to
    write its marker and no crash would ever be counted.
    """
    fresh = tmp_path / "logs-not-yet-there"
    monkeypatch.setattr(logging_util, "log_dir", lambda: str(fresh))

    folder = crash_recovery._folder()

    assert folder == str(fresh)
    assert os.path.isdir(folder), "the marker folder was not created on demand"


def test_a_log_folder_that_raises_falls_back_under_the_home_directory(
        tmp_path, monkeypatch):
    """A read-only or unreadable log folder may not stop the diagnosis.

    ``log_dir()`` raises on a portable install whose ``SPACR_LOG_DIR`` points
    somewhere unwritable. Recovery is the one subsystem that has to survive
    that: if it propagated the error, the crash-loop escape hatch would fail
    exactly on the broken machines it exists for.
    """
    def _boom():
        raise OSError("read-only file system")

    monkeypatch.setattr(logging_util, "log_dir", _boom)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    folder = crash_recovery._folder()

    assert folder == str(tmp_path / "home" / ".spacr" / "logs")
    assert os.path.isdir(folder)


def test_a_log_folder_that_answers_with_nothing_falls_back_too(tmp_path,
                                                               monkeypatch):
    """An empty answer is as useless as an exception and must be treated so.

    An embedding host can stub ``log_dir`` out and return an empty string.
    Joining the marker name onto that would put ``running.marker`` in the
    process's current directory, so every ``cd`` would look like a fresh
    machine and a real crash loop would never be noticed.
    """
    monkeypatch.setattr(logging_util, "log_dir", lambda: "")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    folder = crash_recovery._folder()

    assert folder == str(tmp_path / "home" / ".spacr" / "logs")
    assert os.path.isdir(folder)


# ---------------------------------------------------------------------------
# reading a counter somebody else has written
# ---------------------------------------------------------------------------

def test_a_counter_that_is_not_a_number_reads_as_no_crashes(logs):
    """Garbage on disk must not switch the backdrop off for good.

    The counter is a plain text file in the user's log folder; an editor, a
    half-written flush or a sync tool can leave anything in it. Reading that
    as "many crashes" would strip the animated backdrop from a machine that
    never crashed, with no way for the user to connect cause and effect.
    """
    with open(_counter_file(logs), "w") as handle:
        handle.write("banana")
    assert crash_recovery._read_counter() == 0
    assert crash_recovery.should_start_without_the_backdrop() is False

    # ...and a number that IS readable still counts, so the zero above is a
    # judgement about the garbage and not a reader that always returns 0.
    with open(_counter_file(logs), "w") as handle:
        handle.write("7\n")
    assert crash_recovery._read_counter() == 7
    assert crash_recovery.should_start_without_the_backdrop() is True


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
# a filesystem that will not co-operate
# ---------------------------------------------------------------------------

def test_a_counter_that_cannot_be_written_is_logged_not_raised(logs, caplog):
    """A launch may not die because it could not record that one died.

    Here the counter path is occupied by a directory -- what a botched
    restore or an rsync of a stale tree leaves behind. ``note_that_a_launch
    _began`` still has to return the number it worked out, because the caller
    uses it to decide whether to build the backdrop at all.
    """
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    os.mkdir(_counter_file(logs))
    # A leftover marker: the previous run died, so this launch counts it.
    with open(_marker_file(logs), "w") as handle:
        handle.write("4321")

    unclean = crash_recovery.note_that_a_launch_began()

    assert unclean == 1, "the crash was not counted for this launch"
    assert "could not record the unclean-exit count" in caplog.text
    # The count could not be persisted, so the next launch starts from zero.
    assert crash_recovery._read_counter() == 0


def test_a_marker_that_cannot_be_written_still_counts_the_crash(logs, caplog):
    """The number handed to the caller is what turns the backdrop off.

    If the marker write threw, spaCR would abort during recovery and the
    user would be back to an application that will not open -- the exact
    failure this module exists to escape.
    """
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    # A directory where the marker file belongs: it "exists", so the previous
    # run looks unclean, and it cannot be opened for writing.
    os.mkdir(_marker_file(logs))

    unclean = crash_recovery.note_that_a_launch_began()

    assert unclean == 1
    assert "could not write the running marker" in caplog.text
    assert crash_recovery._read_counter() == 1, (
        "the crash was not remembered for the next launch")
    assert os.path.isdir(_marker_file(logs)), (
        "the obstruction was silently replaced")


def test_a_marker_that_cannot_be_removed_still_resets_the_count(logs, caplog):
    """A clean shutdown must clear the count even when the marker is stuck.

    Otherwise one unremovable marker would pin the count at its high-water
    mark: every later launch would find it, add one, and the backdrop would
    stay off for ever on a machine that is now shutting down cleanly.
    """
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    crash_recovery._write_counter(3)
    assert crash_recovery.should_start_without_the_backdrop() is True
    os.mkdir(_marker_file(logs))

    crash_recovery.note_a_clean_shutdown()

    assert "could not remove the running marker" in caplog.text
    assert crash_recovery._read_counter() == 0
    assert crash_recovery.should_start_without_the_backdrop() is False


def test_a_shutdown_with_no_marker_at_all_is_not_an_error(logs, caplog):
    """Two clean shutdowns in a row, or a marker a cleaner ate, are normal.

    ``note_a_clean_shutdown`` is called from teardown paths that can run
    twice. The second call finds nothing to remove; treating that as a
    failure would spam the log and, worse, could skip the reset that follows.
    """
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    crash_recovery._write_counter(2)
    assert crash_recovery.should_start_without_the_backdrop() is True
    assert not os.path.exists(_marker_file(logs))

    crash_recovery.note_a_clean_shutdown()

    assert crash_recovery._read_counter() == 0
    assert "could not remove the running marker" not in caplog.text
    assert crash_recovery.should_start_without_the_backdrop() is False


# ---------------------------------------------------------------------------
# the whole cycle, over the real folder
# ---------------------------------------------------------------------------

def test_a_launch_writes_its_pid_and_a_clean_stop_takes_it_away(logs):
    """The marker is the whole mechanism: it must be written and removed.

    The pid inside it is what tells a person reading the log folder which
    process claimed the marker; a marker that is never removed would make
    every subsequent launch look like a crash.
    """
    assert crash_recovery.note_that_a_launch_began() == 0

    with open(_marker_file(logs)) as handle:
        assert handle.read() == str(os.getpid())

    crash_recovery.note_a_clean_shutdown()

    assert not os.path.exists(_marker_file(logs))
    assert crash_recovery.should_start_without_the_backdrop() is False


def test_two_deaths_in_a_row_over_the_real_folder_drop_the_backdrop(logs):
    """The count has to survive across processes, which means across files.

    The earlier crash-loop test stubs ``_folder`` out, so nothing proves the
    arithmetic works through the file the next process will actually read.
    """
    crash_recovery.note_that_a_launch_began()          # dies
    assert crash_recovery.note_that_a_launch_began() == 1
    assert crash_recovery.should_start_without_the_backdrop() is False

    assert crash_recovery.note_that_a_launch_began() == 2
    with open(_counter_file(logs)) as handle:
        assert handle.read() == "2"
    assert crash_recovery.should_start_without_the_backdrop() is True


def test_the_caller_may_hand_in_the_count_it_already_read(logs):
    """The launch path reads the count once and must not race with itself.

    ``note_that_a_launch_began`` returns the number and the caller passes it
    straight back in. If the parameter were ignored and the disk re-read, a
    write that failed -- the case above -- would silently disagree with the
    number the caller was told.
    """
    crash_recovery._write_counter(0)

    assert crash_recovery.should_start_without_the_backdrop(0) is False
    assert crash_recovery.should_start_without_the_backdrop(1) is False
    assert crash_recovery.should_start_without_the_backdrop(2) is True
    assert crash_recovery.should_start_without_the_backdrop(9) is True
    # ...and with nothing handed in it falls back to the file, which still
    # says zero even though the answers above said True.
    assert crash_recovery.should_start_without_the_backdrop() is False


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
