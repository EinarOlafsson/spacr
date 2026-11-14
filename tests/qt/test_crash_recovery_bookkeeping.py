"""Table-driven tests of the crash-loop bookkeeping over a real folder.

``spacr.qt.crash_recovery`` runs before the window exists, so nothing it does
may raise: a log folder that cannot be found, a counter replaced by a folder,
a marker left by a dead run must all end with spaCR starting and the right
count.  ``_folder`` runs for real here (pointed at a temporary log folder, or
falling back under a temporary home); each case starts from a clean folder.
"""
from __future__ import annotations

import logging
import os

import pytest

from spacr import logging_util
from spacr.qt import crash_recovery as CR

pytestmark = pytest.mark.qt

LOGGER = "spacr.qt.crash_recovery"


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Home, lock folder and SPACR_HOME moved off the user's real ones."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("SPACR_HOME", raising=False)
    monkeypatch.setenv("SPACR_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.setattr(logging_util, "_portable_root", lambda: None)
    return tmp_path


def _fresh(isolated, monkeypatch, name):
    folder = isolated / name
    folder.mkdir()
    monkeypatch.setattr(logging_util, "log_dir", lambda: str(folder))
    return folder


def _read(path):
    if os.path.isdir(path):
        return "<dir>"
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _messages(caplog):
    return [(r.levelno, r.getMessage()) for r in caplog.records
            if r.name == LOGGER]


def test_folder_keeps_markers_beside_the_logs_or_under_home(
        subtests, isolated, monkeypatch):
    """The log folder is used and made on demand; no usable answer falls back."""
    fresh = isolated / "logs-not-yet-there"
    fallback = str(isolated / "home" / ".spacr" / "logs")

    def boom():
        raise OSError("read-only file system")

    cases = [
        ("a first-ever launch makes the log folder", lambda: str(fresh),
         str(fresh)),
        ("a log folder that raises falls back under home", boom, fallback),
        ("a log folder that answers with nothing falls back too", lambda: "",
         fallback),
        ("a log folder that answers None falls back too", lambda: None,
         fallback),
    ]
    for case_id, log_dir, expected in cases:
        with subtests.test(msg=case_id):
            monkeypatch.setattr(logging_util, "log_dir", log_dir)
            assert CR._folder() == expected
            assert os.path.isdir(expected)


def test_a_launch_counts_the_runs_that_died_before_it(
        subtests, isolated, monkeypatch, caplog):
    """Exact count returned, counter and marker left, and lines logged per state."""
    pid = str(os.getpid())
    cases = [
        # (id, counter before, marker before, returned, counter after,
        #  marker after, drop backdrop?, log lines)
        ("a first launch writes its pid and counts nothing", None, None, 0,
         None, pid, False, []),
        ("a leftover marker is one death", None, "4321", 1, "1", pid, False,
         []),
        ("a second death in a row drops the backdrop", "1", "4321", 2, "2", pid,
         True, []),
        ("a garbage counter plus a leftover marker counts one", "banana", "9",
         1, "1", pid, False, []),
        ("a counter that cannot be written is logged, not raised", "<dir>",
         "4321", 1, "<dir>", pid, False,
         [(logging.DEBUG, "could not record the unclean-exit count")]),
        ("a marker that is a folder is no crash and switches detection off",
         None, "<dir>", 0, None, "<dir>", False,
         [(logging.WARNING, "crash detection is disabled: {marker} exists but "
           "is not a file, so spaCR can neither write nor clear its running "
           "marker"),
          (logging.DEBUG, "could not write the running marker")]),
    ]
    for index, (case_id, counter, marker, returned, counter_after,
                marker_after, drop, logs) in enumerate(cases):
        with subtests.test(msg=case_id):
            folder = _fresh(isolated, monkeypatch, f"launch-{index}")
            counter_path = str(folder / CR._COUNTER)
            marker_path = str(folder / CR._MARKER)
            for path, content in ((counter_path, counter), (marker_path, marker)):
                if content == "<dir>":
                    os.mkdir(path)
                elif content is not None:
                    with open(path, "w", encoding="utf-8") as handle:
                        handle.write(content)
            caplog.clear()
            with caplog.at_level(logging.DEBUG, logger=LOGGER):
                got = CR.note_that_a_launch_began()
            assert got == returned
            assert _read(counter_path) == counter_after
            assert _read(marker_path) == marker_after
            assert CR.should_start_without_the_backdrop(got) is drop
            assert _messages(caplog) == [
                (level, text.format(marker=marker_path)) for level, text in logs]
            CR._release_all_locks()
    with subtests.test(msg="a process that launches again keeps its lock and counts its marker"):
        folder = _fresh(isolated, monkeypatch, "launch-again")
        lock = os.path.join(CR._locks_folder(), CR._INSTANCE_LOCK_NAME)
        try:
            assert CR.note_that_a_launch_began() == 0
            assert list(CR._HELD_LOCKS) == [lock]
            assert CR.note_that_a_launch_began() == 1
            assert list(CR._HELD_LOCKS) == [lock]
            assert (_read(str(folder / CR._COUNTER)),
                    _read(str(folder / CR._MARKER))) == ("1", pid)
        finally:
            CR._release_all_locks()


def test_a_clean_shutdown_clears_the_count_and_the_marker(
        subtests, isolated, monkeypatch, caplog):
    """The count resets to 0 whatever state the marker is in; only a stuck one logs."""
    cases = [
        # (id, counter before, marker before, marker after, log lines)
        ("a launch's own marker is taken away", "3", "pid", None, []),
        ("a shutdown with no marker at all is not an error", "2", None, None,
         []),
        ("a marker that cannot be removed still resets the count", "3",
         "<dir>", "<dir>",
         [(logging.DEBUG, "could not remove the running marker")]),
    ]
    for index, (case_id, counter, marker, marker_after, logs) in enumerate(cases):
        with subtests.test(msg=case_id):
            folder = _fresh(isolated, monkeypatch, f"shutdown-{index}")
            counter_path = str(folder / CR._COUNTER)
            marker_path = str(folder / CR._MARKER)
            if marker == "<dir>":
                os.mkdir(marker_path)
            elif marker == "pid":
                assert CR.note_that_a_launch_began() == 0
                assert _read(marker_path) == str(os.getpid())
            with open(counter_path, "w", encoding="utf-8") as handle:
                handle.write(counter)
            assert CR.should_start_without_the_backdrop() is True
            caplog.clear()
            with caplog.at_level(logging.DEBUG, logger=LOGGER):
                CR.note_a_clean_shutdown()
            assert _read(counter_path) == "0"
            assert _read(marker_path) == marker_after
            assert CR.should_start_without_the_backdrop() is False
            assert _messages(caplog) == logs


def test_the_backdrop_decision_reads_the_count_handed_in_or_on_disk(
        subtests, isolated, monkeypatch):
    """Two or more unclean exits drop the backdrop; unreadable counters are 0."""
    folder = _fresh(isolated, monkeypatch, "decision")
    counter_path = str(folder / CR._COUNTER)
    on_disk = [
        # (id, file content or None, read count, drop?)
        ("no counter file is no crash", None, 0, False),
        ("a counter that is not a number reads as no crashes", "banana", 0,
         False),
        ("an empty counter reads as no crashes", "", 0, False),
        ("a negative counter is clamped to zero", "-5", 0, False),
        ("one crash keeps the backdrop", "1\n", 1, False),
        ("exactly the threshold drops it", "2", 2, True),
        ("a readable large count drops it", "7\n", 7, True),
    ]
    for case_id, content, count, drop in on_disk:
        with subtests.test(msg=case_id):
            if os.path.exists(counter_path):
                os.remove(counter_path)
            if content is not None:
                with open(counter_path, "w", encoding="utf-8") as handle:
                    handle.write(content)
            assert CR._read_counter() == count
            assert CR.should_start_without_the_backdrop() is drop

    with subtests.test(msg="writes are clamped at zero"):
        for value, text in ((-4, "0"), (3, "3"), (0, "0")):
            CR._write_counter(value)
            assert _read(counter_path) == text

    with subtests.test(msg="a handed-in count wins over the file"):
        CR._write_counter(0)
        assert [CR.should_start_without_the_backdrop(n)
                for n in (0, 1, 2, 9)] == [False, False, True, True]
        assert CR.should_start_without_the_backdrop() is False
        assert CR.CRASHES_BEFORE_DROPPING_THE_BACKDROP == 2
