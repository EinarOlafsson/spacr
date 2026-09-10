"""``spacr.qt.crash_recovery``: notice repeated launch crashes, drop the backdrop.

THIS MODULE HAD NO TESTS, which is the wrong way round for code whose whole
job runs when things are already going wrong. The crash it recovers from is a
segfault in native code -- Qt's render thread or the GL driver -- where there
is no Python frame to report and no ``except`` that can run, so the ONLY
evidence is the marker file this module keeps. If the bookkeeping is wrong,
the user gets an application that will not open and a backdrop setting they
cannot reach to turn off.

Each test here pins a decision the module's own docstrings argue for, because
those are the parts a later edit would quietly reverse.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

from spacr.qt import crash_recovery as cr


@pytest.fixture()
def markers(tmp_path, monkeypatch):
    """Point the module's marker folder at a private directory."""
    monkeypatch.setattr(cr, "_folder", lambda: str(tmp_path))
    return tmp_path


def test_a_clean_run_records_nothing_to_recover_from(markers):
    """Begin, shut down cleanly, begin again: no crash was seen."""
    assert cr.note_that_a_launch_began() == 0
    cr.note_a_clean_shutdown()
    assert cr.note_that_a_launch_began() == 0
    assert not cr.should_start_without_the_backdrop()


def test_a_launch_that_never_shut_down_is_counted(markers):
    """The marker left behind IS the evidence; there is no other."""
    cr.note_that_a_launch_began()          # dies here, no clean shutdown
    assert cr.note_that_a_launch_began() == 1
    assert cr.note_that_a_launch_began() == 2


def test_two_crashes_drop_the_backdrop_and_one_does_not(markers):
    """Two in a row is a pattern; one is an accident.

    The threshold is the module's own constant rather than a literal here,
    so raising it stays a one-line decision instead of a two-file edit.
    """
    cr.note_that_a_launch_began()
    unclean = cr.note_that_a_launch_began()
    assert unclean == 1
    assert not cr.should_start_without_the_backdrop(unclean)
    unclean = cr.note_that_a_launch_began()
    assert unclean >= cr.CRASHES_BEFORE_DROPPING_THE_BACKDROP
    assert cr.should_start_without_the_backdrop(unclean)


def test_a_clean_shutdown_resets_rather_than_decrements(markers):
    """"Is spaCR crashing right now", not "how often has it ever crashed".

    A total that only grew would eventually disable the backdrop on a
    machine where it works perfectly.
    """
    cr.note_that_a_launch_began()
    cr.note_that_a_launch_began()
    cr.note_that_a_launch_began()
    assert cr.should_start_without_the_backdrop()
    cr.note_a_clean_shutdown()
    assert cr._read_counter() == 0
    assert not cr.should_start_without_the_backdrop()


def test_a_directory_on_the_marker_name_does_not_count_as_a_crash(markers):
    """The bug this module's longest comment records, as a test.

    `os.path.exists` is also True for a DIRECTORY, and `os.remove` cannot
    delete one. A stray directory on the marker path was therefore read as
    "the last run died" on every single launch, and no clean shutdown could
    clear it -- the user lost the backdrop permanently, with no crash and no
    setting to point at.
    """
    os.makedirs(os.path.join(str(markers), cr._MARKER), exist_ok=True)
    assert cr.note_that_a_launch_began() == 0
    assert cr.note_that_a_launch_began() == 0
    assert not cr.should_start_without_the_backdrop()


def test_the_obstruction_is_reported_rather_than_passed_over(markers, caplog):
    """Crash detection is OFF while something else owns the name, and says so.

    "the whole failure this entry records is a mechanism that was wrong
    about itself and never mentioned it" -- so silence here is the defect.
    """
    os.makedirs(os.path.join(str(markers), cr._MARKER), exist_ok=True)
    with caplog.at_level("WARNING"):
        cr.note_that_a_launch_began()
    assert any("crash detection is disabled" in r.message
               for r in caplog.records), [r.message for r in caplog.records]


def test_an_unreadable_counter_is_not_evidence_of_a_crash(markers):
    """A counter that cannot be read must read as zero, not as a crash."""
    path = os.path.join(str(markers), cr._COUNTER)
    with open(path, "w") as handle:
        handle.write("not a number")
    assert cr._read_counter() == 0
    assert not cr.should_start_without_the_backdrop()


def test_dropping_the_backdrop_is_this_process_only(markers, monkeypatch):
    """NOT a saved preference: the user did not choose it and must not undo it.

    Writing it to the store would turn a diagnosis into a setting they
    never made and cannot explain, and the next clean run is supposed to
    bring the backdrop back on its own.
    """
    monkeypatch.delenv("SPACR_NO_GL", raising=False)
    monkeypatch.delenv("SPACR_NO_BACKDROP", raising=False)
    cr.take_the_backdrop_out_of_this_launch()
    assert os.environ["SPACR_NO_GL"] == "1"
    assert os.environ["SPACR_NO_BACKDROP"] == "1"

    # AND THE STORED PREFERENCE IS UNTOUCHED. The env var is what turns the
    # backdrop off; the saved setting still says the user wants it, so the
    # next clean run restores it with nothing for them to undo. A future
    # edit that "helpfully" persists the diagnosis fails here rather than in
    # somebody's settings file.
    from spacr.qt import preferences as prefs
    assert prefs.get_ambient_enabled() is False, (
        "the env var must be what answers while it is set")
    monkeypatch.delenv("SPACR_NO_BACKDROP", raising=False)
    assert prefs.get_ambient_enabled() is True, (
        "clearing the env var must restore the user's own setting, which "
        "means the diagnosis was never written to the store")


def test_the_counter_survives_a_write_failure_without_raising(markers,
                                                             monkeypatch):
    """Startup and shutdown must not fail over a bookkeeping file.

    A read-only filesystem is a real deployment -- a shared install, a
    container, a home directory over a full disk -- and losing the crash
    counter there is a lost diagnosis. Losing the APPLICATION over it
    would be a lost session.
    """
    before = cr._read_counter()

    def refuse(*_a, **_k):
        raise OSError("read-only filesystem")

    monkeypatch.setattr("builtins.open", refuse)
    cr._write_counter(3)          # must not raise
    cr.note_a_clean_shutdown()    # must not raise

    # AND THE COUNTER IS UNCHANGED, which is the half a "does not raise"
    # test leaves out: swallowing the error must not also invent a value.
    monkeypatch.undo()
    assert cr._read_counter() == before, (
        "a failed write changed what the counter reads back, so the next "
        "start would act on a number nothing wrote")


def test_the_marker_folder_falls_back_when_the_log_dir_is_unavailable(
        monkeypatch, tmp_path):
    """A broken log directory must not stop crash detection.

    `_folder` is the one piece the rest of the module cannot work without,
    and it runs before anything has been set up -- so it takes the log
    directory when there is one and its own path under the home folder when
    there is not. Both arms create the directory, because every caller
    assumes it exists.
    """
    import spacr.logging_util as logging_util

    monkeypatch.setattr(logging_util, "log_dir",
                        lambda: str(tmp_path / "logs"))
    assert cr._folder() == str(tmp_path / "logs")
    assert os.path.isdir(cr._folder())

    def explode():
        raise RuntimeError("no log directory on this machine")

    monkeypatch.setattr(logging_util, "log_dir", explode)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: p.replace("~", str(tmp_path / "home")))
    fallback = cr._folder()
    assert fallback.endswith(os.path.join(".spacr", "logs")), fallback
    assert os.path.isdir(fallback)


def test_a_marker_that_cannot_be_removed_still_clears_the_count(markers,
                                                                monkeypatch):
    """Shutdown must finish even when the marker will not go.

    The count is what `should_start_without_the_backdrop` reads, so
    clearing it is the part that matters; a marker left behind costs one
    spurious unclean exit on the next launch, and raising here would cost
    the shutdown.
    """
    cr.note_that_a_launch_began()
    cr.note_that_a_launch_began()

    def refuse(_path):
        raise PermissionError("marker is locked by another process")

    monkeypatch.setattr(os, "remove", refuse)
    cr.note_a_clean_shutdown()          # must not raise
    assert cr._read_counter() == 0
