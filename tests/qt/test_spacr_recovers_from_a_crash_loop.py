"""When spaCR keeps dying on launch, it starts without what kills it."""
from __future__ import annotations

import os

import pytest

from spacr.qt import crash_recovery


@pytest.fixture
def markers(tmp_path, monkeypatch):
    """Keep the markers out of the user's real log folder."""
    monkeypatch.setattr(crash_recovery, "_folder", lambda: str(tmp_path))
    monkeypatch.delenv("SPACR_NO_GL", raising=False)
    monkeypatch.delenv("SPACR_NO_BACKDROP", raising=False)
    try:
        yield tmp_path
    finally:
        # ``take_the_backdrop_out_of_this_launch`` writes these directly to
        # ``os.environ``.  Remove that simulated crash state before the next
        # test starts; the monkeypatch fixture will then restore any value
        # the parent process genuinely supplied.
        os.environ.pop("SPACR_NO_GL", None)
        os.environ.pop("SPACR_NO_BACKDROP", None)


def test_a_clean_run_leaves_nothing_behind(markers):
    """The ordinary case: start, stop, no count."""
    assert crash_recovery.note_that_a_launch_began() == 0
    crash_recovery.note_a_clean_shutdown()
    assert crash_recovery.should_start_without_the_backdrop() is False


def test_one_unclean_exit_is_not_yet_a_pattern(markers):
    """A kill -9, a lid closing and a crash look the same after one."""
    crash_recovery.note_that_a_launch_began()      # dies: no clean shutdown
    assert crash_recovery.note_that_a_launch_began() == 1
    assert crash_recovery.should_start_without_the_backdrop() is False


def test_two_in_a_row_drops_the_backdrop(markers):
    """The crash log shows repeated native aborts on the drawing path."""
    crash_recovery.note_that_a_launch_began()
    crash_recovery.note_that_a_launch_began()
    assert crash_recovery.note_that_a_launch_began() == 2
    assert crash_recovery.should_start_without_the_backdrop() is True


def test_a_clean_run_brings_it_back(markers):
    """The user must not have to undo a diagnosis they never made."""
    for _ in range(4):
        crash_recovery.note_that_a_launch_began()
    assert crash_recovery.should_start_without_the_backdrop() is True

    crash_recovery.note_a_clean_shutdown()
    assert crash_recovery.should_start_without_the_backdrop() is False


def test_dropping_it_is_process_local_and_not_saved(markers, monkeypatch):
    """Writing it to the store would be a setting the user cannot explain."""
    from spacr.qt import preferences

    store = {}

    class _Mem:
        def value(self, key, default=None, type=None):
            return store.get(key, default)

        def setValue(self, key, value):
            store[key] = value

        def sync(self):
            pass

    monkeypatch.setattr(preferences, "_settings", lambda: _Mem())

    crash_recovery.take_the_backdrop_out_of_this_launch()

    assert os.environ.get("SPACR_NO_BACKDROP") == "1"
    assert os.environ.get("SPACR_NO_GL") == "1"
    assert store == {}, "the diagnosis was written to the user's preferences"
    # And it takes effect for this process.
    assert preferences.get_ambient_enabled() is False


def test_only_the_backdrop_goes(markers, monkeypatch):
    """Safe mode reads everything as defaults; this must not."""
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "_SAFE_MODE", False)
    crash_recovery.take_the_backdrop_out_of_this_launch()
    assert preferences.in_safe_mode() is False, (
        "a driver crash reset every preference the user had set")


def test_no_gl_canvas_is_built_after_the_drop(markers, monkeypatch):
    """The crash log's Python frames all carry QtOpenGL."""
    from spacr.qt.widgets import fractal_travel

    monkeypatch.setenv("QT_QPA_PLATFORM", "xcb")
    monkeypatch.setenv("DISPLAY", ":0")
    assert fractal_travel.platform_can_do_opengl() is True

    crash_recovery.take_the_backdrop_out_of_this_launch()
    assert fractal_travel.platform_can_do_opengl() is False


def test_a_directory_where_the_marker_goes_is_not_a_crash(markers):
    """310 A1: a *directory* at the marker path is not evidence of a crash.

    ``os.path.exists`` answers True for a directory, and ``os.remove`` cannot
    delete one -- it raises ``IsADirectoryError``, which the shutdown path
    swallows along with everything else. So a stray directory at that name,
    which a botched restore or a sync tool that materialises a name as a
    folder can leave, was counted as an unclean exit on EVERY launch and no
    clean shutdown could ever clear it.

    The cost is that the user loses the animated backdrop permanently, with
    no crash having occurred and no setting they can point at, and the
    misdiagnosis is self-perpetuating: the marker cannot be written either,
    so the next launch finds the same directory and counts again.

    Both halves are asserted, because fixing only the read would leave a
    shutdown that reports success while the path is still occupied.
    """
    stray = os.path.join(str(markers), crash_recovery._MARKER)
    os.mkdir(stray)

    assert crash_recovery.note_that_a_launch_began() == 0
    assert crash_recovery.note_that_a_launch_began() == 0, (
        "a directory must not accumulate unclean exits across launches"
    )

    crash_recovery.note_a_clean_shutdown()
    assert crash_recovery.should_start_without_the_backdrop() is False
    assert os.path.isdir(stray), (
        "the directory was not ours to create and must not be removed"
    )


def test_a_real_stale_marker_file_still_counts(markers):
    """The guard above must not blind the mechanism to what it is for.

    A leftover marker FILE is exactly what an unclean exit leaves, and it has
    to keep counting -- otherwise the A1 fix would disable crash recovery
    rather than correct it.
    """
    with open(os.path.join(str(markers), crash_recovery._MARKER), "w") as fh:
        fh.write("4242")
    assert crash_recovery.note_that_a_launch_began() == 1
