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
    return tmp_path


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
