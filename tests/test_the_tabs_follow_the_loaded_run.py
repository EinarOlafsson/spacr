"""The Cells and Measurements tabs follow the loaded run.

Reported from a real session: "the measurement and cell tabs are always
linked to the last run". Both tabs take zero-argument providers precisely so
they can be re-read -- the scan panel's own docstring says "the tab must not
go on showing the previous run's inputs" -- but the only thing that re-read
them was OPENING the tab. The tab a user is looking at when they load
another run is never opened again, so it kept the previous run's content
while every other view moved.
"""

import inspect

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens import app_screen as A  # noqa: E402


def test_the_handler_exists():
    assert hasattr(A.AppScreen, "_on_loaded_run_changed_refresh_tabs")


def test_it_is_connected_to_the_loaded_run_signal():
    """Wired, not merely defined. `set_loaded_run` is a whole method on the
    Runs panel with no caller at all, which is how this class of bug hides."""
    src = inspect.getsource(A.AppScreen)
    assert "loaded_run_changed.connect(" in src
    assert "_on_loaded_run_changed_refresh_tabs" in src


def test_it_touches_both_tabs():
    body = inspect.getsource(A.AppScreen._on_loaded_run_changed_refresh_tabs)
    assert "_cell_montage" in body
    assert "_scan_panel" in body


def test_the_montage_is_emptied_not_left_full():
    """Its contents answer a coefficient from the PREVIOUS run's table."""
    body = inspect.getsource(A.AppScreen._on_loaded_run_changed_refresh_tabs)
    assert "montage.clear()" in body


def test_neither_tab_can_take_the_run_change_down():
    """A tab that cannot refresh must not break loading a run."""
    body = inspect.getsource(A.AppScreen._on_loaded_run_changed_refresh_tabs)
    assert body.count("except Exception") >= 2


def test_it_survives_a_screen_with_neither_tab():
    """The handler is on the regression screen and must not assume the
    tabs were built -- every other provider on this screen is written that
    way for the same reason."""
    class _Bare:
        _on_loaded_run_changed_refresh_tabs = \
            A.AppScreen._on_loaded_run_changed_refresh_tabs

    bare = _Bare()

    result = bare._on_loaded_run_changed_refresh_tabs({"run": "ols_2"})

    assert result is None
    assert vars(bare) == {}, "the absent tabs must not be synthesised"


def test_a_tab_that_raises_does_not_stop_the_other(qtbot):
    """The Measurements tab still refreshes when the Cells tab throws."""
    seen = []

    class _Angry:
        def clear(self):
            raise RuntimeError("no")

        def refresh(self):
            raise RuntimeError("no")

    class _Willing:
        def refresh(self):
            seen.append("scan")

    class _Bare:
        _cell_montage = _Angry()
        _scan_panel = _Willing()
        _on_loaded_run_changed_refresh_tabs = \
            A.AppScreen._on_loaded_run_changed_refresh_tabs

    _Bare()._on_loaded_run_changed_refresh_tabs({"run": "ols_2"})
    assert seen == ["scan"]


def test_both_are_refreshed_in_the_ordinary_case(qtbot):
    seen = []

    class _Montage:
        def clear(self):
            seen.append("clear")

        def refresh(self):
            seen.append("montage")

    class _Scan:
        def refresh(self):
            seen.append("scan")

    class _Bare:
        _cell_montage = _Montage()
        _scan_panel = _Scan()
        _on_loaded_run_changed_refresh_tabs = \
            A.AppScreen._on_loaded_run_changed_refresh_tabs

    _Bare()._on_loaded_run_changed_refresh_tabs({"run": "ols_2"})
    assert seen == ["clear", "montage", "scan"]


def test_both_tabs_read_the_run_through_one_provider():
    """They must not each work the folder out separately, or they can
    disagree about which run is on screen."""
    src = inspect.getsource(A.AppScreen)
    assert "results_provider=self._results_source_path" in src
    assert "frame_provider=self._scan_source_frame" in src
    assert "self._results_source_path()" in inspect.getsource(
        A.AppScreen._scan_source_frame)
