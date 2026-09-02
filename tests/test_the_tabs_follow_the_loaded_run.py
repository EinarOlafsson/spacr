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


class _LikeTheRealThing:
    """A double whose API is the real widget's, and nothing else.

    THE POINT OF THE WHOLE FILE, learned from issue 116: a double that
    defines whatever the code under test happens to call can only ever agree
    with it. This one raises AttributeError for a name the real class does
    not have -- exactly as the widget did in the user's session -- and
    records what was asked for.
    """

    def __init__(self, real, seen, names):
        self._real = real
        self._seen = seen
        self._names = dict(names)
        self.missing = []
        self.explode = False
        self.explode_on = ""

    def __getattr__(self, name):
        if name.startswith("_") or not hasattr(self._real, name):
            self.missing.append(f"{self._real.__name__}.{name}")
            raise AttributeError(
                f"{self._real.__name__!r} object has no attribute {name!r}")

        def _call(*_args, **_kwargs):
            if self.explode or self.explode_on == name:
                raise RuntimeError("no")
            self._seen.append(self._names.get(name, name))
            return 0

        return _call


def _real_shaped_doubles():
    """``(montage, scan, seen)`` shaped like the widgets the screen holds."""
    from spacr.qt.widgets.cell_montage_view import CellMontageView
    from spacr.qt.widgets.measurement_scan_panel import MeasurementScanPanel

    seen = []
    montage = _LikeTheRealThing(
        CellMontageView, seen, {"clear": "clear", "refresh": "montage"})
    scan = _LikeTheRealThing(
        MeasurementScanPanel, seen, {"refresh_databases": "scan"})
    return montage, scan, seen


def _screen(montage, scan):
    """The handler, bound to an object holding just those two tabs."""
    class _Bare:
        _on_loaded_run_changed_refresh_tabs = \
            A.AppScreen._on_loaded_run_changed_refresh_tabs

    bare = _Bare()
    bare._cell_montage = montage
    bare._scan_panel = scan
    return bare


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


def test_the_montage_is_emptied_not_left_full(qtbot):
    """Its contents answer a coefficient from the PREVIOUS run's table.

    Asserted by CALLING the handler rather than by reading its source: the
    source said ``montage.clear()`` for as long as this file has existed,
    and `CellMontageView` had no such method the whole time.
    """
    montage, scan, seen = _real_shaped_doubles()
    _screen(montage, scan)._on_loaded_run_changed_refresh_tabs({"run": "ols"})
    assert "clear" in seen


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
    montage, scan, seen = _real_shaped_doubles()
    montage.explode = True

    _screen(montage, scan)._on_loaded_run_changed_refresh_tabs({"run": "ols"})

    assert seen == ["scan"]


def test_one_dead_call_does_not_swallow_the_next(qtbot):
    """Both halves of the Cells tab went stale on ONE typo.

    `montage.clear()` and `montage.refresh()` shared a try block, so the
    AttributeError on the first meant the second never ran -- the tab was
    neither emptied nor re-read. Each call answers for itself now.
    """
    montage, scan, seen = _real_shaped_doubles()
    montage.explode_on = "clear"

    _screen(montage, scan)._on_loaded_run_changed_refresh_tabs({"run": "ols"})

    assert seen == ["montage", "scan"]


def test_both_are_refreshed_in_the_ordinary_case(qtbot):
    montage, scan, seen = _real_shaped_doubles()
    _screen(montage, scan)._on_loaded_run_changed_refresh_tabs({"run": "ols"})
    assert seen == ["clear", "montage", "scan"]


def test_the_handler_calls_only_methods_the_real_widgets_have(qtbot):
    """GitHub issue 116, and the reason this file did not catch it.

    Reported on 2026-09-02: "show the cells still not able to pull images
    from selected points on volcano plots", stuck at "reading 4
    database(s)". The attached log had the cause twice per run:

        AttributeError: 'CellMontageView' object has no attribute 'clear'
        AttributeError: 'MeasurementScanPanel' object has no attribute
                        'refresh'

    Both were caught and logged at DEBUG, so nothing reached the user: the
    Cells tab kept the previous run's montage and the Measurements tab never
    re-attached its databases, which is where the Cells tab gets its images.

    The doubles above used to define whatever the handler called, so the
    tests proved the handler calls `clear` on something that has `clear`.
    These read their API from the REAL classes instead, which is the only
    version of this test that could ever have failed.
    """
    from spacr.qt.widgets.cell_montage_view import CellMontageView
    from spacr.qt.widgets.measurement_scan_panel import MeasurementScanPanel

    montage, scan, seen = _real_shaped_doubles()
    _screen(montage, scan)._on_loaded_run_changed_refresh_tabs({"run": "ols"})

    assert seen == ["clear", "montage", "scan"], (
        f"the handler called something the real widgets do not have: "
        f"{montage.missing + scan.missing}")
    assert callable(getattr(CellMontageView, "clear", None))
    assert callable(getattr(CellMontageView, "refresh", None))
    assert callable(getattr(MeasurementScanPanel, "refresh_databases", None))


def test_both_tabs_read_the_run_through_one_provider():
    """They must not each work the folder out separately, or they can
    disagree about which run is on screen."""
    src = inspect.getsource(A.AppScreen)
    assert "results_provider=self._results_source_path" in src
    assert "frame_provider=self._scan_source_frame" in src
    assert "self._results_source_path()" in inspect.getsource(
        A.AppScreen._scan_source_frame)
