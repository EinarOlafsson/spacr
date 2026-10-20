"""Regression's example fetch and results-part preparation, step by step.

The Load test data button fetches on the screen's job runner and the
results part's modules are imported on a worker before the part is built in
steps. Coverage cannot see a worker thread, so each step here is the real
``AppScreen`` method called on the test thread with a stand-in screen that
records what the step asked for.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens import app_screen as screens  # noqa: E402

AppScreen = screens.AppScreen

pytestmark = pytest.mark.qt


class _Recorder:
    """Something with ``emit``/``submit``/``append_stdout`` that records calls."""

    def __init__(self, answer=True, fail=None):
        self.calls = []
        self.answer = answer
        self.fail = fail

    def __call__(self, *args, **kwargs):
        self.calls.append(args)
        if self.fail is not None:
            raise self.fail
        return self.answer

    emit = submit = append_stdout = __call__


class _Button:
    def __init__(self):
        self.enabled = True
        self.text = ""

    def setEnabled(self, value):  # noqa: N802
        self.enabled = value

    def setText(self, value):  # noqa: N802
        self.text = value


def test_progress_is_coalesced_but_each_files_final_count_is_sent(monkeypatch):
    moved = _Recorder()
    screen = SimpleNamespace(_example_fetch_moved=moved, EXAMPLE_PROGRESS_INTERVAL_S=60)
    ticks = iter([100.0, 100.5, 101.0])
    monkeypatch.setattr(screens, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    last = {"at": None}
    AppScreen._report_the_example_fetch(screen, last, "counts.csv", 10, 100)
    AppScreen._report_the_example_fetch(screen, last, "counts.csv", 20, 100)
    AppScreen._report_the_example_fetch(screen, last, "counts.csv", 100, 100)
    assert moved.calls == [("counts.csv", 10, 100), ("counts.csv", 100, 100)]
    assert last["at"] == 101.0


def test_progress_for_a_screen_already_gone_is_dropped(monkeypatch):
    screen = SimpleNamespace(_example_fetch_moved=_Recorder(fail=RuntimeError("deleted")),
                             EXAMPLE_PROGRESS_INTERVAL_S=0.1)
    last = {"at": None}
    AppScreen._report_the_example_fetch(screen, last, "scores.csv", 5, 5)
    assert last["at"] is not None


@pytest.mark.parametrize("raised, error", [
    (None, ""),
    ("example", "the example files could not be verified"),
    ("other", "ValueError"),
])
def test_the_fetch_answers_what_it_got_or_why_not(monkeypatch, raised, error):
    from spacr import example_data

    got = SimpleNamespace(folder="/data/example")
    asked = []

    def fetch(download, kind, progress):
        asked.append((download, kind, progress))
        if raised == "example":
            raise example_data.ExampleDataError("the example files could not be verified")
        if raised == "other":
            raise ValueError()
        return got

    monkeypatch.setattr(example_data, "fetch", fetch)
    report = object()
    outcome = AppScreen._fetch_the_example_screen(True, "regression", report)
    assert outcome == ((got if raised is None else None), error)
    assert asked == [(True, "regression", report)]


def test_a_second_press_while_fetching_starts_nothing():
    jobs = _Recorder(answer=True)
    screen = SimpleNamespace(_jobs=jobs, _fetch_the_example_screen=lambda *a: None,
                             _report_the_example_fetch=lambda *a: None,
                             _on_the_example_screen_fetched=lambda outcome: None)
    assert AppScreen._fetch_the_example_screen_in_background(screen, None, True, "regression") is True
    assert screen._example_fetch_running and screen._example_fetch_button is None
    assert AppScreen._fetch_the_example_screen_in_background(screen, None, True, "regression") is False
    assert len(jobs.calls) == 1


def test_progress_reaches_the_button_only_while_a_named_file_is_fetching():
    shown = []
    screen = SimpleNamespace(_say_the_download_is_moving=lambda *a, **k: shown.append((a, k)))
    AppScreen._on_example_fetch_moved(screen, "counts.csv", 1, 2)
    screen._example_fetch_running = True
    AppScreen._on_example_fetch_moved(screen, "", 1, 2)
    AppScreen._on_example_fetch_moved(screen, "counts.csv", 1, 2)
    assert shown == [(("counts.csv", 1, 2), {"pump": False})]


@pytest.mark.parametrize("outcome, said", [((None, "no network"), "no network\n"),
                                           ("not a tuple", "\n")])
def test_a_failed_fetch_restores_the_button_and_says_why(outcome, said):
    console, button = _Recorder(), _Button()
    button.setEnabled(False)
    screen = SimpleNamespace(_example_fetch_button=button, _example_fetch_running=True,
                             _console=console,
                             _put_the_example_screen_in=lambda got: pytest.fail("nothing was fetched"))
    AppScreen._on_the_example_screen_fetched(screen, outcome)
    assert console.calls == [(said,)]
    assert button.enabled and button.text == "Load test data…"
    assert screen._example_fetch_running is False and screen._example_fetch_button is None


def test_a_fetched_screen_fills_the_slots_then_prepares_the_results():
    steps = []
    got = SimpleNamespace(folder="/data/example")
    screen = SimpleNamespace(_example_fetch_button=None, _example_fetch_running=True,
                             _console=_Recorder(),
                             _put_the_example_screen_in=lambda received: steps.append(("put", received)),
                             _prepare_the_results_part=lambda: steps.append(("prepare",)))
    AppScreen._on_the_example_screen_fetched(screen, (got, ""))
    assert steps == [("put", got), ("prepare",)]


def test_the_results_part_is_prepared_once_and_only_while_owed():
    jobs = _Recorder(answer=True)
    owed = {"value": False}
    screen = SimpleNamespace(_jobs=jobs, _part_is_owed=lambda key: owed["value"],
                             _import_the_modules=AppScreen._import_the_modules,
                             _build_the_prepared_results_part=lambda imported: None,
                             _RESULTS_PART_MODULES=AppScreen._RESULTS_PART_MODULES)
    assert AppScreen._prepare_the_results_part(screen) is False
    owed["value"] = True
    assert AppScreen._prepare_the_results_part(screen) is True
    assert screen._results_part_preparing is True
    assert AppScreen._prepare_the_results_part(screen) is False
    assert len(jobs.calls) == 1
    work, _done = jobs.calls[0]
    assert work.args == (AppScreen._RESULTS_PART_MODULES,)


def test_modules_import_under_the_heavy_import_lock_and_a_broken_one_is_skipped(monkeypatch):
    held = []

    class Lock:
        def __enter__(self):
            held.append(True)

        def __exit__(self, *exc):
            return False

    monkeypatch.setitem(sys.modules, "spacr.qt.app", SimpleNamespace(HEAVY_IMPORT_LOCK=Lock()))
    assert AppScreen._import_the_modules(["json", "spacr.no_such_module_for_this_test"]) == 1
    assert held == [True, True]
    monkeypatch.setitem(sys.modules, "spacr.qt.app", SimpleNamespace())
    assert AppScreen._import_the_modules(["json"]) == 1


def test_the_prepared_part_waits_for_a_running_pipeline():
    built = []
    running = {"value": True}
    screen = SimpleNamespace(_worker_thread_is_running=lambda: running["value"],
                             _build_owed_part_in_steps=lambda key: built.append(key) or True)
    assert AppScreen._build_the_prepared_results_part(screen) is False
    assert screen._results_part_preparing is False and built == []
    running["value"] = False
    assert AppScreen._build_the_prepared_results_part(screen, 3) is True
    assert built == [screens._REGRESSION_RESULTS]


def test_download_progress_on_the_gui_thread_lets_the_window_paint(qapp, monkeypatch):
    from PySide6.QtWidgets import QApplication

    painted = []
    monkeypatch.setattr(QApplication, "processEvents", staticmethod(lambda *a: painted.append(True)))
    button = _Button()
    screen = SimpleNamespace(_example_data_button=button)
    AppScreen._say_the_download_is_moving(screen, "scores.csv", 1, 4)
    assert button.text == "scores.csv — 25%" and painted[:1] == [True]


def test_a_run_whose_part_cannot_be_built_in_steps_is_recorded_now():
    recorded = []
    runs = SimpleNamespace(record_run=lambda *a: recorded.append(a) or "handle")
    screen = SimpleNamespace(_part_is_owed=lambda key: True,
                             _record_the_run_once_built=lambda *a: None,
                             _build_owed_part_in_steps=lambda key, then: False,
                             _sweep_runs=runs)
    assert AppScreen._record_run_in_runs_tab(screen, "run 1", "/data", {"a": 1}) == "handle"
    assert recorded == [("run 1", "/data", {"a": 1})]


def test_the_grid_is_not_raised_on_a_screen_without_figures():
    queued = []
    screen = SimpleNamespace(_queue_figure_grid_refresh=lambda: queued.append(True))
    AppScreen._show_figure_grid_soon(screen)
    assert queued == []
