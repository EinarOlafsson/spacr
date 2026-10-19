"""Edges of the app screen that no other test reaches (item 288).

Each test drives one method of ``AppScreen`` on a stand-in ``self`` that holds
only the attributes the method reads. That is enough to state what the method
decides -- which notice the user is given, whether a refusal is silent, what
is handed back -- without building the 740 widgets of a real screen.
"""
from __future__ import annotations

import os
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from spacr.qt.screens import app_screen as aps                   # noqa: E402
from spacr.qt.screens.app_screen import AppScreen                # noqa: E402

pytestmark = pytest.mark.qt


class _Console:
    def __init__(self):
        self.notices = []
        self.stdout = []

    def append_notice(self, text, **kw):
        self.notices.append(text.format(**kw) if kw else text)

    def append_stdout(self, text, **_kw):
        self.stdout.append(text)


def _ns(**attrs):
    ns = types.SimpleNamespace(_console=_Console(), **attrs)
    ns._same_run_folder = AppScreen._same_run_folder
    ns.MAX_LIVE_RUNS = AppScreen.MAX_LIVE_RUNS
    ns.LIVE_RUNS_NOTE = AppScreen.LIVE_RUNS_NOTE
    ns.live_run_count = lambda: AppScreen.live_run_count(ns)
    ns._say = lambda msg: AppScreen._say(ns, msg)
    ns._the_run_did_not_open = lambda why: AppScreen._the_run_did_not_open(
        ns, why)
    return ns


class _Panel:
    def __init__(self, folder="/runs/a", loads=True, grab=None):
        self._folder = folder
        self._loads = loads
        self.volcano = types.SimpleNamespace(grab=grab or (lambda: None))
        self.parent_cleared = False
        self.deleted = False

    def run_folder(self):
        return self._folder

    def run_name(self):
        return "run-b"

    def load(self, folder):
        self._folder = folder
        return self._loads

    def setParent(self, parent):
        self.parent_cleared = parent is None

    def deleteLater(self):
        self.deleted = True


class _Split:
    def __init__(self):
        self.added = []
        self.sizes = None

    def addWidget(self, panel):
        self.added.append(panel)

    def setSizes(self, sizes):
        self.sizes = sizes


class TestRunsBesideEachOther:

    def test_live_runs_are_counted_by_the_panels_that_exist(self):
        assert AppScreen.live_run_count(types.SimpleNamespace()) == 0
        one = types.SimpleNamespace(_results_panel=object())
        assert AppScreen.live_run_count(one) == 1
        both = types.SimpleNamespace(_results_panel=object(),
                                     _compare_panel=object())
        assert AppScreen.live_run_count(both) == 2

    def test_a_second_run_opens_beside_the_loaded_one(self, monkeypatch):
        from spacr.qt.widgets import regression_results

        made = []

        def factory(parent):
            made.append(_Panel("/x"))
            return made[-1]

        monkeypatch.setattr(regression_results, "RegressionResultsPanel",
                            factory)
        split = _Split()
        ns = _ns(_results_panel=_Panel("/runs/a"), _results_split=split,
                 _compare_panel=None)
        raised = []
        ns._raise_the_results_tab = lambda: raised.append(True)
        assert AppScreen.open_run_beside(ns, {"folder": "/runs/b"}) is True
        assert ns._compare_panel is made[0]
        assert split.added == [made[0]] and split.sizes == [1, 1]
        assert raised == [True]
        assert "run-b is open beside the loaded run" in ns._console.notices[-1]

    def test_a_run_that_will_not_load_is_not_left_beside_the_other(
            self, monkeypatch):
        from spacr.qt.widgets import regression_results

        refused = _Panel(loads=False)
        monkeypatch.setattr(regression_results, "RegressionResultsPanel",
                            lambda parent: refused)
        ns = _ns(_results_panel=_Panel("/runs/a"), _results_split=_Split(),
                 _compare_panel=None)
        assert AppScreen.open_run_beside(ns, "/runs/b") is False
        assert refused.parent_cleared and refused.deleted
        assert ns._compare_panel is None
        assert "No results in /runs/b" in ns._console.notices[-1]

    @pytest.mark.parametrize("record,expect", [
        ({"folder": ""}, "no folder on disk"),
        ("/runs/a", "already on screen"),
    ])
    def test_a_run_that_cannot_open_says_why(self, record, expect):
        ns = _ns(_results_panel=_Panel("/runs/a"), _results_split=_Split(),
                 _compare_panel=None)
        assert AppScreen.open_run_beside(ns, record) is False
        assert expect in ns._console.notices[-1]

    def test_a_third_live_run_is_refused_with_the_reason(self):
        ns = _ns(_results_panel=_Panel("/runs/a"), _results_split=_Split(),
                 _compare_panel=_Panel("/runs/c"))
        assert AppScreen.open_run_beside(ns, "/runs/b") is False
        assert "Two runs can be live" in ns._console.notices[-1]

    def test_nothing_opens_without_a_loaded_run(self):
        ns = _ns(_results_panel=None, _results_split=None)
        assert AppScreen.open_run_beside(ns, "/runs/b") is False
        assert ns._console.notices == []

    def test_closing_with_no_second_run_does_nothing(self):
        assert AppScreen.close_run_beside(
            types.SimpleNamespace(_compare_panel=None)) is False

    def test_closing_keeps_a_photograph_only_when_one_was_taken(self):
        class Photo:
            def isNull(self):
                return False

        photo = Photo()
        ns = types.SimpleNamespace(_compare_panel=_Panel(
            "/runs/b", grab=lambda: photo), _run_photographs={})
        assert AppScreen.close_run_beside(ns) is True
        assert ns._run_photographs[os.path.abspath("/runs/b")] is photo
        assert ns._compare_panel is None

        def broken():
            raise RuntimeError("the plot is gone")

        ns2 = types.SimpleNamespace(_compare_panel=_Panel(
            "/runs/b", grab=broken), _run_photographs={})
        assert AppScreen.close_run_beside(ns2) is True
        assert ns2._run_photographs == {}

    def test_a_photograph_is_looked_up_by_absolute_folder(self):
        ns = types.SimpleNamespace(
            _run_photographs={os.path.abspath("/runs/b"): "still"})
        assert AppScreen.run_photograph(ns, "/runs/b") == "still"
        assert AppScreen.run_photograph(ns, "") is None
        assert AppScreen.run_photograph(ns, "/runs/none") is None

    def test_raising_the_results_tab_survives_a_tab_bar_that_refuses(self):
        shown = []
        card = types.SimpleNamespace(
            show=lambda: shown.append(True),
            folder=types.SimpleNamespace(
                set_shut=lambda *a, **k: shown.append("open")))

        def refuse(_page):
            raise TypeError("not a page of this tab bar")

        ns = types.SimpleNamespace(
            _results_tabs=types.SimpleNamespace(setCurrentWidget=refuse),
            _results_page=object(), _figures_card=card)
        AppScreen._raise_the_results_tab(ns)
        assert shown == [True, "open"]

        quiet = types.SimpleNamespace(_results_tabs=None)
        AppScreen._raise_the_results_tab(quiet)


class TestRemovedRuns:

    def test_a_removed_run_takes_its_photograph_figures_and_view_with_it(self):
        forgotten = []
        photo_key = os.path.abspath("/runs/b")
        beside = _Panel("/runs/b")
        queue = types.SimpleNamespace(
            forget_run=lambda label: forgotten.append(label) or True)
        refreshed = []
        ns = _ns(_results_panel=types.SimpleNamespace(
                     forget_run=lambda folder: forgotten.append(folder)),
                 _compare_panel=beside,
                 _run_photographs={photo_key: "still"},
                 _figure_queue=queue)
        ns.close_run_beside = lambda: setattr(ns, "_compare_panel", None)
        ns._queue_figure_grid_refresh = lambda: refreshed.append(True)
        AppScreen._on_runs_removed(
            ns, [{"folder": "/runs/b", "run": "B"}, {"folder": ""}, None])
        assert photo_key not in ns._run_photographs
        assert forgotten == ["/runs/b", "B"] and refreshed == [True]
        assert ns._compare_panel is None

    def test_a_panel_that_cannot_forget_does_not_stop_the_others(self):
        def refuse(_folder):
            raise RuntimeError("busy")

        def refuse_queue(_label):
            raise RuntimeError("busy")

        ns = _ns(_results_panel=types.SimpleNamespace(forget_run=refuse),
                 _compare_panel=None, _run_photographs={},
                 _figure_queue=types.SimpleNamespace(forget_run=refuse_queue))
        AppScreen._on_runs_removed(ns, [{"folder": "/runs/b", "run": "B"}])

    def test_nothing_is_forgotten_without_a_panel_or_records(self):
        AppScreen._on_runs_removed(types.SimpleNamespace(), [{"folder": "x"}])
        AppScreen._on_runs_removed(
            types.SimpleNamespace(_results_panel=object()), [])

    def test_a_loaded_run_change_survives_tabs_that_raise(self):
        def refuse():
            raise RuntimeError("no")

        ns = types.SimpleNamespace(
            _cell_montage=types.SimpleNamespace(clear=refuse, refresh=None),
            _scan_panel=types.SimpleNamespace(refresh_databases=refuse))
        AppScreen._on_loaded_run_changed_refresh_tabs(ns)


class TestShowingATrial:

    def _screen(self, panel, **extra):
        shown = []
        ns = _ns(_results_panel=panel, _figures_card=types.SimpleNamespace(
            show=lambda: shown.append(True)), **extra)
        ns.shown = shown
        return ns

    def test_a_running_trial_says_it_is_still_going(self):
        ns = self._screen(None)
        AppScreen._show_trial(ns, {"run": "T1", "status": "running"})
        assert "T1 is still going" in ns._console.stdout[-1]

    def test_a_trial_that_failed_gives_its_reason(self):
        ns = self._screen(None)
        AppScreen._show_trial(ns, {"trial_id": 4, "status": "error",
                                   "error_type": "ValueError",
                                   "error": "singular"})
        assert "Trial 4 did not produce a regression" in ns._console.stdout[-1]
        assert "singular" in ns._console.stdout[-1]

    def test_a_trial_without_a_folder_on_disk_is_named_not_shown(self, tmp_path):
        ns = self._screen(_Panel("/runs/a"))
        AppScreen._show_trial(ns, {"run": "T2", "folder":
                                   str(tmp_path / "gone")})
        assert "no saved results on disk" in ns._console.stdout[-1]

    def test_the_loaded_trial_is_not_loaded_twice(self, tmp_path):
        ns = self._screen(_Panel(str(tmp_path)))
        AppScreen._show_trial(ns, {"run": "T3", "folder": str(tmp_path)})
        assert "already loaded" in ns._console.stdout[-1]
        assert ns.shown == [True]

    def test_a_trial_load_is_wired_once_and_a_refused_start_stops(
            self, tmp_path):
        connected = []

        class P(_Panel):
            load_finished = types.SimpleNamespace(
                connect=lambda slot: connected.append(slot))

            def start_load(self, folder):
                return False

        ns = self._screen(P("/runs/a"))
        ns._on_trial_loaded = lambda ok: None
        AppScreen._show_trial(ns, {"run": "T4", "folder": str(tmp_path)})
        AppScreen._show_trial(ns, {"run": "T4", "folder": str(tmp_path)})
        assert len(connected) == 1
        assert ns._pending_trial == ("T4", str(tmp_path))
        assert ns.shown == []

    def test_a_trial_with_no_panel_or_no_dict_is_ignored(self):
        AppScreen._show_trial(self._screen(None), "not a record")
        AppScreen._show_trial(self._screen(None), {"run": "T5"})


class TestWorkspaceRestore:

    def test_a_record_without_a_folder_restores_nothing(self):
        report = AppScreen.restore_run_workspace(types.SimpleNamespace(), {})
        assert report == {"restored": [], "skipped": [], "files": []}

    def test_a_folder_without_a_saved_workspace_says_so(self, tmp_path):
        ns = _ns()
        report = AppScreen.restore_run_workspace(ns, str(tmp_path))
        assert report["restored"] == []
        assert "carries no saved workspace" in ns._console.stdout[-1]

    def test_a_workspace_is_restored_and_reported(self, tmp_path, monkeypatch):
        from spacr import workspace

        monkeypatch.setattr(workspace, "load", lambda folder: {"doc": 1})
        monkeypatch.setattr(workspace, "providers", lambda: {})
        monkeypatch.setattr(workspace, "restore",
                            lambda prov, doc, run_dir: {"restored": ["a"]})
        monkeypatch.setattr(workspace, "report_text", lambda r: "1 section")
        ns = _ns()
        report = AppScreen.restore_run_workspace(ns, {"folder": str(tmp_path)})
        assert report == {"restored": ["a"]}
        assert "Restored the workspace of" in ns._console.stdout[-1]
        assert "1 section" in ns._console.stdout[-1]

    def test_a_console_that_raises_falls_back_to_the_log(self):
        def refuse(_text):
            raise RuntimeError("closed")

        ns = types.SimpleNamespace(
            _console=types.SimpleNamespace(append_stdout=refuse))
        AppScreen._say(ns, "hello")
        AppScreen._say(types.SimpleNamespace(), "hello")
