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
from spacr.qt.widgets.console_panel import ConsolePanel          # noqa: E402

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
        asked = []

        def refuse(folder):
            asked.append(("panel", folder))
            raise RuntimeError("busy")

        def refuse_queue(label):
            asked.append(("queue", label))
            raise RuntimeError("busy")

        photo_key = os.path.abspath("/runs/b")
        ns = _ns(_results_panel=types.SimpleNamespace(forget_run=refuse),
                 _compare_panel=None, _run_photographs={photo_key: "still"},
                 _figure_queue=types.SimpleNamespace(forget_run=refuse_queue))
        AppScreen._on_runs_removed(ns, [{"folder": "/runs/b", "run": "B"}])
        assert asked == [("panel", "/runs/b"), ("queue", "B")]
        assert ns._run_photographs == {}

    def test_nothing_is_forgotten_without_a_panel_or_records(self):
        bare = types.SimpleNamespace()
        assert AppScreen._on_runs_removed(bare, [{"folder": "x"}]) is None
        assert vars(bare) == {}
        panel = object()
        without_records = types.SimpleNamespace(_results_panel=panel,
                                                _run_photographs={"x": 1})
        assert AppScreen._on_runs_removed(without_records, []) is None
        assert vars(without_records) == {"_results_panel": panel,
                                         "_run_photographs": {"x": 1}}

    def test_a_loaded_run_change_survives_tabs_that_raise(self):
        asked = []

        def refuse():
            asked.append(len(asked))
            raise RuntimeError("no")

        ns = types.SimpleNamespace(
            _cell_montage=types.SimpleNamespace(clear=refuse, refresh=None),
            _scan_panel=types.SimpleNamespace(refresh_databases=refuse))
        assert AppScreen._on_loaded_run_changed_refresh_tabs(ns) is None
        assert asked == [0, 1], "the measurements tab was not asked after " \
            "the cells tab raised"


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
        for record in ("not a record", {"run": "T5"}):
            ns = self._screen(None)
            assert AppScreen._show_trial(ns, record) is None
            assert ns._console.stdout == [] and ns.shown == []
            assert not hasattr(ns, "_pending_trial")


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
        import logging

        records = []
        handler = logging.Handler()
        handler.emit = records.append
        aps.LOG.addHandler(handler)
        level = aps.LOG.level
        aps.LOG.setLevel(logging.INFO)
        try:
            AppScreen._say(ns, "hello")
            AppScreen._say(types.SimpleNamespace(), "again")
        finally:
            aps.LOG.removeHandler(handler)
            aps.LOG.setLevel(level)
        logged = [r.getMessage() for r in records
                  if r.levelno == logging.INFO]
        assert logged == ["hello", "again"]


class _Outcome(_Console):
    _show_error_report_result = ConsolePanel._show_error_report_result

    def __init__(self):
        super().__init__()
        self.entries = []
        self.usage_refreshed = 0
        self._pending_report_entries = {}

    def fill_ai_entry(self, entry, text):
        self.entries.append(text)

    def refresh_ai_usage(self):
        self.usage_refreshed += 1


class TestWhatBecameOfTheAnonymousReport:

    def _sent(self, outcome, entry="entry"):
        ns = types.SimpleNamespace(_console=_Outcome(), _ai_report_entry=entry)
        AppScreen._on_anonymous_report_sent(ns, outcome)
        return ns._console

    def test_the_explanation_and_a_lighter_tier_are_both_shown(self):
        console = self._sent({"explained": True, "status": "sent",
                              "issue_number": 7,
                              "explanation": {"text": "Because.",
                                              "tier": "light"}})
        assert console.entries[0].startswith("Because.")
        assert "lighter model" in console.entries[0]
        assert console.usage_refreshed == 1

    def test_a_used_up_allowance_is_said_in_the_console(self):
        console = self._sent({"explained": True, "status": "sent",
                              "issue_number": 7,
                              "explanation": {"refused": "quota"}})
        assert "used up" in console.entries[0]
        assert "daily AI allowance used" in console.notices[-1]

    def test_an_explanation_that_failed_says_it_could_not_explain(self):
        console = self._sent({"explained": True, "status": "sent",
                              "issue_number": 7, "explanation": {}})
        assert "could not explain" in console.entries[0]
        assert "without AI summary)" in console.notices[-1]

    def test_the_relay_text_is_filled_when_the_provider_gave_none(self):
        console = self._sent({"ai_summary": "relay words", "status": "queued",
                              "quota": True})
        assert console.entries[0].startswith("relay words")
        assert "daily limit resets" in console.notices[-1]

    def test_a_usage_meter_that_raises_does_not_hide_the_answer(self):
        console = _Outcome()

        def refuse():
            raise RuntimeError("meter")

        console.refresh_ai_usage = refuse
        ns = types.SimpleNamespace(_console=console, _ai_report_entry="e")
        AppScreen._on_anonymous_report_sent(
            ns, {"explained": True, "explanation": {"text": "ok"}})
        assert console.entries == ["ok"]

    def test_a_console_that_cannot_show_the_answer_still_reports_the_issue(
            self):
        console = _Outcome()

        def refuse(entry, text):
            raise RuntimeError("closed")

        console.fill_ai_entry = refuse
        ns = types.SimpleNamespace(_console=console, _ai_report_entry="e")
        AppScreen._on_anonymous_report_sent(
            ns, {"explained": True, "explanation": {"text": "ok"},
                 "status": "sent", "issue_number": 3})
        assert "issue #3" in console.notices[-1]

    @pytest.mark.parametrize("outcome,expect", [
        ({"status": "sent", "issue_number": 5, "deduplicated": True},
         "already has issue #5"),
        ({"status": "sent", "issue_number": 5,
          "ai_summary_attached": True}, "with AI summary"),
        ({"status": "sent", "issue_number": 5}, "Reported anonymously as"),
        ({"status": "queued"}, "will retry later"),
        ({"status": "dropped", "detail": "bad"}, "did not accept"),
        ({"status": "dropped"}, "unknown"),
    ])
    def test_each_status_gets_its_own_sentence(self, outcome, expect):
        assert expect in self._sent(outcome, entry=None).notices[-1]


class TestNamingTheScreen:

    def test_an_app_table_entry_that_cannot_be_read_falls_back_to_the_title(
            self, monkeypatch):
        import sys

        bogus = types.SimpleNamespace(APPS=[(), ("mask", "Mask")])
        monkeypatch.setitem(sys.modules, "spacr.qt.app", bogus)
        ns = types.SimpleNamespace(app_key="mask")
        assert AppScreen._screen_name(ns) == "Mask"
        ns = types.SimpleNamespace(app_key="unlisted_key")
        bogus.APPS = [()]
        assert AppScreen._screen_name(ns) == aps.APP_TITLES.get(
            "unlisted_key", "unlisted_key")


class TestSettingsForMyData:

    def _screen(self, why_not=""):
        ns = types.SimpleNamespace(
            _console=_Console(),
            the_advisor_can_run=lambda: why_not,
            _settings_model=types.SimpleNamespace(
                collect=lambda: {"dependent_variable": "y"}),
            _tables_for_the_advisor=lambda values: (["c.csv"], ["s.csv"]),
            _reading_with_the_last_run=lambda reading, values: reading,
            apply_settings_dict=lambda chosen: len(chosen))
        return ns

    def _patch(self, monkeypatch, run_note="", run_folder="", chosen=None):
        from spacr import settings_advisor

        reading = types.SimpleNamespace(
            trouble=["thin table"], run_note=run_note, run_folder=run_folder)
        monkeypatch.setattr(settings_advisor, "read_the_screen",
                            lambda counts, scores, dep: reading)
        monkeypatch.setattr(
            settings_advisor, "advise_that_runs",
            lambda r, answers: types.SimpleNamespace(
                as_settings=lambda: chosen or {}))

    def test_a_refusal_is_said_and_nothing_is_changed(self):
        ns = self._screen("No tables attached.")
        assert AppScreen.settings_for_my_data(ns) == {}
        assert ns._console.stdout == ["No tables attached.\n"]

    def test_answers_are_applied_and_counted(self, monkeypatch):
        self._patch(monkeypatch, run_note="Last run was thin.",
                    chosen={"penalty": 1})
        ns = self._screen()
        assert AppScreen.settings_for_my_data(
            ns, answers={"a": 1}) == {"penalty": 1}
        text = "".join(ns._console.stdout)
        assert "thin table" in text and "Last run was thin." in text
        assert "1 setting(s) written" in text

    def test_the_run_folder_is_named_when_there_is_no_note(self, monkeypatch):
        self._patch(monkeypatch, run_folder="/runs/a", chosen={})
        ns = self._screen()
        assert AppScreen.settings_for_my_data(ns, answers={}) == {}
        text = "".join(ns._console.stdout)
        assert "diagnostics of /runs/a" in text
        assert "never shown" in text

    def test_a_declined_proposal_changes_nothing(self, monkeypatch):
        from spacr.qt.widgets import settings_advisor_dialog

        self._patch(monkeypatch)

        class Declined:
            def __init__(self, *a, **k):
                pass

            def exec(self):
                return 0

        monkeypatch.setattr(settings_advisor_dialog, "SettingsAdvisorDialog",
                            Declined)
        ns = self._screen()
        assert AppScreen.settings_for_my_data(ns) == {}
        assert "Nothing was changed" in ns._console.stdout[-1]


class TestPermutationBanner:

    def test_a_count_that_is_not_a_number_is_reported_not_calculated(self):
        ns = types.SimpleNamespace(_console=_Console())
        note = AppScreen._say_what_the_permutation_will_do(
            ns, {"guide_permutations": "many"})
        assert "configuration is invalid" in note
        assert "'many'" in note
        assert "configuration is invalid" in ns._console.notices[-1]

    def test_a_count_below_one_is_reported_as_invalid(self):
        ns = types.SimpleNamespace(_console=_Console())
        note = AppScreen._say_what_the_permutation_will_do(
            ns, {"guide_permutations": 0})
        assert "guide_permutations=0" in note

    def test_a_valid_count_states_the_smallest_attainable_p_value(self):
        ns = types.SimpleNamespace(_console=_Console())
        note = AppScreen._say_what_the_permutation_will_do(
            ns, {"guide_permutations": 99})
        assert "99 within-plateID" in note and "0.01" in note


class TestGlyphWarmUp:

    def test_a_console_without_the_warm_up_is_left_alone(self):
        ns = types.SimpleNamespace(_console=None)
        assert AppScreen._warm_console_glyphs(ns) is None
        assert vars(ns) == {"_console": None}
        plain = types.SimpleNamespace(_console=types.SimpleNamespace())
        assert AppScreen._warm_console_glyphs(plain) is None
        assert vars(plain._console) == {}

    def test_a_console_that_raises_runtime_error_is_not_fatal(self):
        asked = []

        def refuse():
            asked.append(True)
            raise RuntimeError("deleted")

        assert AppScreen._warm_console_glyphs(types.SimpleNamespace(
            _console=types.SimpleNamespace(warm_notice_glyphs=refuse))) is None
        assert asked == [True]

    def test_a_deleted_qt_console_is_not_warmed(self):
        from PySide6.QtCore import QObject
        import shiboken6

        warmed = []

        class Console(QObject):
            def warm_notice_glyphs(self):
                warmed.append(True)

        console = Console()
        AppScreen._warm_console_glyphs(types.SimpleNamespace(_console=console))
        assert warmed == [True]
        shiboken6.delete(console)
        AppScreen._warm_console_glyphs(types.SimpleNamespace(_console=console))
        assert warmed == [True]


class TestSettingsPathErrors:

    def test_an_empty_traceback_is_not_a_settings_error(self):
        assert AppScreen._is_a_settings_path_error("", {}) is False
        assert AppScreen._is_a_settings_path_error("\n \n", {}) is False

    def test_a_configuration_error_always_is(self):
        assert AppScreen._is_a_settings_path_error(
            "Traceback\nConfigurationError: src is wrong", {}) is True

    def test_other_errors_are_not(self):
        assert AppScreen._is_a_settings_path_error(
            "Traceback\nValueError: no", {}) is False

    def test_a_missing_file_without_a_path_is_not(self):
        assert AppScreen._is_a_settings_path_error(
            "Traceback\nFileNotFoundError: gone", {}) is False

    def test_a_missing_file_inside_a_settings_folder_is(self):
        tb = ("Traceback\nFileNotFoundError: [Errno 2] No such file or "
              "directory: '/data/exp/plate1/img.tif'")
        settings = {"src": ["/data/exp", 3], "n": 4, "note": "x"}
        assert AppScreen._is_a_settings_path_error(tb, settings) is True

    def test_a_missing_file_elsewhere_is_a_fault(self):
        tb = ("Traceback\nFileNotFoundError: [Errno 2] No such file or "
              "directory: '/usr/lib/thing.so'")
        assert AppScreen._is_a_settings_path_error(
            tb, {"src": "/data/exp"}) is False


class TestClassifyingAndExplaining:

    def test_a_classifier_that_raises_makes_the_failure_a_fault(
            self, monkeypatch):
        from spacr import crashreport

        def refuse(traceback_text):
            raise RuntimeError("no")

        monkeypatch.setattr(crashreport, "classify_failure", refuse)
        ns = types.SimpleNamespace(
            _is_a_settings_path_error=AppScreen._is_a_settings_path_error,
            _settings_snapshot=lambda: {})
        assert AppScreen._kind_of_failure(ns, "tb") == crashreport.KIND_FAULT

    def test_a_settings_path_error_is_a_user_error(self):
        from spacr import crashreport

        ns = types.SimpleNamespace(
            _is_a_settings_path_error=lambda tb, s: True,
            _settings_snapshot=lambda: {})
        assert AppScreen._kind_of_failure(ns, "tb") == \
            crashreport.KIND_USER_ERROR

    def test_an_explanation_check_that_raises_is_no(self, monkeypatch):
        from spacr import crashreport

        def refuse(*a):
            raise RuntimeError("no")

        monkeypatch.setattr(crashreport, "mode_explains", refuse)
        assert AppScreen._the_failure_is_explained_after_the_run(
            types.SimpleNamespace()) is False

    def test_the_consoles_provider_wins_and_a_broken_lookup_is_none(self):
        chosen = object()
        ns = types.SimpleNamespace(_console=types.SimpleNamespace(
            _current_provider=lambda: chosen))
        assert AppScreen._report_summary_provider(ns) is chosen
        broken = types.SimpleNamespace(_console=None)
        assert AppScreen._report_summary_provider(broken) is None


class TestStoppingTheRefusalWatch:

    def test_unreadable_records_are_skipped_and_the_first_site_is_kept(self):
        import logging

        class Bad:
            def getMessage(self):
                raise ValueError("bad")

        class Good:
            spacr_call_site = ["a.py", 3]

            def getMessage(self):
                return "refused"

        class Problem:
            spacr_problems = ["p1"]

        handler = logging.NullHandler()
        listener = logging.NullHandler()
        ns = types.SimpleNamespace(
            _refusal_watch=(handler, [Bad(), Good()], listener, [Problem()]),
            _refusal_call_site=[], _settings_problems=[])
        assert AppScreen._stop_watching_refusals(ns) == ["refused"]
        assert ns._refusal_call_site == ["a.py", 3]
        assert ns._settings_problems == ["p1"]
        assert ns._refusal_watch is None
        assert AppScreen._stop_watching_refusals(ns) == []


class _Jobs:
    def __init__(self, outcomes):
        self.thread = None
        self.outcomes = outcomes

    def job(self):
        assert self.thread is not None
        self.thread.join(timeout=5)
        assert not self.thread.is_alive()
        return next((outcome for outcome in reversed(self.outcomes)
                     if not outcome.get("pending")), None)


class _ReportConsole(_Outcome):
    _run_function = "mask"

    def __init__(self, as_text_fails=False):
        super().__init__()
        self.begun = []
        self.filled = []
        self.as_text_fails = as_text_fails

    def as_text(self):
        if self.as_text_fails:
            raise RuntimeError("closed")
        return "log"

    def begin_ai_entry(self, thinking=True):
        self.begun.append(thinking)
        return "entry"

    def fill_ai_entry(self, entry, text):
        self.filled.append(text)


class TestSendingTheAnonymousReport:

    @pytest.fixture(autouse=True)
    def _owned_dispatch(self, monkeypatch, tmp_path):
        from spacr import crashreport

        self._outcomes = []
        self._jobs = _Jobs(self._outcomes)
        self._batcher = crashreport._ErrorBatcher()
        monkeypatch.setattr(crashreport, "_reports_dir", lambda: tmp_path / "reports")
        monkeypatch.setattr(crashreport, "_ERROR_BATCHER", self._batcher)
        monkeypatch.setattr(crashreport, "_ERROR_DISPATCH_SEEN", {})
        monkeypatch.setattr(crashreport, "_ERROR_OCCURRENCE_POLICIES", {})
        monkeypatch.setattr(crashreport, "_ERROR_POLICY_CACHE", {})
        yield
        if self._jobs.thread is not None:
            self._jobs.thread.join(timeout=5)
            assert not self._jobs.thread.is_alive()
        if self._batcher.timer is not None:
            self._batcher.timer.cancel()
            self._batcher.timer.join(timeout=5)
            assert not self._batcher.timer.is_alive()

    def _screen(self, console=None, started=None):
        jobs = self._jobs
        ns = types.SimpleNamespace(
            _last_error_text="Traceback\nValueError: x",
            _last_error_console_text="", _last_error_occurrence="",
            _last_error_problems=["a", "b"],
            _run_settings={"dry_run": True}, _run_started_at=started,
            _console=console or _ReportConsole(), _jobs=jobs, app_key="mask",
            _kind_of_failure=lambda tb: "fault",
            _on_anonymous_report_sent=lambda outcome: None,
            _screen_name=lambda: "Mask",
            _report_summary_provider=lambda: object())
        ns.jobs = jobs
        self._console = ns._console
        return ns

    def _patch(self, monkeypatch, **overrides):
        from spacr import crashreport
        from spacr.qt import preferences, terms
        from spacr.qt.ai import providers, settings

        calls = {"queued": [], "provider_requests": []}
        monkeypatch.setattr(crashreport, "network_allowed", lambda: True)
        terms.record_agreement()
        preferences.set_preferred_provider("claude")
        preferences.set_issue_prompt_mode("always")
        settings.set_route_errors_through_ai(True)
        settings.set_auto_file_issues(True)
        crashreport.set_error_report_mode("all")
        original_summary = crashreport.public_summary
        original_dispatch = crashreport._dispatch_error

        def public_summary(*args, **kwargs):
            calls["summary"] = dict(kwargs)
            return original_summary(*args, **kwargs)

        def dispatch(*args, **kwargs):
            self._jobs.thread = original_dispatch(*args, **kwargs)
            return self._jobs.thread

        def deliver(outcome):
            self._outcomes.append(dict(outcome))
            self._console._show_error_report_result(outcome)

        def enqueue(summary, **kwargs):
            calls["queued"].append(dict(summary))
            return None

        def send_queued(queued, summary, timeout, *, allowed):
            assert allowed()
            calls["submitted"] = {"status": "sent", "summary": dict(summary)}
            return {"status": "sent"}

        table = {"public_summary": public_summary,
                 "_dispatch_error": dispatch,
                 "_deliver_error_result": deliver,
                 "_enqueue": enqueue,
                 "_send_queued": send_queued}
        table.update(overrides)
        for name, fn in table.items():
            monkeypatch.setattr(crashreport, name, fn)
        def report_summary_result(summary, provider, timeout):
            calls["provider_requests"].append(dict(summary))
            return "why", ""

        monkeypatch.setattr(providers, "_report_summary_result", report_summary_result)
        return calls

    def test_nothing_is_sent_without_an_error_text(self):
        ns = self._screen()
        ns._last_error_text = ""
        assert AppScreen._send_the_anonymous_report(ns) is False

    def test_settings_that_cannot_be_read_send_nothing(self, monkeypatch):
        def refuse():
            raise RuntimeError("no")

        from spacr import crashreport

        calls = self._patch(monkeypatch, error_report_mode=refuse)
        assert AppScreen._send_the_anonymous_report(self._screen()) is False
        assert crashreport._ERROR_POLICY_CACHE["affirmed_policy"] == {}
        assert crashreport._automatic_error_policy("fault")[1:] == (False, False)
        assert self._jobs.thread is None and self._outcomes == []
        assert calls["queued"] == [] and "submitted" not in calls

    def test_neither_explaining_nor_sending_does_nothing(self, monkeypatch):
        self._patch(monkeypatch, mode_explains=lambda mode=None: False,
                    mode_sends=lambda kind, mode: False)
        assert AppScreen._send_the_anonymous_report(self._screen()) is False

    def test_a_saved_explanation_is_shown_and_travels_with_the_report(
            self, monkeypatch):
        calls = self._patch(monkeypatch, saved_explanation=lambda key: {
            "text": "kept", "tier": "full", "cause": "root"})
        console = _ReportConsole(as_text_fails=True)
        ns = self._screen(console, started=1.0)
        assert AppScreen._send_the_anonymous_report(ns) is True
        outcome = ns.jobs.job()
        assert console.begun == [False] and console.filled == ["kept"]
        assert outcome["explained"] is True and outcome["kind"] == "fault"
        assert outcome["ai_summary_attached"] is True
        sent = calls["submitted"]["summary"]
        assert sent["ai_summary"] == "kept" and sent["cause"] == "root"
        assert calls["provider_requests"] == []
        assert calls["summary"]["message"].startswith(
            "The run did not start: 2 settings problems.")
        assert not {"run", "run_settings", "log_text"} & calls["summary"].keys()
        assert not {"run", "run_settings", "log_text", "elapsed_s"} & sent.keys()
        assert ns._run_settings == {"dry_run": True} and ns._run_started_at == 1.0
        assert ns._last_error_text == "Traceback\nValueError: x"

    def test_a_saved_lookup_that_raises_asks_the_ai_afresh(self, monkeypatch):
        def refuse(key):
            raise RuntimeError("disk")

        calls = self._patch(monkeypatch, saved_explanation=refuse)
        ns = self._screen()
        assert AppScreen._send_the_anonymous_report(ns) is True
        outcome = ns.jobs.job()
        assert ns._console.begun == [True]
        assert outcome["explanation"]["text"] == "why"
        assert calls["submitted"]["summary"]["ai_summary"] == "why"
        assert len(calls["provider_requests"]) == 1

    def test_an_ai_that_raises_is_recorded_and_the_report_still_goes(
            self, monkeypatch):
        from spacr.qt.ai import providers

        self._patch(monkeypatch)

        def refuse(summary, provider):
            raise RuntimeError("offline")

        monkeypatch.setattr(providers, "explain_failure", refuse)
        ns = self._screen()
        AppScreen._send_the_anonymous_report(ns)
        outcome = ns.jobs.job()
        assert outcome["explanation"]["refused"] == "error"
        assert outcome["ai_summary_attached"] is False

    def test_a_summary_that_cannot_be_built_stays_local_without_a_queue_claim(
            self, monkeypatch):
        def refuse(**kw):
            raise OSError("no disk")

        calls = self._patch(monkeypatch, public_summary=refuse)
        ns = self._screen()
        AppScreen._send_the_anonymous_report(ns)
        assert ns.jobs.job() is None
        assert calls["queued"] == [] and "submitted" not in calls
        assert self._outcomes == [] and ns._console.notices == []
        assert ns._last_error_text == "Traceback\nValueError: x"


class TestSmallHelpers:

    def test_a_host_that_is_gone_has_no_arrivals_to_sweep(self):
        asked, arrived = [], []

        class Gone:
            def children(self):
                asked.append(True)
                raise RuntimeError("deleted")

        translator = types.SimpleNamespace(_on_arrival=arrived.append)
        assert aps._LateCaptionTranslator._on_arrivals_in(
            translator, Gone()) is None
        assert asked == [True] and arrived == []

    def test_a_viewport_with_no_split_has_no_floor_to_sync(self):
        ns = types.SimpleNamespace(_split=None)
        assert aps._RuntimeViewport._sync_floor(ns) is None
        assert vars(ns) == {"_split": None}

    def test_a_catalogue_that_cannot_be_read_means_no_animation(
            self, monkeypatch):
        import sys

        broken = types.ModuleType("spacr.qt.setting_animations")
        monkeypatch.setitem(sys.modules, "spacr.qt.setting_animations", broken)
        assert aps._setting_has_an_animation("anything") is False
        assert aps._setting_has_an_animation("") is False

    def test_the_fractal_is_paused_backdrop_by_backdrop(self, monkeypatch):
        class Good:
            def pause(self):
                return True

        class Bad:
            def pause(self):
                raise RuntimeError("gone")

        monkeypatch.setattr(aps, "_each_fractal_backdrop",
                            lambda screen: iter([Good(), Bad(), Good()]))
        assert aps._pause_the_fractal(object()) == 2

    def test_the_stretch_index_is_the_spring_or_the_end(self, qtbot):
        from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

        host = QWidget()
        qtbot.addWidget(host)
        layout = QVBoxLayout(host)
        layout.addWidget(QLabel("a"))
        assert AppScreen._index_before_the_stretch(layout) == 1
        layout.addStretch(1)
        assert AppScreen._index_before_the_stretch(layout) == 1

    def test_the_object_grid_skips_spacers_and_falls_back_to_the_stretch(
            self, qtbot):
        from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

        host = QWidget()
        qtbot.addWidget(host)
        layout = QVBoxLayout(host)
        layout.addStretch(1)
        layout.addWidget(QLabel("plain"))
        ns = types.SimpleNamespace(
            OBJECT_GRID_BEFORE="Cell Segmentation",
            _index_before_the_stretch=AppScreen._index_before_the_stretch)
        assert AppScreen._index_for_the_object_grid(ns, layout) == 0
        heading = QLabel("heading")
        heading.setProperty("settingsCategorySource", "Cell Segmentation")
        layout.insertWidget(0, heading)
        assert AppScreen._index_for_the_object_grid(ns, layout) == 0

    def test_an_ambient_whose_shutdown_raises_is_still_removed(self):
        def refuse():
            raise RuntimeError("busy")

        widget = types.SimpleNamespace(
            shutdown=refuse, set_animating=lambda on: None,
            deleteLater=lambda: None, setParent=lambda p: None,
            hide=lambda: None)
        ns = types.SimpleNamespace(
            _ambient=widget, _ambient_applied=1,
            _sync_page_palette=lambda: None, update=lambda: None)
        AppScreen._remove_ambient(ns)
        assert ns._ambient is None and ns._ambient_applied is None


class TestPickersAndChoosers:

    def test_closing_the_plate_map_leaves_the_field_alone(self, monkeypatch):
        from spacr.qt.widgets import plate_map_picker

        class Closed:
            def __init__(self, before, parent=None):
                pass

            def exec(self):
                return 0

        monkeypatch.setattr(plate_map_picker, "PlateMapPicker", Closed)
        field = types.SimpleNamespace(text=lambda: "A1")
        assert AppScreen.pick_wells_for(types.SimpleNamespace(), field) == ""

    @pytest.mark.parametrize("route,called", [
        ("", None), ("stream", "measure"), ("crops", "annotate")])
    def test_the_test_data_route_decides_which_example_is_fetched(
            self, route, called):
        seen = []
        ns = types.SimpleNamespace(
            load_the_measure_example=lambda ask=None: seen.append("measure")
            or {"m": 1},
            load_the_annotate_example=lambda ask=None: seen.append("annotate")
            or {"a": 1})
        chooser = types.SimpleNamespace(chosen=route, exec=lambda: 0)
        out = AppScreen.choose_the_test_data(ns, chooser=chooser)
        assert seen == ([called] if called else [])
        assert out == ({} if not called else {"m": 1} if called == "measure"
                       else {"a": 1})
        if route:
            assert ns._test_data_route == route


class TestOwedParts:

    def test_a_step_that_raises_is_logged_and_dropped(self):
        def steps():
            raise ValueError("a step broke")
            yield

        ns = types.SimpleNamespace()
        ns.__dict__["_parts_stepping"] = {"p": steps()}
        AppScreen._step_owed_part(ns, "p")
        assert "p" not in ns._parts_stepping


class TestHeadingsAndSplitters:

    def test_headings_whose_sections_cannot_be_listed_count_as_shown(
            self, monkeypatch):
        from spacr.qt.widgets import section

        def refuse(_section):
            raise RuntimeError("detached")

        monkeypatch.setattr(section, "_sections_below", refuse)
        assert AppScreen._every_heading_below_is_hidden(object(), set()) \
            is False

    def test_a_saved_splitter_state_is_restored(self, monkeypatch, qtbot):
        from PySide6.QtWidgets import QSplitter, QWidget
        from spacr.qt.widgets import console_panel

        splitter = QSplitter()
        qtbot.addWidget(splitter)
        for _ in range(2):
            splitter.addWidget(QWidget())
        state = splitter.saveState()
        restored = []
        monkeypatch.setattr(console_panel, "get_split_state",
                            lambda key: state)
        monkeypatch.setattr(console_panel, "set_split_state",
                            lambda key, value: None)
        monkeypatch.setattr(splitter, "restoreState",
                            lambda value: restored.append(value) or True)
        ns = types.SimpleNamespace(app_key="mask",
                                   RUNTIME_SPLIT_SUFFIX=".runtime",
                                   _runtime_splitter=None)
        AppScreen._remember_runtime_splitter(ns, splitter)
        assert restored == [state]


class TestRegressionTabs:

    def test_no_paired_data_widget_means_no_databases(self):
        ns = types.SimpleNamespace(
            _settings_model=types.SimpleNamespace(_widgets={}))
        assert AppScreen._attached_database_rows(ns) == []

    def test_opening_the_runs_tab_reloads_the_sweep_destination(self):
        loaded = []
        runs = types.SimpleNamespace(load=lambda folder: loaded.append(folder))
        tabs = types.SimpleNamespace(widget=lambda index: runs)
        ns = types.SimpleNamespace(
            _results_tabs=tabs, _sweep_runs=runs,
            _sweep_destination=lambda: "/runs", _cell_montage=None,
            _scan_panel=None)
        AppScreen._on_results_tab_changed(ns, 3)
        assert loaded == ["/runs"]

    def test_a_selected_guide_opens_a_collapsed_gene_tile(self):
        calls = []

        class Split:
            def sizes(self):
                return [100, 0]

            def is_collapsed(self, name):
                return True

            def set_collapsed(self, name, collapsed, by_user):
                calls.append((name, collapsed, by_user))

            def setSizes(self, sizes):
                calls.append(sizes)

        ns = types.SimpleNamespace(
            _show_regression_graph=lambda: None, _gene_split=Split())
        AppScreen._on_guide_selected(ns, "guide")
        assert calls[0] == ("Gene", False, False)
        assert ns._gene_opened is True


class TestUsageAndDialogs:

    def test_a_gpu_reading_comes_from_gputil_when_nvml_has_none(
            self, monkeypatch):
        import sys

        gpu = types.SimpleNamespace(load=0.5, memoryUtil=0.25)
        fake = types.ModuleType("GPUtil")
        fake.getGPUs = lambda: [gpu]
        monkeypatch.setitem(sys.modules, "GPUtil", fake)
        monkeypatch.setattr(aps, "_nvidia_smi_available", lambda: True)
        monkeypatch.setattr(aps, "_nvml_gpu_usage", lambda: None)
        sample = aps._sample_usage(False)
        assert sample["gpu"] == 50.0 and sample["vram"] == 25.0
        fake.getGPUs = lambda: []
        sample = aps._sample_usage(False)
        assert sample["gpu"] == 0 and sample["vram"] == 0

        def refuse():
            raise RuntimeError("no driver")

        fake.getGPUs = refuse
        assert "gpu" not in aps._sample_usage(False)

    def test_the_ram_guard_dialog_returns_the_choice(self, monkeypatch):
        class Fake(aps._RamGuardDialog):
            def __init__(self, parent, plan):
                self.choice = "use"

            def exec(self):
                return 0

        assert Fake.ask(None, {}) == "use"

    def test_the_free_ram_dialog_says_when_nothing_can_be_closed(
            self, qtbot, monkeypatch):
        from spacr import resource_log

        monkeypatch.setattr(resource_log, "_closable_processes",
                            lambda psutil_module=None: [])
        dialog = aps._FreeRamDialog()
        qtbot.addWidget(dialog)
        assert dialog.boxes == []
