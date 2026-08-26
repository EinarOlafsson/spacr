"""Toggles, layout memory and the run's own reporting, on a partial screen.

Every card on a module screen is optional: the sweep, the live preview, the
UMAP explorer, the hyperparameter panel, the runtime splitter, the two hint
strips. Which of them exists depends on the module, and a toggle wired to a
card that this module does not build is the ordinary case rather than a
mistake.

So each handler here answers the same question -- is the thing I am about to
move actually here? -- and the ones that seed a card from the form have the
second obligation: seeding is a convenience, and failing to seed must not stop
the card from opening.
"""
from __future__ import annotations

import os
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QSplitter, QWidget                 # noqa: E402

from spacr.qt.screens.app_screen import AppScreen                # noqa: E402

pytestmark = pytest.mark.qt


def _boom(*_args, **_kwargs):
    raise RuntimeError("the panel is half built")


@pytest.fixture
def screen(qtbot):
    widget = AppScreen("regression")
    qtbot.addWidget(widget)
    return widget


class TestRememberingTheRuntimeSplit:

    def test_a_screen_with_no_splitter_remembers_nothing(self, screen):
        screen._remember_runtime_splitter(None)

        assert screen._runtime_splitter is None

    def test_a_layout_store_that_is_not_there_leaves_the_default_split(
            self, screen, monkeypatch, qtbot):
        """The default split is the right fallback, not an exception."""
        import builtins

        real_import = builtins.__import__

        def refuse(name, *args, **kwargs):
            if name.endswith("console_panel"):
                raise ImportError("no layout store")
            return real_import(name, *args, **kwargs)

        splitter = QSplitter()
        qtbot.addWidget(splitter)
        monkeypatch.setattr(builtins, "__import__", refuse)

        screen._remember_runtime_splitter(splitter)

        assert screen._runtime_splitter is splitter

    def test_a_saved_blob_from_an_older_layout_restores_nothing(
            self, screen, monkeypatch, qtbot):
        """A state Qt cannot read is not a reason to refuse to draw."""
        from spacr.qt.widgets import console_panel

        splitter = QSplitter()
        qtbot.addWidget(splitter)
        monkeypatch.setattr(console_panel, "get_split_state", _boom)

        screen._remember_runtime_splitter(splitter)

        assert screen._runtime_splitter is splitter

    def test_a_split_that_cannot_be_saved_does_not_break_the_drag(
            self, screen, monkeypatch, qtbot):
        """The handle is being dragged; raising here happens per mouse move."""
        from spacr.qt.widgets import console_panel

        splitter = QSplitter()
        qtbot.addWidget(splitter)
        splitter.addWidget(QWidget())
        splitter.addWidget(QWidget())
        monkeypatch.setattr(console_panel, "get_split_state", lambda _k: None)
        monkeypatch.setattr(console_panel, "set_split_state", _boom)
        screen._remember_runtime_splitter(splitter)

        splitter.splitterMoved.emit(10, 1)


class TestTheHintStrips:

    def test_a_screen_with_no_hint_strip_reserves_nothing(self, screen,
                                                           monkeypatch):
        monkeypatch.setattr(screen, "_hint_strip", None, raising=False)
        monkeypatch.setattr(screen, "_category_hint", None, raising=False)

        screen._sync_hint_strip_height()
        screen._sync_category_hint_height()
        screen.show_category_hint("Paths")
        screen.clear_category_hint()

    def test_a_pinned_category_comes_back_when_the_pointer_leaves(self,
                                                                   screen):
        """The expanded category is what the strip falls back to."""
        screen._category_hint_pinned = "Paths"

        screen.clear_category_hint()

        assert "PATHS" in screen._category_hint.text().upper()


class TestTheOptionalCards:

    def test_a_module_with_no_live_preview_card_ignores_the_toggle(self,
                                                                    screen,
                                                                    monkeypatch):
        monkeypatch.setattr(screen, "_live_preview_card", None, raising=False)

        screen._on_lp_switch(True)

    def test_a_module_with_no_preview_card_ignores_the_toggle(self, screen,
                                                               monkeypatch):
        monkeypatch.setattr(screen, "_preview_card_attr", "", raising=False)

        screen._on_preview_switch(True)

    def test_a_module_with_no_sweep_card_ignores_the_toggle(self, screen,
                                                             monkeypatch):
        monkeypatch.setattr(screen, "_sweep_card", None, raising=False)

        screen._on_sweep_switch(True)

    def test_opening_the_sweep_seeds_it_from_the_form(self, screen,
                                                       monkeypatch, qtbot):
        """It starts from the user's inputs rather than defaults they retype."""
        card = QWidget()
        qtbot.addWidget(card)
        seeded = []
        monkeypatch.setattr(screen, "_sweep_card", card, raising=False)
        monkeypatch.setattr(screen, "_sweep", types.SimpleNamespace(
            apply_settings=seeded.append), raising=False)

        screen._on_sweep_switch(True)

        assert len(seeded) == 1 and isinstance(seeded[0], dict)
        assert not card.isHidden()

    def test_a_sweep_that_will_not_be_seeded_still_opens(self, screen,
                                                          monkeypatch, qtbot):
        """Seeding is a convenience; the card is the control."""
        card = QWidget()
        qtbot.addWidget(card)
        monkeypatch.setattr(screen, "_sweep_card", card, raising=False)
        monkeypatch.setattr(screen, "_sweep",
                            types.SimpleNamespace(apply_settings=_boom),
                            raising=False)

        screen._on_sweep_switch(True)

        assert not card.isHidden()

    def test_a_module_that_is_not_the_image_umap_keeps_one_gpu_answer(
            self, screen, monkeypatch):
        """The switch is the anchor, and only the UMAP screen has one."""
        monkeypatch.setattr(screen, "_hyperparam",
                            types.SimpleNamespace(apply_settings=_boom),
                            raising=False)

        screen._on_umap_gpu_switch(True)

    def test_a_module_with_no_explorer_ignores_the_interactive_toggle(
            self, screen, monkeypatch):
        monkeypatch.setattr(screen, "_umap_explorer", None, raising=False)

        screen._on_interactive_switch(True)

    def test_a_static_figure_with_no_payload_flips_no_switch(self, screen,
                                                              monkeypatch):
        """Clicking a figure that carries no embedding must not open an
        empty explorer."""
        said = []
        screen._console = types.SimpleNamespace(append_notice=said.append)
        monkeypatch.setattr(screen, "_umap_payload_ready", True,
                            raising=False)
        monkeypatch.setattr(screen, "_interactive_switch", None, raising=False)

        screen._on_static_figure_clicked()

        assert said == []


class TestTheAiSwitchDefault:

    def test_a_preference_that_cannot_be_read_leaves_the_switch_alone(
            self, screen, monkeypatch):
        """A preference is not a reason for a module screen to fail to build."""
        from spacr.qt import preferences

        monkeypatch.setattr(preferences, "get_ai_on_by_default", _boom)
        monkeypatch.setattr(screen, "_ai_switch",
                            types.SimpleNamespace(setChecked=_boom),
                            raising=False)

        screen._apply_ai_default()

    def test_a_switch_that_will_not_take_it_is_not_fatal_either(self, screen,
                                                                 monkeypatch):
        from spacr.qt import preferences

        monkeypatch.setattr(preferences, "get_ai_on_by_default", lambda: True)
        monkeypatch.setattr(screen, "_ai_switch",
                            types.SimpleNamespace(setChecked=_boom),
                            raising=False)

        screen._apply_ai_default()


class TestTheConsoleAndPreferencesButtons:

    def test_a_console_that_will_not_be_copied_says_so_in_the_console(
            self, screen, monkeypatch):
        """A clipboard write is silent; a button that seems inert reads broken."""
        errors = []
        screen._console = types.SimpleNamespace(
            copy_all=_boom, append_error=errors.append)

        screen._on_copy_console()

        assert errors and "Could not copy the console" in errors[0]

    def test_preferences_that_will_not_open_is_logged_not_raised(self, screen,
                                                                  monkeypatch,
                                                                  caplog):
        from spacr.qt import preferences

        monkeypatch.setattr(preferences, "PreferencesDialog", _boom)

        with caplog.at_level("ERROR"):
            screen._open_preferences_dialog()

        assert any("could not open Preferences" in record.message
                   for record in caplog.records)

    def test_settings_that_will_not_collect_stop_a_remote_submit(
            self, screen, monkeypatch):
        """A snapshot that could not be read must not be sent to a cluster."""
        from PySide6.QtWidgets import QMessageBox

        warned = []
        monkeypatch.setattr(QMessageBox, "warning",
                            staticmethod(lambda *args, **kwargs: warned.append(args)))
        monkeypatch.setattr(screen._settings_model, "collect", _boom)
        sent = []
        screen.remote_submit_requested.connect(lambda *a: sent.append(a))

        screen._on_remote_submit()

        assert warned and sent == []


class TestTheHeartbeat:

    def test_a_run_that_has_not_recorded_a_start_says_nothing(self, screen,
                                                               monkeypatch):
        said = []
        screen._console = types.SimpleNamespace(append_notice=said.append)
        monkeypatch.setattr(screen, "_thread", types.SimpleNamespace(
            isRunning=lambda: True))
        monkeypatch.setattr(screen, "_run_started_at", None, raising=False)

        screen._on_heartbeat()

        assert said == []

    def test_a_mark_that_has_already_been_spoken_is_not_repeated(self, screen,
                                                                  monkeypatch):
        """One line per mark; the schedule is what makes it readable."""
        import time as _time

        said = []
        screen._console = types.SimpleNamespace(append_notice=said.append)
        monkeypatch.setattr(screen, "_thread", types.SimpleNamespace(
            isRunning=lambda: True))
        monkeypatch.setattr(screen, "_run_started_at", _time.time(),
                            raising=False)

        screen._on_heartbeat()

        assert said == []

    def test_an_ordinary_run_says_only_how_long_it_has_been_going(
            self, screen, monkeypatch):
        """The single-core sentence belongs to a fit, not to every module."""
        import time as _time

        said = []
        screen._console = types.SimpleNamespace(
            append_notice=lambda text, **kwargs: said.append((text, kwargs)))
        monkeypatch.setattr(screen, "_thread", types.SimpleNamespace(
            isRunning=lambda: True))
        monkeypatch.setattr(screen, "_run_started_at",
                            _time.time() - 400.0, raising=False)
        monkeypatch.setattr(screen, "_slow_fit", False, raising=False)

        screen._on_heartbeat()

        assert len(said) == 1
        assert "Still running" in said[0][0]
        assert said[0][1]["elapsed"] == "6 min 40 s"

    def test_a_slow_fit_explains_why_one_core_is_at_a_hundred_percent(
            self, screen, monkeypatch):
        """"cpu at 100 percent" was the observation the user could not read."""
        import time as _time

        said = []
        screen._console = types.SimpleNamespace(
            append_notice=lambda text, **kwargs: said.append(text))
        monkeypatch.setattr(screen, "_thread", types.SimpleNamespace(
            isRunning=lambda: True))
        monkeypatch.setattr(screen, "_run_started_at",
                            _time.time() - 400.0, raising=False)
        monkeypatch.setattr(screen, "_slow_fit", True, raising=False)

        screen._on_heartbeat()

        assert "single-threaded optimisation" in said[0]


class TestReportingTheDesignThatWasRead:

    def test_a_scan_that_returns_nothing_usable_says_nothing(self, screen):
        said = []
        screen._console = types.SimpleNamespace(append_notice=said.append)

        screen._on_design_scanned(None)
        screen._on_design_scanned("not a design")

        assert said == []

    def test_a_partial_design_reports_what_it_could_read_and_why_not_more(
            self, screen):
        """Half an answer with its reason beats no answer at all."""
        said = []
        screen._console = types.SimpleNamespace(
            append_notice=lambda text, **kwargs: said.append((text, kwargs)))

        screen._on_design_scanned(
            {"guides": 4096, "rows": 50000, "note": "no gene column"})

        assert "distinct gRNAs" in said[0][0]
        assert said[0][1]["why"] == "no gene column"

    def test_a_design_scan_that_cannot_be_submitted_is_not_fatal(self, screen,
                                                                  monkeypatch):
        """The scan is a sentence beside the run, not part of it."""
        monkeypatch.setattr(screen._jobs, "submit", _boom)

        screen._announce_the_fit({"src": "/data"})


class TestOpeningARowFromTheRunsTab:

    def test_a_row_that_is_not_a_record_opens_nothing(self, screen):
        screen._show_trial("not a record")

    def test_a_screen_with_no_results_panel_opens_no_row(self, screen,
                                                          monkeypatch):
        said = []
        screen._console = types.SimpleNamespace(append_stdout=said.append)
        monkeypatch.setattr(screen, "_results_panel", None)

        screen._show_trial({"run": "run 3", "status": "ok",
                            "folder": "/data/run"})

        assert said == []

    def test_a_result_arriving_without_a_payload_changes_nothing(self,
                                                                  screen):
        screen._on_pipeline_result(None)
        screen._on_pipeline_result("not a payload")

    def test_a_trial_answer_with_nothing_pending_is_ignored(self, screen,
                                                             monkeypatch):
        """The read is on a worker; its answer can outlive what asked for it."""
        monkeypatch.setattr(screen, "_pending_trial", None, raising=False)

        screen._on_trial_loaded(True)


class TestTheRemainingOptionalParts:

    def test_a_screen_missing_one_caption_host_still_watches_the_others(
            self, screen, monkeypatch):
        """Three hosts, and not every module builds all three.

        The pass is what translates a panel that arrives after the screen;
        skipping the whole install because one host is absent would leave a
        folded module's captions in English for the session.
        """
        monkeypatch.setattr(screen, "_runtime_wrap", None, raising=False)
        monkeypatch.setattr(screen, "_body_splitter", None, raising=False)

        screen._watch_for_late_captions()

        assert screen._late_caption_watcher is not None

    def test_the_live_preview_card_follows_its_toggle(self, screen,
                                                       monkeypatch, qtbot):
        card = QWidget()
        qtbot.addWidget(card)
        card.hide()
        monkeypatch.setattr(screen, "_live_preview_card", card, raising=False)

        screen._on_lp_switch(True)
        assert not card.isHidden()

        screen._on_lp_switch(False)
        assert card.isHidden()

    def test_the_ai_switch_is_left_off_when_the_preference_says_so(
            self, screen, monkeypatch):
        """Off by default means untouched, not explicitly unchecked."""
        from spacr.qt import preferences

        monkeypatch.setattr(preferences, "get_ai_on_by_default",
                            lambda: False)
        monkeypatch.setattr(screen, "_ai_switch",
                            types.SimpleNamespace(setChecked=_boom),
                            raising=False)

        screen._apply_ai_default()

    def test_a_console_that_will_not_take_the_explanation_still_switches(
            self, screen, monkeypatch):
        """The view changing under the user is the thing worth explaining.

        Failing to explain it is a smaller defect than leaving the click
        with no effect at all.
        """
        switched = []
        monkeypatch.setattr(screen, "_umap_payload_ready", True,
                            raising=False)
        monkeypatch.setattr(screen, "_interactive_switch",
                            types.SimpleNamespace(
                                isChecked=lambda: False,
                                setChecked=switched.append),
                            raising=False)
        screen._console = types.SimpleNamespace(append_notice=_boom)

        screen._on_static_figure_clicked()

        assert switched == [True]

    def test_a_finished_run_with_no_results_panel_still_records_its_row(
            self, screen, monkeypatch):
        """The row is what makes the Runs tab navigable, panel or no panel."""
        recorded = {}
        monkeypatch.setattr(screen, "_update_run_in_runs_tab",
                            lambda **fields: recorded.update(fields) or True)
        monkeypatch.setattr(screen, "_say_the_qc_verdict", lambda _p: "")
        monkeypatch.setattr(screen, "_results_panel", None)

        screen._on_pipeline_result({"res_folder": "/data/run", "results": None})

        assert recorded["folder"] == "/data/run"


class TestFilingAnIssueAboutTheLastError:

    @staticmethod
    def _preview(monkeypatch, accepted, report=None):
        """Stand in for the consent dialog, which is the boundary being tested."""
        from PySide6.QtWidgets import QDialog
        from spacr.qt.ai import issue_preview

        class Preview:
            def __init__(self, built, parent=None):
                self.built = built

            def exec(self):
                return QDialog.Accepted if accepted else QDialog.Rejected

            def approved_report(self):
                return report if report is not None else self.built

        monkeypatch.setattr(issue_preview, "IssuePreviewDialog", Preview)

    def test_a_cancelled_preview_sends_nothing_and_says_so(self, screen,
                                                            monkeypatch):
        """The preview IS the consent boundary; closing it is a refusal."""
        from spacr.qt import preferences
        from spacr.qt.ai import issue_report

        said = []
        screen._console = types.SimpleNamespace(append_notice=said.append)
        screen._last_error_text = "Traceback: cellpose exploded"
        monkeypatch.setattr(preferences, "get_issue_prompt_mode",
                            lambda: "ask")
        monkeypatch.setattr(issue_report, "submit_report",
                            lambda _r: pytest.fail("a report was sent"))
        self._preview(monkeypatch, accepted=False)

        screen._on_file_issue()

        assert any("cancelled" in line for line in said)

    def test_an_approved_report_carries_the_settings_that_were_running(
            self, screen, monkeypatch, qtbot):
        """A crash report without the values that caused it is half a report.

        The class_metadata crash arrived that way -- every list setting was
        missing, because only the text boxes were read.
        """
        from PySide6.QtWidgets import QLineEdit
        from spacr.qt import preferences
        from spacr.qt.ai import issue_report

        plain = QLineEdit()
        qtbot.addWidget(plain)
        plain.setText("/data/plate1")
        monkeypatch.setitem(screen._settings_model._widgets,
                            "a_plain_text_setting", plain)

        screen._console = types.SimpleNamespace(append_notice=lambda *a, **k: None)
        screen._last_error_text = "Traceback: cellpose exploded"
        monkeypatch.setattr(preferences, "get_issue_prompt_mode",
                            lambda: "ask")
        built = {}
        monkeypatch.setattr(issue_report, "build_report",
                            lambda text, **kwargs: built.update(kwargs) or {"body": text})
        sent = []
        monkeypatch.setattr(issue_report, "submit_report",
                            lambda report: sent.append(report) or "https://x/1")
        self._preview(monkeypatch, accepted=True)
        monkeypatch.setattr(screen._jobs, "submit",
                            lambda work, done: done(work()))

        screen._on_file_issue()

        assert built["settings"]["a_plain_text_setting"] == "/data/plate1"
        assert sent == [{"body": "Traceback: cellpose exploded"}]

    def test_a_report_that_will_not_send_is_carried_back_as_a_failure(
            self, screen, monkeypatch):
        """Once the call is asynchronous no ``except`` around it can see it.

        A report that silently fails to send is worse than one that fails
        loudly, so the failure travels back as data.
        """
        from spacr.qt import preferences
        from spacr.qt.ai import issue_report

        said = []
        screen._console = types.SimpleNamespace(
            append_notice=lambda text, **kwargs: said.append((text, kwargs)))
        screen._last_error_text = "Traceback: cellpose exploded"
        monkeypatch.setattr(preferences, "get_issue_prompt_mode",
                            lambda: "ask")
        monkeypatch.setattr(issue_report, "build_report",
                            lambda text, **kwargs: {"body": text})
        monkeypatch.setattr(issue_report, "submit_report", _boom)
        self._preview(monkeypatch, accepted=True)
        monkeypatch.setattr(screen._jobs, "submit",
                            lambda work, done: done(work()))

        screen._on_file_issue()

        assert any("auto-file failed" in text for text, _ in said)


class TestBuildingTheResultsSideOfTheScreen:

    def test_a_saved_fold_layout_that_cannot_be_restored_still_gives_the_tab(
            self, qtbot, monkeypatch):
        """The layout is a remembered convenience; the panel is the module.

        It is restored after the last section is added rather than in the
        panel's constructor, so this runs late enough to take a whole screen
        down with it if it were not caught.
        """
        from spacr.qt.widgets import measurement_scan_panel as scan_module

        monkeypatch.setattr(
            scan_module.MeasurementScanPanel, "restore_section_layout", _boom)

        built = AppScreen("regression")
        qtbot.addWidget(built)

        assert built._scan_panel is not None
        assert built._results_tabs is not None

    def test_a_results_side_that_will_not_build_falls_back_to_the_queue(
            self, qtbot, monkeypatch):
        """The run still produces figures, and they still have to be shown.

        The whole left half is built as one unit -- table, tabs, figure
        stack and the Cells view -- so any part of it failing leaves the
        screen with no results panel at all. Without the fallback there
        would be nowhere to put the pictures a finished run streams in.
        """
        from spacr.qt.widgets import cell_montage_view

        monkeypatch.setattr(cell_montage_view, "CellMontageView", _boom)

        built = AppScreen("regression")
        qtbot.addWidget(built)

        assert built._results_panel is None
        assert built._figures_stack is None
        assert built._cell_montage is None
        assert built._figure_queue.parent() is not None
