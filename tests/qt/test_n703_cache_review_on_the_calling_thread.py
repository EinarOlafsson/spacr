"""The cache review's background steps, run on the calling thread.

``tests/qt/test_n703_cache_review_workers.py`` drives the real QThreads.
Coverage cannot trace code a QThread runs, so the preview, the guard and the
apply step are captured here as the closures the window hands its worker and
called directly; the window's handlers are the real ``MainWindow`` methods.
The same file covers the update-plan and backdrop guards a closing or
half-built window takes.
"""
from __future__ import annotations

import logging

import pytest
from PySide6.QtWidgets import QMainWindow, QWidget

from spacr import install_cleanup, run_journal
from spacr.qt import app as qt_app
from spacr.qt import install_consent as consent

pytestmark = pytest.mark.qt


class _Window(QMainWindow):
    """A bare window carrying the real cache-review and update handlers."""

    _review_installation_caches = qt_app.MainWindow._review_installation_caches
    _on_cache_cleanup_finished = qt_app.MainWindow._on_cache_cleanup_finished
    _on_cache_cleanup_preview_ready = qt_app.MainWindow._on_cache_cleanup_preview_ready
    _on_cache_cleanup_applied = qt_app.MainWindow._on_cache_cleanup_applied
    _finish_cache_cleanup_review = qt_app.MainWindow._finish_cache_cleanup_review
    _on_update_plan_prepared = qt_app.MainWindow._on_update_plan_prepared
    _save_pip_update_restart = qt_app.MainWindow._save_pip_update_restart

    def __init__(self):
        super().__init__()
        self._closing = False
        self.started = []

    def _start_cache_cleanup_worker(self, operation, fn):
        """Record the step instead of starting a QThread."""
        self.started.append((operation, fn))


@pytest.fixture
def window(qtbot, monkeypatch):
    widget = _Window()
    qtbot.addWidget(widget)
    shown = {"warnings": [], "notices": []}
    monkeypatch.setattr(qt_app.QMessageBox, "warning", staticmethod(
        lambda _parent, title, text, *_args: shown["warnings"].append((title, text))))
    monkeypatch.setattr(qt_app.QMessageBox, "exec",
                        lambda box: shown["notices"].append(box.text()) or qt_app.QMessageBox.Ok)
    widget.shown = shown
    return widget


def test_a_closing_window_starts_no_review(window):
    window._closing = True
    window._review_installation_caches()
    assert window.started == []


def test_a_second_review_waits_for_the_first(window):
    window._cache_cleanup_busy = True
    window._review_installation_caches()
    assert window.started == []
    assert window.statusBar().currentMessage() == "A cache review is already running."


def test_the_preview_runs_under_the_run_lock_and_reports_a_busy_cache(window, monkeypatch, tmp_path):
    busy, idle = tmp_path / "torch", tmp_path / "news"
    held = []

    def refusal(folder):
        held.append(run_journal._ACTIVE_RUNS_LOCK._is_owned())
        return "a run is using it" if folder == busy else None

    def preview(operation, *, guard):
        with guard([str(idle), str(busy)]) as reason:
            return {"operation_id": operation, "folders": [], "refusal": reason}

    monkeypatch.setattr(run_journal, "_cache_refusal", refusal)
    monkeypatch.setattr(install_cleanup, "preview_optional_cleanup", preview)
    window._review_installation_caches()
    assert window._cache_cleanup_busy is True
    [(operation, prepare)] = window.started
    assert operation == "preview"
    result = prepare()
    assert result["refusal"] == "a run is using it"
    assert len(result["operation_id"]) == 32
    assert held == [True, True]
    assert not run_journal._ACTIVE_RUNS_LOCK._is_owned()


def test_a_finish_from_a_retired_worker_is_ignored(window):
    window._cache_cleanup_worker = object()
    window._cache_cleanup_busy = True
    window._on_cache_cleanup_finished()
    assert window._cache_cleanup_busy is True
    assert window.shown["warnings"] == []


def test_a_review_dialog_that_cannot_open_is_reported_and_released(window, monkeypatch, caplog):
    def broken(preview, parent):
        raise RuntimeError("the dialog could not be built")

    monkeypatch.setattr(consent, "_choose_optional_cleanup", broken)
    window._cache_cleanup_busy = True
    with caplog.at_level(logging.ERROR, logger=qt_app.LOG.name):
        window._on_cache_cleanup_preview_ready({"operation_id": "op", "folders": []})
    assert "Could not display the optional cache review" in caplog.text
    assert window.shown["warnings"] == [("Optional cache cleanup",
                                         "Cache review could not finish. Check the log for details.")]
    assert window._cache_cleanup_busy is False
    assert window.started == []


def test_a_dialog_failing_while_the_window_closes_shows_nothing(window, monkeypatch):
    def broken(preview, parent):
        window._closing = True
        raise RuntimeError("the window is going away")

    monkeypatch.setattr(consent, "_choose_optional_cleanup", broken)
    window._on_cache_cleanup_preview_ready({"operation_id": "op", "folders": []})
    assert window.shown["warnings"] == []


def test_chosen_folders_are_applied_with_the_reviews_own_guard(window, monkeypatch):
    preview = {"operation_id": "op", "folders": []}
    guard = object()
    window._cache_cleanup_guard = guard
    applied = []
    monkeypatch.setattr(consent, "_choose_optional_cleanup", lambda prepared, parent: ("news",))
    monkeypatch.setattr(install_cleanup, "apply_optional_cleanup",
                        lambda prepared, choices, *, guard: applied.append(
                            (prepared, choices, guard)) or {"ok": True, "removed": 1})
    window._on_cache_cleanup_preview_ready(preview)
    [(operation, apply)] = window.started
    assert operation == "apply"
    assert apply() == {"ok": True, "removed": 1}
    assert applied == [(preview, ("news",), guard)]


def test_an_outcome_arriving_after_close_shows_no_notice(window):
    window._closing = True
    window._cache_cleanup_busy = True
    window._on_cache_cleanup_applied({"ok": True, "removed": 3})
    assert window.shown["notices"] == []
    assert window._cache_cleanup_busy is False


@pytest.mark.parametrize("reason, sentence", [
    ("active", "A run is in progress. Try again when it has finished."),
    ("changed", "Cleanup did not complete. Review the folders again before trying cleanup."),
])
def test_a_refused_cleanup_says_why_nothing_was_removed(window, reason, sentence):
    window._on_cache_cleanup_applied({"ok": False, "reason": reason, "removed": 0, "folders": []})
    assert window.shown["notices"] == ["Cached files removed: 0.\n\n" + sentence]


def test_a_late_update_plan_that_cannot_be_disarmed_is_logged(window, monkeypatch, caplog):
    cancelled = []

    class Stuck:
        def __init__(self, plan):
            cancelled.append(plan)

        def cancel(self):
            raise OSError("the handshake file is locked")

    monkeypatch.setattr(install_cleanup, "_FrozenUpdateHandshake", Stuck)
    window._closing = True
    plan = {"handshake": "token", "adapter": "pip-private-posix-v1", "command": ["pip"]}
    with caplog.at_level(logging.ERROR, logger=qt_app.LOG.name):
        window._on_update_plan_prepared(plan)
    assert cancelled == [plan]
    assert "Could not disarm a late update plan" in caplog.text
    assert window.shown["warnings"] == []


def test_an_update_is_cancelled_when_the_settings_cannot_be_saved(window, monkeypatch):
    from PySide6.QtWidgets import QStackedWidget

    from spacr import restart_state

    screen = QWidget()
    screen.app_key = "mask"
    screen._settings_model = type("Model", (), {"collect": lambda self: {"src": "/data"}})()
    window._stack = QStackedWidget()
    window._stack.addWidget(screen)

    def full_disk(**_kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(restart_state, "save", full_disk)
    assert window._save_pip_update_restart() is False
    assert window.shown["warnings"][0][0] == "Updates"
    assert "could not be saved" in window.shown["warnings"][0][1]
    assert not getattr(window, "_frozen_update_restart_saved", False)


def test_a_screen_that_cannot_list_its_children_has_no_backdrop_to_retire():
    class Gone:
        def children(self):
            raise RuntimeError("Internal C++ object already deleted")

    assert qt_app.MainWindow._a_screen_backdrop_built_beside_the_window_one(Gone()) is None
