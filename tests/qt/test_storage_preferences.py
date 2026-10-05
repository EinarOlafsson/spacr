"""The Storage tab: pruning caps, the prune button and the cache manager.

Every test runs against a temporary home and preference store, so nothing
real is read or deleted.
"""
from __future__ import annotations

import os
import time

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QPushButton, QSpinBox, QTableWidget

DAY = 86400.0


@pytest.fixture
def home(tmp_path, monkeypatch, qt_theme_applied):
    from spacr.qt import preferences as prefs
    from spacr.qt import resource_cleanup

    root = tmp_path / "home"
    root.mkdir()
    for name in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(name, str(root))
    monkeypatch.setenv("XDG_CACHE_HOME", str(root / ".cache"))
    monkeypatch.setenv("SPACR_PORTABLE", "0")
    monkeypatch.setenv("SPACR_LOG_DIR", str(root / ".spacr" / "logs"))
    for name in ("CELLPOSE_LOCAL_MODELS_PATH", "HF_HOME", "TORCH_HOME",
                 "SPACR_BACKENDS_DIR", "SPACR_NEWS_CACHE", "SPACR_HOME", "SPACR_RUN_ID"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    store = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: store)
    monkeypatch.setattr(resource_cleanup, "_a_run_is_active", lambda: False)
    return root


@pytest.fixture
def page(home, qtbot, monkeypatch):
    from spacr.qt import preferences as prefs

    asked, told = [], []

    def confirm(title, text, parent=None):
        asked.append(text)
        return page.answer

    monkeypatch.setattr(prefs, "_confirm_storage_action", confirm)
    monkeypatch.setattr(prefs, "_show_storage_result",
                        lambda title, text, parent=None: told.append(text))
    dlg = prefs.PreferencesDialog()
    qtbot.addWidget(dlg)
    page = dlg._storage_page
    page.answer = False
    page.asked, page.told, page.dialog = asked, told, dlg
    return page


def _old_log(home, name="spacr-20200101.log"):
    path = home / ".spacr" / "logs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * 100)
    os.utime(path, (time.time() - 90 * DAY,) * 2)
    return path


def test_the_storage_tab_has_caps_prune_and_caches(page):
    dlg = page.dialog
    assert dlg.findChild(QPushButton, "StoragePruneButton") is not None
    assert dlg.findChild(QTableWidget, "StorageCacheTable").rowCount() == 6
    for kind in ("logs", "run_logs", "run_folders"):
        assert dlg.findChild(QSpinBox, f"StorageKeepDays_{kind}").minimum() == 1


def test_caps_are_saved_with_the_dialog(page):
    from spacr.qt import preferences as prefs

    page.spins["logs"][0].setValue(7)
    page.spins["logs"][1].setValue(0)
    page.save()
    assert prefs._get_storage_caps()["logs"] == (7, 0)


def test_prune_asks_first_and_cancel_keeps_everything(page, home, qtbot):
    log = _old_log(home)
    page.spins["logs"][1].setValue(0)
    page.prune()
    qtbot.waitUntil(lambda: page.prune_button.isEnabled(), timeout=10000)
    assert page.asked and "1 of 1" in page.asked[0]
    assert log.exists()


def test_prune_deletes_after_yes(page, home, qtbot):
    log = _old_log(home)
    page.answer = True
    page.spins["logs"][1].setValue(0)
    page.prune()
    qtbot.waitUntil(lambda: bool(page.told), timeout=10000)
    assert not log.exists()


def test_clear_asks_and_empties_the_selected_cache(page, home, qtbot):
    torch_dir = home / ".cache" / "torch"
    torch_dir.mkdir(parents=True)
    (torch_dir / "w.pt").write_bytes(b"1")
    row = [r["key"] for r in page._rows].index("torch")
    page.table.selectRow(row)
    page.clear()
    assert page.asked and (torch_dir / "w.pt").exists()
    page.answer = True
    page.clear()
    qtbot.waitUntil(lambda: bool(page.told), timeout=10000)
    assert torch_dir.is_dir() and not (torch_dir / "w.pt").exists()


def test_clear_waits_for_a_running_run(page, home, monkeypatch):
    from spacr.qt import resource_cleanup

    monkeypatch.setattr(resource_cleanup, "_a_run_is_active", lambda: True)
    page.table.selectRow(0)
    page.answer = True
    page.clear()
    assert page.told and not page.asked


def test_failed_cache_clear_reports_the_failure_and_keeps_the_cache(
        page, home, qtbot, monkeypatch):
    from spacr import run_journal

    torch_dir = home / ".cache" / "torch"
    torch_dir.mkdir(parents=True)
    cached = torch_dir / "w.pt"
    cached.write_bytes(b"1")

    def unavailable(_key):
        raise OSError("cache is unavailable")

    monkeypatch.setattr(run_journal, "_clear_cache", unavailable)
    page.table.selectRow([r["key"] for r in page._rows].index("torch"))
    page.answer = True
    page.clear()
    qtbot.waitUntil(lambda: bool(page.told), timeout=10000)
    assert cached.exists()
    assert "Removed 0 item(s)." in page.told[0]
    assert "failed" in page.told[0]


def test_move_relocates_after_yes(page, home, tmp_path, qtbot):
    news = home / ".spacr" / "news"
    news.mkdir(parents=True)
    (news / "releases.json").write_text("[]")
    row = [r["key"] for r in page._rows].index("news")
    page.table.selectRow(row)
    assert page.move_button.isEnabled()
    page.answer = True
    page.relocate(str(tmp_path / "bigdisk"))
    qtbot.waitUntil(lambda: bool(page.told), timeout=10000)
    assert (tmp_path / "bigdisk" / "spacr-news" / "releases.json").exists()
    page.table.selectRow([r["key"] for r in page._rows].index("models"))
    assert not page.move_button.isEnabled()


def test_failed_cache_move_reports_the_reason_and_keeps_the_source(
        page, home, tmp_path, qtbot, monkeypatch):
    from spacr import run_journal

    news = home / ".spacr" / "news"
    news.mkdir(parents=True)
    cached = news / "releases.json"
    cached.write_text("[]")

    def unavailable(_key, _folder):
        raise OSError("destination unavailable")

    monkeypatch.setattr(run_journal, "_relocate_cache", unavailable)
    page.table.selectRow([r["key"] for r in page._rows].index("news"))
    page.answer = True
    page.relocate(str(tmp_path / "bigdisk"))
    qtbot.waitUntil(lambda: bool(page.told), timeout=10000)
    assert cached.exists()
    assert "Not moved: destination unavailable" in page.told[0]


@pytest.fixture
def clearer(page):
    return page.dialog._log_clearer


def _logs(home):
    logs = home / ".spacr" / "logs"
    runs = logs / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    files = {
        "daily": logs / "spacr-20200101.log",
        "rotated": logs / "spacr-20200101.log.2",
        "run": runs / "run-1.jsonl",
        "crash": logs / "spacr-crash.log",
        "verbose": logs / "spacr-debug.log",
        "other": logs / "running.marker",
    }
    for path in files.values():
        path.write_bytes(b"x" * 100)
    return files


def test_clear_all_logs_lists_each_kind_and_cancel_keeps_them(
        page, clearer, home, qtbot):
    files = _logs(home)
    assert page.dialog.findChild(QPushButton, "LoggingClearAllLogs") is clearer.button
    clearer.clear()
    qtbot.waitUntil(lambda: clearer.button.isEnabled(), timeout=10000)
    text = page.asked[0]
    for line in ("Daily logs: 2 file(s)", "Run logs: 1 file(s)",
                 "Crash logs: 1 file(s)", "Verbose logs: 1 file(s)"):
        assert line in text
    assert all(path.exists() for path in files.values())


def test_clear_all_logs_deletes_but_only_empties_an_open_log(
        page, clearer, home, qtbot):
    import logging

    files = _logs(home)
    handler = logging.FileHandler(str(files["verbose"]))
    logging.getLogger("spacr.test_clear_logs").addHandler(handler)
    try:
        page.answer = True
        clearer.clear()
        qtbot.waitUntil(lambda: bool(page.told), timeout=10000)
        assert "emptied, not deleted" in page.asked[0]
        assert files["verbose"].exists() and files["verbose"].stat().st_size == 0
        for key in ("daily", "rotated", "run", "crash"):
            assert not files[key].exists(), key
        assert files["other"].exists()
        assert "Deleted 4 file(s) and emptied 1" in page.told[0]
    finally:
        logging.getLogger("spacr.test_clear_logs").removeHandler(handler)
        handler.close()


def test_cache_actions_without_a_selection_do_no_work(page, monkeypatch):
    submitted = []
    monkeypatch.setattr(page, "_background",
                        lambda *_args: submitted.append(True))
    page.table.clearSelection()
    page.clear()
    page.relocate("unused")
    assert submitted == []
    assert page.asked == []


def test_cache_move_waits_for_a_run_and_allows_cancel_before_work(
        page, tmp_path, monkeypatch):
    from spacr.qt import preferences, resource_cleanup

    submitted = []
    monkeypatch.setattr(page, "_background",
                        lambda *_args: submitted.append(True))
    page.table.selectRow([r["key"] for r in page._rows].index("news"))
    monkeypatch.setattr(resource_cleanup, "_a_run_is_active", lambda: True)
    page.relocate(str(tmp_path))
    assert page.told and "A run is in progress" in page.told[-1]
    assert page.asked == []

    monkeypatch.setattr(resource_cleanup, "_a_run_is_active", lambda: False)
    monkeypatch.setattr(preferences, "_choose_cache_parent", lambda *_: "")
    page.relocate()
    assert page.asked == []
    page.relocate(str(tmp_path))
    assert page.asked and "Move" in page.asked[-1]
    assert submitted == []


def test_empty_pruning_plan_restores_controls_without_confirmation(page):
    page.prune_button.setEnabled(False)
    page._planned("Prune home folder", None)
    assert page.prune_button.isEnabled()
    page.prune_button.setEnabled(False)
    page._planned("Prune home folder", [{
        "kind": "logs", "delete": [], "count": 3, "total": 300,
    }])
    assert page.prune_button.isEnabled()
    assert page.asked == []
    assert "Nothing is over its caps" in page.told[-1]


def test_failed_and_partly_refused_pruning_restore_controls(page):
    page.prune_button.setEnabled(False)
    page._pruned("Prune home folder", None)
    assert page.prune_button.isEnabled()
    assert page.told == []
    page._pruned("Prune home folder", [(1, 100, ["a log is locked"])])
    assert "Deleted 1 item(s)" in page.told[-1]
    assert "a log is locked" in page.told[-1]


def test_empty_and_failed_log_clear_results_restore_controls(page, clearer):
    clearer.button.setEnabled(False)
    clearer._planned(None)
    assert clearer.button.isEnabled()
    assert "There are no logs to clear" in page.told[-1]
    assert page.asked == []
    previous = list(page.told)
    clearer.button.setEnabled(False)
    clearer._cleared("Clear all logs", None)
    assert clearer.button.isEnabled()
    assert page.told == previous
    clearer._cleared("Clear all logs", (1, 0, 100, ["another log is locked"]))
    assert "another log is locked" in page.told[-1]


def test_a_busy_log_worker_keeps_the_clear_button_retryable(
        page, clearer, monkeypatch):
    from types import SimpleNamespace

    from spacr.qt import preferences

    submitted = []

    def busy(*args):
        submitted.append(args)
        return False

    monkeypatch.setattr(preferences, "_disk_report_runner",
                        lambda: SimpleNamespace(submit=busy))
    clearer.clear()
    assert len(submitted) == 1
    assert clearer.button.isEnabled()
    assert page.asked == []
    assert page.told == []


def test_log_worker_failures_reach_the_live_page_as_a_failed_result(
        clearer, caplog):
    results = []
    with caplog.at_level("WARNING"):
        clearer._finish(results.append, OSError("disk unavailable"))
    assert results == [None]
    assert "clearing the logs failed" in caplog.text


def test_a_log_clear_callback_after_window_destruction_does_nothing(
        page, clearer, qtbot):
    from PySide6.QtCore import Qt
    from shiboken6 import isValid

    page.dialog.setAttribute(Qt.WA_DeleteOnClose)
    page.dialog.show()
    page.dialog.close()
    qtbot.waitUntil(lambda: not isValid(clearer.button))
    results = []
    clearer._finish(results.append, (1, 0, 100, []))
    assert results == []
