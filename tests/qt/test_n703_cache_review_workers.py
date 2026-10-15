"""Real cache-review QThreads use actual app handlers without a full app launch."""
import threading

import pytest
from PySide6.QtCore import QThread, QTimer, Qt
from PySide6.QtWidgets import QApplication, QDialog, QMainWindow, QMessageBox, QPushButton, QTreeWidget

from spacr import install_cleanup as cleanup
from spacr.qt import app as qt_app
from spacr.qt import install_consent as consent


_REAL_DIALOG_EXEC = QDialog.exec


class ReviewWindow(QMainWindow):
    _review_installation_caches = qt_app.MainWindow._review_installation_caches
    _start_cache_cleanup_worker = qt_app.MainWindow._start_cache_cleanup_worker
    _on_cache_cleanup_succeeded = qt_app.MainWindow._on_cache_cleanup_succeeded
    _on_cache_cleanup_failed = qt_app.MainWindow._on_cache_cleanup_failed
    _on_cache_cleanup_finished = qt_app.MainWindow._on_cache_cleanup_finished
    _on_cache_cleanup_preview_ready = qt_app.MainWindow._on_cache_cleanup_preview_ready
    _on_cache_cleanup_applied = qt_app.MainWindow._on_cache_cleanup_applied
    _finish_cache_cleanup_review = qt_app.MainWindow._finish_cache_cleanup_review
    _release_update_workers = qt_app.MainWindow._release_update_workers
    _UPDATE_WORKER_WAIT_MS = 0


@pytest.fixture
def review(qtbot, monkeypatch, tmp_path):
    window = ReviewWindow()
    window._closing = False
    qtbot.addWidget(window)
    window.show()
    original_updater = QThread(window)
    window._update_worker = original_updater
    base = tmp_path / "home" / ".spacr"
    for key, filename in [("news", "releases.json"), ("icons", "button-0123456789abcdef0123.png")]:
        folder = base / key
        folder.mkdir(parents=True)
        (folder / filename).write_bytes(("original-" + key).encode())
    machine = cleanup._Machine(environ={"HOME": str(base.parent)}, platform="linux")
    actual_preview = cleanup.preview_optional_cleanup
    actual_apply = cleanup.apply_optional_cleanup
    state = {"window": window, "base": base, "machine": machine, "gates": [],
             "threads": [], "previews": [], "applies": [], "notices": [], "warnings": [],
             "original_updater": original_updater}
    def preview(operation, *, system=None, guard=None):
        assert QThread.currentThread() is not QApplication.instance().thread()
        if guard is None:
            return actual_preview(operation, system=machine)
        state["threads"].append(QThread.currentThread())
        state["previews"].append((operation, guard))
        gate = state.get("preview_gate")
        if gate is not None:
            state["preview_entered"].set()
            assert gate.wait(5)
        return actual_preview(operation, system=machine, guard=guard)
    def apply(prepared, choices, *, guard):
        assert QThread.currentThread() is not QApplication.instance().thread()
        assert guard is state["previews"][-1][1]
        state["applies"].append((prepared, choices))
        gate = state.get("apply_gate")
        if gate is not None:
            state["apply_entered"].set()
            assert gate.wait(5)
        return actual_apply(prepared, choices, system=machine, guard=guard)
    monkeypatch.setattr(cleanup, "preview_optional_cleanup", preview)
    monkeypatch.setattr(cleanup, "apply_optional_cleanup", apply)
    def notice(box):
        state["notices"].append((box.text(), box.textFormat()))
        return QMessageBox.Ok
    monkeypatch.setattr(qt_app.QMessageBox, "exec", notice)
    monkeypatch.setattr(qt_app.QMessageBox, "warning", staticmethod(
        lambda _parent, title, text, *_args: state["warnings"].append((title, text))))
    yield state
    window._closing = True
    for gate in state["gates"]:
        gate.set()
    worker = getattr(window, "_cache_cleanup_worker", None)
    if worker is not None:
        assert worker.wait(5000)
    QApplication.processEvents()
    from spacr.qt import bridge
    bridge.wait_for_parked_threads(timeout_ms=5000)
    assert bridge.prune_parked_threads() == 0


def blocked(state, phase):
    state[phase + "_gate"] = threading.Event()
    state[phase + "_entered"] = threading.Event()
    state["gates"].append(state[phase + "_gate"])


def choose_with_native_dialog(monkeypatch, state, *, remove):
    monkeypatch.setattr(consent._InstallerCleanupDialog, "exec", _REAL_DIALOG_EXEC)
    original = consent._choose_optional_cleanup
    def choose(preview, parent):
        assert parent is state["window"]
        assert QThread.currentThread() is QApplication.instance().thread()
        assert getattr(parent, "_cache_cleanup_worker", None) is None
        def answer():
            dialog = QApplication.activeModalWidget()
            assert isinstance(dialog, consent._InstallerCleanupDialog)
            tree = dialog.findChild(QTreeWidget, "InstallerCleanupFolders")
            button = dialog.findChild(QPushButton, "InstallerCleanupRemove")
            keep = dialog.findChild(QPushButton, "InstallerCleanupKeep")
            assert keep.isDefault() and not button.isEnabled()
            assert all(tree.topLevelItem(i).checkState(0) == Qt.Unchecked
                       for i in range(tree.topLevelItemCount()))
            if remove:
                tree.topLevelItem(0).setCheckState(0, Qt.Checked)
                assert button.isEnabled()
                button.click()
            else:
                keep.click()
        watchdog = QTimer(parent)
        watchdog.setSingleShot(True)
        def dismiss_stuck_dialog():
            dialog = QApplication.activeModalWidget()
            if isinstance(dialog, consent._InstallerCleanupDialog):
                dialog.reject()
        watchdog.timeout.connect(dismiss_stuck_dialog)
        watchdog.start(2000)
        QTimer.singleShot(0, answer)
        try:
            return original(preview, parent)
        finally:
            watchdog.stop()
            watchdog.deleteLater()
    monkeypatch.setattr(consent, "_choose_optional_cleanup", choose)


def test_blocked_preview_keeps_gui_heartbeat_and_native_keep_never_applies(review, qtbot, monkeypatch):
    blocked(review, "preview")
    choose_with_native_dialog(monkeypatch, review, remove=False)
    beats = []
    timer = QTimer(review["window"])
    timer.timeout.connect(lambda: beats.append(True))
    timer.start(5)
    review["window"]._review_installation_caches()
    qtbot.waitUntil(review["preview_entered"].is_set)
    qtbot.waitUntil(lambda: len(beats) >= 3)
    assert review["window"]._update_worker is review["original_updater"]
    assert review["window"]._cache_cleanup_worker is not review["original_updater"]
    review["preview_gate"].set()
    qtbot.waitUntil(lambda: not review["window"]._cache_cleanup_busy)
    assert not review["applies"] and not review["notices"] and not review["warnings"]
    assert (review["base"] / "news" / "releases.json").read_bytes() == b"original-news"


def test_selected_native_dialog_starts_apply_only_after_preview_thread_finished(review, qtbot, monkeypatch):
    blocked(review, "apply")
    choose_with_native_dialog(monkeypatch, review, remove=True)
    review["window"]._review_installation_caches()
    qtbot.waitUntil(review["apply_entered"].is_set)
    assert review["window"]._cache_cleanup_worker.operation == "apply"
    assert review["window"]._update_worker is review["original_updater"]
    review["apply_gate"].set()
    qtbot.waitUntil(lambda: not review["window"]._cache_cleanup_busy)
    assert len(review["applies"]) == 1 and len(review["notices"]) == 1
    assert "1" in review["notices"][0][0] and review["notices"][0][1] == Qt.PlainText
    assert list((review["base"] / "news").iterdir()) == []
    assert (review["base"] / "icons" / "button-0123456789abcdef0123.png").read_bytes() == b"original-icons"


def test_late_close_drains_blocked_preview_without_dialog_or_apply(review, qtbot, monkeypatch):
    from spacr.qt import bridge
    blocked(review, "preview")
    monkeypatch.setattr(consent, "_choose_optional_cleanup", lambda *_: pytest.fail("late dialog"))
    calls = []
    actual_drain = bridge.drain_thread
    def drain(worker, **kwargs):
        calls.append(worker)
        return actual_drain(worker, **kwargs)
    monkeypatch.setattr(bridge, "drain_thread", drain)
    window = review["window"]
    window._review_installation_caches()
    qtbot.waitUntil(review["preview_entered"].is_set)
    worker = window._cache_cleanup_worker
    window._closing = True
    window._release_update_workers()
    assert calls == [worker] and worker.parent() is None
    review["preview_gate"].set()
    qtbot.waitUntil(lambda: window._cache_cleanup_worker is None)
    assert not review["applies"] and not review["notices"] and not review["warnings"]
    assert (review["base"] / "news" / "releases.json").read_bytes() == b"original-news"


def test_late_close_retains_authorized_apply_worker_and_discards_result_popup(review, qtbot, monkeypatch):
    blocked(review, "apply")
    choose_with_native_dialog(monkeypatch, review, remove=True)
    window = review["window"]
    window._review_installation_caches()
    qtbot.waitUntil(review["apply_entered"].is_set)
    worker = window._cache_cleanup_worker
    window._closing = True
    window._release_update_workers()
    assert worker.parent() is None
    review["apply_gate"].set()
    qtbot.waitUntil(lambda: window._cache_cleanup_worker is None)
    assert len(review["applies"]) == 1 and not review["notices"]
    assert list((review["base"] / "news").iterdir()) == []


def test_preview_failure_finishes_worker_before_showing_failure_and_never_applies(review, qtbot, monkeypatch):
    def failure(*_args, **_kwargs):
        assert QThread.currentThread() is not QApplication.instance().thread()
        raise OSError("disposable preview failure")
    monkeypatch.setattr(cleanup, "preview_optional_cleanup", failure)
    monkeypatch.setattr(consent, "_choose_optional_cleanup", lambda *_: pytest.fail("failed preview dialog"))
    review["window"]._review_installation_caches()
    qtbot.waitUntil(lambda: not review["window"]._cache_cleanup_busy)
    assert review["window"]._cache_cleanup_worker is None
    assert len(review["warnings"]) == 1 and not review["applies"] and not review["notices"]
    assert review["window"]._update_worker is review["original_updater"]


def test_partial_result_uses_plain_text_removed_count_and_literal_retained_path(review):
    path = str(review["base"] / "<retained-user-name>")
    review["window"]._cache_cleanup_busy = True
    review["window"]._on_cache_cleanup_applied({
        "ok": False, "removed": 2, "folders": [{"retained_path": path}]})
    assert len(review["notices"]) == 1
    text, formatting = review["notices"][0]
    assert "2" in text and path in text and formatting == Qt.PlainText
    assert not review["window"]._cache_cleanup_busy
