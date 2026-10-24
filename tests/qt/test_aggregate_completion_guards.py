"""Late and failed GUI completions leave ownership and user state intact."""
from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace
import threading
import weakref

import pytest
import shiboken6
from PySide6.QtCore import QObject, QEvent, QRect, QThread, QTimer, Signal
from PySide6.QtWidgets import QMainWindow, QStackedWidget, QWidget


def test_a_consumed_timer_callback_cannot_run_twice(qapp):
    from spacr.qt import _fire_after

    timer = QTimer()
    calls = []
    box = [lambda: calls.append("once")]
    _fire_after(timer, box)
    _fire_after(timer, box)
    assert calls == ["once"] and box == []


def test_an_unidentified_worker_completion_cannot_settle_a_job(qapp):
    from spacr.qt.job_runner import JobRunner

    class UnidentifiedWorker(QObject):
        finished = Signal(bool)

    runner = JobRunner()
    worker = UnidentifiedWorker()
    results = []
    runner.job_finished.connect(results.append)
    worker.finished.connect(runner._worker_finished)
    worker.finished.emit(True)
    assert results == []
    assert runner.pending_jobs() == 0 and not runner.is_busy()


def test_a_running_job_is_not_retired_by_an_early_sweep(qapp, qtbot):
    from spacr.qt.job_runner import JobRunner

    entered = threading.Event()
    release = threading.Event()
    runner = JobRunner()

    def work():
        entered.set()
        release.wait(5)
        return "done"

    results = []
    assert runner.submit(work, results.append)
    try:
        assert entered.wait(5)
        held = dict(runner._jobs)
        runner._retire_finished_jobs()
        assert runner._jobs == held
        assert runner.pending_jobs() == 1 and runner.is_busy()
    finally:
        release.set()
    qtbot.waitUntil(lambda: not runner._jobs, timeout=10000)
    assert results == ["done"]
    assert runner.pending_jobs() == 0 and not runner.is_busy()


def test_deleted_walkthroughs_are_discarded_without_touching_a_dead_stack(qapp):
    from spacr.qt import walkthrough

    window = QMainWindow()
    overlay = QWidget(window)
    overlay._walkthrough_app_key = "test"
    window._module_walkthroughs = [overlay]
    shiboken6.delete(overlay)
    walkthrough._dismiss_module_walkthroughs(window)
    assert window._module_walkthroughs == []


@pytest.mark.parametrize("module_name", ["fractal_cascade", "fractal_space"])
@pytest.mark.parametrize("dependency,method", [
    ("spacr.logging_util", "_use_spacrs_kernel_cache"),
    ("spacr.qt.widgets.ambient", "_prime_kernels"),
])
def test_optional_kernel_cache_import_failure_does_not_block_rendering(
        module_name, dependency, method, monkeypatch):
    from importlib import import_module

    module = import_module("spacr.qt.widgets." + module_name)
    original = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == dependency:
            raise ImportError("optional cache helper unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    assert getattr(module, method)() is None


def test_missing_numba_version_still_gives_a_stable_cache_identity(tmp_path, monkeypatch):
    from importlib import metadata
    from spacr.qt.widgets import ambient

    def missing(_name):
        raise metadata.PackageNotFoundError("numba")

    monkeypatch.setattr(metadata, "version", missing)
    first = ambient._primed_marker(tmp_path)
    assert first == ambient._primed_marker(tmp_path)
    assert first.parent == tmp_path
    assert first.name.startswith("spacr-ambient-primed-")


@pytest.mark.parametrize("failure", ["fractal", "marker"])
def test_a_failed_priming_stage_never_writes_a_success_marker(
        failure, tmp_path, monkeypatch):
    from spacr.qt.widgets import ambient

    marker = tmp_path / "primed"
    monkeypatch.setattr(ambient, "_numba_cache_directory", lambda: tmp_path)
    monkeypatch.setattr(ambient, "_warm_packed_scatter", lambda: None)
    monkeypatch.setattr(ambient._SATIN_COMPILER, "_warm", lambda: None)
    monkeypatch.setattr(ambient, "_PACKED_SCATTER", object())
    monkeypatch.setattr(ambient._SATIN_COMPILER, "kernel", object())
    if failure == "fractal":
        def fail_fractal():
            raise RuntimeError("compiler unavailable")
        monkeypatch.setattr(ambient, "_prime_fractal_kernels", fail_fractal)
    else:
        monkeypatch.setattr(ambient, "_prime_fractal_kernels", lambda: None)
        def fail_marker():
            raise OSError("read-only cache")
        monkeypatch.setattr(ambient, "_primed_marker", lambda _folder: SimpleNamespace(
            touch=fail_marker))
    assert ambient._prime_kernel_cache() == 1
    assert not marker.exists()


def test_failed_worker_priming_is_attempted_once_without_escaping(monkeypatch):
    from spacr.qt.widgets import ambient

    calls = []
    monkeypatch.setattr(ambient, "_PRIMED_ONCE", threading.Event())
    monkeypatch.setattr(ambient, "_on_a_worker_thread", lambda: True)

    def fail():
        calls.append("attempt")
        raise RuntimeError("helper failed")

    monkeypatch.setattr(ambient, "_prime_in_helper_process", fail)
    ambient._prime_before_first_kernel()
    ambient._prime_before_first_kernel()
    assert calls == ["attempt"]


def test_an_unavailable_priming_interpreter_leaves_the_cache_unclaimed(
        tmp_path, monkeypatch):
    import subprocess
    from spacr.qt.widgets import ambient

    calls = []
    monkeypatch.setattr(ambient, "_COMPILER_STOP", threading.Event())
    monkeypatch.setattr(ambient, "_priming_allowed", lambda: True)
    monkeypatch.setattr(ambient, "_numba_cache_directory", lambda: tmp_path)
    marker = tmp_path / "primed"
    monkeypatch.setattr(ambient, "_primed_marker", lambda _folder: marker)

    def missing_interpreter(argv, **options):
        calls.append((argv, options))
        raise FileNotFoundError("interpreter unavailable")

    monkeypatch.setattr(subprocess, "Popen", missing_interpreter)
    assert not ambient._prime_in_helper_process()
    assert len(calls) == 1 and not marker.exists()
    assert calls[0][1]["env"]["NUMBA_CACHE_DIR"] == str(tmp_path)


def test_invalid_event_targets_are_rejected_without_registering_watches(qapp):
    from spacr.qt import gil_priority as gil

    hub = SimpleNamespace(_objects={})
    invalid = object()
    assert not gil._hub_discard_object(hub, invalid, invalid)
    assert gil._hub_object_watchers(hub, invalid) == ()
    assert not gil._hub_ask_object_watchers(hub, invalid, QEvent(QEvent.User), QEvent.User)
    assert not gil._watch_object_events(None, invalid, ())
    assert not gil._stop_watching_object_events(None, invalid)
    assert hub._objects == {}


def test_geometry_replay_skips_deleted_widgets_and_unrequested_event_types(qapp):
    from spacr.qt import gil_priority as gil

    widget = QWidget()
    dead = QWidget()
    shiboken6.delete(dead)
    vanished = QWidget()
    vanished_ref = weakref.ref(vanished)
    del vanished
    hub = SimpleNamespace(_geometry_watched={1: weakref.ref(widget), 2: weakref.ref(dead),
                                            3: vanished_ref}, _object_kinds=frozenset())
    assert [entry[0] for entry in gil._hub_geometry_before(hub)] == [widget]
    previous = widget.geometry()
    widget.setGeometry(QRect(previous.x() + 1, previous.y() + 1,
                             previous.width() + 10, previous.height() + 10))
    gil._hub_geometry_after(hub, [(dead, False, previous), (widget, widget.isHidden(), previous)])
    assert widget.geometry() != previous


def test_a_failing_object_watcher_reports_its_error_and_leaves_the_event_unclaimed(
        qapp, monkeypatch):
    from spacr.qt import gil_priority as gil

    errors = []
    monkeypatch.setattr(gil.sys, "excepthook", lambda *info: errors.append(info))

    class FailingWatcher(QObject):
        def eventFilter(self, watched, event):
            raise RuntimeError("watcher stopped")

    widget, watcher = QWidget(), FailingWatcher()
    hub = gil._application_event_hub_class()(qapp)
    kind = QEvent.Type.User
    gil._hub_add_object(hub, widget, watcher, {kind})
    assert not gil._hub_ask_object_watchers(hub, widget, QEvent(kind), kind)
    assert len(errors) == 1 and str(errors[0][1]) == "watcher stopped"
    gil._hub_discard_object(hub, widget, watcher)
    hub.deleteLater()


def test_an_update_watcher_can_consume_a_paint_request_before_the_paint(qapp):
    from spacr.qt import gil_priority as gil

    calls = []

    class ConsumeUpdate(QObject):
        def eventFilter(self, watched, event):
            calls.append(event.type())
            return True

    widget, watcher = QWidget(), ConsumeUpdate()
    hub = gil._application_event_hub_class()(qapp)
    kind = QEvent.Type.UpdateRequest
    gil._hub_add_object(hub, widget, watcher, {kind})
    assert hub.eventFilter(widget, QEvent(kind))
    assert calls == [kind]
    gil._hub_discard_object(hub, widget, watcher)
    hub.deleteLater()


def test_removing_a_watch_tolerates_both_a_dead_hub_and_dead_widget(qapp, monkeypatch):
    from spacr.qt import gil_priority as gil

    widget, watcher = QWidget(), QObject()
    shiboken6.delete(widget)
    monkeypatch.setattr(gil, "_application_event_hub", lambda *_args, **_kwargs: object())

    def dead_hub(*_args):
        raise RuntimeError("hub was destroyed")

    monkeypatch.setattr(gil, "_hub_discard_object", dead_hub)
    assert not gil._stop_watching_object_events(widget, watcher)


def test_walkthrough_navigation_survives_a_failed_overlay_cleanup(qapp, monkeypatch):
    from spacr.qt import walkthrough

    window = QMainWindow()
    stack = QStackedWidget(window)
    screen = QWidget()
    screen.app_key = "completion-guard-test"
    stack.addWidget(screen)
    window._stack = stack
    handler = walkthrough._WalkthroughHandler(window)

    def already_closed(*_args, **_kwargs):
        raise RuntimeError("overlay was destroyed")

    monkeypatch.setattr(walkthrough, "_dismiss_module_walkthroughs", already_closed)
    monkeypatch.setattr(walkthrough, "was_seen", lambda _key: False)
    handler.on_current_changed(0)
    assert stack.currentWidget() is screen


def test_a_disconnected_destroyed_emitter_is_safe_to_disconnect_again(qapp):
    from spacr.qt.screens.data_manager import _disconnect_quietly

    class Emitter(QObject):
        changed = Signal()

    emitter = Emitter()
    slot = lambda: None
    emitter.changed.connect(slot)
    signal = emitter.changed
    shiboken6.delete(emitter)
    assert _disconnect_quietly(signal, slot) is None


def test_api_links_reject_an_accidental_non_spacr_target(monkeypatch):
    from spacr.qt.screens import settings_model
    from spacr.qt.screens import setting_api_targets

    monkeypatch.setitem(setting_api_targets.SETTING_API_TARGETS, "guard_key",
                        ("unrelated.module", "function", True))
    assert settings_model._mapped_api_target("guard_key") == ("", "")


def test_preview_model_stale_indices_and_empty_values_are_safe(qapp):
    from PySide6.QtCore import QModelIndex, Qt
    from spacr.qt.screens.db_browser import PreviewModel, _WIDTH_SAMPLE_ROWS

    model = PreviewModel()
    model.set_page(["value"], [[None]])
    assert model.flags(QModelIndex()) == Qt.NoItemFlags
    assert model.data(model.index(0, 0), Qt.EditRole) == ""
    stale = model.index(0, 0)
    model.clear()
    assert model.data(stale, Qt.DisplayRole) is None
    assert model.data(stale, Qt.EditRole) is None
    assert model._widest_texts(0) == []
    model.set_page(["value"], [["short"] for _ in range(_WIDTH_SAMPLE_ROWS)])
    assert model._widest_texts(0) == ["short"]
    assert model.append_rows([["outside the width sample"]]) == 1
    assert model._widest_texts(0) == ["short"]
    assert model.rowCount() == _WIDTH_SAMPLE_ROWS + 1


def test_column_sizing_handles_empty_models_and_an_unchanged_search(qapp, qtbot):
    from spacr.qt.screens.db_browser import DbBrowserScreen

    screen = DbBrowserScreen(threaded=False)
    qtbot.addWidget(screen)
    screen._model.set_page(["first", "second"], [])
    screen._autosize_columns()
    widths = [screen._view.columnWidth(i) for i in range(2)]
    screen._autosize_columns()
    assert [screen._view.columnWidth(i) for i in range(2)] == widths
    assert screen._cell_padding("", 0) > 0
    screen._col_search.setText(screen._model.column_filter())
    screen._apply_column_search()
    assert screen._model.visible_columns() == ["first", "second"]
    screen._model.set_page([], [[]])
    assert screen._cell_padding("", 0) > 0


def test_a_late_outlier_scan_cannot_fill_a_destroyed_screen(qapp):
    from spacr.qt.screens.outliers import OutliersScreen, _alive

    screen = QWidget()
    screen._frame = object()
    sentinel = screen._objects = object()
    shiboken6.delete(screen)
    OutliersScreen._on_scanned(screen, object())
    assert screen._objects is sentinel
    assert not _alive(None)


def test_failed_binding_liveness_probes_are_conservative(monkeypatch):
    from spacr.qt import i18n
    from spacr.qt.screens import outliers

    def unavailable(_obj):
        raise RuntimeError("binding is shutting down")

    monkeypatch.setattr(shiboken6, "isValid", unavailable)
    assert not i18n._a11y_alive(object())
    assert not outliers._alive(object())


def test_accessibility_tree_and_label_probes_tolerate_a_teardown_race(qapp, monkeypatch):
    from spacr.qt import i18n
    from PySide6.QtWidgets import QLineEdit

    class VanishedRoot(QObject):
        def findChildren(self, *_args):
            raise RuntimeError("root vanished after its liveness check")

    assert i18n._a11y_live_tree(VanishedRoot()) == []
    assert i18n._a11y_label_beside(None) is None
    parent = QWidget()
    child = QLineEdit(parent)
    original = i18n._a11y_alive
    monkeypatch.setattr(i18n, "_a11y_alive", lambda obj: False if obj is parent else original(obj))
    assert i18n._a11y_label_beside(child) is None


def test_accessibility_naming_skips_widgets_destroyed_during_enumeration(qapp):
    from spacr.qt import i18n
    from PySide6.QtWidgets import QLineEdit, QMenu, QPlainTextEdit

    menu, target = QMenu("Tools"), QLineEdit()
    victims = [menu, target]

    class ClosingEditor(QPlainTextEdit):
        def tabChangesFocus(self):
            for victim in victims[:]:
                shiboken6.delete(victim)
                victims.remove(victim)
            return True

    editor = ClosingEditor()
    assert isinstance(i18n._name_for_assistive_tech([menu, target, editor]), int)
    assert not shiboken6.isValid(menu) and not shiboken6.isValid(target)


def test_accessibility_menu_failure_does_not_abort_the_language_pass(qapp):
    from spacr.qt import i18n
    from PySide6.QtWidgets import QMenu

    class VanishedMenu(QMenu):
        def accessibleName(self):
            raise RuntimeError("menu vanished after enumeration")

    menu = VanishedMenu("Tools")
    assert i18n._name_for_assistive_tech([menu]) == 0


def test_dialog_translation_skips_the_same_kept_language_and_dead_actions(qapp, monkeypatch):
    from spacr.qt import i18n
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import QDialog

    dialog = QDialog()
    dialog.setProperty(i18n.TRANSLATE_ONCE_PER_LANGUAGE, True)
    dialog.setProperty(i18n._TRANSLATED_FOR, i18n.current_language())
    dialog.show()
    calls = []
    original = i18n.retranslate_widget_tree
    monkeypatch.setattr(i18n, "retranslate_widget_tree", lambda _root: calls.append("translated"))
    i18n._translate_shown_dialog(dialog)
    assert calls == []
    monkeypatch.setattr(i18n, "retranslate_widget_tree", original)
    dialog.close()

    class StaleActionRoot(QWidget):
        def findChildren(self, cls, *args):
            if cls is QAction:
                return [self.stale_action]
            return super().findChildren(cls, *args)

    root = StaleActionRoot()
    root.stale_action = QAction("Stale", root)
    shiboken6.delete(root.stale_action)
    assert i18n.retranslate_widget_tree(root) is None
    assert shiboken6.isValid(root) and not shiboken6.isValid(root.stale_action)
