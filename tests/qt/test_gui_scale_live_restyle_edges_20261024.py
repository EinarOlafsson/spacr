"""Edge paths of the live-restyle guard: dead widgets, held restyles and teardown marks."""
import sys
import threading

import pytest

pytest.importorskip("PySide6")
import PySide6.QtCore as QtCore
from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout
from shiboken6 import delete as shiboken_delete

from spacr.qt import gui_scale as gs

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def isolated_guard(monkeypatch):
    monkeypatch.setattr(gs, "_PENDING_SHEETS", {})
    monkeypatch.setattr(gs, "_TORN_DOWN", set())
    monkeypatch.setitem(gs._SHEET_GUARD, "posted", False)
    yield


class _Event(QEvent):
    def type(self):
        raise RuntimeError("event gone")


def _widget(qtbot):
    widget = QWidget()
    qtbot.addWidget(widget)
    return widget


def test_polish_check_refuses_a_missing_or_foreign_target(qtbot):
    widget = _widget(qtbot)
    assert gs._polish_not_yet_begun(None, (widget,)) is False
    assert gs._polish_not_yet_begun(widget, (object(),)) is False
    assert gs._polish_not_yet_begun(widget, (widget,)) is True


def test_polish_check_treats_a_broken_target_as_not_before_polish():
    class Broken:
        def isWindow(self):
            raise RuntimeError("deleted")

    broken = Broken()
    assert gs._polish_not_yet_begun(broken, (broken,)) is False


def test_a_frame_whose_locals_cannot_be_read_is_not_a_delicate_handler():
    class Frame:
        f_code = (lambda self, event: None).__code__

        @property
        def f_locals(self):
            raise OSError("no locals")

    assert gs._frame_handles_a_delicate_event(Frame(), {QEvent.Type.Polish}) is False


def test_a_handler_holding_an_unreadable_event_counts_as_delicate():
    def handler(self, event):
        return None

    class Frame:
        f_code = handler.__code__
        f_locals = {"self": None, "event": _Event(QEvent.Type.None_)}

    assert gs._frame_handles_a_delicate_event(Frame(), {QEvent.Type.Polish}) is True


def test_is_alive_answers_for_none_deleted_and_unimportable(monkeypatch, qtbot):
    widget = QWidget()
    assert gs._is_alive(None) is False
    assert gs._is_alive(widget) is True
    shiboken_delete(widget)
    assert gs._is_alive(widget) is False
    import shiboken6
    with monkeypatch.context() as patch:
        patch.setattr(shiboken6, "isValid", lambda obj: 1 / 0)
        assert gs._is_alive(object()) is False
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "shiboken6", None)
        assert gs._is_alive(object()) is True


def test_address_of_a_non_qt_object_is_none():
    assert gs._address_of(object()) is None


def test_marking_torn_down_ignores_unaddressable_and_repeated_objects(monkeypatch, qtbot):
    gs._mark_torn_down(object())
    assert gs._TORN_DOWN == set()
    widget = _widget(qtbot)
    gs._mark_torn_down(widget)
    address = gs._address_of(widget)
    assert address in gs._TORN_DOWN
    dropped = []
    monkeypatch.setattr(gs, "_drop_pending_sheets_inside", dropped.append)
    gs._mark_torn_down(widget)
    assert dropped == []


def test_marking_an_object_without_a_destroyed_signal_still_records_it(monkeypatch):
    seen = []
    monkeypatch.setattr(gs, "_address_of", lambda obj: 4242)
    monkeypatch.setattr(gs, "_drop_pending_sheets_inside", seen.append)
    marker = object()
    gs._mark_torn_down(marker)
    assert 4242 in gs._TORN_DOWN
    assert seen == [marker]


def test_a_closing_application_is_torn_down_and_a_failing_query_is_ignored(monkeypatch):
    class Closing:
        @staticmethod
        def closingDown():
            return True

    class Failing:
        @staticmethod
        def closingDown():
            raise RuntimeError("no app")

    monkeypatch.setattr(QtCore, "QCoreApplication", Closing)
    assert gs._is_torn_down(object()) is True
    monkeypatch.setattr(QtCore, "QCoreApplication", Failing)
    assert gs._is_torn_down(object()) is False


def test_teardown_walk_handles_none_plain_cyclic_and_deleted_parents(monkeypatch):
    gs._TORN_DOWN.add(-1)
    assert gs._is_torn_down(None) is False
    assert gs._is_torn_down(QObject()) is False

    class Loop:
        def parentWidget(self):
            return self

    monkeypatch.setattr(gs, "_address_of", lambda node: 7)
    assert gs._is_torn_down(Loop()) is False

    class Dead:
        def parentWidget(self):
            raise RuntimeError("deleted")

    assert gs._is_torn_down(Dead()) is True


def test_top_level_of_falls_back_to_the_object(qtbot):
    marker = object()
    assert gs._top_level_of(marker) is marker

    class Dead:
        def window(self):
            raise RuntimeError("deleted")

    dead = Dead()
    assert gs._top_level_of(dead) is dead
    widget = _widget(qtbot)
    assert gs._top_level_of(widget) is widget


def test_dropping_sheets_forgets_the_root_its_children_and_dead_targets(qtbot):
    root = _widget(qtbot)
    child = QWidget(root)
    stranger = _widget(qtbot)
    dead = QWidget()
    gs._PENDING_SHEETS.update({
        1: (root, "a", None), 2: (child, "b", None),
        3: (stranger, "c", None), 4: (dead, "d", None)})
    shiboken_delete(dead)
    assert gs._drop_pending_sheets_inside(root) == 3
    assert list(gs._PENDING_SHEETS) == [3]


def test_dropping_sheets_survives_an_ancestor_query_that_fails(qtbot, monkeypatch):
    root = _widget(qtbot)
    other = _widget(qtbot)

    def boom(widget):
        raise TypeError("bad widget")

    monkeypatch.setattr(root, "isAncestorOf", boom, raising=False)
    gs._PENDING_SHEETS[1] = (other, "x", None)
    assert gs._drop_pending_sheets_inside(root) == 0
    assert 1 in gs._PENDING_SHEETS


def test_post_flush_gives_up_without_qtimer_or_application(monkeypatch):
    with monkeypatch.context() as patch:
        patch.delattr(QtCore, "QTimer")
        gs._post_sheet_flush()
        assert gs._SHEET_GUARD["posted"] is False

    class NoApp:
        @staticmethod
        def instance():
            return None

    monkeypatch.setattr(QtCore, "QCoreApplication", NoApp)
    gs._post_sheet_flush()
    assert gs._SHEET_GUARD["posted"] is False


def test_post_flush_from_a_thread_without_a_live_watcher_stays_unposted(monkeypatch, qapp):
    monkeypatch.setattr(gs, "_on_the_gui_thread", lambda: False)
    monkeypatch.setitem(gs._SHEET_GUARD, "watcher", None)
    gs._post_sheet_flush()
    assert gs._SHEET_GUARD["posted"] is False


def test_widgets_a_restyle_walks_lists_children_and_survives_deletion(qtbot, qapp):
    root = _widget(qtbot)
    child = QWidget(root)
    walked = gs._widgets_a_restyle_walks(root)
    assert walked[0] is root and child in walked
    gone = QWidget()
    shiboken_delete(gone)
    assert gs._widgets_a_restyle_walks(gone) == []
    assert root in gs._widgets_a_restyle_walks(qapp)


def test_flush_does_nothing_from_another_thread(qtbot, monkeypatch):
    widget = _widget(qtbot)
    gs._PENDING_SHEETS[id(widget)] = (widget, "QWidget{}", None)
    result = []
    worker = threading.Thread(target=lambda: result.append(gs._flush_pending_sheets()))
    worker.start()
    worker.join(10)
    assert result == [0]
    assert id(widget) in gs._PENDING_SHEETS


def test_flush_inside_a_delicate_moment_reposts_and_keeps_the_sheets(qtbot, monkeypatch):
    widget = _widget(qtbot)
    gs._PENDING_SHEETS[id(widget)] = (widget, "QWidget{}", None)
    posted = []
    monkeypatch.setattr(gs, "_inside_a_delicate_moment", lambda target=None: True)
    monkeypatch.setattr(gs, "_post_sheet_flush", lambda: posted.append(True))
    assert gs._flush_pending_sheets() == 0
    assert posted == [True]
    assert id(widget) in gs._PENDING_SHEETS


def test_flush_for_one_window_leaves_other_windows_waiting(qtbot, monkeypatch):
    first, second = _widget(qtbot), _widget(qtbot)
    applied = []
    monkeypatch.setattr(gs, "_apply_now", lambda t, text, raw: applied.append((t, text)))
    monkeypatch.setattr(gs, "_inside_a_delicate_moment", lambda target=None: False)
    gs._PENDING_SHEETS[id(first)] = (first, "a{}", None)
    gs._PENDING_SHEETS[id(second)] = (second, "b{}", None)
    assert gs._flush_pending_sheets(first) == 1
    assert applied == [(first, "a{}")]
    assert list(gs._PENDING_SHEETS) == [id(second)]


def test_flush_skips_torn_down_windows_and_widgets(qtbot, monkeypatch):
    window = _widget(qtbot)
    inner = QWidget(window)
    sibling = _widget(qtbot)
    applied = []
    monkeypatch.setattr(gs, "_apply_now", lambda t, text, raw: applied.append(t))
    monkeypatch.setattr(gs, "_inside_a_delicate_moment", lambda target=None: False)
    gs._mark_torn_down(sibling)
    gs._mark_torn_down(inner)
    gs._mark_torn_down(window)
    gs._PENDING_SHEETS[1] = (window, "w{}", None)
    gs._PENDING_SHEETS[2] = (sibling, "s{}", None)
    assert gs._flush_pending_sheets() == 0
    assert applied == []
    gs._TORN_DOWN.discard(gs._address_of(window))
    gs._PENDING_SHEETS[3] = (inner, "i{}", None)
    gs._PENDING_SHEETS[4] = (window, "w{}", None)
    gs._TORN_DOWN.discard(gs._address_of(inner))
    gs._mark_torn_down(inner)
    gs._PENDING_SHEETS[3] = (inner, "i{}", None)
    assert gs._flush_pending_sheets() == 1
    assert applied == [window]


def test_only_a_deferred_delete_marks_an_object(qtbot):
    widget = _widget(qtbot)
    gs._note_sheet_lifecycle(widget, QEvent.Type.Show)
    assert gs._TORN_DOWN == set()
    gs._note_sheet_lifecycle(widget, QEvent.Type.DeferredDelete)
    assert gs._address_of(widget) in gs._TORN_DOWN


def test_watcher_never_consumes_events_and_ignores_unreadable_ones(qtbot, qapp):
    watcher = gs._sheet_guard_watcher_class()()
    victim = _widget(qtbot)
    assert watcher.eventFilter(victim, _Event(QEvent.Type.None_)) is False
    assert watcher.eventFilter(victim, QEvent(QEvent.Type.Show)) is False
    assert gs._TORN_DOWN == set()
    assert watcher.eventFilter(victim, QEvent(QEvent.Type.DeferredDelete)) is False
    assert gs._address_of(victim) in gs._TORN_DOWN


def test_watching_for_teardown_refuses_off_the_gui_thread(qapp):
    result = []
    worker = threading.Thread(target=lambda: result.append(gs._watch_for_teardown()))
    worker.start()
    worker.join(10)
    assert result == [False]


def test_closing_flag_off_resumes_restyling_and_tolerates_no_address(qtbot, monkeypatch):
    window = _widget(qtbot)
    gs._close_without_restyling(window)
    assert gs._address_of(window) in gs._TORN_DOWN
    gs._close_without_restyling(window, closing=False)
    assert gs._TORN_DOWN == set()
    monkeypatch.setattr(gs, "_address_of", lambda obj: None)
    gs._close_without_restyling(window, closing=False)
    assert gs._TORN_DOWN == set()


def test_first_install_wires_every_layer_once(monkeypatch):
    calls = []
    monkeypatch.setattr(gs, "_INSTALLED", False)
    for name in ("_install_sheet_guard", "_install_widget_setters", "_install_layout_setters",
                 "_install_icon_setters", "_install_splitter_setters",
                 "_install_application_sheet"):
        monkeypatch.setattr(gs, name, lambda n=name: calls.append(n))
    assert gs.install_scaling_layer() is True
    assert gs._INSTALLED is True
    assert len(calls) == 6
    assert gs.install_scaling_layer() is False
    assert len(calls) == 6


def test_application_sheet_getter_answers_the_waiting_sheet(qapp, monkeypatch):
    monkeypatch.setitem(gs._ORIGINAL, "guard_raw_app_get", lambda app: "from-qt")
    assert gs._guarded_app_style_sheet(qapp) == "from-qt"
    gs._PENDING_SHEETS[id(qapp)] = (qapp, "QWidget{color:red;}", None)
    assert gs._guarded_app_style_sheet(qapp) == "QWidget{color:red;}"
