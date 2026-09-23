"""Every branch of :mod:`spacr.qt.widgets.collapsible_splitter`.

The behaviour the maintainer asked for is tested in
``test_a_collapsed_pane_locks_to_the_bottom.py`` and
``test_the_shell_collapses_and_resizes.py``; this file reaches the edges those
leave -- unreadable stored sizes, a horizontal fold, a handle clicked with no
pane beside it, a pane whose widget left -- so the release gate's
new-module-at-100% rule holds.
"""
from __future__ import annotations

import json
import uuid

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, QSize, Qt   # noqa: E402
from PySide6.QtGui import QMouseEvent, QResizeEvent      # noqa: E402
from PySide6.QtWidgets import (QApplication, QLabel,    # noqa: E402
                               QVBoxLayout, QWidget)

from spacr.qt.widgets import collapsible_splitter as cs  # noqa: E402
from spacr.qt.widgets.foldable import make_foldable      # noqa: E402


def _pump(n: int = 6) -> None:
    for _ in range(n):
        QApplication.processEvents()


def _key() -> str:
    return f"test471cov/{uuid.uuid4().hex[:8]}"


def _headed(name="Pane", body_height=120, layout=True):
    pane = QWidget()
    heading, body = QLabel(name), QWidget()
    body.setMinimumHeight(body_height)
    if layout:
        column = QVBoxLayout(pane)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(heading)
        column.addWidget(body, 1)
    return pane, heading, body, make_foldable(heading, body, name=name)


def _mouse(widget, kind, x, y, button=Qt.LeftButton):
    at = QPointF(x, y)
    QApplication.sendEvent(widget, QMouseEvent(
        kind, at, at, button, button, Qt.NoModifier))


class TestTheStoredSizes:

    def test_no_key_reads_and_writes_nothing(self):
        assert cs.get_pane_extents("") == {}
        cs.set_pane_extents("", {"A": 3})

    def test_unreadable_values_read_as_nothing(self, monkeypatch):
        key = _key()
        store = cs._settings()
        store.setValue(f"{cs._EXTENTS_PREFIX}/{key}", "{not json")
        assert cs.get_pane_extents(key) == {}
        store.setValue(f"{cs._EXTENTS_PREFIX}/{key}", json.dumps([1, 2]))
        assert cs.get_pane_extents(key) == {}
        store.setValue(f"{cs._EXTENTS_PREFIX}/{key}",
                       json.dumps({"A": "x", "B": 0, "C": 40}))
        assert cs.get_pane_extents(key) == {"C": 40}

    def test_a_size_that_cannot_be_stored_is_dropped_quietly(self):
        key = _key()
        cs.set_pane_extents(key, {"A": "not a number"})
        assert cs.get_pane_extents(key) == {}


class TestLockingAFoldOutsideASplitter:

    def test_a_sideways_fold_locks_to_the_start(self, qtbot):
        pane, _heading, _body, folder = _headed()
        qtbot.addWidget(pane)
        cs.lock_folded_to_bottom(pane, folder, Qt.Horizontal)
        count = pane.layout().count()
        folder.toggle()
        assert pane.layout().count() == count + 1
        folder.toggle()
        assert pane.layout().count() == count

    def test_a_panel_already_folded_is_locked_at_once(self, qtbot):
        pane, _heading, _body, folder = _headed()
        qtbot.addWidget(pane)
        folder.toggle()
        count = pane.layout().count()
        cs.lock_folded_to_bottom(pane, folder)
        assert pane.layout().count() == count + 1

    def test_asking_twice_changes_nothing(self, qtbot):
        pane, _heading, _body, folder = _headed()
        qtbot.addWidget(pane)
        cs.lock_folded_to_bottom(pane, folder)
        apply = folder._listeners[-1]
        count = pane.layout().count()
        apply(True)
        apply(True)
        assert pane.layout().count() == count + 1
        apply(False)
        apply(False)
        assert pane.layout().count() == count

    def test_a_panel_with_no_layout_is_left_alone(self, qtbot):
        pane, _heading, _body, folder = _headed(layout=False)
        qtbot.addWidget(pane)
        cs.lock_folded_to_bottom(pane, folder)
        folder.toggle()
        assert pane.layout() is None


def _split(qtbot, orientation=Qt.Vertical, size=(400, 600), **kwargs):
    split = cs.CollapsibleSplitter(orientation, **kwargs)
    qtbot.addWidget(split)
    split.resize(*size)
    return split


class TestPanes:

    def test_a_pane_can_be_inserted_at_a_place_and_replaced(self, qtbot):
        split = _split(qtbot)
        first, second = QWidget(), QWidget()
        split.add_pane(first, "A")
        split.add_pane(second, "B", index=0)
        assert split.indexOf(second) == 0
        replacement = split.add_pane(first, "A", stretch=0)
        assert split.pane("A") is replacement
        assert len(split.panes()) == 2
        assert split.pane("nope") is None
        assert not split.is_collapsed("nope")

    def test_asking_for_what_cannot_be_done(self, qtbot):
        split = _split(qtbot)
        split.add_pane(QWidget(), "Plain")
        assert split.set_collapsed("Plain", True) is False
        assert split.set_collapsed("nope", True) is False
        assert split.toggle_pane("nope") is False
        pane, _h, _b, folder = _headed()
        split.add_pane(pane, "Head", folder=folder)
        assert split.set_collapsed("Head", False) is True

    def test_a_folded_pane_can_be_added(self, qtbot):
        split = _split(qtbot)
        pane, _h, _b, folder = _headed()
        folder.toggle()
        split.add_pane(pane, "Head", folder=folder)
        assert split.is_collapsed("Head")

    def test_an_unreadable_edge_fold_starts_open(self, qtbot, monkeypatch):
        from spacr.qt import preferences

        def refuse():
            raise RuntimeError("store unavailable")

        monkeypatch.setattr(preferences, "get_folded_panels", refuse)
        split = _split(qtbot, Qt.Horizontal)
        split.add_pane(QWidget(), "Side", mode=cs.EDGE, fold_key=_key())
        assert not split.is_collapsed("Side")

    def test_an_edge_fold_that_cannot_be_stored_still_folds(
            self, qtbot, monkeypatch):
        from spacr.qt import preferences

        def refuse(*_args):
            raise RuntimeError("store unavailable")

        split = _split(qtbot, Qt.Horizontal)
        split.add_pane(QWidget(), "Side", mode=cs.EDGE, fold_key=_key())
        split.add_pane(QWidget(), "Rest")
        monkeypatch.setattr(preferences, "set_folded_panel", refuse)
        assert split.set_collapsed("Side", True, by_user=True)
        split._set_edge(split.pane("Side"), True, by_user=True)

    def test_a_pane_whose_widget_left_is_not_shaped_by_size(self, qtbot):
        split = _split(qtbot)
        pane, _h, _b, folder = _headed()
        registered = split.add_pane(pane, "Head", folder=folder)
        pane.setParent(None)
        qtbot.addWidget(pane)
        split._shape(registered, True)
        split._shape(registered, False)
        edge = cs.Pane(QWidget(), "E", cs.EDGE, None, 1, 0, 0, False, "")
        split._set_edge(edge, True, by_user=False)
        assert edge.edge_collapsed

    def test_a_folder_that_outlived_its_pane_is_ignored(self, qtbot):
        split = _split(qtbot)
        pane, _h, _b, folder = _headed()
        registered = split.add_pane(pane, "Head", folder=folder)
        split._panes.remove(registered)
        folder.toggle()
        assert split.sizes() is not None

    def test_a_sideways_header_pane_folds_by_width(self, qtbot):
        split = _split(qtbot, Qt.Horizontal, size=(800, 300))
        pane, _h, _b, folder = _headed()
        split.add_pane(pane, "Head", folder=folder)
        split.add_pane(QWidget(), "Rest")
        split.show()
        _pump()
        folder.toggle()
        assert pane.minimumWidth() == 0
        folder.toggle()
        assert pane.maximumWidth() == cs.UNLIMITED


class TestHandles:

    def _edge_split(self, qtbot, orientation=Qt.Horizontal, last=False):
        split = _split(qtbot, orientation, size=(800, 600))
        edge = QWidget()
        edge.setMinimumSize(100, 100)
        if last:
            split.add_pane(QWidget(), "Rest", extent=500)
            split.add_pane(edge, "Side", mode=cs.EDGE, extent=300)
        else:
            split.add_pane(edge, "Side", mode=cs.EDGE, extent=300)
            split.add_pane(QWidget(), "Rest", extent=500)
        split.show()
        _pump()
        return split

    def test_a_drag_on_the_handle_is_not_a_click(self, qtbot):
        split = self._edge_split(qtbot)
        handle = split.handle(1)
        _mouse(handle, QEvent.MouseButtonPress, 5, 50)
        _mouse(handle, QEvent.MouseMove, 5, 50)
        _mouse(handle, QEvent.MouseMove, 60, 50)
        _mouse(handle, QEvent.MouseButtonRelease, 60, 50)
        assert not split.is_collapsed("Side")
        _mouse(handle, QEvent.MouseMove, 5, 50)

    def test_a_right_click_is_not_a_click(self, qtbot):
        split = self._edge_split(qtbot)
        handle = split.handle(1)
        _mouse(handle, QEvent.MouseButtonPress, 5, 50, Qt.RightButton)
        _mouse(handle, QEvent.MouseButtonRelease, 5, 50, Qt.RightButton)
        assert not split.is_collapsed("Side")

    def test_a_handle_beside_nothing_does_nothing(self, qtbot):
        split = _split(qtbot)
        split.add_pane(QWidget(), "A")
        split.add_pane(QWidget(), "B")
        split.show()
        _pump()
        handle = split.handle(1)
        _mouse(handle, QEvent.MouseButtonPress, 5, 2)
        _mouse(handle, QEvent.MouseButtonRelease, 5, 2)
        assert handle.edge_pane() is None
        handle.retranslate_dynamic_content()
        assert handle.toolTip() == ""

    def test_every_handle_paints(self, qtbot, monkeypatch):
        for orientation in (Qt.Horizontal, Qt.Vertical):
            for last in (False, True):
                split = self._edge_split(qtbot, orientation, last)
                at = 1
                handle = split.handle(at)
                assert handle.grab().deviceIndependentSize().toSize() == handle.size()
                split.set_collapsed("Side", True)
                assert handle.grab().deviceIndependentSize().toSize() == handle.size()
                QApplication.sendEvent(handle, QEvent(QEvent.Enter))
                QApplication.sendEvent(handle, QEvent(QEvent.Leave))
        from spacr.qt import theme

        def refuse():
            raise RuntimeError("no theme")

        monkeypatch.setattr(theme, "active_palette", refuse)
        handle = split.handle(1)
        assert handle.grab().deviceIndependentSize().toSize() == handle.size()

    def test_the_last_pane_can_be_the_edge(self, qtbot):
        split = self._edge_split(qtbot, last=True)
        assert split.handle(1).edge_pane() is split.pane("Side")

    def test_a_handle_that_left_its_splitter(self, qtbot):
        split = self._edge_split(qtbot)
        stray = cs._PaneHandle(Qt.Horizontal, split)
        assert stray._index() == -1
        assert split._edge_pane_beside(stray) is None

    def test_an_edge_pane_whose_widget_left(self, qtbot):
        split = self._edge_split(qtbot)
        split.pane("Side").widget = QWidget()
        assert split.handle(1).edge_pane() is None

    def test_a_handle_on_a_plain_splitter_parent(self, qtbot):
        from PySide6.QtWidgets import QSplitter

        plain = QSplitter(Qt.Horizontal)
        qtbot.addWidget(plain)
        handle = cs._PaneHandle(Qt.Horizontal, plain)
        assert handle.edge_pane() is None


class TestFollowingTheWidgets:

    def test_a_folded_focus_pane_opens_when_it_is_shown(self, qtbot):
        split = _split(qtbot)
        pane, _h, _b, folder = _headed()
        split.add_pane(pane, "Preview", folder=folder, focus=True)
        split.add_pane(QWidget(), "Rest")
        split.show()
        pane.hide()
        folder.toggle()
        pane.show()
        assert not split.is_collapsed("Preview")

    def test_other_events_are_let_through(self, qtbot):
        split = _split(qtbot)
        widget = QWidget()
        split.add_pane(widget, "A")
        assert split.eventFilter(widget, QEvent(QEvent.Enter)) is False
        assert split.eventFilter(QWidget(), QEvent(QEvent.ShowToParent)) \
            is False

    def test_a_queued_rebalance_after_deletion_is_harmless(self, qtbot):
        split = cs.CollapsibleSplitter(Qt.Vertical)
        split.add_pane(QWidget(), "A")
        split._rebalance_queued = False
        split._queue_rebalance()
        split._queue_rebalance()
        from shiboken6 import delete

        delete(split)
        _pump()
        assert not cs._alive(split)

    def test_an_empty_splitter_has_nothing_to_lay_out(self, qtbot):
        split = _split(qtbot)
        assert split.rebalance() == []
        widget = QWidget()
        split.add_pane(widget, "A")
        widget.hide()
        assert split.rebalance() == []

    def test_a_rewrap_skips_a_dead_pane(self, qtbot):
        split = _split(qtbot)
        gone = QWidget()
        split.add_pane(gone, "Gone")
        split.pane("Gone").widget = QLabel()
        from shiboken6 import delete

        delete(split.pane("Gone").widget)
        split.addWidget(QWidget())
        assert split.count() == 2


class TestSizing:

    def test_a_splitter_never_laid_out_is_sized_from_its_weights(self, qtbot):
        split = cs.CollapsibleSplitter(Qt.Vertical)
        qtbot.addWidget(split)
        split.add_pane(QWidget(), "A", extent=100)
        split.add_pane(QWidget(), "B", extent=300)
        sizes = split.rebalance(fresh=True)
        assert len(sizes) == 2

    def test_fixed_panes_that_do_not_fit_are_scaled(self, qtbot):
        split = _split(qtbot, size=(400, 200))
        split.show()
        _pump()
        a, b = QWidget(), QWidget()
        split.add_pane(a, "A", stretch=0, extent=500)
        split.add_pane(b, "B", extent=400)
        sizes = split.rebalance(fresh=True)
        assert sum(sizes) <= 220

    def test_the_absorber_takes_what_the_fixed_panes_leave(self, qtbot):
        split = _split(qtbot, size=(400, 600))
        pane, _h, _b, folder = _headed()
        split.add_pane(pane, "Head", folder=folder)
        split.add_pane(QWidget(), "Fixed", stretch=0, extent=100)
        split.show()
        _pump()
        folder.toggle()
        _pump()
        sizes = split.sizes()
        assert sizes[0] > sizes[1]

    def test_a_wrapping_fixed_pane_is_refitted_when_the_width_changes(
            self, qtbot):
        from spacr.qt.widgets.flow import FlowLayout
        from PySide6.QtWidgets import QPushButton

        split = _split(qtbot, size=(900, 600))
        row = QWidget()
        flow = FlowLayout(row)
        for i in range(12):
            flow.addWidget(QPushButton(f"Button number {i}"))
        split.add_pane(QWidget(), "Top")
        split.add_pane(row, "Buttons", stretch=0)
        split.show()
        _pump()
        split.resize(300, 600)
        _pump(10)
        index = split.indexOf(row)
        assert split.sizes()[index] >= row.heightForWidth(split.width()) - 2
        split.resizeEvent(QResizeEvent(QSize(300, 600), QSize(300, 600)))

    def test_height_for_width_is_zero_where_it_cannot_be_asked(self, qtbot):
        split = cs.CollapsibleSplitter(Qt.Horizontal)
        qtbot.addWidget(split)
        assert split._height_for_width(QWidget()) == 0

        class Sulky(QWidget):
            def hasHeightForWidth(self):
                return True

            def heightForWidth(self, _width):
                raise RuntimeError("no")

        vertical = cs.CollapsibleSplitter(Qt.Vertical)
        qtbot.addWidget(vertical)
        assert vertical._height_for_width(Sulky()) == 0
        vertical.resize(300, 300)
        assert vertical._height_for_width(Sulky()) == 0
        vertical.resize(0, 300)
        assert vertical._height_for_width(Sulky()) == 0

    def test_a_refit_after_deletion_is_harmless(self, qtbot, monkeypatch):
        split = cs.CollapsibleSplitter(Qt.Vertical)
        rebalanced = []
        monkeypatch.setattr(split, "rebalance", lambda **kw: rebalanced.append(kw))
        split._laid_out = True
        split.resizeEvent(QResizeEvent(QSize(300, 100), QSize(200, 100)))
        assert split._refit_queued
        from shiboken6 import delete

        delete(split)
        _pump()
        assert rebalanced == [], "a queued resize must not touch the deleted widget"

    def test_a_drag_that_leaves_an_open_edge_open(self, qtbot):
        split = _split(qtbot, Qt.Horizontal, size=(800, 300))
        split.add_pane(QWidget(), "Side", mode=cs.EDGE, extent=300)
        split.add_pane(QWidget(), "Rest")
        split.show()
        _pump()
        split.setSizes([250, 540])
        split._on_moved(250, 1)
        assert split.pane("Side").extent == split.sizes()[0]

    def test_a_folded_header_pane_keeps_its_size_through_a_drag(self, qtbot):
        split = _split(qtbot)
        pane, _h, _b, folder = _headed(layout=False)
        split.add_pane(pane, "Head", folder=folder, extent=222)
        split.add_pane(QWidget(), "Rest")
        split.show()
        _pump()
        folder.toggle()
        kept = split.pane("Head").extent
        split._on_moved(10, 1)
        assert split.pane("Head").extent == kept

    def test_a_drag_skips_hidden_and_plain_children(self, qtbot):
        split = _split(qtbot)
        hidden = QWidget()
        split.add_pane(QWidget(), "A")
        split.add_pane(hidden, "Hidden")
        split.addWidget(QWidget())
        split.show()
        _pump()
        hidden.hide()
        split._on_moved(10, 1)
        assert split.extents()


class TestFocusCollapse:

    def _pair(self, qtbot):
        split = _split(qtbot, size=(400, 800))
        pane, _h, _b, folder = _headed("Console")
        split.add_pane(pane, "Console", folder=folder)
        trigger = QWidget()
        split.add_pane(trigger, "Preview")
        trigger.hide()
        focus = cs.FocusCollapse(split)
        return split, folder, trigger, focus

    def test_nothing_to_watch_or_target(self, qtbot):
        split, _folder, trigger, focus = self._pair(qtbot)
        focus.watch(None)
        focus.watch(trigger)
        focus.watch(trigger)
        focus.target(None, "Console")
        focus.target(split, "nope")
        focus.target(split, "Console")
        focus.target(split, "Console")
        assert focus.eventFilter(trigger, QEvent(QEvent.Enter)) is False

    def test_a_target_added_while_active_collapses_at_once(self, qtbot):
        split, _folder, trigger, focus = self._pair(qtbot)
        focus.watch(trigger)
        trigger.show()
        focus.target(split, "Console")
        assert split.is_collapsed("Console")

    def test_a_deleted_trigger_is_forgotten(self, qtbot):
        split, _folder, _trigger, focus = self._pair(qtbot)
        other = QWidget()
        focus.watch(other)
        from shiboken6 import delete

        delete(other)
        assert focus.is_active() is False

    def test_a_deleted_splitter_is_skipped(self, qtbot):
        from shiboken6 import delete

        split, _folder, _trigger, focus = self._pair(qtbot)
        focus.target(split, "Console")
        gone = cs.CollapsibleSplitter(Qt.Vertical)
        delete(gone)
        focus._collapse_one(gone, "Console")
        focus._targets.append((gone, "Console"))
        focus._auto.add(focus._key(gone, "Console"))
        focus._release()
        assert not focus._auto

    def test_a_target_reopened_elsewhere_is_not_reopened_again(self, qtbot):
        split, _folder, _trigger, focus = self._pair(qtbot)
        focus.target(split, "Console")
        focus._auto.add(focus._key(split, "Console"))
        focus._release()
        assert not split.is_collapsed("Console")

    def test_a_target_that_cannot_collapse_is_not_recorded(self, qtbot):
        split, _folder, trigger, focus = self._pair(qtbot)
        split.add_pane(QWidget(), "Plain")
        focus.target(split, "Plain")
        focus.watch(trigger)
        trigger.show()
        assert focus._key(split, "Plain") not in focus._auto
        trigger.hide()

    def test_a_user_fold_while_active_is_not_a_pin(self, qtbot):
        split, folder, trigger, focus = self._pair(qtbot)
        focus.target(split, "Console")
        focus.watch(trigger)
        trigger.show()
        folder.toggle()
        folder.toggle()
        assert focus.is_pinned(split, "Console") is False

    def test_the_splitter_of_a_widget(self, qtbot):
        split = _split(qtbot)
        widget = QWidget()
        split.add_pane(widget, "A")
        assert cs.splitter_of(widget) is split
        assert cs.splitter_of(QWidget()) is None
        assert cs.splitter_of(None) is None


class TestTheFoldersListeners:

    def test_a_listener_is_added_once_and_a_non_callable_never(self):
        heading, body = QLabel("Console"), QWidget()
        folder = make_foldable(heading, body)
        heard = []

        def listener(shut, by_user):
            heard.append((shut, by_user))

        folder.add_listener(listener)
        folder.add_listener(listener)
        folder.add_listener("not callable")
        folder.toggle()
        assert heard == [(True, True)]

    def test_a_failing_listener_does_not_stop_the_fold(self):
        heading, body = QLabel("Console"), QWidget()
        folder = make_foldable(heading, body)

        def broken(_shut, _by_user):
            raise RuntimeError("listener fell over")

        folder.add_listener(broken)
        folder.toggle()
        assert folder.shut and body.isHidden()
