"""Item 471, slice B: the mechanism -- a collapsed pane locks to the bottom,
every edge drags, and collapsing is where a drag stops.

The request: "whenever anything is collapsed it should auto loch to the bottom
of the container it is in (now for example when i collapse the console it
collapses to the middle of the container.)", and "whenever possible make the
container expandable or shrinkable by draging the edges". One mechanism serves
both: :mod:`spacr.qt.widgets.collapsible_splitter`. The module screen that
uses it is tested in ``test_the_shell_collapses_and_resizes.py``.
"""
from __future__ import annotations

import uuid

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent                   # noqa: E402
from PySide6.QtWidgets import (QApplication, QLabel,    # noqa: E402
                               QVBoxLayout, QWidget)

from spacr.qt.widgets.collapsible_splitter import (     # noqa: E402
    EDGE, CollapsibleSplitter, FocusCollapse, get_pane_extents)
from spacr.qt.widgets.foldable import make_foldable     # noqa: E402


def _key(tag: str) -> str:
    """A persistence key no other test (or run) shares."""
    return f"test471/{tag}/{uuid.uuid4().hex[:8]}"


def _pump(n: int = 6) -> None:
    for _ in range(n):
        QApplication.processEvents()


def _headed(name: str, body_height: int = 200):
    """A pane with a clickable heading over a body, as the console is."""
    pane = QWidget()
    column = QVBoxLayout(pane)
    column.setContentsMargins(0, 0, 0, 0)
    heading = QLabel(name)
    body = QWidget()
    body.setMinimumHeight(body_height)
    column.addWidget(heading)
    column.addWidget(body, 1)
    return pane, heading, body, make_foldable(heading, body, name=name)


def _stack(qtbot, *names, height=700, persist_key=""):
    """A vertical splitter of headed panes, shown at a real size."""
    split = CollapsibleSplitter(Qt.Vertical, persist_key=persist_key)
    parts = {}
    for name in names:
        pane, heading, body, folder = _headed(name)
        split.add_pane(pane, name, folder=folder, extent=300)
        parts[name] = (pane, heading, body, folder)
    qtbot.addWidget(split)
    split.resize(500, height)
    split.show()
    _pump()
    return split, parts


def _bottom_of(widget, split) -> int:
    return widget.mapTo(split, QPoint(0, widget.height())).y()


class TestACollapsedPaneLocksToTheBottom:
    """Point 7, the reported bug: the console folded to the MIDDLE."""

    def test_a_folded_last_pane_puts_its_heading_at_the_bottom(self, qtbot):
        split, parts = _stack(qtbot, "Console")
        pane, heading, _body, folder = parts["Console"]
        folder.toggle()
        _pump()
        assert _bottom_of(heading, split) >= split.height() - 4, (
            f"the folded heading ends at {_bottom_of(heading, split)} of "
            f"{split.height()}: that is the console collapsing to the middle")

    def test_the_room_goes_to_the_pane_above(self, qtbot):
        split, parts = _stack(qtbot, "Preview", "Console")
        _pane, heading, _body, folder = parts["Console"]
        before = split.sizes()
        folder.toggle()
        _pump()
        after = split.sizes()
        assert after[1] <= heading.sizeHint().height() + 8
        assert after[0] > before[0] + 150
        assert _bottom_of(heading, split) >= split.height() - 4

    def test_every_folded_heading_stacks_at_the_bottom(self, qtbot):
        split, parts = _stack(qtbot, "Console", "System", "Actions")
        for name in ("Console", "System", "Actions"):
            parts[name][3].toggle()
        _pump()
        headings = [parts[name][1] for name in ("Console", "System",
                                                 "Actions")]
        assert _bottom_of(headings[-1], split) >= split.height() - 4
        tops = [h.mapTo(split, QPoint(0, 0)).y() for h in headings]
        assert tops == sorted(tops)
        assert tops[0] > split.height() // 2, (
            "three folded strips belong at the bottom, not spread over it")

    def test_a_window_resize_does_not_reopen_the_gap(self, qtbot):
        split, parts = _stack(qtbot, "Preview", "Console")
        heading, folder = parts["Console"][1], parts["Console"][3]
        folder.toggle()
        _pump()
        split.resize(500, 1000)
        _pump()
        assert split.sizes()[1] <= heading.sizeHint().height() + 8


class TestDraggingIsTheSameMechanism:
    """The addendum: drag an edge to resize; collapse is the limit."""

    def test_a_drag_past_the_minimum_collapses_the_pane(self, qtbot):
        split, parts = _stack(qtbot, "Preview", "Console")
        folder = parts["Console"][3]
        split.setSizes([split.height(), 0])
        split.splitterMoved.emit(split.height(), 1)
        _pump()
        assert folder.shut, "dragged to nothing, the console must be folded"
        assert split.sizes()[1] > 0, "and its heading must still be there"

    def test_a_collapsed_heading_can_always_be_reopened(self, qtbot):
        split, parts = _stack(qtbot, "Preview", "Console")
        folder, body = parts["Console"][3], parts["Console"][2]
        folder.toggle()
        _pump()
        folder.toggle()
        _pump()
        assert not body.isHidden()
        assert split.sizes()[1] >= 200

    def test_dragged_sizes_survive_a_new_splitter(self, qtbot):
        key = _key("drag")
        first, _ = _stack(qtbot, "Preview", "Console", persist_key=key)
        first.setSizes([250, 440])
        first.splitterMoved.emit(250, 1)
        stored = get_pane_extents(key)
        assert stored.get("Console", 0) >= 400
        second, _ = _stack(qtbot, "Preview", "Console", persist_key=key)
        _pump()
        assert abs(second.sizes()[1] - first.sizes()[1]) <= 12

    def test_sizes_are_kept_by_name_not_by_position(self, qtbot):
        """A pane added to a screen later must not scramble the old sizes."""
        key = _key("byname")
        first, _ = _stack(qtbot, "Preview", "Console", persist_key=key)
        first.setSizes([250, 440])
        first.splitterMoved.emit(250, 1)
        assert set(get_pane_extents(key)) == {"Preview", "Console"}


class TestTheSettingsColumnCollapsesToTheLeft:

    def _body(self, qtbot, fold_key=""):
        split = CollapsibleSplitter(Qt.Horizontal)
        settings, runtime = QWidget(), QWidget()
        settings.setMinimumWidth(200)
        split.add_pane(settings, "Settings", mode=EDGE, fold_key=fold_key,
                       extent=400)
        split.add_pane(runtime, "Runtime", stretch=2, extent=800)
        qtbot.addWidget(split)
        split.resize(1200, 600)
        split.show()
        _pump()
        return split, settings

    def _click(self, widget):
        centre = QPointF(widget.width() / 2, widget.height() / 2)
        for kind in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease):
            QApplication.sendEvent(widget, QMouseEvent(
                kind, centre, centre, Qt.LeftButton, Qt.LeftButton,
                Qt.NoModifier))

    def test_a_click_on_the_handle_folds_it_left_and_back(self, qtbot):
        split, _settings = self._body(qtbot)
        width = split.sizes()[0]
        handle = split.handle(1)
        self._click(handle)
        _pump()
        assert split.is_collapsed("Settings")
        assert split.sizes()[0] == 0
        assert split.handle(1).isVisible(), (
            "the handle is the strip that brings the column back")
        self._click(handle)
        _pump()
        assert not split.is_collapsed("Settings")
        assert abs(split.sizes()[0] - width) <= 12

    def test_the_handle_says_what_a_click_does(self, qtbot):
        split, _settings = self._body(qtbot)
        assert "hide" in split.handle(1).toolTip().lower()
        split.set_collapsed("Settings", True, by_user=True)
        assert "show" in split.handle(1).toolTip().lower()

    def test_a_drag_to_nothing_is_a_collapse(self, qtbot):
        split, _settings = self._body(qtbot)
        split.setSizes([0, 1200])
        split.splitterMoved.emit(0, 1)
        assert split.is_collapsed("Settings")
        split.setSizes([300, 900])
        split.splitterMoved.emit(300, 1)
        assert not split.is_collapsed("Settings")

    def test_a_users_collapse_survives_a_restart(self, qtbot):
        key = _key("settings")
        split, _ = self._body(qtbot, fold_key=key)
        split.set_collapsed("Settings", True, by_user=True)
        again, _ = self._body(qtbot, fold_key=key)
        assert again.is_collapsed("Settings")
        again.set_collapsed("Settings", False, by_user=True)

    def test_an_automatic_collapse_is_not_remembered(self, qtbot):
        key = _key("settings-auto")
        split, _ = self._body(qtbot, fold_key=key)
        split.set_collapsed("Settings", True, by_user=False)
        again, _ = self._body(qtbot, fold_key=key)
        assert not again.is_collapsed("Settings")

    def test_a_wrapping_container_inherits_the_pane(self, qtbot):
        """The settings search strip replaces the column with a container."""
        split, settings = self._body(qtbot)
        container = QWidget()
        column = QVBoxLayout(container)
        column.addWidget(QLabel("search"))
        column.addWidget(settings, 1)
        split.insertWidget(0, container)
        assert split.pane("Settings").widget is container
        split.set_collapsed("Settings", True, by_user=True)
        assert split.sizes()[0] == 0


class TestAutoCollapseNeverFightsTheUser:
    """Points 2-4 and WHAT TO BE CAREFUL WITH."""

    def _shell(self, qtbot):
        split, parts = _stack(qtbot, "Preview", "Console", "System",
                              "Actions", height=900)
        preview = parts["Preview"][0]
        preview.hide()
        focus = FocusCollapse(split)
        focus.watch(preview)
        for name in ("Console", "System", "Actions"):
            focus.target(split, name)
        _pump()
        return split, parts, preview, focus

    def test_a_focus_pane_collapses_the_rest_and_gives_it_back(self, qtbot):
        split, _parts, preview, _focus = self._shell(qtbot)
        preview.show()
        _pump()
        assert all(split.is_collapsed(n)
                   for n in ("Console", "System", "Actions"))
        assert split.sizes()[0] > 600, "the preview takes the height"
        preview.hide()
        _pump()
        assert not any(split.is_collapsed(n)
                       for n in ("Console", "System", "Actions"))

    def test_what_the_user_opens_stays_open_for_the_view(self, qtbot):
        split, parts, preview, focus = self._shell(qtbot)
        preview.show()
        parts["Console"][3].toggle()
        assert not split.is_collapsed("Console")
        assert focus.is_pinned(split, "Console")
        preview.hide()
        preview.show()
        _pump()
        assert not split.is_collapsed("Console"), (
            "reopened by hand, then collapsed again by the next preview: "
            "that is the fight item 471 rules out")
        assert split.is_collapsed("System")

    def test_a_drag_open_is_a_pin_too(self, qtbot):
        split, _parts, preview, focus = self._shell(qtbot)
        preview.show()
        _pump()
        split.set_collapsed("Console", False, by_user=True)
        assert focus.is_pinned(split, "Console")

    def test_leaving_the_screen_ends_the_view(self, qtbot):
        split, parts, preview, focus = self._shell(qtbot)
        preview.show()
        parts["Console"][3].toggle()
        focus.end_view()
        focus.begin_view()
        assert split.is_collapsed("Console"), (
            "a new visit starts from the rule, not from last visit's pins")

    def test_opening_a_pane_before_any_preview_is_not_a_pin(self, qtbot):
        split, parts, preview, focus = self._shell(qtbot)
        parts["Console"][3].toggle()
        parts["Console"][3].toggle()
        assert not focus.is_pinned(split, "Console")
        preview.show()
        assert split.is_collapsed("Console")

    def test_what_the_user_folded_is_not_reopened(self, qtbot):
        split, parts, preview, _focus = self._shell(qtbot)
        parts["System"][3].toggle()
        preview.show()
        preview.hide()
        assert split.is_collapsed("System")
        assert not split.is_collapsed("Console")

    def test_an_automatic_fold_is_not_stored(self, qtbot):
        from spacr.qt.preferences import get_folded_panels

        key = _key("console-fold")
        heading, body = QLabel("Console"), QWidget()
        folder = make_foldable(heading, body, persist_key=key)
        folder.set_shut(True, by_user=False)
        assert key not in get_folded_panels()
        folder.set_shut(False, by_user=False)
        folder.toggle()
        assert get_folded_panels().get(key)
        folder.toggle()


