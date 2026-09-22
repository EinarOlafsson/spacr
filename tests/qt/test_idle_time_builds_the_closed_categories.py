"""The closed settings categories are built while the screen sits idle.

A category closed at open waits until it is opened
(``AppScreen._build_a_waiting_heading``), and the first open then pays for the
build: 36-200 ms on a click, measured 2026-09-22. ``_IdlePrebuild`` runs the
same steps beforehand, a few milliseconds at a time, only while nobody is
using the screen. These tests hold both halves:

  * it is the SAME build: a category built in idle time equals one built by a
    click and one built with the panel -- rows, captions, enabled state,
    help, values and the search strip's index -- and a click in the middle of
    an idle build finishes that build rather than starting another;
  * it gets out of the way: input stops it before the next slice, a hidden
    screen does no work, and it is sliced rather than one long build.

Also held here: a greyed setting's NAME says why it is greyed, whichever way
the row was built, and a language pass no longer drops that reason.
"""
from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from PySide6.QtCore import QEvent, QEventLoop, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QCursor, QMouseEvent                    # noqa: E402
from PySide6.QtWidgets import (QApplication, QFormLayout,  # noqa: E402
                               QLabel, QWidget)


def _pump(seconds: float, each=None) -> None:
    app = QApplication.instance()
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        if each is not None:
            each()
        app.processEvents(QEventLoop.AllEvents, 10)
        time.sleep(0.001)


def _window(qtbot, key, *, eager: bool = False, idle: bool = True):
    from spacr.qt import register_self_registering_modules
    from spacr.qt.app import MainWindow
    from spacr.qt.screens import app_screen as aps

    register_self_registering_modules()
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1400, 900)
    window.show()
    _pump(0.3)
    saved_wait = aps.AppScreen._a_category_may_wait
    saved_idle = aps._IdlePrebuild.IDLE_MS
    if eager:
        aps.AppScreen._a_category_may_wait = (
            lambda self, title, keys=(): False)
    if not idle:
        aps._IdlePrebuild.IDLE_MS = 10 ** 8
    try:
        window._on_nav_selected(key)
        _pump(0.1)
    finally:
        aps.AppScreen._a_category_may_wait = saved_wait
    return window, window._screens[key], saved_idle


def _restore_idle(saved) -> None:
    from spacr.qt.screens import app_screen as aps

    aps._IdlePrebuild.IDLE_MS = saved


def _waiting(screen) -> list:
    return [section for section in screen._settings_sections
            if screen._heading_is_waiting(section)]


def _until_built(screen, seconds: float = 20.0) -> None:
    end = time.perf_counter() + seconds
    while _waiting(screen) and time.perf_counter() < end:
        _pump(0.2)


def _rows(screen) -> dict:
    """Every category's rows as a user reads them, keyed by title."""
    model = screen._settings_model
    by_widget = {id(widget): key for key, widget
                 in model._widgets.built_items()}
    found = {}
    for section in screen._settings_sections:
        form = getattr(section, "_form", None)
        if not isinstance(form, QFormLayout):
            continue
        rows = []
        for index in range(form.rowCount()):
            field_item = form.itemAt(index, QFormLayout.FieldRole)
            label_item = form.itemAt(index, QFormLayout.LabelRole)
            field = field_item.widget() if field_item else None
            key = by_widget.get(id(field)) if field is not None else None
            if key is None and field is not None:
                key = next((by_widget[id(child)]
                            for child in field.findChildren(QWidget)
                            if id(child) in by_widget), None)
            label = label_item.widget() if label_item else None
            captions = tuple(
                (each.text(), each.toolTip(), each.isEnabled())
                for each in label.findChildren(QLabel)) \
                if label is not None else ()
            control = model._widgets.built(key) if key else None
            rows.append((
                key, captions, form.isRowVisible(index),
                control.isEnabled() if control is not None else None,
                control.toolTip() if control is not None else "",
                type(field).__name__ if field is not None else None,
                control.palette().color(control.foregroundRole()).name()
                if control is not None else "",
            ))
        found[(section.property("settingsCategorySource"),
               section.isHidden())] = tuple(rows)
    return found


def _search(screen) -> tuple:
    bar = screen._settings_search
    return (sorted(bar.indexed_keys()), sorted(bar.visible_keys()),
            bar.count_text())


def _move_the_mouse(screen, state) -> None:
    now = time.perf_counter()
    if now < state["next"]:
        return
    state["next"] = now + 0.02
    state["n"] = state.get("n", 0) + 1
    QCursor.setPos(300 + state["n"] % 40, 300)
    event = QMouseEvent(
        QEvent.MouseMove, QPointF(200, 200),
        QPointF(screen.mapToGlobal(QPoint(200, 200))),
        Qt.NoButton, Qt.NoButton, Qt.NoModifier)
    QApplication.instance().postEvent(screen, event)


# -- it is the same build -------------------------------------------------------

def test_idle_time_builds_every_closed_category(qtbot):
    _w, screen, saved = _window(qtbot, "classify_merged")
    try:
        assert len(_waiting(screen)) >= 5
        _until_built(screen)
        assert _waiting(screen) == []
        assert not screen._settings_model._widgets.keys_to_come()
        builder = screen.__dict__["_idle_prebuild"]
        assert len(builder.slices_ms) > 5 * 5, "the build was not sliced"
    finally:
        _restore_idle(saved)


def test_a_category_built_in_idle_time_equals_one_built_by_a_click(qtbot):
    for key in ("classify_merged", "regression"):
        _w1, idle_built, saved = _window(qtbot, key)
        _until_built(idle_built)
        _w2, clicked, _ = _window(qtbot, key, idle=False)
        _restore_idle(saved)
        clicked._open_every_waiting_heading()
        _w3, eager, _ = _window(qtbot, key, eager=True, idle=False)
        _restore_idle(saved)
        _pump(0.2)
        assert _rows(idle_built) == _rows(clicked) == _rows(eager), key
        assert (idle_built._settings_model.collect()
                == clicked._settings_model.collect()
                == eager._settings_model.collect()), key
        assert _search(idle_built) == _search(clicked) == _search(eager), key


def test_opening_a_prebuilt_category_builds_nothing(qtbot, monkeypatch):
    from spacr.qt.screens import app_screen as aps

    _w, screen, saved = _window(qtbot, "measure")
    try:
        _until_built(screen)
        ran = []
        real = aps.AppScreen._waiting_heading_steps
        monkeypatch.setattr(aps.AppScreen, "_waiting_heading_steps",
                            lambda self, section: ran.append(section)
                            or real(self, section))
        for section in screen.rendered_settings_sections():
            section.set_expanded(True)
        _pump(0.1)
        assert ran == []
    finally:
        _restore_idle(saved)


def test_a_click_in_the_middle_of_an_idle_build_finishes_it(qtbot):
    _w, screen, saved = _window(qtbot, "regression", idle=False)
    _restore_idle(saved)
    heading = _waiting(screen)[0]
    for _ in range(4):
        assert screen._run_a_step_of(heading)
    assert screen._heading_is_waiting(heading), "four steps built it all"
    heading.set_expanded(True)
    _pump(0.1)
    assert not screen._heading_is_waiting(heading)
    assert "_spacr_opening" not in heading.__dict__
    title = heading.property("settingsCategorySource")
    _w2, eager, _ = _window(qtbot, "regression", eager=True, idle=False)
    _restore_idle(saved)
    mine = {k: v for k, v in _rows(screen).items() if k[0] == title}
    theirs = {k: v for k, v in _rows(eager).items() if k[0] == title}
    assert mine and mine == theirs


def test_a_category_prebuilt_after_a_switch_to_swedish_is_in_swedish(qtbot):
    """Every caption equals a window built whole in Swedish, and some changed."""
    from spacr.qt import i18n
    from spacr.qt.i18n import retranslate_widget_tree

    def captions(screen) -> list:
        found = []
        for section in screen.rendered_settings_sections():
            title = section.property("settingsCategorySource")
            if title not in waited:
                continue
            section.set_expanded(True)
            _pump(0.05)
            found.extend((title, label.text()) for label in
                         section._body.findChildren(QLabel) if label.text())
        return sorted(found)

    before = os.environ.get(i18n.ENV_LANGUAGE)
    window, screen, saved = _window(qtbot, "mask", idle=False)
    _restore_idle(saved)
    waited = {section.property("settingsCategorySource")
              for section in _waiting(screen)}
    try:
        os.environ[i18n.ENV_LANGUAGE] = "sv"
        retranslate_widget_tree(window, "sv")
        screen._prebuild_when_idle()
        _until_built(screen)
        assert _waiting(screen) == []
        _w2, eager, _ = _window(qtbot, "mask", eager=True, idle=False)
        _restore_idle(saved)
        assert captions(screen) == captions(eager)
    finally:
        if before is None:
            os.environ.pop(i18n.ENV_LANGUAGE, None)
        else:
            os.environ[i18n.ENV_LANGUAGE] = before
    _w3, english, _ = _window(qtbot, "mask", eager=True, idle=False)
    _restore_idle(saved)
    assert captions(english) != captions(screen), "nothing was translated"


# -- it gets out of the way ------------------------------------------------------

def test_input_stops_the_build_and_idle_resumes_it(qtbot):
    _w, screen, saved = _window(qtbot, "classify_merged")
    try:
        builder = screen.__dict__["_idle_prebuild"]
        state = {"next": 0.0}
        _pump(2.0, lambda: _move_the_mouse(screen, state))
        assert builder.slices_ms == [], "it built while the mouse moved"
        assert _waiting(screen)
        _until_built(screen)
        assert _waiting(screen) == []
    finally:
        _restore_idle(saved)


def test_a_build_interrupted_by_input_waits_for_the_input_to_stop(qtbot):
    _w, screen, saved = _window(qtbot, "regression")
    try:
        builder = screen.__dict__["_idle_prebuild"]
        end = time.perf_counter() + 10
        while not builder.slices_ms and time.perf_counter() < end:
            _pump(0.01)
        assert builder.slices_ms, "it never started"
        state = {"next": 0.0}
        _move_the_mouse(screen, state)
        _pump(0.02)
        started = len(builder.slices_ms)
        _pump(1.5, lambda: _move_the_mouse(screen, state))
        assert len(builder.slices_ms) == started
        _until_built(screen)
        assert _waiting(screen) == []
    finally:
        _restore_idle(saved)


def test_a_hidden_screen_builds_nothing(qtbot):
    window, screen, saved = _window(qtbot, "measure")
    try:
        other = "classify_merged"
        window._on_nav_selected(other)
        builder = screen.__dict__["_idle_prebuild"]
        count = len(builder.slices_ms)
        waiting = len(_waiting(screen))
        _pump(1.5)
        assert len(builder.slices_ms) == count
        assert len(_waiting(screen)) == waiting
    finally:
        _restore_idle(saved)


def test_no_slice_is_one_whole_category(qtbot):
    """A generous bound: slicing broke, not a slow machine, fails this."""
    _w, screen, saved = _window(qtbot, "regression")
    try:
        _until_built(screen)
        builder = screen.__dict__["_idle_prebuild"]
        assert max(builder.slices_ms) < 100.0, builder.slices_ms
    finally:
        _restore_idle(saved)


# -- the greyed reason is on the name -------------------------------------------

def test_a_greyed_settings_name_says_why_after_the_language_pass(qtbot):
    from spacr.qt.i18n import retranslate_widget_tree

    window, screen, saved = _window(qtbot, "classify_merged", eager=True,
                                    idle=False)
    _restore_idle(saved)
    control = screen._settings_model._widgets["batch_column"]
    assert not control.isEnabled()
    label = control._spacr_setting_label
    assert "batch_correction" in label.toolTip()
    retranslate_widget_tree(window)
    assert "batch_correction" in label.toolTip(), "the language pass dropped it"
    screen._settings_model.set_value_for_key("batch_correction", "combat")
    _pump(0.1)
    assert control.isEnabled()
    assert "is read only when" not in label.toolTip()


def test_the_prebuild_shows_or_opens_nothing_outside_the_categories(qtbot):
    """Panes the user collapsed stay collapsed; closed categories stay shut.

    The idle build only lays out rows inside category bodies nobody can see.
    Everything else on the screen -- console, System, the button strip, the
    settings column, previews and figures, whichever the user has collapsed
    or hidden -- keeps exactly the visibility it had.
    """
    from spacr.qt.widgets.section import Section

    _w, screen, saved = _window(qtbot, "mask")
    try:
        for child in screen._settings_body.findChildren(QWidget):
            if child.isVisible() and child.objectName() in (
                    "ConsolePanel", "RuntimeSystem"):
                child.hide()

        def outside(widget) -> bool:
            node = widget
            while node is not None:
                if isinstance(node, Section):
                    return False
                node = node.parentWidget()
            return True

        before = {id(each): each.isHidden()
                  for each in screen.findChildren(QWidget) if outside(each)}
        shut = {id(section): section.is_expanded()
                for section in screen.rendered_settings_sections()}
        _until_built(screen)
        assert _waiting(screen) == []
        after = {id(each): each.isHidden()
                 for each in screen.findChildren(QWidget)
                 if id(each) in before}
        assert after == {key: value for key, value in before.items()
                         if key in after}
        assert {id(section): section.is_expanded()
                for section in screen.rendered_settings_sections()
                if id(section) in shut} == shut
    finally:
        _restore_idle(saved)


# -- the object pass sets each row once ------------------------------------------

def _visible_rows(screen) -> dict:
    """``key -> row visible`` for every laid-out row in a shown category."""
    model = screen._settings_model
    found = {}
    for key, control in model._widgets.built_items():
        for section in screen.rendered_settings_sections():
            form = getattr(section, "_form", None)
            if not isinstance(form, QFormLayout) or section.isHidden():
                continue
            node = control
            while node is not None and form.getWidgetPosition(node)[0] < 0:
                node = node.parentWidget()
                if node is section:
                    node = None
            if node is not None:
                found[key] = form.isRowVisible(node)
                break
    return found


def test_the_object_pass_ends_where_the_round_trip_ended(qtbot):
    """Setting each row once leaves every shown row where showing it and
    hiding it again left it, under Essentials and after a channel commit."""
    from spacr.qt.settings_search import forget_disclosure

    forget_disclosure("mask")
    _w, screen, saved = _window(qtbot, "mask")
    try:
        _until_built(screen)
        model = screen._settings_model
        field = model._widgets["nucleus_channel"]
        field.setText("1")
        field.editingFinished.emit()
        _pump(0.3)
        once = _visible_rows(screen)
        model.rows_the_screen_hides = None
        model.refresh_object_visibility()
        screen._apply_dimension_visibility()
        screen._refilter_the_settings_search()
        _pump(0.2)
        assert _visible_rows(screen) == once
    finally:
        _restore_idle(saved)
