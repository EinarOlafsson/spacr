"""Helpers of the drv-f lesson drivers (24, 25, 26, 54, 64, 67, 69, 77, 82, 85, 86, 87 and lesson 04 Home).

A host clip must show the pointer doing the actions of its scene. These
helpers add what the sequences of those drivers need beside
:mod:`capture_ui_drv_a` (``Rec``) and :mod:`capture_ui_drv_e` (``scene``,
``snap``, ``guarded``). Outside the GNOME host every call reproduces the old
order exactly (the actions, then ``capture(name)``).
"""
from __future__ import annotations

import capture_ui
from capture_ui_drv_a import Rec
from capture_ui_drv_e import guarded, scene, snap  # noqa: F401  (re-exported for the f drivers)


def switch(capture, settle, current, name, lead=0.8, **capture_kwargs):
    """Close ``current`` (a started :class:`Rec`, or None) and start the clip ``name``.

    For consecutive scenes in which the actions of the next scene follow right
    after the end of the previous one (a dialog that stays open between them).
    """
    if current is not None:
        current.stop()
    return Rec(capture, name, settle, lead, **capture_kwargs).start()


def rest_on(capture, *widgets, dwell=1.2):
    """Host: the pointer visits each visible widget (the controls a narration names)."""
    return capture_ui.point_at(capture, *widgets, dwell=dwell)


def tab_click(capture, tabs, index):
    """Open tab ``index`` of a QTabWidget with the pointer (host) or ``setCurrentIndex``."""
    capture_ui.select_tab(capture, tabs, index)


def click_button(capture, button):
    """Click a push button: host through the pointer, else a synthetic click."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    if capture_ui.live(capture):
        capture_ui.press(capture, button)
        return
    QTest.mouseClick(button, Qt.LeftButton)


def clip_started(capture) -> bool:
    """True while a host clip is recording."""
    host = getattr(capture, 'host', None)
    return host is not None and getattr(host, 'current', None) is not None


class _NoClip:
    """Stands in for a host clip when the still was already taken (Xvfb)."""

    def stop(self):
        pass

    def abort(self):
        pass


def opening(capture, settle, name, lead=0.8):
    """A scene whose old driver took the still BEFORE its actions.

    Xvfb: the still ``capture(name)`` is taken now, exactly as before, and the
    returned object does nothing. Host: the clip starts now, so the actions
    that follow are recorded; call ``.stop()`` where they are finished.
    """
    if capture_ui.live(capture):
        return Rec(capture, name, settle, lead).start()
    capture(name)
    return _NoClip()


def sweep_over(capture, widget, fractions=(0.15, 0.5, 0.85), dwell=1.2, horizontal=0.5):
    """Host: the pointer rests at several heights of a tall widget (a console, a table).

    Reading a long output by eye: top, middle, bottom. No-op outside the host.
    """
    if not capture_ui.live(capture) or widget is None or not widget.isVisible():
        return
    from PySide6.QtCore import QPoint
    h = capture_ui._h()
    for fraction in fractions:
        h.hover(widget, QPoint(int(widget.width() * horizontal), int(widget.height() * fraction)))
        h.pause(capture, dwell)


def open_module_live(capture, window, key, settle, prefer='sidebar'):
    """Open a module by its sidebar row or Home tile; a module without either route is
    opened by the navigation slot and the code route is written to the bypass record."""
    if not capture_ui.live(capture):
        window._on_nav_selected(key)
        return
    try:
        capture_ui.open_module(capture, window, key, settle, prefer=prefer)
    except RuntimeError:
        capture_ui.code_action(capture, f'open {key}: no sidebar row or Home tile')
        window._on_nav_selected(key)
        capture_ui.wait(capture, 0.5, settle)


def fold_button(capture, screen, app_key):
    """The visible fold button of ``app_key`` on a host screen (None when absent)."""
    from spacr.qt.widgets.fold_strip import FoldButton
    for button in screen.findChildren(FoldButton):
        if button.isVisible() and button.app_key == app_key:
            return button
    return None


def prefs_toggle_live(app, window, capture, settle, object_name, on, frames, on_shown=None):
    """Host: spaCR menu -> Preferences..., the tab of ``object_name``, the toggle, Save.

    ``frames`` is (before, after, save): clip names or None. ``before`` ends with
    the toggle in view (before the click), ``after`` holds the click, ``save``
    holds the Save click. A missing clip is merged into the next one that exists:
    the first clip starts before the menu is opened, the last one ends after Save.
    """
    from PySide6.QtWidgets import QDialogButtonBox, QTabWidget, QWidget

    import capture_home_live as h

    order = [name for name in frames if name]
    state = {'rec': Rec(capture, order[0], settle).start(), 'next': 1}

    def advance():
        """Close the clip being recorded; open the next one, if any."""
        state['rec'].stop()
        if state['next'] < len(order):
            state['rec'] = Rec(capture, order[state['next']], settle).start()
            state['next'] += 1
            return True
        state['rec'] = None
        return False

    def steps(dialog):
        try:
            toggle = dialog.findChild(QWidget, object_name)
            tabs = dialog.findChild(QTabWidget, 'PreferencesTabs')
            if toggle is None or tabs is None:
                raise RuntimeError(f'Preferences has no {object_name} toggle')
            for index in range(tabs.count()):
                if tabs.widget(index).isAncestorOf(toggle):
                    capture_ui.select_tab(capture, tabs, index)
                    break
            scroll = h.scroll_area_of(toggle)
            if scroll is not None:
                h.scroll_into_view(capture, scroll, toggle)
            h.pause(capture, 0.8)
            if frames[0]:
                advance()                          # the dialog at the toggle, then the click
            if toggle.isChecked() != on:
                h.click(toggle)
                h.wait_until(capture, lambda: toggle.isChecked() == on, 5, 'toggle did not change')
            if on_shown is not None:
                on_shown(dialog, toggle)
            h.pause(capture, 1.0)
            saves = [box.button(QDialogButtonBox.Save) for box in dialog.findChildren(QDialogButtonBox)]
            saves = [b for b in saves if b is not None and b.isVisible()]
            if len(saves) != 1:
                raise RuntimeError('Preferences has no unique Save button')
            if frames[2]:
                advance()
            capture_ui.point_at(capture, saves[0], dwell=1.0)
            h.click(saves[0])
            h.pause(capture, 1.5)
            if state['rec'] is not None:
                state['rec'].stop()
                state['rec'] = None
        except BaseException:
            if state['rec'] is not None:
                state['rec'].abort()
            raise

    h.open_preferences(app, window, capture, steps)


def prefs_tabs_live(app, window, capture, settle, items):
    """Host: spaCR menu -> Preferences..., then one clip per ``(frame, label_object_name)``.

    The first clip starts before the menu is opened; each later one starts when
    the previous ended. Every clip shows the pointer clicking the tab that holds
    the label and resting on it. The dialog is closed with Cancel. Returns the
    ``{frame: error}`` of labels that were not found.
    """
    from PySide6.QtWidgets import QTabWidget, QWidget

    import capture_home_live as h

    problems = {}
    state = {'rec': Rec(capture, items[0][0], settle).start()}

    def steps(dialog):
        try:
            tabs = dialog.findChild(QTabWidget, 'PreferencesTabs')
            for position, (frame, name) in enumerate(items):
                if position:
                    state['rec'] = Rec(capture, frame, settle).start()
                label = dialog.findChild(QWidget, name)
                if label is None:
                    problems[frame] = f'no {name}'
                else:
                    for index in range(tabs.count()):
                        if tabs.widget(index).isAncestorOf(label):
                            capture_ui.select_tab(capture, tabs, index)
                            break
                    h.pause(capture, 0.6)
                    capture_ui.point_at(capture, label, dwell=1.5)
                state['rec'].stop()
                state['rec'] = None
            h.close_preferences(dialog, 'cancel')
        except BaseException:
            if state['rec'] is not None:
                state['rec'].abort()
            raise

    h.open_preferences(app, window, capture, steps)
    return problems


def open_overlay_settings(capture, settle, panel):
    """The Plaque preview overlay settings dialog.

    Xvfb: ``panel.open_overlay_settings()`` as before. Host: a right click on the
    preview picture opens its overlay menu and the pointer chooses
    ``Overlay settings...``; with no picture to click the call is made by code and
    written to the bypass record (the clip gate then reports it).
    """
    if not capture_ui.live(capture):
        return panel.open_overlay_settings()
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QMenu, QWidget

    h = capture_ui._h()
    pictures = [w for w in panel.findChildren(QWidget)
                if hasattr(w, 'context_requested') and w.isVisible() and w.width() > 100]
    if pictures:
        QTest.mouseClick(max(pictures, key=lambda w: w.width() * w.height()), Qt.RightButton)
        menu = h.wait_until(capture, lambda: QApplication.activePopupWidget(), 5, 'overlay menu did not open')
        if isinstance(menu, QMenu):
            h.pause(capture, 0.5)
            action = next((a for a in menu.actions() if a.objectName() == 'PlaqueOverlaySettings'), None)
            if action is not None:
                h.click(menu, menu.actionGeometry(action).center())
                h.wait_until(capture, lambda: panel._overlay_dialog is not None
                             and panel._overlay_dialog.isVisible(), 5, 'overlay settings did not open')
                return panel._overlay_dialog
            h.key(menu, Qt.Key_Escape)
    capture_ui.code_action(capture, 'open the overlay settings: the right-click menu was not reachable')
    return panel.open_overlay_settings()
