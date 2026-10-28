"""Lesson 05 (Home) and the app scenes of lesson 04, driven through the REAL user interface.

GNOME capture host only (``run_native_capture.py --host gnome``). Every change
a scene makes goes through the human pointer and keyboard of
:mod:`capture_host_adapter` (QTest calls, which the adapter turns into real
glides, presses, drags, wheel turns and key presses):

* Preferences opens from the spaCR menu -> Preferences..., the dialog the
  application itself builds and runs (``exec()``); tabs, sections, combo
  boxes, sliders, spin boxes and buttons are clicked, dragged and typed into;
* modules open by clicking their sidebar rows, Home tiles are hovered, the
  Home column handle is clicked, the settings splitter is dragged and the
  column text is scaled with Ctrl + wheel;
* nothing calls ``setCurrentIndex``, ``setChecked``, ``setValue``,
  ``setText``, ``menu.popup()`` or builds a dialog in code.

Scenes keep their names and their clip durations (``--clip-durations``):
each clip lasts at least its narration time; when real pointer travel needs
longer, the clip runs longer, never shorter.

A dialog opened by a real click runs its own event loop (``exec()``); the
steps inside it run from that loop (:func:`inside_next_dialog`), and their
clips use ``capture.begin(name)`` / ``capture.end()`` so a recording may span
the click that opened the dialog.

Every scene function returns the regions the staging focus map needs, in the
3840x2160 hand-off frame.
"""
from __future__ import annotations

import time

def _qt():
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    return QPoint, Qt, QTest


def host_of(capture):
    host = getattr(capture, 'host', None)
    if host is None:
        raise RuntimeError('The live Home scenes record only in the GNOME capture host')
    return host


def pause(capture, seconds):
    host_of(capture).pump.run(seconds)


def wait_until(capture, predicate, timeout=20.0, what='condition'):
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() > deadline:
            raise TimeoutError(what)
        pause(capture, 0.05)


def hold_until(capture, seconds_into_clip):
    """Pump until the current clip has run ``seconds_into_clip`` seconds."""
    host = host_of(capture)
    while host.current is not None and host.elapsed() < seconds_into_clip:
        pause(capture, 0.02)


def clip_seconds(capture):
    host = host_of(capture)
    return host.current['seconds'] if host.current else 0.0


def click(widget, pos=None):
    QPoint, Qt, QTest = _qt()
    QTest.mouseClick(widget, Qt.LeftButton, Qt.KeyboardModifier.NoModifier, pos if pos is not None else QPoint())


def hover(widget, pos=None):
    QPoint, Qt, QTest = _qt()
    QTest.mouseMove(widget, pos if pos is not None else widget.rect().center())


def key(widget, qt_key, modifier=None):
    QPoint, Qt, QTest = _qt()
    QTest.keyClick(widget, qt_key, modifier if modifier is not None else Qt.KeyboardModifier.NoModifier)


def type_text(widget, text):
    QPoint, Qt, QTest = _qt()
    QTest.keyClicks(widget, text)


def wheel(capture, widget, notches, pos=None, modifiers=()):
    """Glide over ``widget`` and turn the real wheel (``+`` = down)."""
    adapter = host_of(capture).adapter
    QPoint, Qt, QTest = _qt()
    point = adapter.target_point(widget, pos)
    adapter.call('move_to', *point)
    for name in modifiers:
        adapter.call('key_down', name)
    try:
        adapter.call('scroll', int(notches))
    finally:
        for name in reversed(modifiers):
            adapter.call('key_up', name)
    adapter.note('wheel', notches=notches, point=point, modifiers=list(modifiers))
    pause(capture, 0.15)


def drag(capture, widget, start, delta, *, duration=None):
    """A real drag: press at ``start`` (widget coordinates), move by ``delta`` pixels, release."""
    adapter = host_of(capture).adapter
    x0, y0 = adapter.target_point(widget, start)
    adapter.call('move_to', x0, y0)
    pause(capture, 0.12)
    adapter.call('button_down', 'left')
    pause(capture, 0.12)
    if duration:
        # A given duration never makes a long drag faster than the pointer
        # peak the clip gate allows (human_input.Human.duration_for).
        from PySide6.QtWidgets import QApplication
        width = QApplication.primaryScreen().geometry().width() or 2560
        distance = (delta[0] ** 2 + delta[1] ** 2) ** 0.5
        duration = max(duration, 1.15 * 1.875 * distance / (0.011 * width * 30.0))
    kwargs = {'duration': duration} if duration else {}
    adapter.call('move_to', x0 + delta[0], y0 + delta[1], **kwargs)
    pause(capture, 0.1)
    adapter.call('button_up', 'left')
    adapter.note('drag', start=[x0, y0], delta=list(delta))
    pause(capture, 0.2)


def drag_qdrag(capture, widget, start, delta, *, duration=None):
    """A real drag whose press starts a Qt QDrag (a list entry dragged to a drop zone).

    ``drag`` presses, moves and releases from ONE call stack that pumps events while it waits. A QDrag opens
    its own nested event loop in the middle of that pumping and does not return until the drop, so the outer
    call can never send the release and the recording hangs (lesson 58, 05_area_histogram, no stall on record).
    Here every step is sent without waiting (``adapter.start``) and the next one is scheduled by a Qt timer,
    which keeps running inside the nested loop and delivers the release that ends it."""
    from PySide6.QtCore import QTimer
    adapter = host_of(capture).adapter
    x0, y0 = adapter.target_point(widget, start)
    kwargs = {}
    if duration:
        from PySide6.QtWidgets import QApplication
        width = QApplication.primaryScreen().geometry().width() or 2560
        distance = (delta[0] ** 2 + delta[1] ** 2) ** 0.5
        kwargs = {'duration': max(duration, 1.15 * 1.875 * distance / (0.011 * width * 30.0))}
    ops = [('move_to', (x0, y0), {}, 0.12), ('button_down', ('left',), {}, 0.12),
           ('move_to', (x0 + delta[0], y0 + delta[1]), kwargs, 0.1), ('button_up', ('left',), {}, 0.2)]
    state = {'index': 0, 'seq': None, 'done': False, 'error': None}

    def step():
        if state['seq'] is not None:
            try:
                if not adapter.finished(state['seq']):
                    QTimer.singleShot(5, step)
                    return
            except BaseException as error:
                state['error'], state['done'] = error, True
                return
            state['seq'] = None
            QTimer.singleShot(int(ops[state['index'] - 1][3] * 1000), step)
            return
        if state['index'] >= len(ops):
            state['done'] = True
            return
        name, args, kw, _after = ops[state['index']]
        state['index'] += 1
        state['seq'] = adapter.start(name, *args, **kw)
        QTimer.singleShot(5, step)

    QTimer.singleShot(0, step)
    wait_until(capture, lambda: state['done'], 60 + (kwargs.get('duration') or 0), 'the drag did not finish')
    if state['error'] is not None:
        raise state['error']
    adapter.note('drag', start=[x0, y0], delta=list(delta))


def _choose_combo_by_keys(capture, combo, view, target):
    """Choose on the already opened popup using the normal native keyboard route."""
    from PySide6.QtCore import Qt
    key(view, Qt.Key_Home)
    for _ in range(target):
        key(view, Qt.Key_Down)
    key(view, Qt.Key_Return)
    wait_until(capture, lambda: not view.isVisible(), 8, 'combo list did not close')
    if combo.currentIndex() != target:
        raise RuntimeError('%s still shows item %d, not %d' % (
            combo.objectName() or 'combo', combo.currentIndex(), target))


def choose_combo(capture, combo, index=None):
    """Click a combo box, then click an item in its list (default: the current one).

    Live, the popup opens and closes on the compositor's schedule (under load a click can land before it is
    mapped, or the list stays up a moment after the item click), so each step waits longer and retries once,
    as a user would, before it raises."""
    view = combo.view()
    target = combo.currentIndex() if index is None else index
    for attempt in range(3):
        if not view.isVisible():
            click(combo)
        try:
            wait_until(capture, lambda: view.isVisible(), 8, 'combo list did not open')
            break
        except TimeoutError:
            if attempt == 2:
                raise TimeoutError('combo list did not open (combo %r enabled=%s visible=%s items=%d current=%r window_active=%s)' % (
                    combo.objectName() or combo.accessibleName(), combo.isEnabled(), combo.isVisible(), combo.count(),
                    combo.currentText(), combo.window().isActiveWindow()))
    pause(capture, 0.5)
    for attempt in range(3):
        if not view.isVisible():
            break                                    # the popup closed already (a click that closed it early)
        item = combo.model().index(target, combo.modelColumn())
        from PySide6.QtCore import Qt
        if (not item.isValid() or not item.flags() & Qt.ItemIsEnabled
                or not item.flags() & Qt.ItemIsSelectable):
            raise RuntimeError(f'Combo item {target} is not enabled and selectable')
        for _ in range(combo.count() + 1):
            rect = view.visualRect(item)
            port = view.viewport().rect()
            visible = rect.intersected(port)
            point = visible.center()
            if (not visible.isEmpty() and rect.top() >= port.top()
                    and rect.bottom() <= port.bottom() and view.indexAt(point) == item):
                break
            before = (view.viewport().mapToGlobal(rect.topLeft()), rect.size(),
                      view.viewport().mapToGlobal(port.topLeft()), port.size())
            wheel(capture, view.viewport(), 1 if rect.center().y() > port.bottom() else -1,
                  port.center())
            after_rect, after_port = view.visualRect(item), view.viewport().rect()
            after = (view.viewport().mapToGlobal(after_rect.topLeft()), after_rect.size(),
                     view.viewport().mapToGlobal(after_port.topLeft()), after_port.size())
            if after == before:
                # Qt menu-style popups can clip leading rows without a moving scrollbar.
                # Use the existing genuine keyboard selection before any unreachable click.
                _choose_combo_by_keys(capture, combo, view, target)
                return
        else:
            raise RuntimeError(f'Combo item {target} is outside its popup viewport')
        adapter = host_of(capture).adapter
        reason = adapter.reachable(view.viewport(), adapter.target_point(view.viewport(), point))
        if reason is not None:
            raise RuntimeError(f'Combo item {target} is not reachable: {reason}')
        click(view.viewport(), point)
        try:
            wait_until(capture, lambda: not view.isVisible(), 8, 'combo list did not close')
            break
        except TimeoutError:
            if attempt == 2:
                from PySide6.QtCore import Qt
                key(view, Qt.Key_Return)                 # last resort: the same choice by keyboard
                wait_until(capture, lambda: not view.isVisible(), 8, 'combo list did not close')
    if index is not None and combo.currentIndex() != target:
        # The click landed on a neighbouring row (a long list scrolled under the pointer: resnet18 was chosen and the
        # box showed squeezenet1_1). Choose it again with the keyboard on the focused box, as a user can.
        from PySide6.QtCore import Qt
        click(combo)
        wait_until(capture, lambda: view.isVisible(), 8, 'combo list did not open')
        _choose_combo_by_keys(capture, combo, view, target)


def set_slider(capture, slider, value):
    """Drag the handle towards ``value``, then finish with arrow keys (a user does both)."""
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtWidgets import QStyle, QStyleOptionSlider
    option = QStyleOptionSlider()
    slider.initStyleOption(option)
    handle = slider.style().subControlRect(QStyle.CC_Slider, option, QStyle.SC_SliderHandle, slider)
    span = slider.width() - handle.width() if slider.orientation() == Qt.Horizontal else slider.height() - handle.height()
    now = QStyle.sliderPositionFromValue(slider.minimum(), slider.maximum(), slider.value(), span)
    goal = QStyle.sliderPositionFromValue(slider.minimum(), slider.maximum(), value, span)
    delta = (goal - now, 0) if slider.orientation() == Qt.Horizontal else (0, goal - now)
    if abs(goal - now) >= 3:
        drag(capture, slider, handle.center(), delta)
    for _ in range(abs(slider.maximum() - slider.minimum()) + 1):
        if slider.value() == value:
            break
        key(slider, Qt.Key_Right if slider.value() < value else Qt.Key_Left)
    if slider.value() != value:
        raise RuntimeError(f'{slider.objectName()} stopped at {slider.value()}, not {value}')


def type_number(capture, spin, value):
    """Click a spin box, select its text and type a new value, then Tab."""
    from PySide6.QtCore import Qt
    click(spin)
    key(spin, Qt.Key_A, Qt.ControlModifier)
    type_text(spin, str(value))
    key(spin, Qt.Key_Tab)
    pause(capture, 0.2)


def scroll_bar_wheel(capture, scroll, notches):
    """Turn the wheel over the scroll area's own scroll bar.

    Over the page a wheel turn can land on a slider, spin box or combo box and
    change its value; over the scroll bar it only scrolls (as a careful user
    does). Falls back to the viewport edge when the bar is hidden.
    """
    from PySide6.QtCore import QPoint
    bar = scroll.verticalScrollBar()
    if bar.isVisible():
        wheel(capture, bar, notches, QPoint(bar.width() // 2, bar.height() // 2))
    else:
        viewport = scroll.viewport()
        wheel(capture, viewport, notches, QPoint(viewport.width() - 6, viewport.height() // 2))


def drag_scroll_bar(capture, bar, value):
    """Drag the scroll bar handle towards ``value`` (a real drag)."""
    from PySide6.QtWidgets import QStyle, QStyleOptionSlider
    option = QStyleOptionSlider()
    bar.initStyleOption(option)
    handle = bar.style().subControlRect(QStyle.CC_ScrollBar, option, QStyle.SC_ScrollBarSlider, bar)
    groove = bar.style().subControlRect(QStyle.CC_ScrollBar, option, QStyle.SC_ScrollBarGroove, bar)
    travel = max(1, groove.height() - handle.height())
    span = max(1, bar.maximum() - bar.minimum())
    dy = int(round((value - bar.value()) * travel / span))
    if dy:
        drag(capture, bar, handle.center(), (0, dy))


def scroll_into_view(capture, scroll, widget, margin=40):
    """Bring ``widget`` into the viewport: wheel over the scroll bar, or drag its handle."""
    from PySide6.QtCore import QPoint
    viewport = scroll.viewport()
    bar = scroll.verticalScrollBar()
    if not widget.isVisible():
        raise RuntimeError(f'{widget.objectName() or type(widget).__name__} is not shown, cannot scroll to it')
    state = {}
    for _ in range(60):
        top = widget.mapTo(viewport, QPoint(0, 0)).y()
        bottom = top + widget.height()
        room = min(margin, max(0, (viewport.height() - widget.height()) // 2))
        state = {'top': top, 'bottom': bottom, 'viewport_h': viewport.height(), 'bar': bar.value(),
                 'bar_range': [bar.minimum(), bar.maximum()]}
        if top >= room and bottom <= viewport.height() - room:
            return
        if widget.height() >= viewport.height() - 2 and top <= 2 and bottom >= viewport.height() - 2:
            return                                # taller than the view: it fills it
        direction = -1 if top < room else 1
        if (direction < 0 and bar.value() == bar.minimum()) or (direction > 0 and bar.value() == bar.maximum()):
            return
        before = bar.value()
        scroll_bar_wheel(capture, scroll, direction)
        pause(capture, 0.1)
        if bar.value() == before and top >= 0 and bottom <= viewport.height():
            return                                    # fully shown; the margin is cosmetic
        if bar.value() == before:
            # The wheel did not move it: drag the handle to the target instead.
            # A QScrollArea scroll bar counts pixels of content.
            target = before + (top - room) if direction < 0 else before + (bottom - (viewport.height() - room))
            drag_scroll_bar(capture, bar, max(bar.minimum(), min(bar.maximum(), target)))
            if bar.value() == before:
                break
    raise RuntimeError(f'Could not scroll {widget.objectName() or type(widget).__name__} into view: {state}')


def scroll_area_of(widget):
    from PySide6.QtWidgets import QScrollArea
    scroll = widget
    while scroll is not None and not isinstance(scroll, QScrollArea):
        scroll = scroll.parentWidget()
    return scroll


class _ShowHook:
    """Run ``steps(widget)`` from the event loop of the next shown top-level widget matching ``match``."""

    def __init__(self, app, match, steps):
        from PySide6.QtCore import QEvent, QObject, QTimer
        from PySide6.QtWidgets import QWidget
        self.done = {}
        hook = self

        class Filter(QObject):
            def eventFilter(self, watched, event):
                if (event.type() == QEvent.Show and isinstance(watched, QWidget) and watched.isWindow()
                        and match(watched)
                        and 'queued' not in hook.done):
                    hook.done['queued'] = True
                    QTimer.singleShot(0, lambda: hook._run(watched, steps))
                return False

        self.filter = Filter()
        self.app = app
        app.installEventFilter(self.filter)

    def _run(self, widget, steps):
        try:
            self.done['result'] = steps(widget)
        except BaseException as error:              # noqa: BLE001 - re-raised by finish()
            self.done['error'] = error
            try:
                widget.reject()
            except Exception:                         # noqa: BLE001
                widget.close()
        finally:
            self.done['finished'] = True

    def finish(self, capture, timeout=600):
        try:
            wait_until(capture, lambda: self.done.get('finished'), timeout, 'dialog steps did not finish')
        finally:
            self.app.removeEventFilter(self.filter)
        if 'error' in self.done:
            raise self.done['error']
        return self.done.get('result')


def inside_next_dialog(app, match, steps):
    return _ShowHook(app, match, steps)


def handoff(capture, widget, window=None):
    from capture_geometry import capture_rect
    from PySide6.QtCore import QPoint
    host = host_of(capture)
    if window is None:
        origin = widget.mapToGlobal(QPoint(0, 0))
        return host.to_handoff([origin.x(), origin.y(), widget.width(), widget.height()])
    return host.to_handoff(capture_rect(widget, window))


def union(*rects):
    rects = [r for r in rects if r]
    x0 = min(r[0] for r in rects)
    y0 = min(r[1] for r in rects)
    x1 = max(r[0] + r[2] for r in rects)
    y1 = max(r[1] + r[3] for r in rects)
    return [x0, y0, x1 - x0, y1 - y0]


# -- Preferences, opened from the spaCR menu ---------------------------------------
def _menu_action(window, title='spaCR'):
    bar = window.menuBar()
    actions = bar.actions()
    action = next((a for a in actions if a.text().replace('&', '') == title), None) or next(
        (a for a in actions if title in a.text().replace('&', '')), None)
    if action is None:
        raise RuntimeError(f'No {title} menu: {[a.text() for a in actions]}')
    return bar, action


def is_preferences(widget):
    from PySide6.QtWidgets import QDialog, QTabWidget
    return isinstance(widget, QDialog) and widget.findChild(QTabWidget, 'PreferencesTabs') is not None


def open_preferences(app, window, capture, steps):
    """Click spaCR -> Preferences...; run ``steps(dialog)`` inside the dialog; return its result.

    ``steps`` must leave the dialog closed (it ends with a real click on
    Cancel or Save); :func:`close_preferences` does that.
    """
    from PySide6.QtWidgets import QApplication, QMenu
    bar, action = _menu_action(window)
    click(bar, bar.actionGeometry(action).center())
    menu = wait_until(capture, lambda: QApplication.activePopupWidget(), 5, 'spaCR menu did not open')
    if not isinstance(menu, QMenu):
        raise RuntimeError('The spaCR menu did not open')
    pause(capture, 0.4)
    item = next(a for a in menu.actions() if a.text().replace('&', '').startswith('Preferences'))
    hook = inside_next_dialog(app, is_preferences, steps)
    click(menu, menu.actionGeometry(item).center())
    result = hook.finish(capture)
    wait_until(capture, lambda: not any(is_preferences(w) and w.isVisible()
                                        for w in QApplication.topLevelWidgets()), 10,
               'Preferences did not close')
    return result


def close_preferences(dialog, button='cancel'):
    from PySide6.QtWidgets import QAbstractButton
    target = next(b for b in dialog.findChildren(QAbstractButton)
                  if b.isVisible() and b.text().replace('&', '').lower() == button)
    click(target)


TAB_TITLES = {'PreferencesTabGeneral': 'General', 'PreferencesTabAppearance': 'Appearance',
              'PreferencesTabPerformance': 'Performance', 'PreferencesTabModules': 'Modules',
              'PreferencesTabStorage': 'Storage'}


def show_tab(capture, dialog, object_name):
    """Click the Preferences tab holding the page ``object_name``; returns (tabs, page).

    Pages can be filled in only when their tab is first opened, so the tab is
    found by its page when that exists, else by its (English) title.
    """
    from PySide6.QtWidgets import QTabWidget, QWidget
    tabs = wait_until(capture, lambda: dialog.findChild(QTabWidget, 'PreferencesTabs'), 20, 'PreferencesTabs')
    page = dialog.findChild(QWidget, object_name)
    if page is not None:
        index = next(i for i in range(tabs.count()) if tabs.widget(i) is page or tabs.widget(i).isAncestorOf(page))
    else:
        title = TAB_TITLES.get(object_name)
        titles = [tabs.tabText(i).replace('&', '') for i in range(tabs.count())]
        if title not in titles:
            raise RuntimeError(f'Preferences has no {object_name}; tabs {titles}')
        index = titles.index(title)
    if tabs.currentIndex() != index:
        click(tabs.tabBar(), tabs.tabBar().tabRect(index).center())
        wait_until(capture, lambda: tabs.currentIndex() == index, 5, f'tab {object_name} did not open')
    page = wait_until(capture, lambda: dialog.findChild(QWidget, object_name), 20, f'{object_name} page')
    pause(capture, 0.6)
    return tabs, page


def sections_of(page):
    from spacr.qt.widgets.section import Section
    return {s.title().casefold(): s for s in page.findChildren(Section)}


def toggle_section(capture, section, expanded):
    if section.is_expanded() != expanded:
        scroll = scroll_area_of(section)
        if scroll is not None:
            scroll_into_view(capture, scroll, section.header())
        click(section.header())
        try:
            wait_until(capture, lambda: section.is_expanded() == expanded, 8, f'section {section.title()} did not toggle')
        except TimeoutError:
            # A click that lands while the form is still re-laying out can miss the header: bring it into view
            # again and press it once more (a user would), then allow longer under load.
            if section.is_expanded() != expanded:
                if scroll is not None:
                    scroll_into_view(capture, scroll, section.header())
                click(section.header())
                wait_until(capture, lambda: section.is_expanded() == expanded, 20, f'section {section.title()} did not toggle')
        pause(capture, 0.4)


def record_preferences_live(app, window, capture, settle):
    """Lesson 05 Preferences scenes (09, 13, 13b-13i) in ONE real Preferences session."""
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtWidgets import (QAbstractButton, QApplication, QComboBox, QDialogButtonBox,
                                   QMessageBox, QSlider, QWidget)
    from spacr.qt import preferences

    focus = {}
    proof = {}
    deferred = []

    def steps(dialog):
        settle(0.8)
        focus['dialog'] = handoff(capture, dialog)
        proof['dialog_rect_host'] = [dialog.x(), dialog.y(), dialog.width(), dialog.height()]
        # 09: Performance profile -- the combo box clicked, its list shown, the current profile kept.
        selector = dialog.findChild(QComboBox, 'PerformanceLevel')
        perf_page = next(w for w in _ancestors(selector) if (w.objectName() or '').startswith('PreferencesTab'))
        show_tab(capture, dialog, perf_page.objectName())
        hover(selector)
        pause(capture, 1.2)
        choose_combo(capture, selector)
        hold_until(capture, clip_seconds(capture))
        capture.end()

        # 13: General
        capture.begin('13_preferences_general')
        _, general = show_tab(capture, dialog, 'PreferencesTabGeneral')
        focus['13_preferences_general'] = handoff(capture, dialog)
        for control in [w for w in general.findChildren(QComboBox) if w.isVisible()][:1] + \
                       [w for w in general.findChildren(QSlider) if w.isVisible()][:2]:
            hover(control)
            pause(capture, 1.5)
        capture.end()

        # 13b: Appearance
        capture.begin('13b_preferences_appearance')
        _, appearance = show_tab(capture, dialog, 'PreferencesTabAppearance')
        focus['13b_preferences_appearance'] = handoff(capture, dialog)
        for control in [w for w in appearance.findChildren(QWidget)
                        if w.isVisible() and w.objectName() in ('TooltipDelay', 'ShowTooltips')][:2]:
            hover(control)
            pause(capture, 1.5)
        capture.end()

        sections = sections_of(appearance)
        for needed in ('theme', 'animation'):
            if needed not in sections:
                raise RuntimeError(f'Appearance has no {needed} section: {sorted(sections)}')
        scroll = scroll_area_of(sections['theme'])

        # 13c: Theme opened by a click on its header
        capture.begin('13c_appearance_theme')
        focus['13c_appearance_theme'] = handoff(capture, dialog)
        toggle_section(capture, sections['animation'], False)
        toggle_section(capture, sections['theme'], True)
        if scroll is not None:
            for _ in range(4):
                scroll_bar_wheel(capture, scroll, 1)
                pause(capture, 0.8)
        capture.end()

        # 13d: Animation; the settings-window motion controls stay independent (real arrow keys)
        capture.begin('13d_appearance_animation')
        focus['13d_appearance_animation'] = handoff(capture, dialog)
        toggle_section(capture, sections['theme'], False)
        toggle_section(capture, sections['animation'], True)
        pairs = (('PopupBackdropSpeed', 'AmbientSpeed'), ('PopupBackdropSize', 'AmbientSize'),
                 ('PopupBackdropResolution', 'AmbientResolution'), ('PopupBackdropDensity', 'AmbientDensity'))
        independence = {}
        if scroll is not None:
            for _ in range(3):
                scroll_bar_wheel(capture, scroll, 1)
                pause(capture, 1.2)
        for settings_name, main_name in pairs:
            settings, main = dialog.findChild(QWidget, settings_name), dialog.findChild(QWidget, main_name)
            if settings is None or main is None or not settings.isVisible():
                independence[settings_name] = {'shown': False}
                continue
            if not settings.isEnabled():
                # Tutorials record settings windows on plain glass (popup
                # backdrop off), so its animation controls are disabled.
                independence[settings_name] = {'shown': True, 'enabled': False}
                hover(settings)
                pause(capture, 1.0)
                continue
            if scroll is not None:
                scroll_into_view(capture, scroll, settings)
            hover(settings)
            before, main_before = settings.value(), main.value()
            key(settings, Qt.Key_Right)
            edited = settings.value()
            key(settings, Qt.Key_Left)
            independence[settings_name] = {'before': before, 'edited': edited,
                                           'main_unchanged': main.value() == main_before,
                                           'restored': settings.value() == before}
            if edited == before or main.value() != main_before or settings.value() != before:
                raise RuntimeError(f'Settings animation is not independent: {settings_name}')
            pause(capture, 1.0)
        proof['settings_controls_independent'] = independence
        hold_until(capture, clip_seconds(capture) - 3)
        if scroll is not None:
            scroll_into_view(capture, scroll, sections['animation'].header())
        capture.end()
        toggle_section(capture, sections['animation'], False)

        # 13h / 13i: Apply previews, the question offers Keep or Revert
        buttons = dialog.findChild(QDialogButtonBox)
        apply_button = buttons.button(QDialogButtonBox.Apply)
        slider = dialog.findChild(QSlider, 'PaneOpacity')
        original = round(preferences.get_pane_opacity() * 100)
        preview = 55 if original != 55 else 60
        proof['original_opacity_pct'] = original

        def question():
            box = getattr(dialog, '_apply_confirmation', None)
            return box if isinstance(box, QMessageBox) and box.isVisible() else None

        def answer(text):
            box = question()
            target = next(b for b in box.buttons() if b.text().replace('&', '') == text)
            click(target)
            wait_until(capture, lambda: question() is None, 10, f'{text} did not close the question')

        capture.begin('13h_preferences_apply')
        toggle_section(capture, sections['theme'], True)      # Page opacity lives under Theme
        if scroll is not None:
            scroll_into_view(capture, scroll, slider)
        set_slider(capture, slider, preview)
        pause(capture, 0.6)
        click(apply_button)
        box = wait_until(capture, question, 10, 'Apply did not ask Keep or Revert')
        settle(0.3)
        focus['13h_preferences_apply'] = union(handoff(capture, dialog), handoff(capture, box))
        if abs(preferences.get_pane_opacity() - preview / 100) > 1e-9:
            raise RuntimeError('Apply did not preview the edited opacity')
        for b in box.buttons():
            hover(b)
            pause(capture, 1.0)
        hold_until(capture, clip_seconds(capture) - 2.5)
        answer('Revert')
        if round(preferences.get_pane_opacity() * 100) != original:
            raise RuntimeError('Revert did not restore the saved opacity')
        proof['revert_verified'] = True
        capture.end()

        capture.begin('13i_preferences_kept')
        set_slider(capture, slider, original)
        click(apply_button)
        box = wait_until(capture, question, 10, 'Apply did not ask Keep or Revert')
        focus['13i_preferences_kept'] = union(handoff(capture, dialog), handoff(capture, box))
        pause(capture, 1.5)
        answer('Keep')
        if round(preferences.get_pane_opacity() * 100) != original or not dialog.isVisible():
            raise RuntimeError('Keep did not keep the value with Preferences open')
        proof['keep_verified'] = True
        capture.end()
        toggle_section(capture, sections['theme'], False)

        # 13e: Modules tab, scrolled to Session and Update channel; Show alpha
        # features (off) must stay out of view. At its default size the whole
        # Modules page fits, so the dialog is first made shorter by dragging
        # its bottom edge (a user resizing the window). If the app does not
        # allow that, 13e is not recorded and the run fails at the end.
        session = dialog.findChild(QWidget, 'RestoreLastSession')
        channel = dialog.findChild(QWidget, 'UpdateChannel')
        toggle = dialog.findChild(QWidget, 'ShowAlphaFeatures')
        _, modules = show_tab(capture, dialog, 'PreferencesTabModules')
        mscroll = scroll_area_of(session)

        def alpha_in_view():
            viewport = mscroll.viewport()
            y = toggle.mapTo(viewport, QPoint(0, 0)).y()
            return toggle.isVisible() and -toggle.height() < y < viewport.height()

        if alpha_in_view():
            needed = toggle.mapTo(mscroll.viewport(), QPoint(0, 0)).y() - session.mapTo(
                mscroll.viewport(), QPoint(0, 0)).y()
            shrink = mscroll.viewport().height() - needed + 30
            before_h = dialog.height()
            drag(capture, dialog, QPoint(dialog.width() // 2, dialog.height() - 3), (0, -shrink), duration=1.0)
            settle(0.5)
            proof['modules_resize'] = {'before_h': before_h, 'after_h': dialog.height(), 'asked': shrink}
            scroll_into_view(capture, mscroll, session, margin=4)
        # Show the Modules tab back on top before the clip, so the clip never shows the toggle.
        if alpha_in_view():
            deferred.append('13e_session_updates not recorded: Show alpha features cannot be kept out of '
                            f'view on the Modules tab of the real dialog ({proof.get("modules_resize")})')
        else:
            capture.begin('13e_session_updates')
            focus['13e_session_updates'] = union(handoff(capture, session), handoff(capture, channel))
            for control in (session, channel):
                hover(control)
                pause(capture, 1.5)
            if alpha_in_view():
                raise RuntimeError('Show alpha features came into view during 13e')
            proof['restore_last_session'] = session.isChecked()
            proof['update_channel'] = channel.currentText()
            capture.end()
        if proof.get('modules_resize'):
            # Give the window its height back (bottom edge dragged down again).
            grown = proof['modules_resize']['before_h'] - dialog.height()
            if grown > 0:
                drag(capture, dialog, QPoint(dialog.width() // 2, dialog.height() - 3), (0, grown), duration=1.0)
                settle(0.5)
            proof['modules_resize']['restored_h'] = dialog.height()

        # 13f / 13g: Storage, then Prune now's question, cancelled
        capture.begin('13f_storage')
        _, storage_page = show_tab(capture, dialog, 'PreferencesTabStorage')
        focus['13f_storage'] = handoff(capture, dialog)
        storage = dialog._storage_page
        keep, cap = storage.spins['run_folders']
        for control in (keep, cap):
            hover(control)
            pause(capture, 1.2)
        capture.end()

        old = (keep.value(), cap.value())
        capture.begin('13g_storage_prune')
        type_number(capture, keep, 1)
        type_number(capture, cap, 0)
        prune = dialog.findChild(QAbstractButton, 'StoragePruneButton')
        asked = {}

        def answer_prune(box):
            settle(0.6)
            asked['question'] = [box.windowTitle(), box.text(), box.informativeText()]
            focus['13g_storage_prune'] = union(handoff(capture, dialog), handoff(capture, box))
            for b in box.buttons():
                hover(b)
                pause(capture, 1.0)
            hold_until(capture, clip_seconds(capture) - 2.0)
            cancel = [b for b in box.buttons()
                      if box.buttonRole(b) in (QMessageBox.RejectRole, QMessageBox.NoRole)]
            if not cancel:
                raise RuntimeError('Prune question has no Cancel/No button')
            click(cancel[0])
            asked['cancel'] = cancel[0].text()

        hook = inside_next_dialog(app, lambda w: isinstance(w, QMessageBox), answer_prune)
        if storage_page is not None and scroll_area_of(prune) is not None:
            scroll_into_view(capture, scroll_area_of(prune), prune)
        click(prune)
        hook.finish(capture, timeout=180)
        proof['prune'] = asked
        capture.end()
        type_number(capture, keep, old[0])
        type_number(capture, cap, old[1])
        close_preferences(dialog, 'cancel')
        return proof

    if capture.begin('09_performance') is False:
        raise RuntimeError('Record the Preferences scenes together (09 opens the dialog)')
    result = open_preferences(app, window, capture, steps)
    focus['09_performance'] = focus.get('dialog')
    focus.pop('dialog', None)
    if deferred:
        # Not raised: the other clips still get their gate verdicts. The
        # missing scene keeps the lesson from staging (stage_lesson needs it).
        print('NOT RECORDED: ' + '; '.join(deferred), flush=True)
        result = {**(result or {}), 'not_recorded': deferred}
    return {'focus': focus, 'proof': result}


def _ancestors(widget):
    while widget is not None:
        yield widget
        widget = widget.parentWidget()


# -- Home navigation scenes ----------------------------------------------------------
def sidebar_row(window, key):
    rows = [row for row in window._sidebar._rows if row.key == key and row.isVisible()]
    if len(rows) != 1:
        raise RuntimeError(f'No unique visible sidebar route for {key}')
    return rows[0]


def open_from_sidebar(capture, window, key):
    click(sidebar_row(window, key))
    wait_until(capture, lambda: key == '__home__' or window._stack.currentWidget() is window._screens.get(key),
               15, f'sidebar click did not open {key}')
    pause(capture, 1.0)


def _button_like(screen, *texts):
    from PySide6.QtWidgets import QAbstractButton
    found = []
    for button in screen.findChildren(QAbstractButton):
        label = button.text().replace('&', '').replace('…', '').strip().lower()
        if button.isVisible() and any(label.startswith(t.lower()) for t in texts):
            found.append(button)
    return found


def record_home_navigation_live(app, window, capture, settle):
    """Lesson 05 Home/navigation scenes; returns the focus regions (hand-off frame)."""
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtWidgets import QAbstractButton, QApplication
    from spacr.qt import app as gui
    from spacr.qt.command_palette import CommandPalette

    home = window._startup
    tabs = home._tabs
    focus = {}

    # Home tiles: the tab is chosen by a click before the clip (so the tile is
    # on screen when the clip starts); in the clip the pointer goes to the tile.
    hidden = []
    for key_, _, _, _ in gui.tiled_apps():
        tiles = [b for b in home.findChildren(QAbstractButton) if b.property('moduleAppKey') == key_]
        if key_ not in {row[0] for row in gui.tiled_apps(gui.visible_apps())}:
            # Not on Home with the capture profile (Feature maturity: Show
            # alpha off hides it); reported, never forced visible.
            hidden.append(key_)
            continue
        for index in range(tabs.count()):
            choices = [tile for tile in tiles if tabs.widget(index).isAncestorOf(tile)]
            if choices:
                if tabs.currentIndex() != index:
                    click(tabs.tabBar(), tabs.tabBar().tabRect(index).center())
                    wait_until(capture, lambda: tabs.currentIndex() == index, 5, 'Home tab did not open')
                    settle(0.4)
                tile = max(choices, key=lambda b: b.width() * b.height())
                capture('home_module_' + key_, actions=[(0.6, 'point at the tile', lambda t=tile: hover(t))])
                break
        else:
            raise RuntimeError('Current Home has no visible tile for ' + key_)

    # 06: Mask opened from the sidebar, inside the clip.
    capture('06_mask_host', actions=[(0.8, 'open Mask from the sidebar',
                                      lambda: open_from_sidebar(capture, window, 'mask'))])
    screen = window._screens['mask']

    def show_settings():
        from spacr.qt.widgets.section import Section
        sections = [s for s in screen.findChildren(Section) if s.isVisible()]
        folded = [s for s in sections if not s.is_expanded()]
        if folded:
            toggle_section(capture, folded[0], True)
            pause(capture, 2.0)
        for button in _button_like(screen, 'Load test data')[:1]:
            hover(button)
            pause(capture, 2.0)

    capture('06b_mask_settings', actions=[(0.8, 'expand a settings section, point at Load test data', show_settings)])

    def show_actions():
        targets = (_button_like(screen, 'Import settings')[:1]
                   + [b for b in (getattr(screen, '_btn_run', None), getattr(screen, '_btn_stop', None))
                      if b is not None and b.isVisible()])
        console = getattr(screen, '_console', None)
        for widget in targets + ([console] if console is not None and console.isVisible() else []):
            hover(widget)
            pause(capture, 2.5)

    capture('06c_mask_actions', actions=[(0.8, 'point at Import settings, Run, Stop and the console', show_actions)])

    # 14: drag the settings divider wider, then Ctrl + wheel over the right column.
    body = getattr(screen, '_body_splitter', None)
    runtime = getattr(screen, '_runtime_wrap', None)
    if body is None or not body.isVisible():
        raise RuntimeError('mask has no visible settings splitter')
    handle = body.handle(1)
    from spacr.qt.live_zoom import _COLUMN_FILTER_ATTRIBUTE
    column = getattr(app, _COLUMN_FILTER_ATTRIBUTE, None)
    scale_before = column.scale() if column is not None else None
    sizes_before = body.sizes()
    focus['14_resize_panels'] = {'handle': handoff(capture, handle, window),
                                 'column': handoff(capture, runtime, window) if runtime is not None else None}
    shift = int(round(420 * window.width() / 3840))

    def resize():
        # A human drag: press the divider, move it in a few small steps with
        # the button held (every motion event reaches the splitter), release.
        adapter = host_of(capture).adapter
        x0, y0 = adapter.target_point(handle, handle.rect().center())
        adapter.call('move_to', x0, y0)
        pause(capture, 0.15)
        adapter.call('button_down', 'left')
        pause(capture, 0.15)
        steps = 6
        for i in range(1, steps + 1):
            adapter.call('move_to', x0 + shift * i / steps, y0, duration=0.22)
            pause(capture, 0.06)
        adapter.call('button_up', 'left')
        adapter.note('drag', start=[x0, y0], delta=[shift, 0], steps=steps)
        pause(capture, 1.0)
        if runtime is not None:
            # Ctrl held, then single wheel notches with a pause between them.
            adapter.call('move_to', *adapter.target_point(runtime, None))
            adapter.call('key_down', 'Control_L')
            try:
                for _ in range(3):
                    adapter.call('scroll', -1)                    # wheel up = larger text
                    pause(capture, 0.45)
            finally:
                adapter.call('key_up', 'Control_L')
            adapter.note('wheel', notches=-3, modifiers=['Control_L'], one_at_a_time=True)

    capture('14_resize_panels', actions=[(1.0, 'drag the divider, Ctrl + wheel the column', resize)])
    proof = {'splitter_before': sizes_before, 'splitter_after': body.sizes(),
             'column_scale_before': scale_before, 'column_scale_after': column.scale() if column else None}
    # Put it back the way a user would (not recorded).
    drag(capture, body.handle(1), body.handle(1).rect().center(), (-shift, 0))
    if runtime is not None and column is not None:
        for _ in range(6):
            if abs(column.scale() - scale_before) < 1e-6:
                break
            wheel(capture, runtime, 1 if column.scale() > scale_before else -1, None, modifiers=('Control_L',))
    proof['restored'] = {'splitter': body.sizes(), 'column_scale': column.scale() if column else None}

    # 10/11: back on Home (not recorded), the right-hand column and its fold handle.
    open_from_sidebar(capture, window, '__home__')
    split = home._aside_split
    if home._aside_collapsed():
        raise RuntimeError('The Home column starts folded')
    focus['10_home_column'] = handoff(capture, home._aside, window)

    def tour_column():
        for widget in (home._aside, home._aside_controls):
            hover(widget)
            pause(capture, 3.0)

    capture('10_home_column', actions=[(1.0, 'point along the column and its Text size slider', tour_column)])
    index = split.indexOf(home._aside)
    focus['11_column_folded_open_handle'] = handoff(capture, split.handle(index), window)

    def fold():
        click(split.handle(index), split.handle(index).rect().center())
        wait_until(capture, home._aside_collapsed, 5, 'the Home column did not fold')
        focus['11_column_folded'] = handoff(capture, split.handle(index), window)

    def unfold():
        click(split.handle(index), split.handle(index).rect().center())
        wait_until(capture, lambda: not home._aside_collapsed(), 5, 'the Home column did not come back')

    seconds = float(getattr(capture, 'clip_seconds_of', lambda n: 20)('11_column_folded'))
    capture('11_column_folded', actions=[(1.0, 'fold the column', fold),
                                         (max(2.0, seconds - 5.0), 'bring it back', unfold)])

    # 08: Help search, typed.
    from spacr.qt.help_search import field_of
    field = field_of(window)
    if field is None or not field.isVisible():
        raise RuntimeError('The Help search field is not visible')

    def search():
        click(field)
        key(field, Qt.Key_A, Qt.ControlModifier)
        type_text(field, 'Performance')
        wait_until(capture, lambda: field.results() and field.popup().isVisible(), 30, 'no Performance results')
        settle(0.6)
        focus['08_help_search'] = handoff(capture, field.popup(), window)
        hover(field.popup())
        pause(capture, 2.0)

    seconds = float(getattr(capture, 'clip_seconds_of', lambda n: 20)('08_help_search'))
    capture('08_help_search', actions=[(0.8, 'type Performance into Help search', search),
                                       (max(2.0, seconds - 2.5), 'close the results',
                                        lambda: key(field, Qt.Key_Escape))])
    key(field, Qt.Key_A, Qt.ControlModifier)
    key(field, Qt.Key_Backspace)
    settle(0.3)

    # 12: Ctrl+K command palette; it runs its own event loop. Nothing is
    # driven re-entrantly from inside the open call: every step below is one
    # short timer callback in the palette loop that sends one real key (or
    # checks a condition) and returns, so the loop runs between keys exactly
    # as it does for a typing user.
    from PySide6.QtCore import QTimer
    palette_steps = {}
    adapter = host_of(capture).adapter
    began = capture.begin('12_command_palette')
    pause(capture, 0.8)

    def palette_open():
        return next((w for w in QApplication.topLevelWidgets()
                     if isinstance(w, CommandPalette) and w.isVisible()), None)

    def run_steps(steps, index=0, pending=None):
        if pending is not None:
            try:
                if not adapter.finished(pending):
                    QTimer.singleShot(10, lambda: run_steps(steps, index, pending))
                    return
            except BaseException as error:              # noqa: BLE001
                palette_steps['error'] = error
                index = len(steps)
        if index >= len(steps) or 'error' in palette_steps:
            palette_steps['finished'] = True
            return
        kind, value = steps[index]
        if kind in ('key', 'type'):
            seq = adapter.start('press' if kind == 'key' else 'type_text', value)
            QTimer.singleShot(10, lambda: run_steps(steps, index + 1, seq))
        elif kind == 'until':                            # (predicate, timeout, what)
            predicate, timeout, what = value
            started = palette_steps.setdefault(('until', index), time.monotonic())
            if predicate():
                QTimer.singleShot(0, lambda: run_steps(steps, index + 1))
            elif time.monotonic() - started > timeout:
                palette_steps['error'] = TimeoutError(what)
                palette_steps['finished'] = True
            else:
                QTimer.singleShot(30, lambda: run_steps(steps, index, None))
        else:                                            # ('call', fn)
            value()
            QTimer.singleShot(0, lambda: run_steps(steps, index + 1))

    def remember_region():
        palette_steps['region'] = handoff(capture, palette_open())

    host = host_of(capture)
    def after(seconds):
        mark = {}
        return lambda: time.monotonic() - mark.setdefault('t', time.monotonic()) >= seconds

    steps = [('until', (lambda: palette_open() is not None, 30, 'the command palette did not open')),
             ('until', (after(0.6), 5, 'pause'))]
    for ch in 'Meas':
        steps += [('type', ch), ('until', (after(0.14), 5, 'typing pause'))]
    steps += [('until', (lambda: palette_open() is not None and palette_open()._list.count() > 0,
                         10, 'command palette listed nothing')),
              ('call', remember_region),
              ('until', (lambda: host.current is None or host.elapsed() >= clip_seconds(capture) - 2.0,
                         600, 'palette hold')),
              ('key', 'Escape'),
              ('until', (lambda: palette_open() is None, 10, 'Escape did not close the command palette'))]
    QTimer.singleShot(0, lambda: run_steps(steps))
    # The Ctrl+K press returns when the palette loop ends (Escape above).
    key(window, Qt.Key_K, Qt.ControlModifier)
    wait_until(capture, lambda: palette_steps.get('finished'), 120, 'palette steps did not finish')
    if 'error' in palette_steps:
        raise palette_steps['error']
    if began:
        capture.end()
    if palette_open() is not None:
        raise RuntimeError('Escape did not close the command palette')
    focus['12_command_palette'] = palette_steps.get('region')
    proof['modules_not_on_home'] = hidden
    return {'focus': focus, 'proof': proof}
