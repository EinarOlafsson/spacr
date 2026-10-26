"""One call per user action, done for real in the GNOME capture host and as before elsewhere.

The scene drivers used to change the interface directly: ``setCurrentIndex``,
``setChecked``, ``setValue``, ``setText``, ``setSizes``, ``ensureWidgetVisible``,
``show_index``, ``selectFile``, ``set_value_for_key``, dialog ``resize``/``move``.
In a screen recording those changes happen with nobody touching anything.

Each function here takes the scene's ``capture`` callback. In the capture
host (``capture.host`` is set) it performs the action as a user does it --
a click, a drag, a wheel turn, typed text -- through
:mod:`capture_host_adapter` (via :mod:`capture_home_live`). Outside the host
(the Xvfb recorder) it does exactly what the driver did before, so existing
recordings are reproducible.
"""
from __future__ import annotations

import time


def live(capture) -> bool:
    return getattr(capture, 'host', None) is not None


def _h():
    import capture_home_live as h
    return h


def scroll_to(capture, scroll, widget, margin=8):
    """Bring ``widget`` into view inside ``scroll``."""
    if live(capture):
        _h().scroll_into_view(capture, scroll, widget, margin=margin)
    else:
        scroll.ensureWidgetVisible(widget)


def scroll_to_value(capture, scroll, value, orientation='vertical'):
    """Scroll to a scroll-bar value (0 = top/left)."""
    bar = scroll.verticalScrollBar() if orientation == 'vertical' else scroll.horizontalScrollBar()
    if not live(capture):
        bar.setValue(value)
        return
    value = max(bar.minimum(), min(bar.maximum(), int(value)))
    h = _h()
    for _ in range(80):
        if abs(bar.value() - value) <= max(2, bar.singleStep()):
            return
        before = bar.value()
        if orientation == 'vertical':
            h.scroll_bar_wheel(capture, scroll, -1 if value < before else 1)
        if bar.value() == before:
            h.drag_scroll_bar(capture, bar, value)
            if bar.value() == before:
                return


def splitter_sizes(capture, splitter, sizes):
    """Give a splitter these sizes: a drag of its first handle in the host."""
    if not live(capture):
        splitter.setSizes(list(sizes))
        return
    from PySide6.QtCore import Qt
    current = splitter.sizes()
    total = sum(current)
    want = int(round(sizes[0] * total / max(1, sum(sizes))))
    delta = want - current[0]
    if abs(delta) < 4:
        return
    handle = splitter.handle(1)
    move = (delta, 0) if splitter.orientation() == Qt.Horizontal else (0, delta)
    _h().drag(capture, handle, handle.rect().center(), move, duration=0.9)


def dialog_geometry(capture, dialog, size=None, position=None):
    """Old 4K recorder layouts resized/moved dialogs; in the host the app's own geometry stays."""
    if live(capture):
        return {'kept_application_geometry': [dialog.x(), dialog.y(), dialog.width(), dialog.height()]}
    if size is not None:
        dialog.resize(*size)
    if position is not None:
        dialog.move(position)
    return None


def select_tab(capture, tabs, index):
    if not live(capture):
        tabs.setCurrentIndex(index)
        return
    h = _h()
    if tabs.currentIndex() != index:
        h.click(tabs.tabBar(), tabs.tabBar().tabRect(index).center())
        h.wait_until(capture, lambda: tabs.currentIndex() == index, 5, 'tab did not open')
    h.pause(capture, 0.4)


def select_combo(capture, combo, index):
    if not live(capture):
        combo.setCurrentIndex(index)
        return
    if combo.currentIndex() != index:
        _h().choose_combo(capture, combo, index)


def select_row(capture, view, row):
    """A list/table row clicked (``setCurrentRow`` outside the host)."""
    if not live(capture):
        view.setCurrentRow(row)
        return
    item = view.item(row)
    _wheel_item_into_view(capture, view, item)
    _h().click(view.viewport(), view.visualItemRect(item).center())
    _h().pause(capture, 0.3)


def _wheel_item_into_view(capture, view, item):
    """Turn the wheel over a list until ``item`` is inside its viewport."""
    h = _h()
    for _ in range(60):
        rect = view.visualItemRect(item)
        port = view.viewport().rect()
        if port.contains(rect.center()):
            return
        horizontal = view.flow() == view.Flow.LeftToRight if hasattr(view, 'flow') else False
        ahead = (rect.center().x() > port.right()) if horizontal else (rect.center().y() > port.bottom())
        bar = view.horizontalScrollBar() if horizontal else view.verticalScrollBar()
        before = bar.value()
        h.wheel(capture, view.viewport(), 1 if ahead else -1, port.center())
        if bar.value() == before:
            return


def set_checked(capture, widget, state):
    if not live(capture):
        widget.setChecked(bool(state))
        return
    if widget.isChecked() != bool(state):
        _h().click(widget)
        _h().wait_until(capture, lambda: widget.isChecked() == bool(state), 5,
                        f'{widget.objectName() or type(widget).__name__} did not toggle')


def set_number(capture, spin, value):
    if not live(capture):
        spin.setValue(value)
        return
    if spin.value() != value:
        _h().type_number(capture, spin, value)


def set_text(capture, edit, text, *, commit=True):
    """Replace the text of a line or plain-text edit."""
    if not live(capture):
        if hasattr(edit, 'setPlainText') and not hasattr(edit, 'setText'):
            edit.setPlainText(text)
        else:
            edit.setText(text)
        return
    from PySide6.QtCore import Qt
    h = _h()
    h.click(edit)
    h.key(edit, Qt.Key_A, Qt.ControlModifier)
    if text:
        h.type_text(edit, text)
    else:
        h.key(edit, Qt.Key_Backspace)
    if commit and not hasattr(edit, 'toPlainText'):
        h.key(edit, Qt.Key_Tab)
    h.pause(capture, 0.2)


def show_figure(capture, queue, index):
    """Show figure ``index`` of a FigureQueue: click its thumbnail in the host."""
    if not live(capture):
        queue.show_index(index)
        return
    lst = queue._list
    item = lst.item(index)
    if item is None:
        return
    h = _h()
    card = queue.parentWidget()
    while card is not None and type(card).__name__ != 'Card':
        card = card.parentWidget()
    body = getattr(card, 'body', None)
    title = getattr(card, 'title_label', None)
    if body is not None and not body.isVisible() and title is not None and title.isVisible():
        # The Figures card is folded: open it from its title, as a user does.
        h.click(title)
        try:
            h.wait_until(capture, body.isVisible, 5, 'the Figures card did not open')
        except TimeoutError:
            pass
    if not lst.isVisible():
        # The Figure list pane is folded at the edge: unfold it with its handle.
        split = getattr(queue, '_body_split', None)
        if split is not None:
            handle = split.handle(1)
            h.click(handle, handle.rect().center())
            try:
                h.wait_until(capture, lst.isVisible, 5, 'the Figure list did not unfold')
            except TimeoutError:
                pass
    if not lst.isVisible():
        # Name the hidden ancestor, so the failing layout can be fixed.
        chain, widget = [], lst
        while widget is not None:
            chain.append(f"{type(widget).__name__}:{widget.objectName() or '-'}"
                         f"{'' if widget.isVisible() else '(hidden)'}"
                         f"[{widget.width()}x{widget.height()}]")
            widget = widget.parentWidget()
        split = getattr(queue, '_body_split', None)
        raise RuntimeError('The figure list is not visible to click: '
                           f"sizes={split.sizes() if split is not None else None} "
                           f"chain={' < '.join(chain[:12])}")
    _wheel_item_into_view(capture, lst, item)
    h.click(lst.viewport(), lst.visualItemRect(item).center())
    h.pause(capture, 0.5)


def console_to_end(capture, console):
    if not live(capture):
        console.jump_to_the_end()
        return
    jump = getattr(console, '_jump', None)
    if jump is not None and jump.isVisible():
        _h().click(jump)
        _h().pause(capture, 0.3)


def choose_file(capture, dialog, path):
    """A Qt file dialog: type the path into its file-name field and press Enter."""
    if not live(capture):
        dialog.selectFile(str(path))
        dialog.accept()
        return
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QLineEdit
    h = _h()
    edits = [e for e in dialog.findChildren(QLineEdit) if e.isVisible() and e.isEnabled()]
    name = next((e for e in edits if e.objectName() == 'fileNameEdit'), edits[0] if edits else None)
    if name is None:
        raise RuntimeError('The file dialog has no file-name field')
    h.click(name)
    h.key(name, Qt.Key_A, Qt.ControlModifier)
    h.type_text(name, str(path))
    h.pause(capture, 0.4)
    h.key(name, Qt.Key_Return)


def _settings_section(widget):
    w = widget.parentWidget()
    while w is not None and 'Section' not in [c.__name__ for c in type(w).__mro__]:   # Section or a subclass of it
        w = w.parentWidget()
    return w


def fill_setting(capture, screen, key, value):
    """Set one module setting through its control, the way a user edits the form."""
    model = screen._settings_model
    if not live(capture):
        return model.set_value_for_key(key, value)
    from PySide6.QtWidgets import (QAbstractButton, QCheckBox, QComboBox, QDoubleSpinBox,
                                   QLineEdit, QSpinBox)
    from spacr.qt.settings_search import DISCLOSURE_NAME
    h = _h()
    widget = model._widgets.get(key)
    if widget is None:
        return False
    if model.collect().get(key) == value:
        return True                 # already set: a user changes nothing here
    binding = getattr(screen, '_object_grid_binding', None)
    if binding is not None and key in binding.owned_keys():
        return _fill_grid_cell(capture, screen, key, value)
    if not widget.isVisible():
        toggle = screen.findChild(QAbstractButton, DISCLOSURE_NAME)
        if toggle is not None and toggle.isVisible() and not toggle.isChecked():
            h.click(toggle)                      # Essentials -> All settings
            h.pause(capture, 1.5)
    sections = []
    s = _settings_section(widget)
    while s is not None:
        sections.append(s)
        s = _settings_section(s)
    for section in reversed(sections):          # outermost first
        if hasattr(section, 'is_expanded') and not section.is_expanded():
            if not section.header().isVisible():
                raise RuntimeError(f'{key}: its settings section {section.title()!r} is not reachable')
            h.toggle_section(capture, section, True)
    # Since nightly a48cadef58 a settings section builds its rows over later event-loop turns: give the row
    # a bounded moment to appear (and re-read the widget, which a deferred build may replace).
    for _ in range(40):
        widget = model._widgets.get(key, widget)
        if widget.isVisible():
            break
        h.pause(capture, 0.25)
    if not widget.isVisible() and sections:
        # A section reports itself expanded while its body is still hidden (the body was shown before a deferred
        # build finished): close and open the hidden section again, as a user would, then wait once more.
        for section in reversed(sections):
            body = getattr(section, '_body', None)
            if body is not None and not body.isVisible():
                h.toggle_section(capture, section, False)
                h.toggle_section(capture, section, True)
        for _ in range(40):
            widget = model._widgets.get(key, widget)
            if widget.isVisible():
                break
            h.pause(capture, 0.25)
    if not widget.isVisible():
        states = [(sec.title(), sec.is_expanded(), getattr(sec, '_body', None) is not None and sec._body.isVisible())
                  for sec in sections]
        chain, w = [], widget
        while w is not None and len(chain) < 8:
            if not w.isVisibleTo(w.parentWidget()) if w.parentWidget() is not None else not w.isVisible():
                chain.append(f'{type(w).__name__}:{w.objectName() or "-"}')
            w = w.parentWidget()
        toggle = screen.findChild(QAbstractButton, DISCLOSURE_NAME)
        raise RuntimeError(f'The {key} control cannot be shown (hidden ancestors {chain}; disclosure '
                           f'{None if toggle is None else (toggle.isVisible(), toggle.isChecked())}; sections {states})')
    scroll = getattr(screen, '_settings_scroll', None) or h.scroll_area_of(widget)
    if scroll is not None:
        h.scroll_into_view(capture, scroll, widget, margin=8)
    if isinstance(widget, QCheckBox) or (isinstance(widget, QAbstractButton) and widget.isCheckable()):
        set_checked(capture, widget, bool(value))
    elif isinstance(widget, QSpinBox):
        set_number(capture, widget, int(value))
    elif isinstance(widget, QDoubleSpinBox):
        if float(widget.value()) != float(value):
            h.type_number(capture, widget, value)
    elif isinstance(widget, QComboBox):
        index = widget.findData(value)
        if index < 0:
            index = widget.findText(str(value))
        if index < 0:
            raise RuntimeError(f'{key}: no item {value!r} in the combo box')
        select_combo(capture, widget, index)
    elif isinstance(widget, QLineEdit):
        set_text(capture, widget, '' if value is None else str(value))
    elif type(widget).__name__ == '_ListEditor' and isinstance(value, (list, tuple)):
        _fill_chips(capture, widget, value)
    elif type(widget).__name__ == 'ClassEditorWidget' and isinstance(value, dict):
        # The Classes editor: pick the column, type the class name and the value, press Add --
        # once per class, the way a user builds the list.
        for name, rule in value.items():
            if not isinstance(rule, dict):
                raise RuntimeError(f'{key}: class {name!r} has no column/value rule')
            if any(getattr(r, 'name', None) == name for r in getattr(widget, '_rules', [])):
                continue
            if widget.column.lineEdit() is not None:
                set_text(capture, widget.column.lineEdit(), str(rule['column']))
            set_text(capture, widget.class_field, str(name))
            set_text(capture, widget.value_field, str(rule['value']))
            if not widget._add_typed.isEnabled():
                raise RuntimeError(f'{key}: the Add button for class {name!r} is not usable')
            h.click(widget._add_typed)
            h.pause(capture, 0.3)
        def _typed(rules):
            # The editor stores what was typed, so 1 and '1' are the same class value.
            return {n: (r.get('column'), str(r.get('value'))) for n, r in (rules or {}).items()}
        if _typed(model.collect().get(key)) != _typed(value):
            raise RuntimeError(f'{key}: the Classes editor holds {model.collect().get(key)!r}, not {value!r}')
    else:
        edits = [e for e in widget.findChildren(QLineEdit) if e.isVisible() and e.isEnabled()]
        if len(edits) == 1 and isinstance(value, (str, int, float)):
            set_text(capture, edits[0], str(value))
        else:
            raise RuntimeError(f'{key}: no user route for a {type(widget).__name__} control')
    h.pause(capture, 0.2)
    return True


def wait_worker(screen, settle, timeout=120):
    """The pipeline worker of a screen after Run. The run starts in deferred parts, so right after the
    click the attribute may be unset (AttributeError) or None; wait for it, bounded."""
    import time as _time
    deadline = _time.monotonic() + timeout
    worker = getattr(screen, '_worker', None)
    while worker is None and _time.monotonic() < deadline:
        settle(.2)
        worker = getattr(screen, '_worker', None)
    return worker


def wait(capture, seconds, settle):
    """Pause the scene (the event loop keeps running)."""
    if live(capture):
        _h().pause(capture, seconds)
    else:
        settle(seconds)


def open_module(capture, window, key, settle, prefer='sidebar'):
    """Open a module through its visible navigation, including organism assay tiles.

    ``prefer='tile'`` uses the Home tile when Home is showing (``From Home,
    open Mask``); otherwise the sidebar row, falling back to the other route.
    """
    if not live(capture):
        window._on_nav_selected(key)
        return
    from PySide6.QtWidgets import QAbstractButton
    from spacr.qt.screens.organism_screen import APP_KEY, FOLDED_APPS
    h = _h()
    rows = [row for row in window._sidebar._rows if row.key == key and row.isVisible()]
    on_home = window._stack.currentWidget() is window._startup
    if key != '__home__' and prefer == 'tile' and on_home:
        rows = []
    if key in FOLDED_APPS:
        # Assays have no standalone Home tile: enter their public organism
        # guide, then click its actual assay tile inside the same recording.
        open_module(capture, window, APP_KEY, settle, prefer=prefer)
        guide = window._screens[APP_KEY]
        choices = [button for button in guide.findChildren(QAbstractButton)
                   if button.property('organismModuleKey') == key]
        if len(choices) != 1 or not choices[0].isVisible() or not choices[0].isEnabled():
            raise RuntimeError(f'No unique visible enabled organism assay tile opens {key}')
        scroll_to(capture, guide._module_scroll, choices[0])
        h.pause(capture, 0.4)
        h.click(choices[0])
    elif rows:
        h.click(rows[0])
    else:
        if window._stack.currentWidget() is not window._startup:
            home = [row for row in window._sidebar._rows if row.key == '__home__' and row.isVisible()]
            if home:
                h.click(home[0])
                h.pause(capture, 1.0)
        start = window._startup
        tabs = start._tabs
        tiles = [b for b in start.findChildren(QAbstractButton) if b.property('moduleAppKey') == key]
        for index in range(tabs.count()):
            choices = [t for t in tiles if tabs.widget(index).isAncestorOf(t)]
            if choices:
                select_tab(capture, tabs, index)
                h.click(max(choices, key=lambda b: b.width() * b.height()))
                break
        else:
            raise RuntimeError(f'No sidebar row or Home tile opens {key} (hidden with alpha off?)')
    deadline = time.monotonic() + 60
    while window._screens.get(key) is None or window._stack.currentWidget() is not window._screens.get(key):
        if time.monotonic() > deadline:
            raise TimeoutError(f'{key} did not open')
        h.pause(capture, 0.1)


def close_dialog(capture, dialog, prefer=('Cancel', 'Close', 'Done', 'OK')):
    """Close a dialog with its own button (host) -- else ``close()`` as before."""
    if not live(capture):
        dialog.close()
        return
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QAbstractButton
    h = _h()
    from shiboken6 import isValid as _valid
    if not _valid(dialog) or not dialog.isVisible():
        return
    buttons = {b.text().replace('&', '').strip(): b for b in dialog.findChildren(QAbstractButton)
               if b.isVisible() and b.isEnabled()}
    for label in prefer:
        if label in buttons:
            h.click(buttons[label])
            break
    else:
        h.key(dialog, Qt.Key_Escape)
    from shiboken6 import isValid
    h.wait_until(capture, lambda: not isValid(dialog) or not dialog.isVisible(), 10, 'the dialog did not close')


def answer_input(capture, dialog, text):
    """A QInputDialog: replace its text by typing, then OK."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QLineEdit
    edit = dialog.findChild(QLineEdit)
    if not live(capture):
        edit.selectAll()
        QTest.keyClicks(edit, text)
        return
    h = _h()
    h.click(edit)
    h.key(edit, Qt.Key_A, Qt.ControlModifier)
    h.type_text(edit, text)


def set_slider(capture, slider, value):
    if not live(capture):
        slider.setValue(value)
        return
    if slider.value() != value:
        _h().set_slider(capture, slider, value)


def search_settings(capture, bar, query):
    """The settings search field: typed (host), else ``set_query``."""
    if not live(capture):
        bar.set_query(query)
        return
    from PySide6.QtCore import Qt
    h = _h()
    h.click(bar._input)
    h.key(bar._input, Qt.Key_A, Qt.ControlModifier)
    if query:
        h.type_text(bar._input, query)
    else:
        h.key(bar._input, Qt.Key_Backspace)
    h.pause(capture, 0.4)


def settings_level(capture, bar, level):
    """Essentials / All settings: the disclosure switch clicked (host), else ``set_level``."""
    if not live(capture):
        bar.set_level(level)
        return
    if bar.level() != level:
        _h().click(bar._disclosure)
        _h().pause(capture, 1.0)


def _fill_grid_cell(capture, screen, key, value):
    """A per-object table cell: double-click it, type the value, Enter (a switch cell: click)."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QAbstractButton
    from spacr.object_settings_table import _settings_key
    from spacr.qt.settings_search import DISCLOSURE_NAME
    h = _h()
    grid = screen._object_grid
    if not grid.isVisible():
        toggle = screen.findChild(QAbstractButton, DISCLOSURE_NAME)
        if toggle is not None and toggle.isVisible() and not toggle.isChecked():
            h.click(toggle)
            h.pause(capture, 1.5)
    model = grid._model
    table = grid._table
    found = None
    for row in range(model.rowCount()):
        for col, obj in enumerate(model.objects()):
            if _settings_key(obj, model.question_at(row)) == key:
                found = (row, col)
    if found is None:
        raise RuntimeError(f'{key}: no cell in the per-object table')
    index = table.model().index(*found)
    mapper = getattr(table.model(), 'mapFromSource', None)
    if mapper is not None:
        index = mapper(model.index(*found))
    scroll = getattr(screen, '_settings_scroll', None)
    if scroll is not None:
        h.scroll_into_view(capture, scroll, grid, margin=8)
    rect = table.visualRect(index)
    port = table.viewport()
    if isinstance(model.value_at(model.question_at(found[0]), model.objects()[found[1]]), bool):
        h.click(port, rect.center())
    else:
        from PySide6.QtTest import QTest
        QTest.mouseDClick(port, Qt.LeftButton, Qt.KeyboardModifier.NoModifier, rect.center())
        h.pause(capture, 0.4)
        editor = port.focusWidget() if hasattr(port, 'focusWidget') else None
        target = editor or port
        h.key(target, Qt.Key_A, Qt.ControlModifier)
        h.type_text(target, str(value))
        h.key(target, Qt.Key_Return)
    h.pause(capture, 0.4)
    if screen._settings_model.collect().get(key) != value:
        raise RuntimeError(f'{key}: the table cell did not take {value!r}')
    return True


def focus(capture, widget):
    """Give a control the keyboard focus: a click on it (host), else ``setFocus``."""
    if not live(capture):
        widget.setFocus()
        return
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QComboBox
    if QApplication.focusWidget() is widget:
        return
    _h().click(widget, widget.visibleRegion().boundingRect().center()
               if not widget.visibleRegion().isEmpty() else None)
    _h().pause(capture, 0.15)
    if isinstance(widget, QComboBox) and widget.view().isVisible():
        # A click on a combo box opens its list; the driver goes on with the
        # keyboard, so close the list and keep the focus (as a user would).
        _h().key(widget.view(), Qt.Key_Escape)
        _h().pause(capture, 0.15)


def press(capture, button):
    """A button pressed: clicked by the pointer (host), else ``click()``."""
    if not live(capture):
        button.click()
        return
    _h().click(button)


def open_live_settings(capture, panel):
    """A live preview's ``Live settings…``: its button clicked (host)."""
    if not live(capture):
        panel.open_live_settings()
        return
    _h().click(panel._live_settings_btn)
    _h().wait_until(capture, lambda: getattr(panel, '_live_settings_dialog', None) is not None
                    and panel._live_settings_dialog.isVisible(), 10, 'Live settings did not open')


def file_dialog_dir(capture, dialog, directory):
    """Go to a folder in a Qt file dialog.

    Host: nothing is done here; the caller types the full path of the file or
    folder it wants into the picker (what a user pasting a path does).
    """
    if not live(capture):
        dialog.setDirectory(str(directory))
        return
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QFileDialog, QLineEdit
    from pathlib import Path as _Path
    # Typing a folder and Enter is unreliable across picker modes (it can
    # choose the folder, or race the completer); callers type full paths.
    return
    h = _h()
    name = dialog.findChild(QLineEdit, 'fileNameEdit') or next(
        e for e in dialog.findChildren(QLineEdit) if e.isVisible())
    h.click(name)
    h.key(name, Qt.Key_A, Qt.ControlModifier)
    h.type_text(name, str(directory))
    h.pause(capture, 0.3)
    popup = QApplication.activePopupWidget()
    if popup is not None and popup.isVisible():
        h.key(popup, Qt.Key_Escape)          # close the completer's suggestions
        h.pause(capture, 0.2)
    h.key(name, Qt.Key_Return)
    try:
        h.wait_until(capture, lambda: _Path(dialog.directory().absolutePath()) == _Path(str(directory)),
                     5, 'the file dialog did not go to the folder')
    except TimeoutError:
        pass
    h.pause(capture, 0.3)
    if name.text():
        h.key(name, Qt.Key_A, Qt.ControlModifier)
        h.key(name, Qt.Key_Backspace)


def reject_dialog(capture, dialog):
    """Dismiss a dialog with its own Cancel/No/Close button (host), else ``reject()``."""
    if not live(capture):
        dialog.reject()
        return
    try:
        close_dialog(capture, dialog, prefer=('Cancel', 'No', 'Close', 'Abort'))
    except Exception:                                   # noqa: BLE001 - failure path: do not hang
        from shiboken6 import isValid
        if isValid(dialog) and dialog.isVisible():
            dialog.reject()


def accept_dialog(capture, dialog):
    """Accept a dialog with its own OK/Save/Open button (host), else ``accept()``."""
    if not live(capture):
        dialog.accept()
        return
    close_dialog(capture, dialog, prefer=('OK', 'Save', 'Open', 'Accept', 'Apply', 'Done', 'Yes'))


def settings_modified_only(capture, bar, state):
    if not live(capture):
        bar.set_modified_only(state)
        return
    if bar.modified_only() != bool(state):
        _h().click(bar._modified)
        _h().pause(capture, 0.5)


def wheel_at(capture, widget, notches, pos=None, modifiers=()):
    """Real wheel turns over ``widget`` (host); ``+`` = towards the user (down)."""
    _h().wheel(capture, widget, notches, pos, modifiers)


def code_action(capture, what):
    """Record that the scene changes the UI by code because no user route exists here.

    In the host this is an adapter bypass (the clip gate then fails the clip),
    never a silent change; outside the host it does nothing.
    """
    if not live(capture):
        return
    import time as _time
    adapter = capture.host.adapter
    entry = {'t': _time.monotonic(), 'qtest': 'code', 'reason': what}
    adapter.bypasses.append(entry)
    adapter.note('bypass', qtest='code', reason=what)


def when_modal(capture, delay_ms, handler, timeout=30.0):
    """Run ``handler`` once the modal dialog or menu that a click opens is showing.

    Outside the host this is ``QTimer.singleShot(delay_ms, handler)``: a
    synthetic click opens the dialog at once. A real pointer first has to
    glide to the button, so in the host the handler waits for the dialog
    (or runs anyway after ``timeout`` seconds, as before when none opens).
    """
    from PySide6.QtCore import QTimer
    if not live(capture):
        QTimer.singleShot(delay_ms, handler)
        return
    from PySide6.QtWidgets import QApplication
    import time as _time
    deadline = _time.monotonic() + timeout

    def poll():
        modal = QApplication.activeModalWidget() or QApplication.activePopupWidget()
        if (modal is not None and modal.isVisible()) or _time.monotonic() > deadline:
            QTimer.singleShot(250, handler)      # let it finish showing
            return
        QTimer.singleShot(150, poll)
    QTimer.singleShot(delay_ms, poll)


def _fill_strip(capture, strip, want):
    """One chip strip, in the narration order: keep the leading chips that already match, click the x of every
    chip after them (and of unwanted ones), then type the rest, each its own Enter. The order matters for a
    range such as [3500, 100000] (minimum, maximum)."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QToolButton
    h = _h()
    want = [str(v) for v in want]
    keep = 0
    for have_text, want_text in zip(strip.values(), want):
        if have_text != want_text:
            break
        keep += 1
    for chip in list(strip._chips)[keep:][::-1]:           # from the last chip backwards
        close = chip.findChild(QToolButton)
        h.click(close)
        h.pause(capture, 0.3)
    for text in want[keep:]:
        h.click(strip._entry)
        h.type_text(strip._entry, text)
        h.key(strip._entry, Qt.Key_Return)
        h.pause(capture, 0.3)
    if strip.values() != want:
        raise RuntimeError(f'The chip list holds {strip.values()}, not {want}')


def _fill_chips(capture, editor, value):
    """A chip list (e.g. Channels 0 1 2 3), or a list of groups (one chip strip per inner list)."""
    h = _h()
    nested = bool(value) and all(isinstance(v, (list, tuple)) for v in value)
    if any(isinstance(v, (list, tuple)) for v in value) and not nested:
        raise RuntimeError('No user route here for a mixed list setting')
    if not nested:
        if len(editor._strips) != 1:
            raise RuntimeError('No user route here for a grouped editor holding a flat value')
        _fill_strip(capture, editor._strips[0], value)
        return
    # A list of groups: the editor's own footer turns a flat list into groups ("Use groups") and adds
    # one group per press ("Add group"); a surplus group is removed by clearing its chips.
    if not editor._nested:
        h.click(editor._footer)
        h.pause(capture, 0.4)
    while len(editor._strips) < len(value):
        h.click(editor._footer)
        h.pause(capture, 0.4)
    while len(editor._strips) > len(value) and len(editor._strips) > 1:
        strip = editor._strips[-1]
        before = len(editor._strips)
        _fill_strip(capture, strip, [])
        h.pause(capture, 0.4)
        if len(editor._strips) >= before:
            raise RuntimeError('A surplus group did not go when its chips were cleared')
    for strip, group in zip(list(editor._strips), value):
        _fill_strip(capture, strip, group)
    got = [[str(x) for x in strip.values()] for strip in editor._strips]
    if got != [[str(x) for x in group] for group in value]:
        raise RuntimeError(f'The grouped list holds {got}, not {value}')


def accept_file_dialog(capture, dialog, button=None):
    """Accept a Qt file dialog whose file-name field holds the wanted path.

    Host: close the completer's suggestions (Escape on the popup), then press
    Enter in the field -- the dialog's own default action, which works for
    open, save and folder pickers alike, and never targets a hidden button.
    Outside the host: click ``button`` (the Open/Save button) as before.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    if not live(capture):
        QTest.mouseClick(button, Qt.LeftButton)
        return
    from PySide6.QtWidgets import QApplication, QLineEdit
    from shiboken6 import isValid
    h = _h()
    popup = QApplication.activePopupWidget()
    if popup is not None and popup.isVisible():
        capture.host.adapter.dismiss_popup(popup)       # pick the suggestion, never Escape
        h.pause(capture, 0.2)
    name = dialog.findChild(QLineEdit, 'fileNameEdit')
    h.key(name, Qt.Key_Return)
    try:
        h.wait_until(capture, lambda: not isValid(dialog) or not dialog.isVisible(), 5, 'file dialog still open')
    except TimeoutError:
        if button is not None and button.isVisible() and button.isEnabled():
            h.click(button)


def console_end_all(capture, console, settle=None):
    """Show the end of every console block (host: the console's Jump to the end)."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    if not live(capture):
        for block, _, _ in console._pipeline_console_blocks():
            block.setFocus()
            QTest.keyClick(block, Qt.Key_End, Qt.ControlModifier)
        return
    console_to_end(capture, console)


def navigate_image(capture, screen, step):
    """Next/previous image on Make Masks.

    Host: the Prev/Next buttons are clicked when a pointer can reach them; at
    2560x1440 the pinned tool panel (MakeMasksToolPin) covers them, so the
    documented PageDown/PageUp shortcuts are used instead (reported to the app).
    """
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    button = screen._btn_next if step > 0 else screen._btn_prev
    if not live(capture):
        QTest.mouseClick(button, Qt.LeftButton)
        return
    start = getattr(screen, '_current_index', None)
    count = len(getattr(screen, '_image_files', []) or [])
    def _current_name():
        files = getattr(screen, '_image_files', None) or []
        index = getattr(screen, '_current_index', None)
        return files[index] if index is not None and 0 <= index < len(files) else None
    start_name = _current_name()
    adapter = capture.host.adapter
    point = adapter.target_point(button, None)
    if adapter.reachable(button, point) is None:
        _h().click(button)
    else:
        canvas = getattr(screen, '_canvas', None) or screen
        _h().key(canvas, Qt.Key_PageDown if step > 0 else Qt.Key_PageUp)
    # Return only once Make Masks shows the requested field: the click is
    # handled asynchronously, and a caller reading the canvas too early sees
    # the previous image (the YOLO reload check failed that way).
    expected = None
    if start is not None and count:
        expected = min(count - 1, max(0, start + (1 if step > 0 else -1)))   # no wrap at the ends
    if expected is not None and expected != start:
        # Done when the screen shows the requested index OR a different field than before. Next after a Recrop retires
        # the parent and opens its child at the SAME index, so the index alone never reaches start + 1.
        _h().wait_until(capture, lambda: ((getattr(screen, '_current_index', None) == expected
                                           or _current_name() != start_name)
                                          and not getattr(screen, '_loading', False)
                                          and getattr(getattr(screen, '_canvas', None), 'image', None) is not None),
                        60, f'Make Masks did not load field {expected} (index {getattr(screen, "_current_index", None)}, '
                            f'name {_current_name()!r}, was {start_name!r}, loading={getattr(screen, "_loading", None)})')
        _h().pause(capture, 0.3)


# --- Scene clips that contain their own actions --------------------------------
#
# A clip that only shows a finished state, with the pointer at rest, does not
# show a human doing the work. ``clip`` starts the recording BEFORE a scene's
# actions in the host, so the pointer is seen performing them; elsewhere the
# body runs first and the finished state is captured exactly as before.

from contextlib import contextmanager


@contextmanager
def clip(capture, name, *, lead=0.8):
    """Record ``name`` around the actions in the ``with`` body.

    Host: the recording starts, the starting state shows for ``lead`` seconds,
    the body performs the scene, and the clip stops once its planned duration
    has passed (a body that takes longer makes the clip longer; the host facts
    record the wall time). Yields True while a host clip is recording.
    Xvfb recorder: the body runs, then ``capture(name)`` as before.
    """
    if not live(capture):
        yield False
        capture(name)
        return
    started = bool(capture.begin(name))
    if started:
        _h().pause(capture, lead)
    try:
        yield started
    finally:
        if started:
            capture.end()


def clip_due(capture) -> bool:
    """Host: the clip being recorded has reached its planned duration."""
    host = getattr(capture, 'host', None)
    current = getattr(host, 'current', None) if host is not None else None
    return current is not None and host.elapsed() >= current['seconds']


def point_at(capture, *widgets, dwell=1.2):
    """Host: move the pointer to each visible widget in turn and rest there.

    The narration names these controls; the viewer sees the pointer go to
    each one. No-op outside the host and for hidden widgets.
    """
    if not live(capture):
        return []
    h = _h()
    shown = []
    for widget in widgets:
        if widget is None or not widget.isVisible():
            continue
        h.hover(widget)
        h.pause(capture, dwell)
        shown.append(widget.objectName() or type(widget).__name__)
    return shown


def _open_sections_of(capture, widget):
    """Press the headings of the collapsed settings sections that hold ``widget`` (outermost first), as a user does."""
    h = _h()
    sections = []
    s = _settings_section(widget)
    while s is not None:
        sections.append(s)
        s = _settings_section(s)
    for section in reversed(sections):
        if hasattr(section, 'is_expanded') and not section.is_expanded() and section.header().isVisible():
            h.toggle_section(capture, section, True)


def point_at_settings(capture, screen, keys, dwell=1.2):
    """Host: scroll each named setting into view and rest the pointer on it."""
    if not live(capture):
        return []
    model = getattr(screen, '_settings_model', None)
    scroll = getattr(screen, '_settings_scroll', None)
    widgets = getattr(model, '_widgets', {}) if model is not None else {}
    shown = []
    for key in keys:
        widget = widgets.get(key)
        if widget is None:
            if _point_at_grid_cell(capture, screen, key, dwell):
                shown.append(key)
            continue
        if not widget.isVisible():
            _open_sections_of(capture, widget)
        if not widget.isVisible():
            continue                        # still hidden (another settings level): nothing to point at, nothing to scroll to
        if scroll is not None and not widget.visibleRegion().boundingRect().height() >= widget.height() // 2:
            scroll_to(capture, scroll, widget)
        if widget.isVisible():
            point_at(capture, widget, dwell=dwell)
            shown.append(key)
    return shown


def _point_at_grid_cell(capture, screen, key, dwell):
    """Rest the pointer on a per-object table cell, when the table is shown."""
    grid = getattr(screen, '_object_grid', None)
    if grid is None or not grid.isVisible():
        return False
    from spacr.object_settings_table import _settings_key
    model, table = grid._model, grid._table
    found = next(((row, col) for row in range(model.rowCount())
                  for col, obj in enumerate(model.objects())
                  if _settings_key(obj, model.question_at(row)) == key), None)
    if found is None:
        return False
    index = model.index(*found)
    mapper = getattr(table.model(), 'mapFromSource', None)
    if mapper is not None:
        index = mapper(index)
    h = _h()
    scroll = getattr(screen, '_settings_scroll', None)
    if scroll is not None:
        h.scroll_into_view(capture, scroll, grid, margin=8)
    h.hover(table.viewport(), table.visualRect(index).center())
    h.pause(capture, dwell)
    return True
