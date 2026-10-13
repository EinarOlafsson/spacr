"""Record Model Zoo's optional source headings and Mask generation's
"Measure diameters…" popup (items 503, 504, 533) through real controls.

Headings are clicked like a user does; bioimage.io rows need the network,
so run this part with the recording namespace online. The diameter popup
is opened from Mask generation's per-object model cell, the only place it
lives, and measures a private copy of real example fields on the CPU.
"""
from __future__ import annotations

import time
from pathlib import Path


def record_source_headings(app, zoo, capture, settle, timeout, facts):
    """Turn on the Cellpose 3 and bioimage.io headings, capture, turn them off."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    strip = zoo.sources

    def visible_rows():
        table = zoo._table
        return [row for row in range(table.rowCount()) if not table.isRowHidden(row)]

    def toggle(name, on):
        heading = strip.heading(name)
        if strip.is_on(name) != on:
            QTest.mouseClick(heading, Qt.LeftButton)
            settle(0.6)
        if strip.is_on(name) != on:
            raise RuntimeError(f'The {name} heading did not turn {"on" if on else "off"}')

    # Headings remember their state between sessions; start from the default.
    toggle('cellpose3', False)
    toggle('bioimage.io', False)
    before = len(visible_rows())
    toggle('cellpose3', True)
    zoo._table.scrollToBottom()
    settle(0.5)
    shown = len(visible_rows())
    if shown <= before:
        raise RuntimeError('The cellpose3 heading showed no Cellpose 3 models')
    facts['cellpose3_rows'] = shown - before
    capture('03b_cellpose3_models')
    toggle('cellpose3', False)

    toggle('bioimage.io', True)
    deadline = time.monotonic() + min(timeout, 180)
    while len(visible_rows()) <= before:
        if time.monotonic() >= deadline:
            import spacr.model_zoo as catalogue
            sources = sorted({catalogue.source_of(e) for e in getattr(zoo, '_entries', [])})
            raise RuntimeError(f'bioimage.io listed no models: on={strip.enabled()} '
                               f'rows={zoo._table.rowCount()} visible={len(visible_rows())} '
                               f'before={before} sources={sources} home={Path.home()}')
        settle(0.5)
    settle(1.0)
    facts['bioimageio_rows'] = len(visible_rows()) - before
    zoo._table.scrollToBottom()
    settle(0.5)
    capture('03c_bioimageio_models')
    toggle('bioimage.io', False)
    zoo._table.scrollToTop()
    settle(0.5)


def record_measure_diameters(app, window, stage, capture, settle, write_json, captures, timeout, source):
    """Mask generation -> per-object Model cell -> Measure diameters… -> Measure."""
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QLineEdit

    from spacr.qt.prerun import DiameterDialog
    from spacr.qt.widgets.model_zoo_picker import ModelZooPicker
    from spacr.qt.widgets.object_settings_grid import MODEL_QUESTION, ObjectSettingsGrid

    window._on_nav_selected('mask')
    deadline = time.monotonic() + 60
    while window._screens.get('mask') is None:
        if time.monotonic() > deadline:
            raise TimeoutError('Mask did not open')
        settle(0.1)
    settle(2)
    mask = window._screens['mask']
    field = mask._settings_model._widgets['src']
    edit = field if isinstance(field, QLineEdit) else field.findChild(QLineEdit)
    mask._settings_scroll.ensureWidgetVisible(edit)
    settle(0.3)
    QTest.mouseClick(edit, Qt.LeftButton)
    QTest.keyClick(edit, Qt.Key_A, Qt.ControlModifier)
    QTest.keyClicks(edit, str(source))
    QTest.keyClick(edit, Qt.Key_Tab)
    settle(1.0)
    # The example's channels: nucleus 0, cell 1, pathogen 2 (its settings file).
    for key, value in (('nucleus_channel', '0'), ('cell_channel', '1'), ('pathogen_channel', '2')):
        widget = mask._settings_model._widgets[key]
        line = widget if isinstance(widget, QLineEdit) else widget.findChild(QLineEdit)
        mask._settings_scroll.ensureWidgetVisible(line)
        QTest.mouseClick(line, Qt.LeftButton)
        QTest.keyClick(line, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(line, value)
        QTest.keyClick(line, Qt.Key_Tab)
        settle(0.4)
    capture('09b_mask_source_and_channels')
    # Essentials hides the per-object table; show every setting.
    disclosure = mask._settings_search._disclosure
    QTest.mouseClick(disclosure, Qt.LeftButton)
    settle(1.0)

    grids = mask.findChildren(ObjectSettingsGrid)
    if not grids:
        raise RuntimeError('Mask generation has no per-object settings table')
    grid = grids[0]
    # Open the folded category that holds the table with its own header.
    widget = grid.parentWidget()
    while widget is not None and not grid.isVisible():
        if hasattr(widget, 'is_expanded') and hasattr(widget, '_header') and not widget.is_expanded():
            QTest.mouseClick(widget._header, Qt.LeftButton)
            settle(0.5)
        widget = widget.parentWidget()
    if not grid.isVisible():
        raise RuntimeError('The per-object settings table did not become visible')
    model = grid._model
    row = next(r for r in range(model.rowCount()) if model.question_at(r) == MODEL_QUESTION)
    column = model.objects().index('cell')
    table = grid._table
    index = table.model().index(row, column)
    mapper = getattr(table.model(), 'mapFromSource', None)
    if mapper is not None:
        index = mapper(model.index(row, column))
    mask._settings_scroll.ensureWidgetVisible(table)
    table.scrollTo(index)
    settle(0.5)

    errors, facts = [], {}

    def in_picker():
        picker = app.activeModalWidget()
        if not isinstance(picker, ModelZooPicker):
            if time.monotonic() < deadline_picker[0]:
                QTimer.singleShot(200, in_picker)
            else:
                errors.append('The Model zoo popup did not open from the model cell')
            return
        try:
            picker.resize(2200, 1400)
            picker.move(window.pos().x() + 820, window.pos().y() + 330)
            settle(1.5)
            if picker.diameter_button is None or not picker.diameter_button.isVisible():
                raise RuntimeError('The Model zoo popup has no Measure diameters button')
            capture('10_mask_model_zoo_popup')
            QTest.mouseClick(picker.diameter_button, Qt.LeftButton)
            settle(1.0)
            dialog = next((w for w in app.topLevelWidgets()
                           if isinstance(w, DiameterDialog) and w.isVisible()), None)
            if dialog is None:
                raise RuntimeError('Measure diameters did not open its popup')
            dialog.resize(1500, 1000)
            dialog.move(window.pos().x() + 1520, window.pos().y() + 560)
            settle(0.5)
            capture('11_measure_diameters_popup')
            panel = dialog.panel
            QTest.mouseClick(panel._btn_measure, Qt.LeftButton)
            settle(0.5)
            end = time.monotonic() + timeout
            while panel.busy:
                if time.monotonic() > end:
                    raise TimeoutError('The diameter estimate did not finish')
                settle(0.5)
            settle(1.0)
            capture('12_measured_diameters')
            QTest.mouseClick(dialog.close_button, Qt.LeftButton)
            settle(0.5)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
        finally:
            picker.reject()

    deadline_picker = [time.monotonic() + 30]
    QTimer.singleShot(300, in_picker)
    rect = table.visualRect(index)
    QTest.mouseClick(table.viewport(), Qt.LeftButton, pos=rect.center())
    settle(1.0)
    if errors:
        raise RuntimeError('; '.join(errors))
    write_json(captures / 'measure_diameters_acceptance.json', {
        'accepted': True, 'opened_from': 'Mask generation per-object Model cell (cell)',
        'source_folder': str(source), 'measured': True, **facts})
