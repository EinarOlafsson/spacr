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

import shiboken6

import capture_ui
from capture_ui_drv_a import Rec, modal_wait, table_into_view, type_text


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
    bar = zoo._table.verticalScrollBar()
    # Host: each clip holds the pointer clicking the heading and the wheel
    # turning the list to the new rows.
    with Rec(capture, '03b_cellpose3_models', settle):
        toggle('cellpose3', True)
        capture_ui.scroll_to_value(capture, zoo._table, bar.maximum())
        settle(0.5)
        shown = len(visible_rows())
        if shown <= before:
            raise RuntimeError('The cellpose3 heading showed no Cellpose 3 models')
        facts['cellpose3_rows'] = shown - before
    toggle('cellpose3', False)

    with Rec(capture, '03c_bioimageio_models', settle):
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
        capture_ui.scroll_to_value(capture, zoo._table, bar.maximum())
        settle(0.5)
    toggle('bioimage.io', False)
    capture_ui.scroll_to_value(capture, zoo._table, 0)
    settle(0.5)


def record_measure_diameters(app, window, stage, capture, settle, write_json, captures, timeout, source):
    """Mask generation -> per-object Model cell -> Measure diameters… -> Measure."""
    from PySide6.QtCore import QPoint, Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QLineEdit

    from spacr.qt.prerun import DiameterDialog
    from spacr.qt.widgets.model_zoo_picker import ModelZooPicker
    from spacr.qt.widgets.object_settings_grid import MODEL_QUESTION, ObjectSettingsGrid

    # Host: the clip starts before Mask is opened, so the viewer sees the
    # sidebar click, the Source typed and the channel numbers filled in.
    setup = Rec(capture, '09b_mask_source_and_channels', settle).start()
    try:
        capture_ui.open_module(capture, window, 'mask', settle)
        deadline = time.monotonic() + 60
        while window._screens.get('mask') is None:
            if time.monotonic() > deadline:
                raise TimeoutError('Mask did not open')
            settle(0.1)
        settle(2)
        mask = window._screens['mask']
        field = mask._settings_model._widgets['src']
        edit = field if isinstance(field, QLineEdit) else field.findChild(QLineEdit)
        capture_ui.scroll_to(capture, mask._settings_scroll, edit)
        settle(0.3)
        type_text(capture, edit, source)
        settle(1.0)
        # The example's channels: nucleus 0, cell 1, pathogen 2 (its settings file).
        for key, value in (('nucleus_channel', '0'), ('cell_channel', '1'), ('pathogen_channel', '2')):
            widget = mask._settings_model._widgets[key]
            line = widget if isinstance(widget, QLineEdit) else widget.findChild(QLineEdit)
            capture_ui.scroll_to(capture, mask._settings_scroll, line)
            type_text(capture, line, value)
            settle(0.4)
    except BaseException:
        setup.abort()
        raise
    setup.stop()
    # Host: 10_mask_model_zoo_popup starts before "All settings" is switched
    # on and stops inside the popup (see in_picker).
    popup_rec = Rec(capture, '10_mask_model_zoo_popup', settle).start()
    try:
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
        capture_ui.scroll_to(capture, mask._settings_scroll, table)
        table_into_view(capture, table, index.row(), index.column())
        settle(0.5)
    except BaseException:
        popup_rec.abort()
        raise

    errors, facts = [], {}
    deadline_picker = [time.monotonic() + (90 if capture_ui.live(capture) else 30)]

    def in_picker():
        picker = app.activeModalWidget()
        if not isinstance(picker, ModelZooPicker):
            if time.monotonic() < deadline_picker[0]:
                QTimer.singleShot(200, in_picker)
            else:
                popup_rec.abort()
                errors.append('The Model zoo popup did not open from the model cell')
            return
        diameters = Rec(capture, '11_measure_diameters_popup', settle)
        measured = Rec(capture, '12_measured_diameters', settle)
        try:
            capture_ui.dialog_geometry(capture, picker, (2200, 1400),
                                       window.pos() + QPoint(820, 330))
            settle(1.5)
            if picker.diameter_button is None or not picker.diameter_button.isVisible():
                raise RuntimeError('The Model zoo popup has no Measure diameters button')
            popup_rec.stop()
            diameters.start()
            QTest.mouseClick(picker.diameter_button, Qt.LeftButton)
            settle(1.0)
            dialog = next((w for w in app.topLevelWidgets()
                           if isinstance(w, DiameterDialog) and w.isVisible()), None)
            if dialog is None:
                raise RuntimeError('Measure diameters did not open its popup')
            capture_ui.dialog_geometry(capture, dialog, (1500, 1000),
                                       window.pos() + QPoint(1520, 560))
            settle(0.5)
            diameters.point(dialog.panel._btn_measure, dwell=1.5)
            diameters.stop()
            panel = dialog.panel
            measured.start()
            QTest.mouseClick(panel._btn_measure, Qt.LeftButton)
            settle(0.5)
            end = time.monotonic() + timeout
            while panel.busy:
                if time.monotonic() > end:
                    raise TimeoutError('The diameter estimate did not finish')
                settle(0.5)
            settle(1.0)
            measured.stop()
            QTest.mouseClick(dialog.close_button, Qt.LeftButton)
            settle(0.5)
        except Exception as exc:  # noqa: BLE001
            for rec in (popup_rec, diameters, measured):
                rec.abort()
            errors.append(str(exc))
        finally:
            capture_ui.reject_dialog(capture, picker)

    modal_wait(capture, 300, in_picker)
    # The picker refreshes the per-object grid, so an index taken before it opened
    # may point into a reset model or a deleted table (exit 139 on 2026-10-29).
    grid = next(g for g in mask.findChildren(ObjectSettingsGrid) if shiboken6.isValid(g))
    table = grid._table
    row = next(r for r in range(grid._model.rowCount()) if grid._model.question_at(r) == MODEL_QUESTION)
    index = grid._model.index(row, grid._model.objects().index('cell'))
    mapper = getattr(table.model(), 'mapFromSource', None)
    if mapper is not None:
        index = mapper(index)
    if not index.isValid():
        raise RuntimeError('The model cell is not in the refreshed per-object table')
    rect = table.visualRect(index)
    QTest.mouseClick(table.viewport(), Qt.LeftButton, pos=rect.center())
    settle(1.0)
    if errors:
        raise RuntimeError('; '.join(errors))
    write_json(captures / 'measure_diameters_acceptance.json', {
        'accepted': True, 'opened_from': 'Mask generation per-object Model cell (cell)',
        'source_folder': str(source), 'measured': True, **facts})
