"""Record Make Masks' curation verdicts, Upload data, folder consolidation and
Organize for Measure through the real controls (items 396, 593, 598, 600).

Every file operated on is a private copy under the recording stage. Upload
data is opened and filled but never sent: the recording namespace has no
network and Upload is not pressed. Modal boxes are answered from a timer,
exactly as a user answers them, after their frame is captured.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path


def _modal(app, settle, kind, handle, errors, *, wait=20.0):
    """Answer the next modal widget of ``kind`` with ``handle(widget)``."""
    from PySide6.QtCore import QTimer

    deadline = time.monotonic() + wait

    def poll():
        widget = app.activeModalWidget()
        if isinstance(widget, kind) and widget.isVisible():
            try:
                handle(widget)
            except Exception as exc:  # noqa: BLE001 - reported by the caller
                errors.append(f'{kind.__name__}: {exc}')
                widget.reject()
            return
        if time.monotonic() >= deadline:
            errors.append(f'No {kind.__name__} appeared')
            if widget is not None:
                widget.reject()
            return
        QTimer.singleShot(150, poll)

    QTimer.singleShot(300, poll)


def _press(box, role_button):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    button = box.button(role_button)
    if button is None:
        raise RuntimeError('The box has no such button')
    QTest.mouseClick(button, Qt.LeftButton)


def record_curation(app, window, screen, captures, capture, settle, write_json, timeout):
    """Keep one field and Discard the next; read the verdicts back from the CSV."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    folder = Path(screen._folder)
    first = screen._image_files[screen._current_index]

    def ready():
        deadline = time.monotonic() + timeout
        while screen._loading or screen._canvas.image is None:
            if time.monotonic() >= deadline:
                raise TimeoutError('The next field did not load')
            settle(0.1)
        settle()

    for button in (screen._btn_keep, screen._btn_discard, screen._btn_skip, screen._btn_contribute):
        if not button.isVisible():
            raise RuntimeError(f'{button.text()} is not visible on the bottom row')
    QTest.mouseMove(screen._btn_keep)
    QTest.mouseClick(screen._btn_keep, Qt.LeftButton)
    ready()
    second = screen._image_files[screen._current_index]
    if second == first:
        raise RuntimeError('Keep did not move to the next field')
    capture('20_curate_keep')
    QTest.mouseMove(screen._btn_discard)
    QTest.mouseClick(screen._btn_discard, Qt.LeftButton)
    ready()
    settle(0.5)
    capture('21_curate_discard')
    ledger = folder / 'csv' / 'keep_discard.csv'
    if not ledger.is_file():
        raise RuntimeError('Keep/Discard did not write csv/keep_discard.csv')
    with ledger.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    text = ledger.read_text()
    if Path(first).stem not in text or Path(second).stem not in text:
        raise RuntimeError('The verdict ledger does not name both fields')
    write_json(captures / 'curation_acceptance.json', {
        'accepted': True, 'kept': first, 'discarded': second,
        'ledger_rows': len(rows), 'ledger_columns': list(rows[0]) if rows else [],
        'skip_enabled_without_queue': screen._btn_skip.isEnabled(),
        'files_deleted': False})


def record_upload(app, window, screen, captures, capture, settle, write_json):
    """Open Upload data, choose the curated fields and name the dataset; never send."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from spacr.qt.widgets.model_share_dialog import CURATED_SOURCE

    sent = []
    QTest.mouseClick(screen._btn_contribute, Qt.LeftButton)
    settle(0.8)
    dialog = getattr(screen, '_contribute_dialog', None)
    if dialog is None or not dialog.isVisible():
        raise RuntimeError('Upload data did not open its dialog')
    # Belt and braces: the namespace has no network, and this replaces the
    # sender so even a stray press could not reach the dataset.
    dialog._upload = lambda folder, target: sent.append(target) or ''
    dialog.resize(1500, 1250)
    dialog.move(window.pos().x() + 1150, window.pos().y() + 420)
    settle()
    curated = dialog.source_buttons.get(CURATED_SOURCE)
    if curated is not None and curated.isEnabled():
        QTest.mouseClick(curated, Qt.LeftButton)
        settle()
    edit = dialog.name_edit
    QTest.mouseClick(edit, Qt.LeftButton)
    QTest.keyClick(edit, Qt.Key_A, Qt.ControlModifier)
    QTest.keyClicks(edit, 'toxoplasma ER cells')
    settle(0.8)
    link = dialog.target_label.text()
    if 'huggingface.co/datasets' not in link:
        raise RuntimeError('Naming the dataset did not show its destination link')
    capture('22_upload_data')
    dialog.close()
    settle()
    if sent:
        raise RuntimeError('Upload data sent something during the recording')
    write_json(captures / 'upload_acceptance.json', {
        'accepted': True, 'dialog_opened': True, 'upload_pressed': False,
        'sent': False, 'destination_shown': True})


def prepare_nested(stage, sources):
    """Two private layouts from the same real fields.

    ``by_well``: one ER image per field in a subfolder per well, which
    Make Masks offers to consolidate. ``scope_export``: one subfolder per
    channel, as microscopes export, and ``cell_masks`` beside it.
    """
    import tempfile

    import numpy as np
    import tifffile

    parent = Path(tempfile.mkdtemp(prefix='organize-', dir=stage / 'make_masks_runs'))
    by_well = parent / 'by_well'
    export = parent / 'scope_export'
    masks = parent / 'cell_masks'
    masks.mkdir(parents=True)
    rows = []
    for number, source in enumerate(sources, start=1):
        merged = np.load(source, mmap_mode='r')
        field = f'E01_f{number:02d}'
        for plane, channel in ((0, 'Hoechst'), (1, 'ER'), (2, 'Toxo')):
            target = export / channel / f'{field}.tif'
            target.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(target, np.asarray(merged[..., plane]))
        tifffile.imwrite(masks / f'{field}.tif', np.asarray(merged[..., 4]))
        well = by_well / ('E01' if number == 1 else 'E02')
        well.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(well / 'field_1.tif', np.asarray(merged[..., 1]))
        rows.append({'source': str(source), 'field': field,
                     'planes': {'Hoechst': 0, 'ER': 1, 'Toxo': 2, 'cell mask': 4}})
    return parent, by_well, export, masks, rows


def record_consolidation(app, window, screen, stage, by_well, captures, capture, settle, write_json, timeout):
    """Open a folder whose images sit in subfolders and accept Consolidate."""
    import os

    from PySide6.QtCore import Qt, QUrl
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QDialogButtonBox, QFileDialog, QLineEdit, QMessageBox

    errors = []

    def choose(dialog):
        dialog.resize(1300, 950)
        dialog.setSidebarUrls([QUrl.fromLocalFile(str(stage))])
        # Browse to the parent (as a sidebar click would) and type the name.
        dialog.setDirectory(str(by_well.parent))
        settle(0.5)
        edit = dialog.findChild(QLineEdit, 'fileNameEdit')
        QTest.mouseClick(edit, Qt.LeftButton)
        QTest.keyClick(edit, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(edit, by_well.name)
        settle(0.2)
        if edit.text() != by_well.name:
            raise RuntimeError(f'The folder picker holds {edit.text()!r}, not the prepared folder')
        capture('23a_folder_picker_nested')
        _modal(app, settle, QMessageBox, consolidate, errors)
        QTest.mouseClick(dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Open), Qt.LeftButton)

    def consolidate(box):
        settle(0.3)
        if 'Consolidate' not in box.windowTitle() and 'subfolder' not in box.text():
            raise RuntimeError(f'Unexpected box: {box.windowTitle()}')
        capture('23_consolidate_offer')
        _press(box, QMessageBox.Yes)

    _modal(app, settle, QFileDialog, choose, errors)
    previous = Path.cwd()
    try:
        os.chdir(stage)
        QTest.mouseClick(screen._btn_open, Qt.LeftButton)
    finally:
        os.chdir(previous)
    output = by_well.parent / (by_well.name + '_renamed')
    deadline = time.monotonic() + timeout
    while not errors and (screen._folder_job_running() or
                          Path(screen._folder or '') != output or
                          screen._loading or screen._canvas.image is None):
        if time.monotonic() >= deadline:
            raise TimeoutError('The consolidated folder did not open')
        settle(0.2)
    if errors:
        raise RuntimeError('; '.join(errors))
    settle()
    copied = sorted(p.name for p in output.glob('*.tif'))
    if len(copied) != 2 or not (output / 'rename_manifest.csv').is_file():
        raise RuntimeError('Consolidation did not copy both images with a manifest')
    if not all((by_well / w / 'field_1.tif').is_file() for w in ('E01', 'E02')):
        raise RuntimeError('Consolidation changed the original folders')
    capture('24_consolidated_opened')
    return {'output': str(output), 'copied': copied, 'originals_untouched': True}


def record_organize(app, window, screen, export, masks, captures, capture, settle, write_json, timeout):
    """Organize for Measure: a column per channel folder, a cell-mask column, Detect sets, Apply."""
    import numpy as np
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QMessageBox

    from spacr.qt.widgets.channel_sort_dialog import ExampleSetsDialog
    from spacr.qt.widgets.organize_for_measure import OrganizeForMeasureDialog

    errors, facts = [], {}

    def fill(dialog):
        settle(0.5)
        dialog.resize(2700, 1650)
        dialog.move(window.pos().x() + 570, window.pos().y() + 260)
        settle()
        # Point the popup at the microscope export, so the sorted folder
        # lands beside it rather than beside the folder open in the editor.
        dialog.source_edit.clear()
        QTest.mouseClick(dialog.source_edit, Qt.LeftButton)
        dialog.source_edit.setText(str(export))
        dialog.source_edit.editingFinished.emit()
        settle(0.5)
        capture('25_organize_open')
        channels = {}
        for name in ('Hoechst', 'ER', 'Toxo'):
            # What a drop of each channel folder on the new-column zone does.
            index = dialog.add_column('channel')
            dialog.add_files(index, [str(export / name)])
            channels[name] = index
            settle(0.3)
        er_channel = dialog._channel_of(channels['ER'])
        mask_column = dialog.add_column('mask', role='cell', of_channel=er_channel)
        dialog.add_files(mask_column, [str(masks)])
        # The column editors are rebuilt with deleteLater; a nested event
        # loop does not run deferred deletes, so flush them before painting.
        from PySide6.QtCore import QCoreApplication, QEvent
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        settle(0.5)
        facts['rows'] = len(dialog.rows)
        facts['columns'] = [c.kind for c in dialog.columns]
        if facts['rows'] != 2 or any(not all(row) for row in dialog.rows):
            raise RuntimeError(f'The table did not match two complete rows: {dialog.rows}')
        capture('26_organize_table')
        _modal(app, settle, ExampleSetsDialog, examples, errors)
        QTest.mouseClick(dialog.detect_button, Qt.LeftButton)
        settle(0.5)
        index = dialog.view_box.findData('both')
        dialog.view_box.setCurrentIndex(index)
        dialog.size_slider.setValue(dialog.size_slider.maximum())
        # Item 611: High thumbnails for checking rows by eye (display only).
        dialog.quality_box.setCurrentIndex(dialog.quality_box.findData('high'))
        settle(2.5)
        capture('28_organize_images')
        _modal(app, settle, QMessageBox, confirm, errors)
        QTest.mouseClick(dialog.apply_button, Qt.LeftButton)

    def examples(sets_dialog):
        settle(0.8)
        if 'sets' not in sets_dialog.windowTitle().lower():
            raise RuntimeError(f'Unexpected dialog {sets_dialog.windowTitle()}')
        capture('27_detect_sets_check')
        sets_dialog.accept()

    def confirm(box):
        settle(0.3)
        facts['plan'] = box.text()
        capture('29_move_and_merge')
        _press(box, QMessageBox.Yes)

    _modal(app, settle, OrganizeForMeasureDialog, fill, errors, wait=30)
    QTest.mouseClick(screen._btn_organize, Qt.LeftButton)
    if errors:
        raise RuntimeError('; '.join(errors))
    dialog = screen._organize_dialog
    if dialog is None or dialog.plan is None:
        raise RuntimeError('Organize for Measure was not applied')
    dest = Path(dialog.plan.dest)
    deadline = time.monotonic() + timeout
    while screen._folder_job_running() or Path(screen._folder or '') != dest / 'C01' or \
            screen._loading or screen._canvas.image is None:
        if time.monotonic() >= deadline:
            raise TimeoutError('The sorted channel folder did not open')
        settle(0.2)
    settle(1.0)
    merged = sorted(dest.glob('merged/*.npy'))
    if len(merged) != 2:
        raise RuntimeError(f'Expected two merged arrays, found {len(merged)}')
    array = np.load(merged[0], mmap_mode='r')
    if array.shape[-1] != 4:
        raise RuntimeError(f'Merged array has {array.shape[-1]} planes, expected 3 images + 1 mask')
    capture('30_organized_for_measure')
    write_json(captures / 'organize_acceptance.json', {
        'accepted': True, 'rows': facts['rows'], 'columns': facts['columns'],
        'plan_summary': facts.get('plan'), 'destination': str(dest),
        'channel_folders': sorted(p.name for p in dest.iterdir() if p.is_dir()),
        'merged_arrays': [p.name for p in merged], 'merged_planes': int(array.shape[-1]),
        'detect_sets_confirmed': True, 'files_moved_not_copied': True})
