"""Demonstrate FEATURES with genuine file pickers and the real Measure runner."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time


def record_features_run(app, features, captures, capture, settle, write_json, timeout):
    """Assign the two prepared ER fields, measure them, and inspect the database.

    All application changes use visible controls. The input hashes and database
    checks are observations; no table rows or measurement results are injected.
    """
    from PySide6.QtCore import Qt, QTimer, QUrl
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QCheckBox, QDialogButtonBox, QFileDialog, QLineEdit, QSpinBox
    from spacr.qt.widgets.collapsible_section import CollapsibleSection

    inputs = json.loads((captures / 'inputs.json').read_text())
    stage = captures.parent.parent
    table = features.inputs

    def until(predicate, description):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() >= deadline:
                raise TimeoutError(description)
            settle(0.1)

    def type_text(widget, value):
        if not widget.isVisible() or not widget.isEnabled():
            raise RuntimeError('The input control is not visible and enabled')
        widget.setFocus()
        QTest.keyClick(widget, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(widget, str(value))
        QTest.keyClick(widget, Qt.Key_Tab)
        settle()

    QTest.mouseClick(table._clear, Qt.LeftButton)
    type_text(table._channels, 1)
    type_text(table._plate, 'maskdemo')
    assert table.table().n_channels == 1
    assert table.table().ordered_roles() == ('cell',)

    def choose_file(row, column, path, frame):
        failures, accepted = [], []

        def choose():
            dialog = app.activeModalWidget()
            try:
                if not isinstance(dialog, QFileDialog):
                    raise RuntimeError('The table did not open the real file picker')
                dialog.accepted.connect(lambda: accepted.append(True))
                dialog.resize(1300, 950)
                dialog.setSidebarUrls([QUrl.fromLocalFile(str(stage))])
                edit = dialog.findChild(QLineEdit, 'fileNameEdit')
                settle(0.2)
                QTest.mouseClick(edit, Qt.LeftButton)
                QTest.keyClick(edit, Qt.Key_A, Qt.ControlModifier)
                if edit.selectedText() != edit.text():
                    raise RuntimeError('The file picker did not select its previous entry')
                QTest.keyClicks(edit, str(path))
                if edit.text() != str(path):
                    raise RuntimeError('The file picker did not take the exact source path')
                if frame:
                    capture(frame, desktop=True)
                QTest.mouseClick(dialog.findChild(QDialogButtonBox).button(
                    QDialogButtonBox.Open), Qt.LeftButton)
            except Exception as error:
                failures.append(str(error))
                if dialog is not None:
                    dialog.reject()

        def stop_waiting():
            dialog = app.activeModalWidget()
            if dialog is not None and not accepted:
                failures.append('The file picker did not accept the chosen file')
                dialog.reject()

        cell = table._grid.item(row, column)
        table._grid.scrollToItem(cell)
        settle()
        position = table._grid.visualItemRect(cell).center()
        if not table._grid.viewport().rect().contains(position):
            raise RuntimeError('The assignment cell is outside the visible table')
        previous = Path.cwd()
        try:
            os.chdir(stage)
            QTimer.singleShot(350, choose)
            QTimer.singleShot(12000, stop_waiting)
            QTest.mouseClick(table._grid.viewport(), Qt.LeftButton, pos=position)
            QTest.mouseDClick(table._grid.viewport(), Qt.LeftButton, pos=position)
        finally:
            os.chdir(previous)
        if failures or not accepted:
            raise RuntimeError('; '.join(failures) or 'The file picker was cancelled')
        settle()

    for index, entry in enumerate(inputs):
        QTest.mouseClick(table._add_row, Qt.LeftButton)
        settle()
        choose_file(index, 3, entry['image'], '07_features_choose_image' if index == 0 else None)
        choose_file(index, 4, entry['mask'], '08_features_choose_cell_mask' if index == 0 else None)
    assert not table.problems(), table.problems()
    assert not table.unassigned()
    for row, entry in zip(table.table().rows, inputs):
        assert row.channels == {0: entry['image']}
        assert row.masks == {'cell': entry['mask']}
    capture('09_features_assigned_table', desktop=True)

    changes = {'n_jobs': 1, 'cell_min_size': 1, 'save_png': False}
    for key, value in changes.items():
        widget = features.settings._widgets[key]
        for section in features.findChildren(CollapsibleSection):
            if section.isAncestorOf(widget):
                features._settings_area.ensureWidgetVisible(section._header)
                settle()
                if not section.is_expanded():
                    QTest.mouseClick(section._header, Qt.LeftButton)
                    settle()
        features._settings_area.ensureWidgetVisible(widget)
        settle()
        if isinstance(widget, QSpinBox):
            type_text(widget, value)
            assert widget.value() == value, key
        elif isinstance(widget, QCheckBox):
            if widget.isChecked() != value:
                QTest.mouseClick(widget, Qt.LeftButton)
                settle()
            assert widget.isChecked() == value, key
        else:
            raise RuntimeError(f'Unsupported visible setting control: {key}')
        capture(f'10_features_setting_{key}', desktop=True)
    collected = features.settings.collect()
    assert all(collected[key] == value for key, value in changes.items())
    assert collected['save_measurements'] and not collected['test_mode']
    for section in features.findChildren(CollapsibleSection):
        if section.is_expanded():
            features._settings_area.ensureWidgetVisible(section._header)
            settle()
            QTest.mouseClick(section._header, Qt.LeftButton)
            settle()

    results = []
    features.run_finished.connect(results.append)
    assert features.run_button.isVisible() and features.run_button.isEnabled()
    QTest.mouseClick(features.run_button, Qt.LeftButton)
    capture('11_features_measuring', desktop=True)
    until(lambda: bool(results), 'FEATURES did not finish the real measurement')
    result = results[-1]
    if not isinstance(result, dict) or not result.get('db_exists'):
        raise RuntimeError('FEATURES did not produce its measurements database')
    database = Path(result['db_path'])
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as connection:
        tables = [row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        if 'cell' not in tables:
            raise RuntimeError('The completed measurement has no cell table')
        cell_rows = connection.execute('SELECT COUNT(*) FROM cell').fetchone()[0]
        # Match every object to the actual label plane and raw ER pixels.
        # Field identity matters too: a malformed plate name can preserve
        # the row count while collapsing two fields onto one database key.
        import math
        import numpy as np
        import tifffile

        checked_cells = 0
        for number, entry in enumerate(inputs, 1):
            labels = tifffile.imread(entry['mask'])
            intensity = tifffile.imread(entry['image'])
            counts = np.bincount(labels.ravel())
            sums = np.bincount(labels.ravel(), weights=intensity.ravel())
            expected = set(map(int, np.flatnonzero(counts))) - {0}
            rows = connection.execute(
                'SELECT object_label, plateID, rowID, columnID, fieldID, '
                'cell_area, cell_channel_0_mean_intensity FROM cell '
                'WHERE file_name=?', (f'maskdemo_A01_{number}',)).fetchall()
            if {int(row[0]) for row in rows} != expected or len(rows) != len(expected):
                raise RuntimeError('The database lost, duplicated or added a cell label')
            for label, plate, row, column, field, area, mean in rows:
                if (plate, row, column, field) != ('maskdemo', 'r1', 'c1', f'f{number}'):
                    raise RuntimeError('The database did not preserve the assigned field identity')
                if area != counts[label] or not math.isclose(
                        mean, sums[label] / counts[label], rel_tol=1e-10, abs_tol=1e-6):
                    raise RuntimeError('Cell area or mean intensity differs from the source pixels')
            checked_cells += len(rows)
        if checked_cells != cell_rows:
            raise RuntimeError('The database contains extra cells outside the assigned fields')
    if cell_rows < 1 or len(result['stems']) != len(inputs):
        raise RuntimeError('The measurement returned no cells or omitted a field')
    for entry in inputs:
        for key in ('image', 'mask'):
            assert hashlib.sha256(Path(entry[key]).read_bytes()).hexdigest() == entry[key + '_sha256']
    settle()
    capture('12_features_measurements_saved', desktop=True)
    proof = {'accepted': True, 'interaction': 'visible file pickers, settings and Measure button',
             'fields': len(inputs), 'channels': 1, 'mask_roles': ['cell'],
             'settings_chosen_visibly': changes, 'test_mode': False,
             'database': str(database), 'database_sha256': hashlib.sha256(database.read_bytes()).hexdigest(),
             'database_tables': tables, 'measured_cell_rows': cell_rows,
             'source_files_unchanged': True,
             'every_cell_area_and_mean_verified_against_source_pixels': True,
             'distinct_field_identities_verified': True, 'result': result}
    write_json(captures / 'features_measurement_acceptance.json', proof)
    return proof
