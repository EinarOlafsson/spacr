"""Check the unchanged Plate Viewer workflow without replacing its old media.

The current real UI runs on a byte-identical private copy. Per-well means and
counts are independently read with SQLite; no application worker is replaced.
The old CSV and all recorded statistics must still match, including blank wells.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import tempfile
import time

from capture_database import (SOURCE, SOURCE_SHA256, _digest, _readonly,
                              prepare_database_copy, require_unchanged_source)

AUTHOR = Path('/mnt/firecuda2/Claude/toxoplasma_projects/tutorials')
LESSON = '33_plate_viewer'
FEATURE = 'cell_channel_1_mean_intensity'
COLUMNS = ('plateID', 'well', 'rowID', 'columnID', 'row_index', 'column_index',
           'n', 'value', 'ring', 'is_edge')


def source_wells(database):
    """Independent SQL aggregates, without the app's ID parsing or groupby."""
    with _readonly(database) as con:
        rows = con.execute(
            'SELECT plateID,rowID,columnID,COUNT(*),'
            'AVG(cell_channel_1_mean_intensity) FROM cell '
            'GROUP BY plateID,rowID,columnID').fetchall()
    result = {}
    for plate, row, column, count, mean in rows:
        key = (str(plate), str(row), str(column))
        if key in result or not math.isfinite(mean):
            raise ValueError('Unexpected duplicate or nonfinite independent well')
        result[key] = {'n': count, 'mean': mean}
    if len(result) != 196 or sum(v['n'] for v in result.values()) != 81969:
        raise ValueError('The frozen real plate has different well/object identities')
    return result


def verify_wells(records, independent, grouping, minimum):
    expected = {k: v for k, v in independent.items() if v['n'] >= minimum}
    seen = set()
    for row in records:
        key = tuple(str(row[k]) for k in ('plateID', 'rowID', 'columnID'))
        if key not in expected or key in seen:
            raise ValueError('Wrong or duplicate returned well identity')
        seen.add(key)
        wanted = expected[key]
        value = wanted['n'] if grouping == 'count' else wanted['mean']
        if (int(row['n']) != wanted['n']
                or not math.isclose(float(row['value']), value, rel_tol=1e-12, abs_tol=1e-10)):
            raise ValueError('Returned well count or value differs from independent SQL')
    if seen != set(expected):
        raise ValueError('Missing wells; a blank must not be exported as zero')
    return {'wells': len(seen), 'dropped': len(independent) - len(seen),
            'all_well_identities_counts_values_match_sql': True}


def report_values(report):
    values = {key: getattr(report, key) for key in
              ('plate', 'n_wells', 'n_dropped_min_count', 'edge_detected',
               'pct_difference', 'cliffs_delta', 'p_value', 'summary')}
    values.update({axis + '_rho': report.gradient(axis).spearman_rho
                   for axis in ('row', 'column')})
    return values


def verify_report(actual, original):
    if actual != original:
        raise ValueError('The actual current spatial report differs from the old recording')


def verify_export(path, records):
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        if reader.fieldnames != list(COLUMNS) or len(rows) != len(records):
            raise ValueError('Wrong exported columns or row count')
    for got, expected in zip(rows, records):
        for key in COLUMNS:
            if key in ('value', 'row_index', 'column_index', 'n', 'ring'):
                same = float(got[key]) == float(expected[key])
            else:
                same = got[key] == str(expected[key])
            if not same:
                raise ValueError('The old/current export differs in identity, order or value')
    return {'rows': len(rows), 'columns': len(COLUMNS), 'all_values_and_order_exact': True,
            'sha256': _digest(path), 'path': str(path)}


def record_plate_retention(app, window, screen, stage, captures, capture,
                           settle, write_json, timeout):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QFileDialog, QDialogButtonBox, QLineEdit
    from shiboken6 import isValid
    from spacr.qt.screens.plate_view import PlateViewScreen
    from spacr.qt.widgets.fold_strip import FoldButton

    acceptance = Path(captures) / 'retention_acceptance.json'
    proof = {'lesson': LESSON, 'accepted': False, 'media_regenerated': False,
             'published': False, 'scope': 'Recorded Plate Viewer operations, not biological validation'}
    write_json(acceptance, proof)
    deadline = time.monotonic() + timeout
    panel = None
    original = None

    def tick():
        if time.monotonic() > deadline:
            raise TimeoutError('The bounded Plate Viewer check timed out')

    def click(widget):
        tick()
        if not widget.isVisible() or not widget.isEnabled() or widget.visibleRegion().isEmpty():
            raise RuntimeError('An actual requested Plate Viewer control is not usable')
        QTest.mouseClick(widget, Qt.LeftButton,
                         pos=widget.visibleRegion().boundingRect().center())
        settle(.2)

    def idle():
        # Let the real coalescing timer start, then wait for real workers.
        settle(.25)
        while panel.is_busy() or panel.active_jobs() or panel._recompute_timer.isActive():
            tick()
            settle(.1)
        if panel.last_error:
            raise RuntimeError(panel.last_error)

    def fill(widget, text):
        click(widget)
        QTest.keyClick(widget, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(widget, str(text))

    def combo(widget, text):
        index = widget.findText(text)
        if index < 0:
            raise RuntimeError('The current combo no longer offers the recorded setting')
        if widget.currentIndex() == index:
            return
        click(widget)
        # Navigate the OPEN popup view, not the combo itself. The latter
        # starts intermediate table reads and can disable itself mid-gesture.
        view = widget.view()
        if not view.isVisible():
            raise RuntimeError('The actual combo popup did not open')
        QTest.keyClick(view, Qt.Key_Home)
        for _ in range(index):
            QTest.keyClick(view, Qt.Key_Down)
        QTest.keyClick(view, Qt.Key_Return)
        idle()
        proof.setdefault('combo_checks', []).append({'requested': text,
            'actual': widget.currentText(), 'popup_keyboard_selection': True})
        if widget.currentText() != text:
            capture('failed_combo_selection')
            raise RuntimeError(f'The actual combo selected {widget.currentText()!r}, not {text!r}')

    try:
        parent = Path(stage) / 'plate_retention_runs'
        parent.mkdir(exist_ok=True)
        private = Path(tempfile.mkdtemp(prefix='example-', dir=parent))
        original = prepare_database_copy(SOURCE, private / 'measurements.db',
                                         expected_sha256=SOURCE_SHA256)
        proof['source'] = original
        independent = source_wells(private / 'measurements.db')
        baseline = json.loads((AUTHOR / 'production' / LESSON / 'source_manifest.json').read_text())
        if baseline['source_sha256_before'] != SOURCE_SHA256 or baseline['source_sha256_after'] != SOURCE_SHA256:
            raise ValueError('The old recording used a different source')
        folds = [w for w in screen.findChildren(FoldButton)
                 if w.app_key == 'plate_view' and w.isVisible()]
        if len(folds) != 1:
            raise RuntimeError('Graph Builder has no unique visible Plate Viewer fold')
        capture('01_graph_builder_host')
        click(folds[0])
        panels = [w for w in window.findChildren(PlateViewScreen) if w.isVisible()]
        if len(panels) != 1 or not panels[0]._threaded:
            raise RuntimeError('The real threaded Plate Viewer fold did not open')
        panel = panels[0]
        if not panel.link.filter.is_empty:
            raise RuntimeError('This retained unfiltered walkthrough requires a fresh empty shared filter')
        capture('02_current_fold')
        fill(panel._path_edit, private / 'measurements.db')
        click(panel._btn_open)
        idle()
        combo(panel._table_combo, 'cell')
        combo(panel._value_combo, FEATURE)
        combo(panel._grouping_combo, 'mean')
        fill(panel._min_count_box, '100')
        QTest.keyClick(panel._min_count_box, Qt.Key_Return)
        click(panel._btn_render)
        idle()
        proof['states'] = {}

        def inspect(name, phase=None):
            records = panel._layout_df.to_dict('records')
            item = baseline[name]
            result = verify_wells(records, independent, item['grouping'], item['min_count'])
            actual = report_values(panel._report)
            verify_report(actual, item['report'])
            result['report'] = actual
            result['status'] = panel._status.text()
            result['current_threaded_result_matches_original'] = True
            proof['states'][phase or name] = result
            capture('03_' + (phase or name))
            return records

        inspect('initial')
        combo(panel._grouping_combo, 'count')
        inspect('count')
        fill(panel._min_count_box, '350')
        QTest.keyClick(panel._min_count_box, Qt.Key_Return)
        idle()
        records = inspect('filtered')
        proof['old_export'] = verify_export(
            AUTHOR / 'generated/plate_viewer/plate1_cell_counts_min350.csv', records)
        well_checks = []
        for row, column, blank in ((7, 7, False), (3, 14, True)):
            point = panel._grid.cell_rect(row, column).center().toPoint()
            QTest.mouseClick(panel._grid, Qt.LeftButton, pos=point)
            settle(.2)
            text = panel.well_info_text()
            if ('blank' in text.lower()) != blank:
                raise ValueError('The actual well click did not distinguish measured and blank wells')
            well_checks.append({'row': row, 'column': column, 'blank': blank, 'text': text})
            capture('04_blank_well' if blank else '04_measured_well')
        proof['well_clicks'] = well_checks

        exported = private / 'current_plate1_cell_counts_min350.csv'
        accepted, errors, dialogs = [], [], []
        opener, watchdog = QTimer(window), QTimer(window)
        for timer in (opener, watchdog):
            timer.setSingleShot(True)

        def handle():
            dialog = app.activeModalWidget()
            try:
                if not isinstance(dialog, QFileDialog):
                    raise RuntimeError('The actual Export button did not open a Qt file picker')
                dialogs.append(dialog)
                dialog.accepted.connect(lambda: accepted.append(True))
                fill(dialog.findChild(QLineEdit, 'fileNameEdit'), exported)
                capture('05_actual_export_picker')
                click(dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Save))
            except Exception as error:
                errors.append(str(error))
                if isinstance(dialog, QFileDialog) and isValid(dialog):
                    dialog.reject()

        def stalled():
            errors.append('The actual export picker timed out')
            for dialog in dialogs:
                if isValid(dialog) and dialog.isVisible():
                    dialog.reject()

        opener.timeout.connect(handle)
        watchdog.timeout.connect(stalled)
        opener.start(400)
        watchdog.start(15000)
        try:
            click(panel._btn_export)
        finally:
            for timer in (opener, watchdog):
                timer.stop()
                timer.deleteLater()
        if errors or not accepted:
            raise RuntimeError('; '.join(errors) or 'The actual export picker was not accepted')
        idle()
        proof['current_export'] = verify_export(exported, records)
        capture('06_exported')
        # Positive restoration: lowering the minimum must return the 28 wells.
        fill(panel._min_count_box, '100')
        QTest.keyClick(panel._min_count_box, Qt.Key_Return)
        idle()
        inspect('count', 'restored_count')
        proof['restored_wells'] = len(panel._layout_df)
        proof['host_app_key'] = 'graph_builder'
        proof['app_key'] = 'plate_view'
        proof['new_features_not_claimed'] = ['Cross-module Local Data Filter']
        proof['old_file_picker_was_injected'] = True
        proof['current_export_picker_is_actual'] = True
        proof['accepted'] = True
    except Exception as error:
        proof['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if panel is not None:
            until = time.monotonic() + 30
            while (panel.is_busy() or panel.active_jobs()) and time.monotonic() < until:
                settle(.1)
            proof['workers'] = {'busy': panel.is_busy(), 'active_jobs': panel.active_jobs(),
                                'forcibly_stopped': False}
            if panel.is_busy() or panel.active_jobs():
                proof['accepted'] = False
        if original is not None:
            try:
                require_unchanged_source(SOURCE, original['source_bundle'])
                if _digest(original['database']) != SOURCE_SHA256:
                    raise ValueError('Private database bytes changed')
                proof['original_and_private_database_unchanged'] = True
            except Exception as error:
                proof['accepted'] = False
                proof['preservation_error'] = str(error)
        write_json(acceptance, proof)
        if proof['accepted'] is not True and 'error' not in proof:
            raise RuntimeError('The final Plate Viewer retention guards failed')
