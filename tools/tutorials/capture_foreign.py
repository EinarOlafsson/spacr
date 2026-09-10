"""Record actual Import Project decisions on exact, isolated microscopy copies.

Measure the neutral CSV from the same labels as the TIFFs, never an independently
versioned annotation database. Area is px²; physical calibration is unknown.
"""
from __future__ import annotations

import csv
import hashlib
import sqlite3
import tempfile
import time
from pathlib import Path

FIELDS = ('plate1_E01_1_1', 'plate1_L02_9_1')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure_labels(image, labels):
    """Return exact positive-label areas and means; zero is background."""
    import numpy as np
    if image.shape != labels.shape or image.ndim != 2:
        raise ValueError('Image and labels must have identical two-dimensional shapes')
    counts = np.bincount(labels.ravel())
    sums = np.bincount(labels.ravel(), weights=image.ravel())
    ids = np.flatnonzero(counts[1:]) + 1
    if not len(ids):
        raise ValueError('There are no labelled objects to import')
    return [(int(i), int(counts[i]), float(sums[i] / counts[i])) for i in ids]


def prepare_inputs(stage):
    import numpy as np
    import tifffile
    parent = stage / 'foreign_runs'
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='example-', dir=parent))
    images, masks = run / 'images', run / 'cell_masks'
    images.mkdir()
    masks.mkdir()
    records, rows = [], []
    for index, stem in enumerate(FIELDS, 1):
        source = stage / 'example_data/plate1/merged' / (stem + '.npy')
        merged = np.load(source, mmap_mode='r')
        if merged.ndim != 3 or merged.shape[-1] != 7 or merged.dtype != np.uint16:
            raise ValueError('Expected the downloaded seven-plane uint16 Measure example')
        image, labels = merged[..., 1], merged[..., 4]
        measured = measure_labels(image, labels)
        name = f'fov{index:02d}'
        target, mask = images / (name + '_C1.tif'), masks / (name + '_cell_mask.tif')
        tifffile.imwrite(target, image)
        tifffile.imwrite(mask, labels)
        if not np.array_equal(tifffile.imread(target), image) or not np.array_equal(tifffile.imread(mask), labels):
            raise RuntimeError('Neutral TIFF preparation changed original pixels')
        for label, area, mean in measured:
            rows.append({'ImageNumber': target.name, 'ObjectNumber': label,
                         'Area_px2': area, 'MeanIntensity_ER': mean, 'cell_area': area})
        records.append({'source': str(source), 'source_sha256': digest(source),
                        'image': str(target), 'image_sha256': digest(target),
                        'mask': str(mask), 'mask_sha256': digest(mask),
                        'neutral_stem': name, 'original_stem': stem,
                        'image_plane': 1, 'mask_plane': 4,
                        'objects': len(measured), 'shape': list(image.shape)})
    table = run / 'measurements.csv'
    with table.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return run, images, masks, table, records, rows


def compare_object_measurements(expected, actual):
    """Verify values by (source image, label), not just a bag of areas."""
    import math
    if len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError('Imported object identities do not match the prepared rows')
    for key, (area, mean) in expected.items():
        got_area, got_mean = actual[key]
        if area != got_area or not math.isclose(mean, got_mean, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError('Imported measurements changed their value or object assignment')


def verify_output(destination, records, rows):
    import numpy as np
    import tifffile
    with (destination / 'images/conversion_map.csv').open(newline='') as stream:
        conversion = list(csv.DictReader(stream))
    if len(conversion) != len(records):
        raise ValueError('Conversion map does not contain exactly the imported images')
    by_stem = {}
    for record in records:
        mapping = [m for m in conversion if m['source'] == record['image']]
        if len(mapping) != 1:
            raise ValueError('An input image has no unique conversion-map entry')
        mapping = mapping[0]
        stem = f"{mapping['plate']}_{mapping['well']}_{mapping['field']}"
        by_stem[stem] = Path(record['image']).name
        image, mask = tifffile.imread(record['image']), tifffile.imread(record['mask'])
        merged = np.load(destination / 'merged' / (stem + '.npy'), mmap_mode='r')
        if merged.shape != (*image.shape, 2) or not np.array_equal(merged[..., 0], image) or not np.array_equal(merged[..., 1], mask):
            raise ValueError('Imported merged pixels or labels differ from the actual source')
        if not np.array_equal(tifffile.imread(mapping['target_path']), image):
            raise ValueError('Converted intensity pixels differ from the actual source')
    expected = {(r['ImageNumber'], int(r['ObjectNumber'])):
                (float(r['Area_px2']), float(r['MeanIntensity_ER'])) for r in rows}
    with sqlite3.connect((destination / 'measurements/measurements.db').as_uri() + '?mode=ro', uri=True) as conn:
        loaded = conn.execute('SELECT file_name,object_label,foreign_reviewed_area_px2,foreign_meanintensity_er FROM foreign_cell').fetchall()
    actual = {(by_stem.get(stem, stem), int(label)): (area, mean)
              for stem, label, area, mean in loaded}
    if len(actual) != len(loaded):
        raise ValueError('Imported measurement identities are duplicated')
    compare_object_measurements(expected, actual)
    return {'per_object_areas_and_means_preserved': True,
            'converted_and_merged_pixels_and_labels_preserved': True,
            'new_well_ids_are_not_original_acquisition_wells': True}


def record_foreign(app, window, screen, stage, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QFileDialog, QLineEdit, QDialogButtonBox
    from spacr.qt.widgets.fold_strip import FoldButton
    from spacr.qt.screens.foreign import FOLDED_APPS

    run, images, masks, table, records, rows = prepare_inputs(stage)
    destination, mapping = run / 'imported_project', run / 'reviewed_mapping.csv'
    write_json(captures / 'inputs.json', {
        'run': str(run), 'records': records, 'rows': len(rows),
        'measurement_source': 'Exact areas and means recomputed from copied image/label planes',
        'area_units': 'px^2', 'intensity_units': 'raw image counts',
        'physical_calibration': None,
        'neutral_names_do_not_preserve_wells_without_provenance': True})

    def fill(widget, value):
        widget.setFocus()
        QTest.keyClick(widget, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(widget, str(value))
        QTest.keyClick(widget, Qt.Key_Tab)
        settle(0.1)

    def click(button):
        if not button.isVisible() or not button.isEnabled():
            raise RuntimeError(f'Control is not usable: {button.text()}')
        QTest.mouseClick(button, Qt.LeftButton)
        settle(0.3)

    def wait_job(name):
        deadline = time.monotonic() + timeout
        while screen._busy:
            if time.monotonic() >= deadline:
                raise TimeoutError(f'{name}: {screen.status_text()}')
            settle(0.1)
        settle(0.4)

    def show_report_end():
        scrollbar = screen._report.verticalScrollBar()
        QTest.keyClick(scrollbar, Qt.Key_End)
        settle()
        if scrollbar.value() != scrollbar.maximum():
            raise RuntimeError('The actual report scrollbar did not reach the final lines')

    def choose(button, path, frame, save=False):
        errors, accepted = [], []
        def handle():
            dialog = app.activeModalWidget()
            try:
                if not isinstance(dialog, QFileDialog):
                    raise RuntimeError('Expected the real Qt file picker')
                dialog.accepted.connect(lambda: accepted.append(True))
                dialog.resize(1400, 950)
                fill(dialog.findChild(QLineEdit, 'fileNameEdit'), path)
                capture(frame)
                box = dialog.findChild(QDialogButtonBox)
                QTest.mouseClick(box.button(QDialogButtonBox.Save if save else QDialogButtonBox.Open), Qt.LeftButton)
            except Exception as exc:
                errors.append(str(exc))
                if dialog is not None:
                    dialog.reject()
        def reject_stalled():
            dialog = app.activeModalWidget()
            if dialog is not None and not accepted:
                errors.append('File picker never accepted the requested path')
                dialog.reject()
        QTimer.singleShot(500, handle)
        QTimer.singleShot(12000, reject_stalled)
        click(button)
        if errors or not accepted:
            raise RuntimeError('; '.join(errors) or 'File selection was cancelled')

    def edit_target(target):
        row = screen._model.row_of('Area_px2')
        if row < 0:
            raise RuntimeError('The pixel-area measurement has no mapping row')
        index = screen._model.index(row, 1)
        if screen._table.model() is not screen._model:
            index = screen._table.model().mapFromSource(index)
        screen._table.scrollTo(index)
        settle()
        QTest.mouseClick(screen._table.viewport(), Qt.LeftButton,
                         pos=screen._table.visualRect(index).center())
        QTest.keyClick(screen._table, Qt.Key_F2)
        settle()
        editors = [e for e in screen._table.findChildren(QLineEdit) if e.isVisible()]
        if len(editors) != 1:
            write_json(captures / 'mapping_editor_problem.json', {
                'source_row': row, 'view_row': index.row(), 'view_column': index.column(),
                'current_row': screen._table.currentIndex().row(),
                'current_column': screen._table.currentIndex().column(),
                'edit_triggers': str(screen._table.editTriggers()),
                'flags': str(index.flags()),
                'focused_widget': type(app.focusWidget()).__name__,
                'line_edits': len(editors)})
            capture('mapping_editor_problem')
            raise RuntimeError('Selecting the target and pressing F2 did not open the mapping editor')
        fill(editors[0], target)

    for key in FOLDED_APPS:
        buttons = [b for b in screen.findChildren(FoldButton) if b.app_key == key and b.isVisible()]
        if len(buttons) != 1:
            raise RuntimeError(f'Expected one visible Import fold: {key}')
        QTest.mouseMove(buttons[0])
        settle(0.8)
        capture('02_fold_' + key)
    choose(screen._btn_pick_images, images, '03_choose_images')
    fill(screen._dst_edit, destination)
    choose(screen._btn_pick_mask, masks, '04_choose_masks')
    if screen._object_box.currentData() != 'cell':
        raise RuntimeError('The visible mask class is not Cell')
    click(screen._btn_add_mask)
    choose(screen._btn_pick_table, table, '05_choose_table')
    fill(screen._scale_edit, '')
    capture('06_inputs_unknown_calibration')
    if screen.on_conflict() != 'refuse':
        raise RuntimeError('Expected the default refusal policy for name collisions')
    click(screen._btn_preview)
    wait_job('Preview with conservative inferred names')
    if screen.plan() is None or not screen.plan().ok or destination.exists():
        raise RuntimeError('The real inferred proposal is not a non-writing valid plan')
    capture('06b_safe_inferred_names')
    # Inference already prefixes foreign columns. Demonstrate a REAL refusal
    # by explicitly editing one target to a reserved spaCR measurement name.
    edit_target('cell_area')
    initial = screen.plan()
    write_json(captures / 'preview_refused.json', {'report': screen.report_text(),
               'status': screen.status_text(), 'destination_exists': destination.exists()})
    if initial is None or initial.ok or screen.can_import() or destination.exists():
        raise RuntimeError('The real name conflict was not refused before writing')
    capture('07_conflict_refused')
    screen._conflict_box.setFocus()
    QTest.keyClick(screen._conflict_box, Qt.Key_Home)
    for _ in range(screen._conflict_box.findData('rename')):
        QTest.keyClick(screen._conflict_box, Qt.Key_Down)
    QTest.keyClick(screen._conflict_box, Qt.Key_Tab)
    settle()
    plan = screen.plan()
    if plan is None or not plan.ok or destination.exists():
        raise RuntimeError('The reviewed Rename policy did not yield a non-writing valid plan')
    if plan.join.rows_total != len(rows) or plan.join.rows_matched != len(rows):
        raise RuntimeError('Not every actual measured object matched its copied label')
    if plan.join.n_objects_unmeasured:
        raise RuntimeError('The demonstration unexpectedly omits labelled objects')
    capture('08_reviewed_joins')
    show_report_end()
    capture('08b_join_report')
    edit_target('foreign_reviewed_area_px2')
    if screen.plan().target_for('Area_px2') != 'foreign_reviewed_area_px2':
        raise RuntimeError('The visible mapping edit did not update the actual plan')
    capture('09_mapping_edited')
    choose(screen._btn_save_map, mapping, '10_save_mapping', save=True)
    if not mapping.is_file():
        raise RuntimeError('The accepted Save mapping did not write a file')
    saved_hash = digest(mapping)
    edit_target('foreign_unsaved_area_px2')
    if screen.plan().target_for('Area_px2') != 'foreign_unsaved_area_px2':
        raise RuntimeError('The unsaved mapping change was not actually applied')
    capture('10b_unsaved_mapping_change')
    choose(screen._btn_load_map, mapping, '11_load_mapping')
    if screen.plan().target_for('Area_px2') != 'foreign_reviewed_area_px2' or digest(mapping) != saved_hash:
        raise RuntimeError('Mapping reload changed the reviewed file or target')
    capture('12_mapping_reloaded')
    click(screen._btn_import)
    wait_job('Import')
    result = screen.result()
    write_json(captures / 'import_report.json', {'report': screen.report_text(),
               'status': screen.status_text(), 'destination': str(destination)})
    if result is None or not result.is_complete or result.n_fields != 2:
        raise RuntimeError('The actual two-field import did not complete')
    capture('13_imported')
    show_report_end()
    capture('14_import_scope')
    db = Path(result.db_path)
    with sqlite3.connect(db.as_uri() + '?mode=ro', uri=True) as conn:
        names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        imported = conn.execute('SELECT COUNT(*) FROM foreign_cell').fetchone()[0]
        values = sorted(conn.execute('SELECT foreign_reviewed_area_px2 FROM foreign_cell').fetchall())
    if imported != len(rows) or values != sorted((r['Area_px2'],) for r in rows):
        raise RuntimeError('Imported pixel-area values or row count changed')
    output_checks = verify_output(destination, records, rows)
    for entry in records:
        for key in ('source', 'image', 'mask'):
            if digest(Path(entry[key])) != entry[key + '_sha256']:
                raise RuntimeError(f'Import changed an original input: {key}')
    write_json(captures / 'scientific_acceptance.json', {
        'accepted': True, 'fields': result.n_fields, 'measurement_rows': imported,
        'all_rows_matched': True, 'unmeasured_objects': 0, 'physical_calibration': None,
        'area_values_preserved': True, 'source_inputs_unchanged': True,
        'destination': str(destination), 'tables': names, 'mapping_sha256': saved_hash,
        'folds_shown': list(FOLDED_APPS), 'published': False, **output_checks})
    print(f'Accepted real Import Project: {imported} measured labels in two fields', flush=True)
