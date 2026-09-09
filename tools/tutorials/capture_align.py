"""Record a real stitch of the preserved, explicitly emulated tile example.

The nine old tutorial tiles are exact crops from one acquired microscopy field.
They are NOT nine independently acquired images or validation of the OPS fold.
Only the application's visible controls create the new plan and stitched output.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import time


def record_align(app, window, screen, stage, captures, capture, settle, write_json, timeout):
    import numpy as np
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QFileDialog, QDialogButtonBox, QLineEdit
    from spacr.align import read_coordinates

    write_json(captures / 'scientific_acceptance.json', {
        'accepted': False, 'reason': 'The actual tile stitch is not yet checked', 'published': False,
    })
    authoring = stage.parent
    manifest = json.loads((authoring / 'derived/align_stitch/source_manifest.json').read_text())
    source_path = authoring.parent / 'test_datasets/spacr/tutorials/orig/test/stack/test_E11_2_1.npy'
    source = np.load(source_path, mmap_mode='r')
    if source.shape != (2000, 2000, 4) or source.dtype != np.uint16:
        raise RuntimeError('The preserved acquired field has changed its shape or dtype')
    tiles = authoring / 'derived/align_stitch/tiles'
    originals = {str(source_path): hashlib.sha256(source_path.read_bytes()).hexdigest()}
    records = []
    for entry in manifest['tiles']:
        path = tiles / Path(entry['path']).name
        data = np.load(path, mmap_mode='r')
        y, x = entry['source_yx']
        if data.shape != (800, 800, 4) or data.dtype != np.uint16 or not np.array_equal(
                data, source[y:y + 800, x:x + 800, :]):
            raise RuntimeError('A preserved tutorial tile is not its exact acquired source crop')
        originals[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append({'path': str(path), 'field': entry['field'], 'source_yx': [y, x]})
    if len(records) != 9 or len(list(tiles.glob('*.npy'))) != 9:
        raise RuntimeError('The known 3 by 3 example requires exactly nine original tiles')
    root = stage / 'align_runs'
    root.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='example-', dir=root))
    destination = work / 'stacks'
    database = work / 'measurements.db'
    write_json(captures / 'input_manifest.json', {
        'original_hashes': originals, 'tiles': records, 'output_directory': str(work),
        'disclosure': 'Nine exact crops from ONE real field emulate a tiled acquisition; not nine acquired fields',
        'new_test_data_download_button': False, 'ops_validation': False,
    })

    def fill(edit, value):
        QTest.mouseClick(edit, Qt.LeftButton)
        QTest.keyClick(edit, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(edit, str(value))
        QTest.keyClick(edit, Qt.Key_Tab)
        settle(.1)

    def choose(box, text):
        index = box.findText(text)
        if index < 0:
            raise RuntimeError('The actual settings lack ' + text)
        box.setFocus()
        QTest.keyClick(box, Qt.Key_Home)
        for _ in range(index):
            QTest.keyClick(box, Qt.Key_Down)
        QTest.keyClick(box, Qt.Key_Tab)

    accepted, errors = [], []
    def select_source():
        dialog = app.activeModalWidget()
        try:
            if not isinstance(dialog, QFileDialog):
                raise RuntimeError('The real Choose tile folder dialog did not open')
            dialog.accepted.connect(lambda: accepted.append(True))
            watchdog = QTimer(dialog)
            watchdog.setSingleShot(True)
            watchdog.timeout.connect(dialog.reject)
            watchdog.start(12000)
            fill(dialog.findChild(QLineEdit, 'fileNameEdit'), tiles)
            capture('02_preserved_real_pixel_tiles')
            QTest.mouseClick(dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Open), Qt.LeftButton)
        except Exception as exc:
            errors.append(str(exc))
            if dialog is not None:
                dialog.reject()
    QTimer.singleShot(400, select_source)
    QTest.mouseClick(screen._btn_pick_src, Qt.LeftButton)
    if errors or not accepted or Path(screen._src_edit.text()) != tiles:
        raise RuntimeError('; '.join(errors) or 'The actual folder selection was not accepted')
    fill(screen._rows_box.lineEdit(), 3)
    fill(screen._cols_box.lineEdit(), 3)
    fill(screen._overlap_box.lineEdit(), .25)
    choose(screen._order_combo, 'row-major')
    fill(screen._ref_box.lineEdit(), 1)
    fill(screen._conf_box.lineEdit(), .30)
    fill(screen._radius_box.lineEdit(), 1)
    choose(screen._blend_combo, 'feather')
    fill(screen._budget_box.lineEdit(), 64)
    fill(screen._dst_edit, destination)
    fill(screen._db_edit, database)
    if screen._overwrite_box.isChecked():
        QTest.mouseClick(screen._overwrite_box, Qt.LeftButton)
    settings = screen.settings()
    expected = {'grid': (3, 3), 'overlap': .25, 'reference_channel': 1,
                'min_confidence': .30, 'neighbour_radius': 1, 'max_buffer_bytes': 64 << 20,
                'blend': 'feather', 'order': 'row-major', 'overwrite': False}
    if any(settings[key] != value for key, value in expected.items()):
        raise RuntimeError('The recorded controls did not reach the expected real settings')
    if screen._btn_write.isEnabled():
        raise RuntimeError('Writing should require an actual computed plan')
    capture('03_actual_layout_and_budget')

    def wait():
        deadline = time.monotonic() + timeout
        while screen.is_busy() or screen.active_jobs():
            if time.monotonic() >= deadline:
                raise TimeoutError('The real alignment job did not finish')
            settle(.1)
        settle(.5)
        if screen.last_error:
            raise RuntimeError(screen.last_error)
    QTest.mouseClick(screen._btn_plan, Qt.LeftButton)
    wait()
    plan = screen.plan()
    if plan is None or len(plan.placements) != 9 or plan.n_registered != 9 or plan.n_nominal or plan.unplaced:
        raise RuntimeError('The known-geometry example did not register all nine actual tiles')
    if not screen._btn_write.isEnabled() or destination.exists() or database.exists():
        raise RuntimeError('Plan must enable Write without already creating output')
    capture('04_registered_plan_not_yet_written')
    expected_positions = {Path(r['path']).name: r['source_yx'] for r in records}
    for placement in plan.placements:
        y, x = expected_positions[Path(placement.tile.path).name]
        if abs(placement.y - plan.origin[0] - y) > .01 or abs(placement.x - plan.origin[1] - x) > .01:
            raise RuntimeError('The actual solved geometry differs from the known crop origins')
    index, rectangle = screen._layout_view.tile_rects()[4]
    QTest.mouseClick(screen._layout_view, Qt.LeftButton, pos=rectangle.center().toPoint())
    settle(.4)
    clicked = next(p for p in plan.placements if p.tile.index == index)
    if Path(clicked.tile.path).name not in screen._tile_label.text():
        raise RuntimeError('The actual tile selection did not display its measured details')
    capture('05_actual_tile_detail')
    QTest.mouseClick(screen._btn_write, Qt.LeftButton)
    wait()
    result = screen.result()
    if result is None or result.n_written != 9 or result.n_skipped:
        raise RuntimeError('The stitch did not write all nine input tiles')
    stack = Path(result.stack_path)
    output = np.load(stack, mmap_mode='r')
    if output.shape != source.shape or output.dtype != source.dtype:
        raise RuntimeError('The actual mosaic changed the known image shape or dtype')
    # Integer feathering can round by one intensity unit. Compare every pixel
    # and channel, in bounded rows, not merely the output dimensions or mean.
    maximum_error = 0
    for y in range(0, 2000, 100):
        delta = np.abs(output[y:y + 100].astype(np.int32) - source[y:y + 100].astype(np.int32))
        maximum_error = max(maximum_error, int(delta.max()))
    if maximum_error > 1:
        raise RuntimeError(f'The actual mosaic differs from the known source by {maximum_error} intensity units')
    coordinates = read_coordinates(database)
    if len(coordinates) != 9 or result.peak_buffer_bytes > 64 << 20:
        raise RuntimeError('Coordinate count or measured band-buffer budget is wrong')
    from align_evidence import check_coordinate_rows
    check_coordinate_rows(records, coordinates.to_dict('records'), stack)
    for path, digest in originals.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise RuntimeError('The stitch modified an original source or preserved tile')
    capture('06_real_stitch_complete')
    write_json(captures / 'stitch_evidence.json', {
        'settings': settings, 'registered': plan.n_registered, 'nominal': plan.n_nominal,
        'max_residual_pixels': plan.max_residual, 'all_crop_origins_match_within_pixels': .01,
        'stack_path': str(stack), 'stack_sha256': hashlib.sha256(stack.read_bytes()).hexdigest(),
        'stack_shape': list(output.shape), 'dtype': str(output.dtype),
        'all_pixels_and_channels_maximum_absolute_error': maximum_error,
        'coordinate_rows': len(coordinates), 'coordinate_records': coordinates.to_dict('records'),
        'coordinate_source_identity_geometry_and_output_verified': True,
        'peak_band_buffer_bytes': result.peak_buffer_bytes, 'band_rows': result.band_rows,
        'total_process_ram_is_not_the_band_buffer': True, 'original_hashes_unchanged': True,
        'plan_report': screen._report_view.toPlainText(), 'published': False,
    })
    write_json(captures / 'scientific_acceptance.json', {
        'accepted': True, 'reason': 'Real GUI stitch recovered known crop geometry and every source pixel within integer rounding',
        'emulated_tiles_not_independent_acquisitions': True, 'ops_validation': False, 'published': False,
    })
    print(f'Actual stitch: nine tiles, {output.shape}, maximum pixel error {maximum_error}', flush=True)
