"""Record Format Converter on the two preserved real Foreign-lesson TIFFs.

The neutral fov names acquire demonstration coordinates, not original wells.
These are single-channel 2-D planes, so this recording does not validate vendor
readers, multichannel stacks, Z projection or time-series conversion. Resume's
header checks are distinguished from this recorder's independent pixel checks.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import time


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_converter_run_status(records, *, operation, planned_targets,
                               verified_outputs, planned_fields, resumed_fields,
                               n_sources, n_written, n_existing,
                               previous_records=()):
    """Validate the newest ledger entry against this independently checked run.

    A complete first conversion attempts every source and writes every target.
    A full checkpoint resume attempts nothing, so its legitimate ledger status
    is ``empty``. That is accepted only with exact target coverage, successful
    independent pixel checks, and every planned field demonstrably reused.
    Prior entries must be preserved with exactly one new, distinct run appended.
    """
    targets = tuple(planned_targets)
    fields = tuple(planned_fields)
    resumed = tuple(resumed_fields)
    if (not targets or len(set(targets)) != len(targets)
            or not fields or len(set(fields)) != len(fields)):
        raise ValueError('The conversion must have unique planned outputs and fields')
    if (set(verified_outputs) != set(targets)
            or any(proof.get('all_pixels_exact') is not True
                   for proof in verified_outputs.values())):
        raise ValueError('Every planned output must be independently verified')
    if (not isinstance(records, list) or len(records) != len(previous_records) + 1
            or records[:-1] != list(previous_records)
            or not isinstance(records[-1], dict)):
        raise ValueError('The ledger must preserve its history and append exactly one run')
    last = records[-1]
    run_id = last.get('run_id')
    if (not isinstance(run_id, str) or not run_id
            or any(entry.get('run_id') == run_id for entry in previous_records)):
        raise ValueError('The latest operation needs a distinct run identity')
    if last.get('name') != 'convert_to_yokogawa_plan':
        raise ValueError('The latest ledger entry is not the converter operation')
    counts = (n_sources, n_written, n_existing,
              last.get('n_attempted'), last.get('n_succeeded'), last.get('n_failed'))
    if any(type(value) is not int or value < 0 for value in counts) or n_sources == 0:
        raise ValueError('Conversion counts must be exact nonnegative integers')
    if last['n_failed'] != 0 or last.get('failures') != []:
        raise ValueError('The latest converter operation recorded a failure')
    if operation == 'convert':
        if (n_written != len(targets) or n_existing != 0 or resumed
                or last.get('status') != 'complete'
                or last['n_attempted'] != n_sources or last['n_succeeded'] != n_sources
                or last.get('success_by_stage') != {'convert': n_sources}):
            raise ValueError('The latest operation did not complete all new conversions')
    elif operation == 'resume':
        if (n_written != 0 or n_existing != len(targets)
                or len(resumed) != len(fields) or set(resumed) != set(fields)
                or last.get('status') != 'empty'
                or last['n_attempted'] != 0 or last['n_succeeded'] != 0
                or last.get('success_by_stage') != {}):
            raise ValueError('An empty ledger is valid only for verified reuse of every planned output')
    else:
        raise ValueError(f'Unknown expected converter operation: {operation}')
    return {'operation': operation, 'run_id': run_id, 'ledger_status': last['status'],
            'ledger_entries': len(records), 'independently_verified_outputs': len(targets)}


def record_converter(app, window, screen, stage, captures, capture, settle,
                     write_json, timeout):
    """Use real pickers, Preview, Convert and Resume; fail closed on mismatch."""
    import numpy as np
    import tifffile
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QFileDialog, QDialogButtonBox, QLineEdit
    from spacr.qt.screens.convert import ConvertScreen
    from spacr.qt.widgets.fold_strip import FoldButton

    # Reach the module through its current parent, not a hidden Home route.
    buttons = [button for button in screen.findChildren(FoldButton)
               if button.app_key == 'convert' and button.isVisible()]
    if len(buttons) != 1 or not buttons[0].isEnabled():
        raise RuntimeError('Import must expose one usable Format Converter fold')
    QTest.mouseClick(buttons[0], Qt.LeftButton)
    settle(2)
    children = [child for child in window.findChildren(ConvertScreen) if child.isVisible()]
    if len(children) != 1:
        raise RuntimeError('The actual Format Converter fold did not open')
    screen = children[0]
    capture('01b_current_converter_fold')

    stage, captures = Path(stage), Path(captures)
    write_json(captures / 'scientific_acceptance.json', {
        'accepted': False, 'reason': 'The actual conversion has not been verified',
        'published': False,
    })
    foreign_capture = stage / 'captures/foreign_release_v2'
    foreign_acceptance = foreign_capture / 'scientific_acceptance.json'
    if not json.loads(foreign_acceptance.read_text())['accepted']:
        raise RuntimeError('The real Foreign input capture was not accepted')
    # The completed capture currently calls this inputs.json. Accept the
    # explicit manifest name too, but never generate replacement inputs.
    input_manifest = foreign_capture / 'input_manifest.json'
    if not input_manifest.is_file():
        input_manifest = foreign_capture / 'inputs.json'
    prior = json.loads(input_manifest.read_text())
    records = prior['records']
    if (len(records) != 2 or {r['neutral_stem'] for r in records} != {'fov01', 'fov02'}
            or len({Path(r['image']).parent for r in records}) != 1):
        raise RuntimeError('Expected exactly the two preserved Foreign TIFF inputs')
    images = Path(records[0]['image']).parent
    if {p.name for p in images.iterdir()} != {Path(r['image']).name for r in records}:
        raise RuntimeError('The source directory contains unexpected conversion inputs')
    originals = {str(input_manifest): _digest(input_manifest),
                 str(foreign_acceptance): _digest(foreign_acceptance)}
    expected = {}
    for record in sorted(records, key=lambda r: r['neutral_stem']):
        source, image = Path(record['source']), Path(record['image'])
        for key in ('source', 'image'):
            path = Path(record[key])
            actual_hash = _digest(path)
            if actual_hash != record[key + '_sha256']:
                raise RuntimeError(f'The accepted Foreign input has changed: {path}')
            originals[str(path)] = actual_hash
        acquired = np.load(source, mmap_mode='r')
        plane = tifffile.imread(image)
        if (record['image_plane'] != 1 or image.name != record['neutral_stem'] + '_C1.tif'
                or acquired.ndim != 3 or acquired.shape[-1] != 7
                or plane.ndim != 2 or plane.dtype != np.uint16
                or acquired.dtype != plane.dtype or list(plane.shape) != record['shape']
                or plane.shape != acquired.shape[:2]):
            raise RuntimeError('The preserved TIFF is not the recorded single image plane')
        for y in range(0, plane.shape[0], 128):
            if not np.array_equal(plane[y:y + 128], acquired[y:y + 128, :, 1]):
                raise RuntimeError('The preserved TIFF differs from its acquired source plane')
        index = int(record['neutral_stem'][3:])
        expected[str(image)] = {
            'source': str(image), 'source_relpath': image.name,
            'target': f'plate1_A01_T0001F{index:03d}L01A01Z01C01.tif',
            'plate': 'plate1', 'well': 'A01', 'field': index,
            'channel': 1, 'z': 1, 't': 1,
            'source_plate': images.name, 'source_well': 'A01',
            'source_field': record['neutral_stem'], 'source_channel': 'C1',
            'source_z': '1', 'source_t': '1', 'z_handling': 'keep',
            'n_z_planes': 1, 'n_timepoints': 1,
            'plateID': 'plate1', 'rowID': 'r1', 'columnID': 'c1',
            'fieldID': f'f{index}', 'prc': 'plate1_r1_c1',
            'prcf': f'plate1_r1_c1_f{index}',
        }
        del plane, acquired

    runs = stage / 'converter_runs'
    runs.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='example-', dir=runs))
    destination = work / 'converted'
    destination.mkdir()  # A new empty directory selectable by the real picker.
    write_json(captures / 'input_manifest.json', {
        'reused_foreign_manifest': str(input_manifest), 'records': records,
        'original_hashes': originals, 'destination': str(destination),
        'acquired_image_plane_zero_based': 1, 'source_tiff_plane_zero_based': 0,
        'assigned_output_channel_one_based': 1,
        'neutral_names_are_not_original_acquisition_coordinates': True,
        'new_download': False, 'new_artificial_images': False,
        'z_projection_or_time_series_demonstrated': False,
    })

    def fill(edit, value):
        if edit is None:
            raise RuntimeError('The actual dialog has no filename editor')
        edit.setFocus()
        QTest.keyClick(edit, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(edit, str(value))
        QTest.keyClick(edit, Qt.Key_Tab)
        settle(.1)

    def click(button):
        if not button.isVisible() or not button.isEnabled():
            raise RuntimeError(f'The actual control is not usable: {button.text()}')
        QTest.mouseClick(button, Qt.LeftButton)
        settle(.2)

    def choose_directory(button, path, frame):
        accepted, errors = [], []

        def handle():
            dialog = app.activeModalWidget()
            try:
                if not isinstance(dialog, QFileDialog):
                    raise RuntimeError('The actual directory picker did not open')
                dialog.accepted.connect(lambda: accepted.append(True))
                watchdog = QTimer(dialog)
                watchdog.setSingleShot(True)
                watchdog.timeout.connect(dialog.reject)
                watchdog.start(12000)
                dialog.resize(1400, 950)
                fill(dialog.findChild(QLineEdit, 'fileNameEdit'), path)
                capture(frame)
                buttons = dialog.findChild(QDialogButtonBox)
                QTest.mouseClick(buttons.button(QDialogButtonBox.Open), Qt.LeftButton)
            except Exception as exc:
                errors.append(str(exc))
                if dialog is not None:
                    dialog.reject()

        QTimer.singleShot(400, handle)
        click(button)
        if errors or not accepted:
            raise RuntimeError('; '.join(errors) or 'The real directory picker was cancelled')

    def choose_option(box, value):
        index = box.findData(value)
        if index < 0:
            raise RuntimeError(f'The actual converter lacks the option {value}')
        box.setFocus()
        QTest.keyClick(box, Qt.Key_Home)
        for _ in range(index):
            QTest.keyClick(box, Qt.Key_Down)
        QTest.keyClick(box, Qt.Key_Tab)
        settle(.1)
        if box.currentData() != value:
            raise RuntimeError('The visible option did not reach the requested value')

    def wait_job(label):
        deadline = time.monotonic() + timeout
        while screen.is_busy() or screen.active_jobs():
            if time.monotonic() >= deadline:
                raise TimeoutError(f'{label}: {screen.status_text()}')
            settle(.1)
        settle(.4)
        if screen.last_error:
            raise RuntimeError(screen.last_error)

    choose_directory(screen._btn_pick_src, images, '02_real_source_folder')
    choose_directory(screen._btn_pick_dst, destination, '03_new_private_destination')
    if Path(screen.source_path()) != images or Path(screen.destination_path()) != destination:
        raise RuntimeError('The accepted folder choices do not match the intended paths')
    choose_option(screen._layout_box, 'flat')
    choose_option(screen._z_box, 'keep')
    choose_option(screen._plate_box, 'index')
    if screen.resume_enabled():
        click(screen._resume)
    if screen.resume_enabled() or screen.plan() is not None or any(destination.iterdir()):
        raise RuntimeError('The first run requires Resume off and an empty destination')
    capture('04_explicit_flat_single_plane_settings')
    click(screen._btn_preview)
    wait_job('Read-only Preview')
    plan = screen.plan()
    if (plan is None or not plan.ok or plan.unreadable or plan.n_sources != 2
            or len(plan.mappings) != 2 or screen.preview_row_count() != 2
            or not screen.can_convert() or any(destination.iterdir())):
        raise RuntimeError('Preview did not produce exactly two non-writing mappings')
    if {m.source for m in plan.mappings} != set(expected):
        raise RuntimeError('Preview assigned a different set of source identities')
    for mapping in plan.mappings:
        if mapping.plane != (0, 0, 0):
            raise RuntimeError('A single-plane input was assigned another array plane')
        row = mapping.to_row(str(destination), 'planned')
        if any(str(row.get(key)) != str(value)
               for key, value in expected[mapping.source].items()):
            raise RuntimeError('Preview changed a field, channel or source identity')
    if screen._model.flags(screen._model.index(0, 0)) & Qt.ItemIsEditable:
        raise RuntimeError('The preview table unexpectedly allows source mapping edits')
    capture('05_preview_no_output_written')
    scrollbar = screen._table.horizontalScrollBar()
    QTest.keyClick(scrollbar, Qt.Key_End)
    settle(.2)
    capture('06_preview_source_channel_and_plane_provenance')
    QTest.keyClick(scrollbar, Qt.Key_Home)
    write_json(captures / 'preview_evidence.json', {
        'summary': screen.summary_text(), 'status': screen.status_text(),
        'rows': [m.to_row(str(destination), 'planned') for m in plan.mappings],
        'destination_files': [], 'preview_table_read_only': True,
        'layout': screen.layout_mode(), 'z_handling': screen.z_handling(),
        'plate_naming': screen.plate_naming(), 'resume': screen.resume_enabled(),
    })

    def verify_outputs(result, status, previous_records=()):
        if result is None or not result.is_complete or result.failed or result.skipped:
            raise RuntimeError('The real conversion did not complete')
        map_path = Path(result.map_path)
        if map_path != destination / 'conversion_map.csv':
            raise RuntimeError('The generated mapping CSV is outside the new destination')
        with map_path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 2 or {r['source'] for r in rows} != set(expected):
            raise RuntimeError('The mapping CSV does not identify both actual inputs exactly once')
        outputs = {}
        for row in rows:
            if any(row.get(key) != str(value) for key, value in expected[row['source']].items()):
                raise RuntimeError('The written map changed a source or output identity')
            target = destination / row['target']
            if row['target_path'] != str(target) or row['status'] != status:
                raise RuntimeError('The written map has the wrong destination or status')
            metadata = json.loads(row['meta_json'])
            if metadata.get('reader') != 'tifffile' or metadata.get('axes') != 'YX':
                raise RuntimeError('The mapping does not retain the actual 2-D TIFF provenance')
            source_pixels, output_pixels = tifffile.imread(row['source']), tifffile.imread(target)
            if (source_pixels.dtype != output_pixels.dtype
                    or source_pixels.shape != output_pixels.shape
                    or not np.array_equal(source_pixels, output_pixels)):
                raise RuntimeError('Conversion changed source pixels, dtype or shape')
            outputs[str(target)] = {'sha256': _digest(target),
                                    'shape': list(output_pixels.shape),
                                    'dtype': str(output_pixels.dtype),
                                    'all_pixels_exact': True}
            del source_pixels, output_pixels
        if {p.name for p in destination.glob('*.tif')} != {r['target'] for r in rows}:
            raise RuntimeError('Unexpected or missing output TIFFs')
        run_status_path = map_path.with_suffix('.run_status.json')
        run_status = json.loads(run_status_path.read_text())
        ledger_check = check_converter_run_status(
            run_status, operation='convert' if status == 'converted' else 'resume',
            planned_targets=[str(destination / row['target']) for row in expected.values()],
            verified_outputs=outputs,
            planned_fields=[f"{row['plate']}/{row['well']}/f{row['field']:04d}"
                            for row in expected.values()],
            resumed_fields=result.resumed_fields, n_sources=plan.n_sources,
            n_written=result.n_written, n_existing=len(result.existing),
            previous_records=previous_records)
        checkpoint = Path(result.checkpoint_path)
        if checkpoint.parent != destination or not checkpoint.is_file():
            raise RuntimeError('No actual field checkpoint was written in the destination')
        return {'mapping_rows': rows, 'mapping_sha256': _digest(map_path),
                'outputs': outputs, 'run_status': run_status, 'ledger_check': ledger_check,
                'checkpoint_path': str(checkpoint), 'checkpoint_sha256': _digest(checkpoint),
                'n_written': result.n_written, 'n_existing': len(result.existing),
                'resumed_fields': list(result.resumed_fields)}

    click(screen._btn_convert)
    wait_job('Convert')
    result = screen.result()
    if result is None or result.n_written != 2 or result.existing:
        raise RuntimeError('The first conversion did not write exactly two new TIFFs')
    first = verify_outputs(result, 'converted')
    capture('07_actual_conversion_complete')
    first['summary'], first['status'] = screen.summary_text(), screen.status_text()
    write_json(captures / 'conversion_evidence.json', first)

    click(screen._resume)
    if not screen.resume_enabled():
        raise RuntimeError('The real Resume toggle did not enable checkpoint reuse')
    capture('08_resume_enabled')
    click(screen._btn_convert)
    wait_job('Resume')
    resumed = screen.result()
    if (resumed is None or resumed is result or resumed.n_written != 0
            or len(resumed.existing) != 2 or len(resumed.resumed_fields) != 2):
        raise RuntimeError('Resume did not reuse the two complete fields')
    second = verify_outputs(resumed, 'existing', previous_records=first['run_status'])
    if second['outputs'] != first['outputs']:
        raise RuntimeError('Resume modified the independently verified TIFFs')
    second.update(summary=screen.summary_text(), status=screen.status_text(),
                  application_resume_checks='Readable TIFF metadata/pages, not all pixel values',
                  independent_recorder_check='Every output pixel, dtype and source identity compared')
    capture('09_completed_fields_reused')
    write_json(captures / 'resume_evidence.json', second)
    for path, digest in originals.items():
        if _digest(path) != digest:
            raise RuntimeError('Conversion modified an original image or provenance file')
    write_json(captures / 'scientific_acceptance.json', {
        'accepted': True, 'published': False, 'destination': str(destination),
        'source_fields': 2, 'output_tiffs': 2, 'all_pixels_and_dtypes_exact': True,
        'mapping_source_field_channel_and_plane_identities_verified': True,
        'preview_wrote_no_outputs': True, 'original_hashes_unchanged': True,
        'resume_reused_fields': 2, 'resume_application_checks_full_pixels': False,
        'neutral_names_are_not_original_acquisition_coordinates': True,
        'original_image_plane_zero_based': 1, 'output_channel_one_based': 1,
        'z_projection_or_time_series_demonstrated': False,
        'vendor_reader_validation': False, 'new_download': False,
    })
    print('Actual Format Converter: two real TIFFs, exact pixels and identities; Resume reused both fields',
          flush=True)
