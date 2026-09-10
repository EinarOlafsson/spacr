"""Record the current Model Zoo using real, provenance-checked image crops.

This helper does not launch an app, select an accelerator, inject inference,
download weights or train. The caller owns the host-RAM guard, two-thread
environment and outer process timeout. Returned masks are workflow evidence,
not ground truth, and the tutorial crops are not held-out validation data.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import tempfile
import time


AUTHOR = Path('/mnt/firecuda2/Claude/toxoplasma_projects/tutorials')
SOURCE = Path('/mnt/firecuda2/Claude/toxoplasma_projects/test_datasets/spacr/tutorials/merged')
FIELDS = AUTHOR / 'derived/cellpose_masks'
TRAIN_IMAGES = AUTHOR / 'derived/train_cellpose/train/images'
MODELS = Path('/home/olafsson/.cellpose/models')
PRIMARY = MODELS / 'cpsam'
SECONDARY = MODELS / 'cpsam_v2'
# Frozen preparation recipe and independently checked TIFF file identities.
PAIRS = (
    ('cell_pair_01', 'plate1_B01_1_1.npy', 120, 180,
     '2069121f2a4bf664b99e328444d77a4236ee309c0345f35bdf9631452071fa50'),
    ('cell_pair_02', 'plate1_B01_2_1.npy', 520, 240,
     'cc9df7fe55a085f100d633f9cc83af6aca4b654de2c9208bfc80b3044e88c138'),
    ('cell_pair_03', 'plate1_B01_3_1.npy', 940, 520,
     '91aadd96190734607a2484d7b744f7c08222a84eafce16a1c86659cdd5c1132e'),
)


def _stat(path):
    """Stable file identity; deliberately exclude read-induced access times."""
    value = Path(path).stat()
    return {name: int(getattr(value, name)) for name in
            ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_mode')}


def _digest(path, tick=lambda: None):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            result.update(block)
            tick()
    return result.hexdigest()


def _fingerprint(path, tick):
    before = _stat(path)
    digest = _digest(path, tick)
    if _stat(path) != before:
        raise RuntimeError(f'The file changed while being read: {path}')
    return {'stat': before, 'sha256': digest}


def _directory_state(path):
    """Notice additions or altered direct children without scanning caches."""
    return {item.name: _stat(item) for item in sorted(Path(path).iterdir())}


def record_model_zoo(app, window, screen, stage, captures, capture, settle,
                     write_json, timeout):
    """Use visible controls; accept only actual, fully retired benchmark jobs."""
    stage, captures = Path(stage).resolve(), Path(captures).resolve()
    if not captures.is_relative_to(stage):
        raise ValueError('Model Zoo evidence must stay inside the private stage')
    if not math.isfinite(float(timeout)) or float(timeout) <= 0:
        raise ValueError('A positive, finite recorder timeout is required')
    deadline = time.monotonic() + float(timeout)
    acceptance_path = captures / 'scientific_acceptance.json'
    evidence = {
        'accepted': False, 'published': False,
        'reason': 'The real Model Zoo benchmark has not been verified',
        'lesson': '22_model_zoo', 'held_out_validation': False,
        'mask_accuracy_validated': False, 'training_performed': False,
        'weights_downloaded': False, 'inference_overridden': False,
        'checkpoint_origin_or_training_data_validated': False,
        'device': None,
        'device_note': 'No actual inference device is exposed by BenchmarkResult',
        'resource_guard': 'Caller-owned host-RAM guard and two-thread environment',
    }
    write_json(acceptance_path, evidence)
    zoo = comparison = None
    progress = []
    originals = {}
    directories = {}
    workers = []
    progress_slot = None

    def tick():
        if time.monotonic() >= deadline:
            raise TimeoutError('Model Zoo recorder exceeded its complete lifecycle timeout')
        settle(0)

    def check_originals():
        for name, before in originals.items():
            if _fingerprint(Path(name), tick) != before:
                raise RuntimeError(f'An original input or checkpoint changed: {name}')
        for name, before in directories.items():
            if _directory_state(Path(name)) != before:
                raise RuntimeError(f'An original directory changed: {name}')

    try:
        import numpy as np
        import tifffile
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QFileDialog, QDialogButtonBox, QLineEdit, QScrollArea
        from spacr.qt.screens.model_zoo import ModelZooScreen
        from spacr.qt.screens.model_compare import ModelCompareScreen
        from spacr.qt.widgets.fold_strip import FoldButton

        def expose(widget):
            tick()
            parent = widget.parentWidget()
            while parent is not None:
                if isinstance(parent, QScrollArea):
                    parent.ensureWidgetVisible(widget)
                parent = parent.parentWidget()
            settle(.1)
            if (not widget.isVisible() or not widget.isEnabled()
                    or widget.visibleRegion().isEmpty()):
                raise RuntimeError('A requested control is not visible and usable')

        def click(widget):
            expose(widget)
            QTest.mouseClick(widget, Qt.LeftButton,
                             pos=widget.visibleRegion().boundingRect().center())
            settle(.15)
            tick()

        def fill(edit, value):
            if edit is None:
                raise RuntimeError('The actual picker has no filename editor')
            expose(edit)
            edit.setFocus()
            QTest.keyClick(edit, Qt.Key_A, Qt.ControlModifier)
            QTest.keyClicks(edit, str(value))
            QTest.keyClick(edit, Qt.Key_Tab)
            settle(.1)

        def choose_directory(button, path, frame):
            accepted, errors = [], []
            timers = []

            def handle():
                dialog = app.activeModalWidget()
                try:
                    if not isinstance(dialog, QFileDialog):
                        raise RuntimeError('The real directory picker did not open')
                    dialog.accepted.connect(lambda: accepted.append(True))
                    watchdog = QTimer(dialog)
                    watchdog.setSingleShot(True)
                    watchdog.timeout.connect(dialog.reject)
                    watchdog.start(max(1, min(12000, int((deadline - time.monotonic()) * 1000))))
                    timers.append(watchdog)
                    dialog.resize(1400, 950)
                    fill(dialog.findChild(QLineEdit, 'fileNameEdit'), path)
                    capture(frame)
                    box = dialog.findChild(QDialogButtonBox)
                    accept = None if box is None else box.button(QDialogButtonBox.Open)
                    if accept is None:
                        raise RuntimeError('The real directory picker has no Open button')
                    click(accept)
                except Exception as exc:
                    errors.append(str(exc))
                    if dialog is not None:
                        dialog.reject()

            opener = QTimer(window)
            opener.setSingleShot(True)
            opener.timeout.connect(handle)
            opener.start(400)
            try:
                click(button)
            finally:
                opener.stop()
                opener.deleteLater()
                for timer in timers:
                    try:
                        timer.stop()
                    except RuntimeError:
                        pass  # The completed dialog may already own-deleted it.
            if errors or not accepted:
                raise RuntimeError('; '.join(errors) or 'The directory picker was not accepted')

        def wait_job(widget, label):
            while widget.is_busy() or widget.active_jobs():
                tick()
                settle(.1)
            settle(.3)
            tick()
            if widget.last_error:
                raise RuntimeError(f'{label}: {widget.last_error}')

        def model_item(path):
            entries = zoo.entries()
            matches = []
            for row in range(zoo._table.rowCount()):
                item = zoo._table.item(row, 0)
                index = None if item is None else item.data(Qt.UserRole)
                if index is None or not 0 <= int(index) < len(entries):
                    continue
                entry = entries[int(index)]
                if entry.path and Path(entry.path).resolve() == path.resolve():
                    matches.append(item)
            if len(matches) != 1:
                raise RuntimeError(f'Expected one displayed row for the exact checkpoint: {path}')
            return matches[0]

        def select_model(path, add=False):
            item = model_item(path)
            zoo._table.scrollToItem(item)
            expose(zoo._table)
            rect = zoo._table.visualItemRect(item)
            if not zoo._table.viewport().rect().contains(rect.center()):
                raise RuntimeError('The selected checkpoint row is outside the visible table')
            modifiers = Qt.ControlModifier if add else Qt.NoModifier
            QTest.mouseClick(zoo._table.viewport(), Qt.LeftButton, modifiers, rect.center())
            settle(.3)

        # Full hashes are intentional here: the caller requested unchanged
        # sources/checkpoints after genuine inference. No checkpoint is loaded
        # to produce this before-state, and no cached model is copied or renamed.
        for folder in (MODELS, FIELDS, TRAIN_IMAGES):
            if not folder.is_dir():
                raise RuntimeError(f'The existing input directory is missing: {folder}')
            directories[str(folder)] = _directory_state(folder)
        checkpoints = [PRIMARY] + ([SECONDARY] if SECONDARY.is_file() else [])
        if not PRIMARY.is_file():
            raise RuntimeError('The exact cached cpsam checkpoint is unavailable; do not download it')
        for path in checkpoints:
            with path.open('rb') as handle:
                header = handle.read(4)
            if not (header.startswith(b'PK\x03\x04') or header.startswith(b'\x80')):
                raise RuntimeError(f'The cached model is not a checkpoint: {path}')
            originals[str(path)] = _fingerprint(path, tick)

        expected_names = [pair[0] for pair in PAIRS]
        expected_images = []
        input_records = []
        for name, source_name, y, x, expected_sha in PAIRS:
            source, training, image = SOURCE / source_name, TRAIN_IMAGES / (name + '.tif'), FIELDS / (name + '.tif')
            for path in (source, training, image):
                originals[str(path)] = _fingerprint(path, tick)
            if any(originals[str(path)]['sha256'] != expected_sha for path in (training, image)):
                raise RuntimeError(f'The frozen real TIFF has changed: {name}')
            acquired = np.load(source, mmap_mode='r', allow_pickle=False)
            pixels = tifffile.imread(image)
            trained = tifffile.imread(training)
            if (acquired.shape != (2000, 2000, 7) or acquired.dtype != np.uint16
                    or pixels.shape != (512, 512) or pixels.dtype != np.uint16
                    or trained.dtype != np.uint16 or trained.shape != (512, 512)
                    or not np.array_equal(pixels, acquired[y:y + 512, x:x + 512, 1])
                    or not np.array_equal(pixels, trained)):
                raise RuntimeError(f'The real crop no longer matches its source recipe: {name}')
            expected_images.append(pixels.copy())
            input_records.append({
                'field': name, 'image': str(image), 'training_image': str(training),
                'source': str(source), 'origin_yx': [y, x], 'image_channel_index': 1,
                'shape': [512, 512], 'dtype': 'uint16', 'tiff_sha256': expected_sha,
                'all_pixels_equal_to_source_crop': True, 'reused_training_example': True,
            })
            del acquired, pixels, trained
        work_parent = stage / 'model_zoo_runs'
        work_parent.mkdir(exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix='example-', dir=work_parent))
        evidence['private_artifacts'] = str(work)
        evidence['inputs'] = input_records
        evidence['originals_before'] = originals
        write_json(captures / 'input_manifest.json', {
            'records': input_records, 'originals_before': originals,
            'directories_before': directories, 'new_artificial_images': False,
            'held_out_validation': False, 'checkpoint_training_provenance_validated': False,
        })

        buttons = [button for button in screen.findChildren(FoldButton)
                   if button.app_key == 'model_zoo' and button.isVisible()]
        if len(buttons) != 1:
            raise RuntimeError('Make Masks must expose one visible Model Zoo FoldButton')
        click(buttons[0])
        settle(.5)
        visible = [child for child in window.findChildren(ModelZooScreen) if child.isVisible()]
        if len(visible) != 1:
            raise RuntimeError('The actual Model Zoo fold did not open visibly')
        zoo = visible[0]
        workers.append(zoo)
        wait_job(zoo, 'Opening the Model Zoo fold')
        if zoo._segment_fn is not None or zoo.result() is not None:
            raise RuntimeError('A fresh Model Zoo with its unmodified inference backend is required')
        capture('01b_current_model_zoo_fold')

        def remember_progress(message):
            progress.append({'seconds': time.monotonic() - started, 'message': str(message)})

        started = time.monotonic()
        progress_slot = remember_progress
        zoo._progress_said.connect(progress_slot)
        choose_directory(zoo._btn_pick_scan, MODELS, '02_actual_model_folder_picker')
        wait_job(zoo, 'Normal model scan')
        if Path(zoo._scan_edit.text()).resolve() != MODELS.resolve():
            raise RuntimeError('The visible model folder does not match the actual cache')
        capture('03_current_catalogue_and_local_models')
        select_model(PRIMARY)
        selected = zoo.selected_entries()
        if (len(selected) != 1 or Path(selected[0].path).resolve() != PRIMARY.resolve()
                or selected[0].kind != 'cellpose' or not selected[0].exists):
            raise RuntimeError('The visible selection is not the exact local cpsam checkpoint')
        entry = selected[0]
        provenance = zoo.detail_text()
        if provenance != entry.describe() or not provenance:
            raise RuntimeError('The visible provenance card differs from the selected entry')
        evidence['model'] = {'path': str(PRIMARY), 'kind': entry.kind,
                             'checksum_state': entry.checksum_state,
                             'trained_on': entry.trained_on, 'trained_by': entry.trained_by,
                             'provenance': provenance}
        expose(zoo._detail)
        capture('04_actual_cpsam_provenance')

        expose(zoo._fields_box)
        zoo._fields_box.setFocus()
        QTest.keyClick(zoo._fields_box, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(zoo._fields_box, '3')
        QTest.keyClick(zoo._fields_box, Qt.Key_Tab)
        wait_job(zoo, 'Setting the field limit')
        if zoo._fields_box.value() != 3:
            raise RuntimeError('The visible field count did not become three')
        choose_directory(zoo._btn_pick_fields, FIELDS, '05_actual_real_fields_picker')
        wait_job(zoo, 'Loading the three real fields')
        if (Path(zoo.fields_folder()).resolve() != FIELDS.resolve()
                or zoo.field_names() != expected_names or len(zoo._images) != 3
                or any(not np.array_equal(got, expected)
                       for got, expected in zip(zoo._images, expected_images))):
            raise RuntimeError('The GUI loaded different fields or pixels')
        capture('06_three_verified_real_fields')
        if zoo._segment_fn is not None or zoo.result() is not None:
            raise RuntimeError('The benchmark must start without an override or stale result')
        click(zoo._btn_test)
        capture('07_actual_test_started')
        wait_job(zoo, 'Actual Cellpose benchmark')
        result = zoo.result()
        if (result is None or result.n_fields != 3 or result.fields != expected_names
                or Path(result.entry.path).resolve() != PRIMARY.resolve()
                or len(result.images) != 3 or len(result.masks) != 3 or len(result.rows) != 3
                or zoo._segment_fn is not None or not math.isfinite(result.seconds)
                or result.seconds < 0 or result.summary not in zoo.summary_text()):
            raise RuntimeError('The actual benchmark lacks the required exact result identities')
        rows = zoo.benchmark_rows()
        if len(rows) != 3 or zoo._bench_table.rowCount() != 3:
            raise RuntimeError('The actual benchmark table does not contain three rows')
        fieldset = hashlib.sha256()
        for name, pixels in zip(expected_names, expected_images):
            fieldset.update(name.encode('utf-8') + b'\x00')
            fieldset.update(str(pixels.shape).encode())
            fieldset.update(str(pixels.dtype).encode())
            fieldset.update(hashlib.sha256(pixels.tobytes(order='C')).digest())
        if result.fieldset != fieldset.hexdigest()[:16]:
            raise RuntimeError('The benchmark field-set identity does not match the verified pixels')
        outputs = []
        for index, (name, pixels, expected, mask, row) in enumerate(zip(
                expected_names, result.images, expected_images, result.masks, result.rows)):
            pixels, mask = np.asarray(pixels), np.asarray(mask)
            if (pixels.shape != (512, 512) or pixels.dtype != np.uint16
                    or not np.array_equal(pixels, expected) or mask.shape != pixels.shape
                    or not np.issubdtype(mask.dtype, np.integer) or np.any(mask < 0)):
                raise RuntimeError(f'The returned image or labels are invalid: {name}')
            count = int(np.count_nonzero(np.unique(mask)))
            if (row.field != name or int(row.n_objects) != count
                    or rows[index] != [name, str(count), row.severity,
                                       ', '.join(row.flags) if row.flags else '-']):
                raise RuntimeError(f'The displayed object count or flags disagree with the mask: {name}')
            output = work / (name + '_cpsam_labels.npy')
            np.save(output, mask, allow_pickle=False)
            outputs.append({'field': name, 'objects': count, 'severity': row.severity,
                            'flags': list(row.flags), 'note': row.note,
                            'shape': list(mask.shape), 'dtype': str(mask.dtype),
                            'mask_evidence': str(output), 'mask_sha256': _digest(output, tick)})
            item = zoo._bench_table.item(index, 0)
            zoo._bench_table.scrollToItem(item)
            expose(zoo._bench_table)
            rect = zoo._bench_table.visualItemRect(item)
            if not zoo._bench_table.viewport().rect().contains(rect.center()):
                raise RuntimeError('The requested result row is not visible')
            QTest.mouseClick(zoo._bench_table.viewport(), Qt.LeftButton, pos=rect.center())
            settle(.3)
            expose(zoo._preview)
            pixmap = zoo._preview.pixmap()
            if zoo._bench_table.currentRow() != index or pixmap is None or pixmap.isNull():
                raise RuntimeError(f'The actual per-field preview is missing: {name}')
            capture(f'08_actual_preview_{index + 1:02d}')
        expose(zoo._summary)
        capture('09_actual_benchmark_summary')
        if len(outputs) != 3 or result.total_objects != sum(row['objects'] for row in outputs):
            raise RuntimeError('The benchmark total does not match all three actual label masks')
        evidence['benchmark'] = {
            'fields': result.fields, 'field_count': result.n_fields,
            'fieldset': result.fieldset, 'fieldset_label': result.fieldset_label,
            'segmentation_seconds': result.seconds, 'total_objects': result.total_objects,
            'summary': result.summary, 'visible_summary': zoo.summary_text(),
            'status': zoo.status_text(), 'rows': outputs,
            'honoured_parameters': result.honoured, 'ignored_parameters': result.ignored,
            'notes': list(result.notes), 'progress_messages': progress,
            'device': None, 'ground_truth_accuracy': False,
        }
        write_json(captures / 'actual_benchmark.json', evidence['benchmark'])

        evidence['comparison_handoff'] = {'performed': False, 'comparison_run': False}
        if SECONDARY in checkpoints:
            select_model(PRIMARY)
            select_model(SECONDARY, add=True)
            chosen = {Path(item.path).resolve() for item in zoo.selected_entries()}
            if chosen != {PRIMARY.resolve(), SECONDARY.resolve()}:
                raise RuntimeError('The two visible selected checkpoint identities are incorrect')
            capture('10_two_actual_cached_models_selected')
            click(zoo._btn_compare)
            settle(.5)
            visible = [child for child in window.findChildren(ModelCompareScreen) if child.isVisible()]
            if len(visible) != 1:
                raise RuntimeError('The actual comparison handoff did not open visibly')
            comparison = visible[0]
            workers.append(comparison)
            wait_job(comparison, 'Comparison handoff field loading, not inference')
            configs = comparison.model_configs()
            if ({Path(config.model).resolve() for config in configs} != chosen
                    or comparison.field_names() != expected_names
                    or Path(comparison.source_folder()).resolve() != FIELDS.resolve()
                    or len(comparison._images) != 3
                    or any(not np.array_equal(got, expected) for got, expected in
                           zip(comparison._images, expected_images))
                    or comparison.report() is not None or comparison._segment_fn is not None):
                raise RuntimeError('Comparison handoff changed the checkpoint or field identities')
            capture('11_actual_comparison_handoff_only')
            evidence['comparison_handoff'] = {
                'performed': True, 'comparison_run': False,
                'models': sorted(str(path) for path in chosen), 'fields': expected_names,
                'ground_truth_accuracy': False,
            }

        for widget in workers:
            wait_job(widget, 'Final worker retirement')
        check_originals()
        evidence.update(accepted=True, reason='Actual bounded benchmark and original preservation verified',
                        originals_unchanged=True, directories_unchanged=True,
                        active_jobs_after=[widget.active_jobs() for widget in workers],
                        elapsed_seconds=time.monotonic() - started)
        write_json(acceptance_path, evidence)
        return evidence
    except BaseException as exc:
        evidence.update(accepted=False, reason=f'{type(exc).__name__}: {exc}',
                        progress_messages=progress)
        # Never close/destroy a running inference widget on a timeout. The
        # dedicated process belongs to the caller's outer timeout/guard.
        lifecycle = []
        for widget in workers:
            try:
                lifecycle.append({'class': type(widget).__name__, 'busy': widget.is_busy(),
                                  'active_jobs': widget.active_jobs()})
            except RuntimeError:
                lifecycle.append({'class': type(widget).__name__, 'unexpectedly_deleted': True})
        evidence['worker_lifecycle'] = lifecycle
        write_json(acceptance_path, evidence)
        raise
    finally:
        if zoo is not None and progress_slot is not None:
            try:
                zoo._progress_said.disconnect(progress_slot)
            except (RuntimeError, TypeError):
                pass
