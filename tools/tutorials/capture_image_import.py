"""Record Import's real Import Images fold using downloaded microscopy files.

This is a UI driver, not a replacement for any application operation. Source
images are copied byte-for-byte into a fresh, bounded tutorial input directory;
the genuine scan, plan save/load, and import buttons perform the workflow.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_example(stage):
    """Select two complete four-channel fields, never generated image content."""
    groups = {}
    for path in sorted((stage / 'example_data/plate1').glob('*.tif')):
        match = re.fullmatch(r'(plate1_[A-Z]\d+_T\d+F\d+)L\d+A\d+Z01C(\d+)\.tif', path.name)
        if match:
            groups.setdefault(match[1], {})[int(match[2])] = path
    complete = [(key, channels) for key, channels in groups.items()
                if set(channels) == {1, 2, 3, 4}]
    selected = []
    for key, channels in complete:
        well = key.split('_')[1]
        field = int(key.rsplit('F', 1)[1])
        if selected and (well[0] == selected[0][0][0] or field == selected[0][1]):
            continue
        selected.append((well, field, channels))
        if len(selected) == 2:
            break
    if len(selected) < 2:
        raise RuntimeError('Download the real Mask example before recording Import Images')
    parent = stage / 'import_images'
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='example-', dir=parent))
    source = run / 'raw_images'
    source.mkdir()
    prepared = run / 'prepared_images'
    prepared.mkdir()
    evidence = []
    for well, field, channels in selected:
        for path in channels.values():
            raw = source / path.name
            # Explicitly prepared COPIES: the raw example has a varying A
            # acquisition token that the current parser cannot assign safely.
            # Its C channel, well and field are retained, never guessed from A.
            copied = prepared / re.sub(r'A\d+(?=Z\d+C\d+\.tif$)', '', path.name)
            shutil.copy2(path, raw)
            shutil.copy2(path, copied)
            sha = digest(path)
            if digest(copied) != sha or digest(raw) != sha:
                raise RuntimeError(f'Input copy changed bytes: {path}')
            evidence.append({'downloaded_source': str(path), 'input': str(copied),
                             'raw_copy': str(raw), 'well': well, 'field': field,
                             'channel': int(re.search(r'C(\d+)\.tif$', path.name)[1]),
                             'sha256': sha, 'bytes': path.stat().st_size})
    return run, source, prepared, evidence


def record_import(app, window, host, stage, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QFileDialog, QLineEdit
    from spacr.qt.screens.image_import import ImageImportScreen
    from spacr.qt.widgets.fold_strip import FoldButton

    buttons = [button for button in host.findChildren(FoldButton)
               if button.app_key == 'import_images' and button.isVisible()]
    if len(buttons) != 1 or not buttons[0].isEnabled():
        raise RuntimeError('Import must expose exactly one usable Import Images fold')
    QTest.mouseMove(buttons[0])
    settle(1)
    capture('02_import_fold')
    QTest.mouseClick(buttons[0], Qt.LeftButton)
    settle(2)
    screens = [screen for screen in window.findChildren(ImageImportScreen)
               if screen.isVisible()]
    if len(screens) != 1:
        raise RuntimeError(f'Fold did not reveal one Import Images screen: {len(screens)}')
    screen = screens[0]
    capture('03_import_images')
    run, raw, source, originals = prepare_example(stage)
    destination = run / 'spacr_project'
    write_json(captures / 'dataset.json', {'subset': True, 'fields': 2,
               'channels_per_field': 4, 'images': originals,
               'raw_source': str(raw), 'source': str(source),
               'preparation': 'Copies omit the A acquisition token only; well, field and C channel retained.',
               'destination': str(destination)})

    def fill(widget, text):
        widget.setFocus()
        QTest.keyClick(widget, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(widget, str(text))
        QTest.keyClick(widget, Qt.Key_Tab)
        settle(0.1)

    def wait_job(label, allow_questions=False):
        deadline = time.monotonic() + timeout
        while screen._busy:
            if time.monotonic() > deadline:
                raise TimeoutError(f'{label}: {screen.status_text()}')
            settle(0.1)
        settle(0.5)
        if screen.last_error and not (allow_questions and screen.plan() is not None):
            raise RuntimeError(f'{label}: {screen.last_error}')

    fill(screen._root_edit, raw)
    fill(screen._dst_edit, destination)
    capture('04_source_options')
    QTest.mouseClick(screen._btn_scan, Qt.LeftButton)
    wait_job('Raw scan', allow_questions=True)
    write_json(captures / 'raw_scan.json', {'report': screen.report_text(),
               'problems': screen.problems(), 'questions': screen.questions(),
               'destination_exists': destination.exists()})
    if not screen.problems() or destination.exists() or screen._btn_import.isEnabled():
        raise RuntimeError('The raw-name ambiguity demonstration no longer matches current source')
    capture('04a_unresolved_acquisition_token')
    # Do not map A to channel: C01 and C04 both use A02 in the real example.
    # Show the prepared-copy input explicitly, rather than claim automatic
    # ingestion of the original naming scheme succeeded.
    fill(screen._root_edit, source)
    QTest.mouseClick(screen._btn_scan, Qt.LeftButton)
    wait_job('Scan')
    if screen.plan() is None or screen.problems() or screen.proposal_row_count() != 8:
        write_json(captures / 'scan_problem.json', {'report': screen.report_text(),
                   'problems': screen.problems(), 'questions': screen.questions()})
        raise RuntimeError('The real example did not yield an unambiguous eight-image plan')
    for entry in originals:
        parsed = screen.plan().files[Path(entry['input']).name]
        if any(parsed.get(key) != entry[key] for key in ('well', 'field', 'channel')):
            raise RuntimeError(f'Proposal changed image identity: {parsed}')
    if destination.exists():
        raise RuntimeError('Scan unexpectedly created the import destination')
    write_json(captures / 'scan.json', {'report': screen.report_text(),
               'status': screen.status_text(), 'questions': screen.questions(),
               'counts': screen.plan().counts(), 'files': screen.plan().files,
               'destination_exists': destination.exists(),
               'columns': screen.proposal_columns()})
    capture('05_scan_proposal')
    screen._table.selectRow(0)
    settle()
    capture('06_check_filename')
    # A portable copied project is more useful for this small demonstration.
    # Leave tile policy unchanged: these real source files are not tiled.
    if screen._link_box.isChecked():
        QTest.mouseClick(screen._link_box, Qt.LeftButton)
    settle()
    capture('07_copy_destination')

    plan_path = run / 'import_plan.json'

    def file_dialog(button, path, frame):
        errors = []
        accepted = []
        deadline = time.monotonic() + 10

        def choose():
            try:
                dialog = app.activeModalWidget()
                if not isinstance(dialog, QFileDialog):
                    if time.monotonic() > deadline:
                        raise RuntimeError('The expected file dialog did not open')
                    QTimer.singleShot(100, choose)
                    return
                dialog.selectFile(str(path))
                edit = dialog.findChild(QLineEdit, 'fileNameEdit')
                if edit is None:
                    raise RuntimeError('No filename editor in the real file dialog')
                edit.setFocus()
                QTest.keyClick(edit, Qt.Key_A, Qt.ControlModifier)
                QTest.keyClicks(edit, str(path))
                dialog.resize(1200, 900)
                settle(0.2)
                capture(frame)
                dialog.accepted.connect(lambda: accepted.append(True))
                QTest.keyClick(edit, Qt.Key_Return)
                def reject_stalled():
                    if not accepted and app.activeModalWidget() is dialog:
                        errors.append('The file dialog did not accept its filename')
                        dialog.reject()
                QTimer.singleShot(3000, reject_stalled)
            except Exception as exc:
                errors.append(str(exc))
                if app.activeModalWidget() is not None:
                    app.activeModalWidget().reject()

        QTimer.singleShot(200, choose)
        QTest.mouseClick(button, Qt.LeftButton)
        settle()
        if errors:
            raise RuntimeError('; '.join(errors))
        if not accepted:
            raise RuntimeError('The file selection was cancelled, not demonstrated')
        if screen.last_error:
            raise RuntimeError(screen.last_error)

    file_dialog(screen._btn_save_plan, plan_path, '08_save_plan_dialog')
    if not plan_path.is_file():
        raise RuntimeError('Save plan did not produce a file')
    capture('09_saved_plan')
    file_dialog(screen._btn_load_plan, plan_path, '10_load_plan_dialog')
    wait_job('Load plan')
    if screen.proposal_row_count() != 8 or screen.problems():
        raise RuntimeError('Loading the saved plan did not restore the source proposal')
    capture('11_reloaded_plan')
    QTest.mouseClick(screen._btn_import, Qt.LeftButton)
    wait_job('Import')
    result = screen.result()
    if result is None or result.written != 8 or result.skipped or result.unverified:
        raise RuntimeError('Import did not produce eight fully accounted-for images')
    outputs = sorted(destination.rglob('*.tif'))
    if len(outputs) != 8 or any(path.is_symlink() for path in outputs):
        raise RuntimeError('Copy mode did not produce eight independent image files')
    expected = sorted(entry['sha256'] for entry in originals)
    actual = sorted(digest(path) for path in outputs)
    if expected != actual or any(digest(Path(entry['input'])) != entry['sha256']
                                 for entry in originals):
        raise RuntimeError('Import altered or lost source image content')
    capture('12_import_complete')
    write_json(captures / 'import_acceptance.json', {'accepted': True,
               'summary': result.summary(), 'status': screen.status_text(),
               'copied_images': len(outputs), 'source_bytes_unchanged': True,
               'outputs_match_input_hashes': True, 'plan_path': str(plan_path),
               'plan_sha256': digest(plan_path),
               'outputs': [{'path': str(path), 'sha256': digest(path)} for path in outputs]})
    return screen
