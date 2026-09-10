"""Check the real plaque assay and its preview routing before writing narration."""
from pathlib import Path
import time

from capture_acceptance import assess_pipeline
from plaque_demo import prepare, require_preserved, verify_database
from replication_demo import digest
from stage_lesson import read


def record_plaque(app, window, screen, stage, captures, capture, settle, write_json, timeout,
                  *, use_zoo_model=False):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QFileDialog, QLineEdit, QDialogButtonBox
    from spacr.qt.widgets.model_zoo_picker import ModelZooPicker

    manifest = prepare(stage)
    root = Path(manifest['root'])
    proof = dict(accepted=False, synthetic=True, manifest=manifest,
        actual_gui_run=False, app_source_modified=False, published=False)
    write_json(captures/'scientific_acceptance.json', proof)
    if screen._console_folder.shut:
        QTest.mouseClick(screen._console_folder.heading, Qt.LeftButton)
        settle(.3)
    if screen._console_folder.shut or not screen._console.isVisible():
        raise ValueError('The actual Console must be visible in the final recording')
    model = Path(__file__).resolve().parents[2]/'spacr/resources/models/toxo_plaque_cyto_e25000_X1120_Y1120.CP_model'
    if not model.is_file():
        raise ValueError('The actual existing bundled plaque checkpoint is missing')
    requested = read(stage.parent/'catalog/24_plaque_settings.json')
    requested.update(src=str(root), plaque_model=str(model), well_detection=False,
                     plate_format=None, well_diameter_mm=None)
    for key, value in requested.items():
        if not screen._settings_model.set_value_for_key(key, value):
            raise ValueError('No actual Plaque setting for '+key)
        settle(.03)
    before = screen._settings_model.collect()
    write_json(captures/'configured_settings.json', before)
    mismatches = {k: (v, before.get(k)) for k, v in requested.items() if before.get(k) != v}
    if mismatches:
        raise ValueError('The actual Plaque settings differ: '+str(mismatches))
    capture('02_existing_masks_configuration')
    bar = screen._settings_search
    original = (bar.query(), bar.level(), bar.modified_only())
    if bar.modified_only():
        QTest.mouseClick(bar._modified, Qt.LeftButton)
    if bar.level() != 'all':
        QTest.mouseClick(bar._disclosure, Qt.LeftButton)
    for i, key in enumerate(('src', 'masks', 'plaque_model', 'well_detection',
                            'plate_format', 'well_diameter_mm'), 3):
        bar._input.setFocus()
        QTest.keyClick(bar._input, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(bar._input, key)
        settle(.3)
        field = screen._settings_model._widgets[key]
        screen._settings_scroll.ensureWidgetVisible(field)
        settle(.2)
        if key not in bar.visible_keys() or not field.isVisible():
            raise ValueError('An actual Plaque setting is hidden: '+key)
        capture(f'{i:02}_setting_{key}')
    bar.set_query(original[0]); bar.set_level(original[1]); bar.set_modified_only(original[2])
    if screen._settings_model.collect() != before:
        raise ValueError('The view-only Plaque tour changed the analysis')
    if screen._ai_switch.isChecked():
        QTest.mouseClick(screen._ai_switch, Qt.LeftButton)
    outcome = dict(finished=False, ok=False, errors=[])
    QTest.mouseClick(screen._btn_run, Qt.LeftButton)
    worker = screen._worker
    if worker is None:
        raise ValueError('The actual Plaque Run button did not launch a worker')
    worker.finished.connect(lambda ok: outcome.update(finished=True, ok=bool(ok)))
    worker.error.connect(lambda message: outcome['errors'].append(str(message)))
    proof['actual_gui_run'] = True
    deadline = time.monotonic()+timeout
    while not outcome['finished'] or screen._worker_thread_is_running():
        if time.monotonic() > deadline:
            QTest.mouseClick(screen._btn_stop, Qt.LeftButton); settle(3)
            raise TimeoutError('The bounded existing-mask Plaque run did not finish')
        settle(.1)
    settle(.5)
    blocks = [text for _, _, text in screen._console._pipeline_console_blocks()]
    write_json(captures/'batch_console.json', blocks)
    write_json(captures/'batch_outcome.json', outcome)
    proof['pipeline'] = assess_pipeline(outcome, blocks, screen._figure_queue.count(), requires_figure=False)
    database = root/'masks/plaques_analysis.db'
    if database.is_file():
        checked, tables = verify_database(database, manifest['references'])
        proof.update(output_checks=checked, output_database=str(database))
        write_json(captures/'actual_output_tables.json', tables)
    require_preserved(manifest['files'])
    proof['all_inputs_preserved'] = True
    for block, _, _ in screen._console._pipeline_console_blocks():
        block.setFocus(); QTest.keyClick(block, Qt.Key_End, Qt.ControlModifier)
    screen._console.jump_to_the_end(); settle(.3)
    capture('09_actual_completion')
    write_json(captures/'scientific_acceptance.json', proof)
    if not proof['pipeline']['accepted'] or not proof.get('output_checks'):
        raise ValueError('The actual Plaque output or console failed acceptance')

    # A preview must use the assay's chosen checkpoint, not merely open a card.
    host = getattr(screen, '_registry_preview', None)
    if host is None:
        raise ValueError('The declared Plaque preview is not attached')
    QTest.mouseClick(host.toggle, Qt.LeftButton)
    settle(.5)
    if not host.panel.isVisible():
        raise ValueError('The actual Plaque preview is hidden')
    proof['preview_routing'] = dict(assay_checkpoint=before['plaque_model'],
        form_model_name=before.get('model_name'), actual_preview=host.panel.current_params(),
        declared_propagation=dict(host._spec.propagation), primed=host._primed)
    capture('10_actual_preview_model')
    write_json(captures/'scientific_acceptance.json', proof)
    if host.panel.current_params()['model'] != str(model):
        proof['preview_not_automatically_matched'] = True
        panel = host.panel
        panel.open_live_settings()
        settle(.4)
        dialog = panel._live_settings_dialog
        dialog.resize(1800, 1250)
        dialog.move(window.geometry().center()-dialog.rect().center())
        settle(.3)
        capture('11_preview_settings_unmatched')
        errors, selected = [], []
        target_key = 'toxoplasma_plaque_v1'
        target_sha = 'eeecd2d6cd5cbb4dddee71564d5f460d26bb07ac125e0b494b7502fea4292d5d'
        selected_paths = []

        def pick_model():
            picker = app.activeModalWidget()
            try:
                if not isinstance(picker, ModelZooPicker):
                    raise ValueError('The real Model zoo picker did not open')
                picker.resize(1800, 1050)
                picker.move(window.geometry().center()-picker.rect().center())
                settle(.3)
                watchdog = QTimer(picker); watchdog.setSingleShot(True)
                watchdog.timeout.connect(picker.reject); watchdog.start(420000 if use_zoo_model else 15000)
                if use_zoo_model:
                    # The genuine Save to field, not a substituted downloader or
                    # a file in the user's model cache. Never bypass a checksum.
                    picker.folder_edit.setFocus()
                    QTest.keyClick(picker.folder_edit, Qt.Key_A, Qt.ControlModifier)
                    QTest.keyClicks(picker.folder_edit, str(stage/'plaque_models'))
                    settle(.3)
                if use_zoo_model:
                    candidate = stage/'plaque_models/cpsam_plaque_r3'
                    if candidate.is_file():
                        if digest(candidate) != target_sha:
                            raise ValueError('The private cached model checksum differs')
                matches = [r for r in range(picker.table.rowCount()) if
                    (picker.table.item(r, 0).text() == target_key if use_zoo_model else
                     picker.table.item(r, 3).toolTip() == str(model))]
                write_json(captures/'model_zoo_choices.json', [[picker.table.item(r, c).text()
                    for c in range(picker.table.columnCount())] for r in range(picker.table.rowCount())])
                if len(matches) != 1:
                    raise ValueError('The actual Model zoo does not uniquely offer the selected bundled file')
                item = picker.table.item(matches[0], 0)
                picker.table.scrollToItem(item)
                QTest.mouseClick(picker.table.viewport(), Qt.LeftButton,
                                 pos=picker.table.visualItemRect(item).center())
                settle(.3)
                capture('12_actual_model_choice')
                if use_zoo_model:
                    entry = picker.selected_entry()
                    if entry.key != target_key or entry.sha256 != target_sha:
                        raise ValueError('The actual selected plaque model or published checksum differs')
                    target = stage/'plaque_models'/entry.name
                    if not target.is_file():
                        if not picker.download_button.isEnabled():
                            raise ValueError('The actual model Download button is disabled')
                        QTest.mouseClick(picker.download_button, Qt.LeftButton)
                        downloaded, failures = [], []
                        if picker._worker is None:
                            raise ValueError('The actual model download did not start')
                        picker._worker.finished.connect(lambda path: downloaded.append(path))
                        picker._worker.failed.connect(lambda error: failures.append(error))
                        deadline = time.monotonic()+360
                        progress_captured = False
                        while picker._thread is not None and time.monotonic()<deadline:
                            settle(.1)
                            if not progress_captured and picker.progress.maximum() == 100 and picker.progress.value()>0:
                                capture('12b_actual_download_progress'); progress_captured=True
                            if failures:
                                raise ValueError('The native download failed: '+str(failures))
                        if picker._thread is not None or not downloaded:
                            raise TimeoutError('The native model download did not complete in six minutes')
                    if not target.is_file() or digest(target) != target_sha:
                        raise ValueError('The downloaded plaque checkpoint bytes fail the published checksum')
                    matches = [r for r in range(picker.table.rowCount())
                               if picker.table.item(r, 0).text() == target_key]
                    if len(matches) != 1:
                        raise ValueError('The downloaded model does not have exactly one picker row')
                    item = picker.table.item(matches[0], 0)
                    QTest.mouseClick(picker.table.viewport(), Qt.LeftButton,
                                     pos=picker.table.visualItemRect(item).center())
                    settle(.2)
                    if picker.selected_entry().key != target_key:
                        raise ValueError('The displayed and selected downloaded model differ')
                    proof['downloaded_model'] = dict(key=target_key, path=str(target),
                        sha256=target_sha, bytes=target.stat().st_size, checksum_verified=True)
                    write_json(captures/'scientific_acceptance.json', proof)
                    capture('12c_actual_download_verified')
                if not picker.use_button.isEnabled():
                    raise ValueError('The existing checkpoint cannot be selected without downloading')
                picker.model_chosen.connect(lambda path: selected_paths.append(path))
                picker.accepted.connect(lambda: selected.append(True))
                QTest.mouseClick(picker.use_button, Qt.LeftButton)
            except Exception as exc:
                errors.append(str(exc))
                if picker is not None:
                    picker.reject()

        QTimer.singleShot(600, pick_model)
        QTest.mouseClick(panel._model_zoo_btn, Qt.LeftButton)
        settle(.5)
        proof['manual_model_selection'] = dict(errors=errors,
            accepted=bool(selected), actual_model=panel.current_params()['model'], selected_paths=selected_paths)
        write_json(captures/'scientific_acceptance.json', proof)
        if use_zoo_model and not errors and selected_paths:
            model = Path(selected_paths[-1])
            if str(model) != proof['downloaded_model']['path']:
                raise ValueError('The preview did not receive the verified download path')
            if not screen._settings_model.set_value_for_key('plaque_model', str(model)):
                raise ValueError('The main Plaque form did not accept the same downloaded model')
            settle(.2)
            proof['manually_matched_assay_checkpoint'] = screen._settings_model.collect()['plaque_model']
            if proof['manually_matched_assay_checkpoint'] != str(model):
                raise ValueError('The actual Plaque model field differs from the preview')
        if errors or not selected or panel.current_params()['model'] != str(model):
            raise ValueError('No verified manual route to the matching preview model: '+str(errors))
        capture('13_preview_checkpoint_matched')
        dialog.close(); settle(.3)
        accepted = []

        def pick_image():
            chooser = app.activeModalWidget()
            try:
                if not isinstance(chooser, QFileDialog):
                    raise ValueError('The actual image file picker did not open')
                watchdog = QTimer(chooser); watchdog.setSingleShot(True)
                watchdog.timeout.connect(chooser.reject); watchdog.start(15000)
                edit = chooser.findChild(QLineEdit, 'fileNameEdit')
                edit.setFocus(); QTest.keyClick(edit, Qt.Key_A, Qt.ControlModifier)
                QTest.keyClicks(edit, str(root/'plate1_A01_control_1.tif'))
                chooser.accepted.connect(lambda: accepted.append(True))
                QTest.mouseClick(chooser.findChild(QDialogButtonBox).button(QDialogButtonBox.Open), Qt.LeftButton)
            except Exception as exc:
                errors.append(str(exc))
                if chooser is not None:
                    chooser.reject()

        QTimer.singleShot(350, pick_image)
        QTest.mouseClick(panel._pick_btn, Qt.LeftButton)
        deadline = time.monotonic()+30
        while getattr(panel, '_image', None) is None and time.monotonic()<deadline:
            settle(.1)
        if errors or not accepted or panel._image is None:
            raise ValueError('The real private image was not opened: '+str(errors))
        capture('14_actual_preview_image')
        preview_errors, completed = [], []
        QTest.mouseClick(panel._run_btn, Qt.LeftButton)
        if panel._worker is None:
            raise ValueError('The native matched-model preview did not start')
        panel._worker.finished_masks.connect(lambda masks, error, token:
            (preview_errors.append(error) if error else completed.append(True)))
        deadline = time.monotonic()+timeout
        while panel._worker.isRunning() and time.monotonic()<deadline:
            settle(.1)
        if panel._worker.isRunning():
            raise TimeoutError('The matched-model preview exceeded the bounded capture time')
        settle(.5)
        proof['matched_model_preview'] = dict(errors=preview_errors, completed=bool(completed),
            status=panel._status.text(), selected_model=panel.current_params()['model'])
        capture('15_actual_matched_preview_outcome')
        write_json(captures/'scientific_acceptance.json', proof)
        if preview_errors or not completed:
            raise ValueError('The actual matched-model preview failed: '+str(preview_errors))
        from plaque_native_views import finish_views
        proof['native_views'] = finish_views(app, window, screen, panel, database,
            manifest, stage, captures, capture, settle, write_json, timeout)
        require_preserved(manifest['files'])
        proof['accepted'] = True
    proof['reason'] = ('Mask reuse, explicitly selected Model Zoo preview, reversible filters and actual '
                       'read-only saved tables verified; synthetic geometry is not biological validation.')
    write_json(captures/'scientific_acceptance.json', proof)
