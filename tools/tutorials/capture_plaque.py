"""Check the real plaque assay and its preview routing before writing narration."""
import capture_ui
from capture_ui_drv_a import modal_wait, search_typed, table_into_view, watchdog_ms
from capture_ui_drv_c import Scenes
from capture_ui_drv_e import guarded, snap
from pathlib import Path
import os
import shutil
import time

from capture_acceptance import assess_pipeline, observe_pipeline_run as observe_plaque_run
from plaque_demo import prepare, require_preserved, verify_database
from replication_demo import digest
from stage_lesson import read


def prepare_bundled_model(stage, source):
    """Copy the actual checkpoint into the private visible capture namespace."""
    stage, source = Path(stage).resolve(), Path(source)
    if not source.is_file():
        raise ValueError('The actual existing bundled plaque checkpoint is missing')
    target = stage / 'plaque_models' / 'bundled' / source.name
    if (not target.parent.resolve().is_relative_to(stage)
            or target.is_symlink()):
        raise ValueError('The private bundled checkpoint must stay inside the capture stage')
    expected = digest(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.samefile(source) or digest(target) != expected:
            raise ValueError('The existing private bundled checkpoint differs or aliases its source')
    else:
        with source.open('rb') as original, target.open('xb') as copied:
            shutil.copyfileobj(original, copied)
    if digest(source) != expected or digest(target) != expected:
        raise ValueError('The private bundled checkpoint copy failed byte verification')
    return dict(source=str(source), copy=str(target), sha256=expected,
                bytes=target.stat().st_size)


def open_preview_settings(capture, settle, panel):
    """The Plaque preview's Settings window: its button clicked (host), or opened directly."""
    if not capture_ui.live(capture):
        return panel.open_settings()
    capture_ui._h().click(panel._settings_btn)
    capture_ui._h().wait_until(capture, lambda: getattr(panel, '_settings_dialog', None) is not None
                               and panel._settings_dialog.isVisible(), 10, 'The preview Settings did not open')
    return panel._settings_dialog


@guarded
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
    if use_zoo_model:
        # Preparation changes only the path: the initial existing-mask run
        # keeps the exact bundled bytes before the later genuine Zoo choice.
        proof['bundled_model_preparation'] = prepare_bundled_model(stage, model)
        model = Path(proof['bundled_model_preparation']['copy'])
        write_json(captures/'scientific_acceptance.json', proof)
    catalog = stage.parent/'catalog/24_plaque_settings.json'
    if not catalog.is_file():
        catalog = Path(__file__).resolve().parent/'authoring/catalog/24_plaque_settings.json'
    requested = read(catalog)
    requested.update(src=str(root), plaque_model=str(model), well_detection=False,
                     plate_format=None, well_diameter_mm=None)
    # Host: every scene clip starts BEFORE the actions that lead to its state.
    S = Scenes(capture, settle)
    S.begin('02_existing_masks_configuration')
    for key, value in requested.items():
        if not capture_ui.fill_setting(capture, screen, key, value):
            raise ValueError('No actual Plaque setting for '+key)
        settle(.03)
    before = screen._settings_model.collect()
    write_json(captures/'configured_settings.json', before)
    mismatches = {k: (v, before.get(k)) for k, v in requested.items() if before.get(k) != v}
    if mismatches:
        raise ValueError('The actual Plaque settings differ: '+str(mismatches))
    S.end('02_existing_masks_configuration')
    bar = screen._settings_search
    original = (bar.query(), bar.level(), bar.modified_only())
    if bar.modified_only():
        QTest.mouseClick(bar._modified, Qt.LeftButton)
    if bar.level() != 'all':
        QTest.mouseClick(bar._disclosure, Qt.LeftButton)
    for i, key in enumerate(('src', 'masks', 'plaque_model', 'well_detection',
                            'plate_format', 'well_diameter_mm'), 3):
        S.begin(f'{i:02}_setting_{key}')
        search_typed(capture, bar, key)
        settle(.3)
        field = screen._settings_model._widgets[key]
        capture_ui.scroll_to(capture, screen._settings_scroll, field)
        settle(.2)
        if key not in bar.visible_keys() or not field.isVisible():
            raise ValueError('An actual Plaque setting is hidden: '+key)
        S.end(f'{i:02}_setting_{key}', field)
    capture_ui.search_settings(capture, bar, original[0]); capture_ui.settings_level(capture, bar, original[1]); capture_ui.settings_modified_only(capture, bar, original[2])
    if screen._settings_model.collect() != before:
        raise ValueError('The view-only Plaque tour changed the analysis')
    if screen._ai_switch.isChecked():
        QTest.mouseClick(screen._ai_switch, Qt.LeftButton)
    S.begin('09_actual_completion')          # host: the clip holds the Run click and the finished console
    try:
        outcome = observe_plaque_run(screen,
            lambda: QTest.mouseClick(screen._btn_run, Qt.LeftButton), settle, timeout)
    except TimeoutError:
        if screen._worker_thread_is_running():
            QTest.mouseClick(screen._btn_stop, Qt.LeftButton); settle(3)
        raise
    proof['actual_gui_run'] = True
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
    if use_zoo_model:
        require_preserved([proof['bundled_model_preparation']])
    proof['all_inputs_preserved'] = True
    capture_ui.console_end_all(capture, screen._console)
    if not capture_ui.live(capture): screen._console.jump_to_the_end()
    settle(.3)
    S.end('09_actual_completion', screen._console)
    write_json(captures/'scientific_acceptance.json', proof)
    if not proof['pipeline']['accepted'] or not proof.get('output_checks'):
        raise ValueError('The actual Plaque output or console failed acceptance')

    # A preview must use the assay's chosen checkpoint, not merely open a card.
    # The screen owns this preview (PreviewSpec.owned_by_screen): its Live switch sits in the actions row
    # and its panel, a PlaquePreviewPanel, is built on first use.
    from spacr.qt.preview_registry import PREVIEWS
    switch = getattr(screen, '_preview_switch', None)
    if switch is None:
        raise ValueError('The Plaque Live switch is not on the screen')
    S.begin('10_actual_preview_model')
    QTest.mouseClick(switch, Qt.LeftButton)
    settle(1.5)
    panel = screen._live_preview
    if panel is None or not panel.isVisible():
        raise ValueError('The actual Plaque preview is hidden')
    wanted = ('plaque_model', 'diameter', 'flow_threshold', 'CP_prob')
    proof['preview_routing'] = dict(assay_checkpoint=before['plaque_model'],
        form_model_name=before.get('model_name'),
        actual_preview={k: v for k, v in panel.current_settings().items() if k in wanted},
        declared_propagation=dict(PREVIEWS[screen.app_key].propagation),
        primed=bool(getattr(screen, '_preview_primed', False)))
    S.end('10_actual_preview_model', panel)
    write_json(captures/'scientific_acceptance.json', proof)
    dialog = None
    # The preview is primed from the form, which holds the bundled Cellpose 3 file here, so it starts on the
    # assay model; the lesson picks the downloaded Model Zoo plaque model in the preview itself.
    if use_zoo_model or panel.current_settings()['plaque_model'] != str(model):
        proof['preview_not_automatically_matched'] = True
        S.begin('11_preview_settings_unmatched')
        dialog = open_preview_settings(capture, settle, panel)
        capture_ui.dialog_geometry(capture, dialog, (1100, 800))
        capture_ui.dialog_geometry(capture, dialog, None, window.geometry().center()-dialog.rect().center())
        settle(.3)
        S.end('11_preview_settings_unmatched', dialog)
        S.begin('12_actual_model_choice')
        errors, selected = [], []
        # The Model Zoo lists one row per model family ('toxoplasma_plaque') with a Version choice; the default
        # version is the current one. SPACR_TUTORIAL_PLAQUE_VERSION (for offline checks) picks another.
        family = 'toxoplasma_plaque'
        chosen = {}
        selected_paths = []

        def pick_model():
            picker = app.activeModalWidget()
            try:
                if not isinstance(picker, ModelZooPicker):
                    raise ValueError('The real Model zoo picker did not open')
                capture_ui.dialog_geometry(capture, picker, (1800, 1050))
                capture_ui.dialog_geometry(capture, picker, None, window.geometry().center()-picker.rect().center())
                settle(.3)
                watchdog = QTimer(picker); watchdog.setSingleShot(True)
                watchdog.timeout.connect(picker.reject); watchdog.start(watchdog_ms(capture, 420000 if use_zoo_model else 15000))
                if use_zoo_model:
                    # The genuine Save to field, not a substituted downloader or
                    # a file in the user's model cache. Never bypass a checksum.
                    capture_ui.focus(capture, picker.folder_edit)
                    QTest.keyClick(picker.folder_edit, Qt.Key_A, Qt.ControlModifier)
                    QTest.keyClicks(picker.folder_edit, str(stage/'app-state/models'))   # the zoo's own folder: a key resolves there
                    settle(.3)
                matches = [r for r in range(picker.table.rowCount()) if
                    (picker.table.item(r, 0).text() == family if use_zoo_model else
                     picker.table.item(r, 3).toolTip() == str(model))]
                write_json(captures/'model_zoo_choices.json', [[(picker.table.item(r, c).text() if picker.table.item(r, c) is not None else
                    getattr(picker.table.cellWidget(r, c), 'currentText', lambda: '')())
                    for c in range(picker.table.columnCount())] for r in range(picker.table.rowCount())])
                if len(matches) != 1:
                    raise ValueError('The actual Model zoo does not uniquely offer the selected bundled file')
                wanted_version = os.environ.get('SPACR_TUTORIAL_PLAQUE_VERSION', '') if use_zoo_model else ''
                if wanted_version:
                    versions = picker.table.cellWidget(matches[0], 4)
                    capture_ui.select_combo(capture, versions, versions.findText(wanted_version))
                    settle(.3)
                item = picker.table.item(matches[0], 0)       # re-read: a version change rewrites the row
                if capture_ui.live(capture):
                    table_into_view(capture, picker.table, matches[0])
                else:
                    picker.table.scrollToItem(item)
                QTest.mouseClick(picker.table.viewport(), Qt.LeftButton,
                                 pos=picker.table.visualItemRect(item).center())
                settle(.3)
                S.end('12_actual_model_choice', picker.table)
                S.begin('12c_actual_download_verified' if use_zoo_model else '13_preview_checkpoint_matched', lead=.3)
                if use_zoo_model:
                    entry = picker.selected_entry()
                    target_key, target_sha = entry.key, entry.sha256
                    chosen['key'] = target_key
                    if not entry.key.startswith(family + '_v') or not target_sha:
                        raise ValueError('The actual selected plaque model has no published checksum: ' + str(entry.key))
                    target = stage/'app-state/models'/entry.name
                    if target.is_file() and digest(target) != target_sha:
                        raise ValueError('The private cached model checksum differs')
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
                                snap(capture, '12b_actual_download_progress'); progress_captured=True
                            if failures:
                                raise ValueError('The native download failed: '+str(failures))
                        if picker._thread is not None or not downloaded:
                            raise TimeoutError('The native model download did not complete in six minutes')
                    if not target.is_file() or digest(target) != target_sha:
                        raise ValueError('The downloaded plaque checkpoint bytes fail the published checksum')
                    matches = [r for r in range(picker.table.rowCount())
                               if picker.table.item(r, 0).text() == family]
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
                    S.end('12c_actual_download_verified', picker.table)
                    S.begin('13_preview_checkpoint_matched', lead=.3)
                if not picker.use_button.isEnabled():
                    raise ValueError('The existing checkpoint cannot be selected without downloading')
                picker.model_chosen.connect(lambda path: selected_paths.append(path))
                picker.accepted.connect(lambda: selected.append(True))
                QTest.mouseClick(picker.use_button, Qt.LeftButton)
            except Exception as exc:
                errors.append(str(exc))
                if picker is not None:
                    capture_ui.reject_dialog(capture, picker)

        modal_wait(capture, 600, pick_model)
        QTest.mouseClick(panel._model_zoo_btn, Qt.LeftButton)
        settle(.5)
        proof['manual_model_selection'] = dict(errors=errors,
            accepted=bool(selected), actual_model=panel.current_settings()['plaque_model'], selected_paths=selected_paths)
        write_json(captures/'scientific_acceptance.json', proof)
        if use_zoo_model and not errors and selected_paths:
            model = Path(selected_paths[-1])
            if str(model) != proof['downloaded_model']['path']:
                raise ValueError('The preview did not receive the verified download path')
            # The zoo writes the model KEY (portable between machines), not the file path.
            if not capture_ui.fill_setting(capture, screen, 'plaque_model', chosen['key']):
                raise ValueError('The main Plaque form did not accept the same downloaded model')
            settle(.2)
            proof['manually_matched_assay_checkpoint'] = screen._settings_model.collect()['plaque_model']
            if proof['manually_matched_assay_checkpoint'] != chosen['key']:
                raise ValueError('The actual Plaque model field differs from the preview')
        expected_model = chosen.get('key') if use_zoo_model else str(model)
        if errors or not selected or panel.current_settings()['plaque_model'] != expected_model:
            raise ValueError('No verified manual route to the matching preview model: '+str(errors))
        S.end('13_preview_checkpoint_matched', panel._model_zoo_btn)
    capture_ui.close_dialog(capture, dialog) if dialog is not None else None
    settle(.3)
    S.begin('14_actual_preview_image')
    wanted_name = 'plate1_A01_control_1.tif'
    deadline = time.monotonic()+60
    while panel.current_path() is None and time.monotonic() < deadline:
        settle(.2)
    if panel.current_path() is None:
        raise ValueError('The Plaque preview did not list the private images')
    names = [panel._picker.itemText(i) for i in range(panel._picker.count())]
    if wanted_name not in names:
        raise ValueError('The private example image is not in the preview list: ' + str(names))
    if panel.current_path().name != wanted_name:
        capture_ui.select_combo(capture, panel._picker, names.index(wanted_name))
        settle(.5)
    if panel.current_path().name != wanted_name:
        raise ValueError('The real private image was not selected in the preview')
    S.end('14_actual_preview_image', panel._picker)
    S.begin('15_actual_matched_preview_outcome')
    QTest.mouseClick(panel._run_btn, Qt.LeftButton)
    settle(2)
    deadline = time.monotonic()+timeout
    while panel.preview_running() and time.monotonic() < deadline:
        settle(.2)
    if panel.preview_running():
        raise TimeoutError('The matched-model preview exceeded the bounded capture time')
    settle(.5)
    completed = panel._plaque_result is not None
    proof['matched_model_preview'] = dict(errors=[] if completed else [panel._status.text()], completed=completed,
        status=panel._status.text(), selected_model=panel.current_settings()['plaque_model'])
    S.end('15_actual_matched_preview_outcome', panel._status)
    write_json(captures/'scientific_acceptance.json', proof)
    if not completed:
        raise ValueError('The actual matched-model preview failed: '+panel._status.text())
    from plaque_native_views import finish_views
    proof['native_views'] = finish_views(app, window, screen, panel, database,
        manifest, stage, captures, capture, settle, write_json, timeout, scenes=S)
    require_preserved(manifest['files'])
    proof['accepted'] = True
    proof['reason'] = ('Mask reuse, explicitly selected Model Zoo preview, its result views and settings, and actual '
                       'read-only saved tables verified; synthetic geometry is not biological validation.')
    write_json(captures/'scientific_acceptance.json', proof)
