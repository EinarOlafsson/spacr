"""Record External Masks using the preserved real Foreign-lesson TIFF pairs.

The existing intensity and label pixels are never regenerated. The neutral
names assign demonstration coordinates, not recovered acquisition wells. Input
groups come only from the real folder picker and are reviewed in its role
controls; no measurement CSV or fabricated result is supplied to the app.
"""
from __future__ import annotations

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


def record_external_masks(app, window, screen, stage, captures, capture,
                          settle, write_json, timeout):
    """Preview without writing, then run and independently verify the project."""
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import (
        QAbstractButton, QComboBox, QDialogButtonBox, QFileDialog, QLineEdit,
        QMessageBox, QSpinBox,
    )
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widgets.channel_mapping import ChannelMappingWidget
    from spacr.qt.widgets.external_mask_inputs import ExternalMaskInputWidget
    from spacr.qt.widgets.fold_strip import FoldButton
    from capture_acceptance import assess_pipeline
    from external_evidence import verify_external_project

    stage, captures = Path(stage), Path(captures)
    write_json(captures / 'scientific_acceptance.json', {
        'accepted': False, 'published': False,
        'reason': 'The actual External Masks project has not been independently verified',
    })
    foreign_capture = stage / 'captures/foreign_release_v2'
    manifest_path = foreign_capture / 'inputs.json'
    acceptance_path = foreign_capture / 'scientific_acceptance.json'
    if json.loads(acceptance_path.read_text()).get('accepted') is not True:
        raise RuntimeError('The preserved Foreign input capture was not accepted')
    manifest = json.loads(manifest_path.read_text())
    records = manifest['records']
    by_name = {record['neutral_stem']: record for record in records}
    if (len(records) != 2 or set(by_name) != {'fov01', 'fov02'}
            or [by_name[name]['objects'] for name in ('fov01', 'fov02')] != [44, 59]
            or manifest.get('rows') != 103):
        raise RuntimeError('Expected the two preserved real fields with 44 and 59 labels')
    originals = {str(manifest_path): _digest(manifest_path),
                 str(acceptance_path): _digest(acceptance_path)}
    for record in records:
        name = record['neutral_stem']
        if (record['image_plane'] != 1 or record['mask_plane'] != 4
                or record['shape'] != [1994, 1994]
                or Path(record['image']).name != name + '_C1.tif'
                or Path(record['mask']).name != name + '_cell_mask.tif'):
            raise RuntimeError('The accepted input plane, shape or neutral identity changed')
        for key in ('source', 'image', 'mask'):
            path = Path(record[key])
            if not path.is_file() or _digest(path) != record[key + '_sha256']:
                raise RuntimeError(f'The accepted original input changed: {path}')
            originals[str(path)] = record[key + '_sha256']
    image_paths = {str(Path(record['image']).resolve()) for record in records}
    mask_paths = {str(Path(record['mask']).resolve()) for record in records}
    image_roots = {Path(path).parent for path in image_paths}
    mask_roots = {Path(path).parent for path in mask_paths}
    if len(image_roots) != 1 or len(mask_roots) != 1:
        raise RuntimeError('The preserved inputs must occupy one image and one mask folder')
    images, masks = next(iter(image_roots)), next(iter(mask_roots))
    for root, expected in ((images, image_paths), (masks, mask_paths)):
        if {str(path.resolve()) for path in root.iterdir()} != expected:
            raise RuntimeError(f'The source folder contains unexpected inputs: {root}')
    runs = stage / 'external_mask_runs'
    runs.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='example-', dir=runs))
    destination = work / 'project'  # The app, not the recorder, must create it.
    if destination.exists():
        raise RuntimeError('The private project destination must not exist')
    write_json(captures / 'input_manifest.json', {
        'reused_foreign_manifest': str(manifest_path), 'records': records,
        'original_hashes': originals, 'destination': str(destination),
        'measurement_csv_imported': False, 'new_images_generated': False,
        'neutral_names_do_not_preserve_original_wells': True,
    })

    def unchanged():
        for path, expected in originals.items():
            if _digest(path) != expected:
                raise RuntimeError(f'The workflow changed an original input: {path}')

    def click(button):
        if not button.isVisible() or not button.isEnabled():
            raise RuntimeError(f'The actual control is not usable: {button.text()}')
        QTest.mouseClick(button, Qt.LeftButton)
        settle(0.2)

    def fill(widget, value):
        if widget is None or not widget.isVisible() or not widget.isEnabled():
            raise RuntimeError('The actual text control is not usable')
        widget.setFocus()
        QTest.keyClick(widget, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(widget, str(value))
        QTest.keyClick(widget, Qt.Key_Tab)
        settle(0.15)

    def select(box, value):
        if not isinstance(box, QComboBox) or not box.isVisible() or not box.isEnabled():
            raise RuntimeError('The actual selection control is not usable')
        index = box.findData(value)
        if index < 0:
            index = box.findText(str(value))
        if index < 0:
            if box.isEditable():
                fill(box.lineEdit(), value)
                return
            raise RuntimeError(f'The actual control has no option {value!r}')
        box.setFocus()
        QTest.keyClick(box, Qt.Key_Home)
        for _ in range(index):
            QTest.keyClick(box, Qt.Key_Down)
        QTest.keyClick(box, Qt.Key_Tab)
        settle(0.15)
        if box.currentIndex() != index:
            raise RuntimeError(f'The actual option did not become selected: {value!r}')

    buttons = [button for button in screen.findChildren(FoldButton)
               if button.app_key == 'external_masks' and button.isVisible()]
    if len(buttons) != 1:
        raise RuntimeError('Import must expose exactly one External Masks fold')
    click(buttons[0])
    settle(2)
    children = [child for child in window.findChildren(AppScreen)
                if child.app_key == 'external_masks' and child.isVisible()]
    if len(children) != 1:
        raise RuntimeError('The actual External Masks fold did not open')
    screen = children[0]
    capture('01b_current_external_masks_fold')
    model = screen._settings_model
    width = sum(screen._body_splitter.sizes())
    screen._body_splitter.setSizes([width // 2, width - width // 2])

    def owning_sections(field):
        # The folded AppScreen has no shell-installed settings search.
        # Follow only its existing widget ancestry: inspecting dormant form
        # rows can itself materialize them, so do not use _row_widgets here.
        registered = {id(section) for section in screen._settings_sections}
        ancestors = []
        parent = field.parentWidget()
        while parent is not None and parent is not screen:
            if id(parent) in registered:
                ancestors.append(parent)
            parent = parent.parentWidget()
        return list(reversed(ancestors))

    def unavailable(key, reason, field=None):
        details = {'key': key, 'reason': reason, 'section_path': []}
        if field is not None:
            details['field'] = {'type': type(field).__name__,
                                'visible': field.isVisible(),
                                'hidden': field.isHidden()}
            details['section_path'] = [
                {'title': section.property('settingsCategorySource'),
                 'expanded': section.is_expanded(),
                 'visible': section.isVisible(),
                 'discarded': bool(section.property('settingsSectionDiscarded'))}
                for section in owning_sections(field)]
        write_json(captures / 'setting_visibility_error.json', details)
        capture('setting_unavailable_' + key)
        raise RuntimeError(f'The actual {key} setting is not exposed: {reason}')

    def expose(key):
        field = model._widgets.get(key)
        if field is None:
            unavailable(key, 'no bound control exists on this form')
        sections = owning_sections(field)
        if not sections:
            unavailable(key, 'no rendered settings section owns the control', field)
        for section in sections:
            if section.property('settingsSectionDiscarded') or not section.isVisible():
                unavailable(key, 'a required section is hidden or discarded', field)
            header = section.header()
            if not section.is_expanded():
                screen._settings_scroll.ensureWidgetVisible(header)
                screen._settings_scroll.horizontalScrollBar().setValue(0)
                settle(0.2)
                if not header.visibleRegion().contains(header.rect().center()):
                    unavailable(key, 'the section header is outside the visible viewport', field)
                click(header)
                if not section.is_expanded():
                    unavailable(key, 'the actual section header did not expand', field)
        screen._settings_scroll.ensureWidgetVisible(field)
        screen._settings_scroll.horizontalScrollBar().setValue(0)
        settle(0.2)
        if not field.isVisible() or not field.visibleRegion().contains(field.rect().center()):
            unavailable(key, 'the control remains hidden or outside the viewport', field)
        return field

    inputs = expose('inputs')
    if not isinstance(inputs, ExternalMaskInputWidget):
        raise RuntimeError('The real input-group editor is not mounted')
    # A retry may have private recorder settings saved. Remove those rows
    # visibly, then obtain every new group from the actual folder picker.
    if inputs.group_count():
        inputs._table.setFocus()
        QTest.keyClick(inputs._table, Qt.Key_A, Qt.ControlModifier)
        click(inputs._remove)
        if inputs.group_count():
            raise RuntimeError('The actual Remove selected did not clear old input groups')

    def choose_folder(path, frame):
        errors, accepted = [], []

        def handle():
            dialog = app.activeModalWidget()
            try:
                if not isinstance(dialog, QFileDialog):
                    raise RuntimeError('Expected the real Qt folder picker')
                dialog.accepted.connect(lambda: accepted.append(True))
                dialog.resize(1400, 950)
                fill(dialog.findChild(QLineEdit, 'fileNameEdit'), path)
                capture(frame)
                box = dialog.findChild(QDialogButtonBox)
                QTest.mouseClick(box.button(QDialogButtonBox.Open), Qt.LeftButton)
            except Exception as exc:
                errors.append(str(exc))
                if dialog is not None:
                    dialog.reject()

        def reject_stalled():
            dialog = app.activeModalWidget()
            if dialog is not None and not accepted:
                errors.append('The folder picker did not accept the requested directory')
                dialog.reject()

        QTimer.singleShot(500, handle)
        QTimer.singleShot(12000, reject_stalled)
        click(inputs._add_folder)
        if errors or not accepted:
            raise RuntimeError('; '.join(errors) or 'Folder selection was cancelled')

    choose_folder(images, '02_choose_real_images')
    choose_folder(masks, '03_choose_real_cell_masks')
    if inputs.group_count() != 2 or inputs.file_count() != 4:
        raise RuntimeError('The two real folders did not produce exactly four grouped TIFFs')

    def row_for(paths):
        groups = inputs.groups()
        matches = []
        for row in range(inputs._table.rowCount()):
            index = inputs._table.item(row, 0).data(Qt.UserRole)
            if index is not None:
                group = groups[int(index)]
                if {str(Path(path).resolve()) for path in group.paths} == paths:
                    matches.append(row)
        if len(matches) != 1:
            raise RuntimeError('An actual input row does not uniquely identify its source files')
        return matches[0]

    image_row, mask_row = row_for(image_paths), row_for(mask_paths)
    select(inputs._table.cellWidget(image_row, 2), 'image')
    select(inputs._table.cellWidget(mask_row, 2), 'mask')
    select(inputs._table.cellWidget(mask_row, 3), 'cell')
    capture('04_review_detected_roles')
    initial_groups = inputs.get_value()
    if (len(initial_groups) != 2
            or {group['role'] for group in initial_groups} != {'image', 'mask'}
            or next(group for group in initial_groups if group['role'] == 'mask')['object_type'] != 'cell'):
        raise RuntimeError('The visible input roles were not retained')
    select(inputs._table.cellWidget(image_row, 2), 'ignore')
    if len(inputs.get_value()) != 1 or inputs.get_value()[0]['role'] != 'mask':
        raise RuntimeError('Ignore did not exclude the image group from the actual settings')
    capture('04b_image_ignored_not_submitted')
    select(inputs._table.cellWidget(image_row, 2), 'image')
    if inputs.get_value() != initial_groups or model.collect().get('inputs') != initial_groups:
        raise RuntimeError('Restoring the actual role did not restore the original input groups')
    capture('04c_image_role_restored')
    write_json(captures / 'input_roles.json', {
        'groups': initial_groups, 'files': inputs.file_count(),
        'image_ignore_image_restored': True, 'groups_obtained_from_real_pickers': True,
    })

    # The shared model writes each real, visibly exposed list editor. Simple
    # controls use ordinary keyboard/click gestures. Never set ``inputs``
    # through the model, and never pass an override into the Run handler.
    preset = {
        'dst': str(destination), 'layout': 'flat', 'z_handling': 'first',
        'plate_naming': 'index', 'recursive': False, 'overwrite': False,
        'channels': [0], 'png_dims': [0],
        'png_channel_mapping': {'r': 0, 'g': 0, 'b': 0},
        'normalize': False, 'cell_min_size': 0, 'cell_max_size': None,
        'cytoplasm': True, 'timelapse': False, 'resume': False,
        'uninfected': True, 'merge_edge_pathogen_cells': False,
        'n_jobs': 1, 'plot': True, 'save_measurements': True,
        'save_png': True, 'crop_mode': ['cell'], 'png_size': [224, 224],
        'radial_dist': False, 'spatial_measurements': False,
        'object_distance_maxima': False, 'object_distance_intensity': False,
        'object_distances': False, 'calculate_correlation': False,
        'homogeneity': False, 'dry_run': False, 'test_mode': False,
        'preview_only': True,
    }
    observations = []

    def set_setting(key, value, frame=None):
        field = expose(key)
        if model.collect().get(key) != value:
            if isinstance(field, QAbstractButton) and isinstance(value, bool):
                click(field)
            elif isinstance(field, QSpinBox):
                fill(field, value)
            elif isinstance(field, QComboBox):
                select(field, value)
            elif isinstance(field, QLineEdit):
                fill(field, value)
            elif isinstance(field, ChannelMappingWidget):
                for channel, index in value.items():
                    fill(field._boxes[channel], index)
            elif not model.set_value_for_key(key, value):
                raise RuntimeError(f'The real setting editor cannot accept {key}={value!r}')
        settle(0.15)
        actual = model.collect().get(key)
        if actual != value:
            raise RuntimeError(f'The real setting did not retain {key}={value!r}: {actual!r}')
        observations.append({'key': key, 'value': actual,
                             'exposure': 'actual_section_headers_and_scroll_area',
                             'section_path': [section.property('settingsCategorySource')
                                              for section in owning_sections(field)],
                             'visible_keys': [name for name, widget in model._widgets.items()
                                              if widget.isVisible() and not widget.visibleRegion().isEmpty()],
                             'widget': type(field).__name__})
        if frame:
            capture(frame)

    for key, value in preset.items():
        set_setting(key, value, '05_setting_' + key)
    preview_settings = model.collect()
    if preview_settings.get('inputs') != initial_groups:
        raise RuntimeError('Configuring measurement settings changed the input assignments')
    write_json(captures / 'preview_settings.json', preview_settings)
    write_json(captures / 'settings_tour.json', {'observations': observations})

    def console_end():
        for block, _, _ in screen._console._pipeline_console_blocks():
            block.setFocus()
            QTest.keyClick(block, Qt.Key_End, Qt.ControlModifier)
        screen._console.jump_to_the_end()
        settle(0.4)

    def run_job(name, requires_figure):
        if screen._worker_thread_is_running():
            raise RuntimeError('An earlier pipeline is still running')
        outcome = {'finished': False, 'ok': False, 'errors': []}
        lines = []
        starting_figures = screen._figure_queue.count()

        def reject_prompt():
            for box in app.topLevelWidgets():
                if isinstance(box, QMessageBox) and box.isVisible():
                    capture(name + '_unexpected_prompt')
                    outcome['errors'].append(box.windowTitle() + ': ' + box.text())
                    box.reject()

        prompt_timer = QTimer()
        prompt_timer.setInterval(500)
        prompt_timer.timeout.connect(reject_prompt)
        prompt_timer.start()
        try:
            if not screen._btn_run.isVisible() or not screen._btn_run.isEnabled():
                raise RuntimeError('The actual Run button is not usable')
            # Connect immediately, before processing GUI events: a preview
            # may finish quickly, but its queued signals must still be seen.
            QTest.mouseClick(screen._btn_run, Qt.LeftButton)
            worker = getattr(screen, '_worker', None)
            if worker is None:
                raise RuntimeError('The actual Run button did not start a pipeline worker')
            worker.finished.connect(lambda ok: outcome.update(finished=True, ok=bool(ok)))
            worker.error.connect(lambda text: outcome['errors'].append(str(text)))
            worker.line_ready.connect(lambda text: lines.append(str(text)))
            deadline = time.monotonic() + timeout
            next_frame = time.monotonic() + 20
            settle(0.5)
            capture(name + '_running')
            while not outcome['finished'] or screen._worker_thread_is_running():
                if time.monotonic() >= deadline:
                    QTest.mouseClick(screen._btn_stop, Qt.LeftButton)
                    settle(3)
                    raise TimeoutError(f'{name} exceeded the bounded recording time limit')
                if time.monotonic() >= next_frame:
                    capture(name + '_progress')
                    next_frame = time.monotonic() + 30
                settle(0.2)
            settle(2)
        finally:
            prompt_timer.stop()
            write_json(captures / (name + '_outcome.json'), outcome)
            write_json(captures / (name + '_worker_lines.json'), lines)
        blocks = [text for _, _, text in screen._console._pipeline_console_blocks()]
        write_json(captures / (name + '_console.json'), blocks)
        count = screen._figure_queue.count() - starting_figures
        acceptance = assess_pipeline(outcome, blocks, count, requires_figure=requires_figure)
        write_json(captures / (name + '_acceptance.json'), acceptance)
        console_end()
        capture(name + '_finished')
        if not acceptance['accepted']:
            raise RuntimeError(f'{name} failed: ' + '; '.join(acceptance['reasons']))
        return outcome, lines, count

    # Share the real console-navigation controls between the genuinely
    # non-writing preview and the later, explicitly post-write result tour.
    readable_tour = {'figures': [], 'console_markers': []}
    usage = screen._usage_card
    if usage.folder is None:
        raise RuntimeError('The actual System card has no native fold control')
    if not usage.folder.shut:
        click(usage.title_label)
    if not usage.folder.shut or usage.body.isVisible():
        raise RuntimeError('The actual System header did not collapse its body')
    runtime = screen._runtime_splitter
    figure_slot = runtime.indexOf(screen._figures_card)
    console_slot = runtime.indexOf(screen._console_wrap)
    if runtime.count() != 2 or {figure_slot, console_slot} != {0, 1}:
        raise RuntimeError('The actual External Masks figure/console splitter changed')
    console = screen._console

    def console_fold(shut):
        if screen._console_folder.shut != shut:
            click(screen._console_header)
        if screen._console_folder.shut != shut:
            raise RuntimeError('The actual Console heading did not change its fold state')

    def runtime_space(for_figures):
        # These are the same native divider positions a user can drag to.
        # Figures has no fold button in this build; do not manufacture one
        # or change its minimum height just to obtain a larger screenshot.
        available = max(sum(runtime.sizes()), runtime.height())
        sizes = [0, 0]
        sizes[figure_slot] = available if for_figures else 1
        sizes[console_slot] = screen._console_header.sizeHint().height() if for_figures else available
        runtime.setSizes(sizes)
        settle(0.5)

    def readable_console():
        console_fold(False)
        runtime_space(False)
        console.set_split_sizes(1400, 80)
        settle(0.5)
        if not console.isVisible() or console._scroll.viewport().height() < 360:
            capture('readability_error_console')
            raise RuntimeError('The actual console viewport remains too short for the readable tour')

    def show_console_marker(marker, frame, *, require_unwritten=False):
        if require_unwritten and destination.exists():
            raise RuntimeError('The non-writing preview frame requires a nonexistent destination')
        readable_console()
        candidates = [(block, text) for block, _, text in console._pipeline_console_blocks()
                      if marker in text]
        if not candidates:
            raise RuntimeError(f'The actual pipeline console has no summary marker: {marker}')
        block, before_text = candidates[-1]
        if not block.isVisible():
            raise RuntimeError('The actual pipeline summary block is folded or hidden')
        # Find in the existing Qt document rather than counting Python
        # characters (Qt uses UTF-16 cursor offsets). Use the last matching
        # occurrence so each frame shows the most recent actual operation.
        found = block.document().find(marker, 0)
        selected = None
        while not found.isNull():
            selected = found
            found = block.document().find(marker, found)
        if selected is None:
            raise RuntimeError('The rendered console document lost its summary marker')
        block.setFocus()
        block.setTextCursor(selected)
        block.centerCursor()
        settle(0.2)
        target = block.viewport().mapTo(console._holder, block.cursorRect().center())
        scrollbar = console._scroll.verticalScrollBar()
        scrollbar.setValue(max(0, min(scrollbar.maximum(),
                                     target.y() - console._scroll.viewport().height() // 3)))
        settle(0.4)
        if not block.viewport().visibleRegion().contains(block.cursorRect().center()):
            capture('readability_error_summary')
            raise RuntimeError('The actual summary marker did not scroll into view')
        if block.toPlainText() != before_text:
            raise RuntimeError('Navigating the actual console changed its text')
        capture(frame)
        if require_unwritten and destination.exists():
            raise RuntimeError('The destination appeared while recording the non-writing preview')
        readable_tour['console_markers'].append({
            'marker': marker, 'frame': frame, 'runtime_sizes': runtime.sizes(),
            'console_chat_sizes': console.split_sizes(),
            'visible_height': console._scroll.viewport().height(),
            'document_text_unchanged': True, 'destination_exists': destination.exists(),
            'nonwriting_frame': require_unwritten,
        })
        write_json(captures / 'readable_tour.json', readable_tour)

    readable_console()
    preview_outcome, preview_lines, preview_figures = run_job('06_preview', False)
    preview_text = ''.join(preview_lines)
    required_plan = (
        'External masks → Measure project (preview; nothing written)',
        'intensity mappings: 2', 'fields ready: 2', 'intensity channels: 1',
        'mask types: cell', 'cell: 2 paired mask(s), merged plane 1',
        'destination: ' + str(destination),
    )
    if (destination.exists() or preview_figures != 0
            or any(part not in preview_text for part in required_plan)
            or 'Blocking problems:' in preview_text):
        raise RuntimeError('Preview did not produce the exact non-writing two-field plan')
    if model.collect() != preview_settings:
        raise RuntimeError('The non-writing preview changed the settings')
    unchanged()
    show_console_marker('External masks → Measure project (preview; nothing written)',
                        '07_preview_plan_no_project_written', require_unwritten=True)
    write_json(captures / 'preview_evidence.json', {
        'destination_exists': False, 'fields': 2, 'intensity_mappings': 2,
        'intensity_channels': 1, 'cell_mask_pairs': 2, 'merged_mask_plane': 1,
        'source_inputs_unchanged': True, 'worker': preview_outcome,
        'readable_plan_frame': '07_preview_plan_no_project_written',
    })

    set_setting('preview_only', False, '08_preview_only_disabled')
    settings = model.collect()
    expected = dict(preview_settings, preview_only=False)
    if settings != expected or destination.exists():
        raise RuntimeError('Only Preview only may change before the first real import')
    write_json(captures / 'configured_settings.json', settings)
    write_json(captures / 'batch_settings.json', settings)
    outcome, run_lines, _ = run_job('09_measure', True)
    if 'Prepared 2 field(s) in ' + str(destination) not in ''.join(run_lines):
        raise RuntimeError('The actual pipeline did not report the completed two-field project')
    if model.collect() != settings:
        raise RuntimeError('The actual run changed the retained UI settings')
    write_json(captures / 'settings_after_run.json', model.collect())

    queue = screen._figure_queue
    figures = []
    width = sum(screen._body_splitter.sizes())
    screen._body_splitter.setSizes([width // 4, width - width // 4])
    console_fold(True)
    runtime_space(True)
    for index, pixmap in enumerate(queue.all_pixmaps()):
        path = captures / f'external_figure_{index:02d}.png'
        if not pixmap.save(str(path), 'PNG'):
            raise RuntimeError(f'Could not preserve the actual figure {index}')
        figures.append({'image': path.name, 'sha256': _digest(path)})
        queue.show_index(index)
        settle(0.5)
        canvas_area = queue._stack.visibleRegion().boundingRect()
        if canvas_area.height() < 500:
            capture('readability_error_figure')
            raise RuntimeError('The actual figure viewport remains too short for the readable tour')
        readable_tour['figures'].append({
            'index': index, 'visible_width': canvas_area.width(),
            'visible_height': canvas_area.height(), 'runtime_sizes': runtime.sizes(),
            'system_folded': usage.folder.shut, 'console_folded': screen._console_folder.shut,
        })
        capture(f'10_external_figure_{index:02d}')
    write_json(captures / 'batch_figures.json', figures)
    show_console_marker('External masks → Measure project (preview; nothing written)',
                        '11a_external_input_plan')
    show_console_marker('Prepared 2 field(s) in ' + str(destination),
                        '11_external_measurement_summary')
    if model.collect() != settings:
        raise RuntimeError('The readable result tour changed the retained settings')
    write_json(captures / 'readable_tour.json', readable_tour)
    evidence = verify_external_project(destination, records, settings=settings)
    write_json(captures / 'output_evidence.json', evidence)
    if evidence.get('accepted') is not True:
        raise RuntimeError('The independent External Masks output verifier did not accept the project')
    unchanged()
    settle(2)
    blocks = [text for _, _, text in screen._console._pipeline_console_blocks()]
    final_acceptance = assess_pipeline(outcome, blocks, queue.count(), requires_figure=True)
    write_json(captures / 'batch_console_after_tour.json', blocks)
    write_json(captures / 'batch_outcome.json', outcome)
    write_json(captures / 'batch_acceptance.json', final_acceptance)
    if not final_acceptance['accepted']:
        raise RuntimeError('The final result tour exposed an incomplete pipeline: '
                           + '; '.join(final_acceptance['reasons']))
    write_json(captures / 'scientific_acceptance.json', {
        'accepted': True, 'published': False, 'destination': str(destination),
        'source_inputs_unchanged': True, 'input_groups_from_real_pickers': True,
        'nonwriting_preview_verified': True, 'originals_sha256': originals,
        'neutral_names_do_not_preserve_original_wells': True,
        'measurement_csv_imported': False, 'pipeline': final_acceptance,
        'outputs': evidence,
    })
    print('Accepted real External Masks: two fields and 103 independently verified cell objects', flush=True)
