"""Record the current Cellpose Workbench: train on the six example pairs, then apply.

The stage must contain the extracted ``Cellpose_training_images_masks`` example
(``training/`` with ``masks/`` and the separate ``apply/`` images). Every step
uses the visible controls: Make Masks' Open folder, the Source folder button,
the Train Run button, the Apply tab, its live preview and its Run button. The
recording checks the input bytes before and after, and reads the checkpoint
the run wrote. A training run of a few epochs demonstrates the workflow; it
is not an accuracy evaluation.
"""
import json
from pathlib import Path
import re
import time

import numpy as np

from build_evaluation_example import sha

MODEL_NAME = 'tutorial_cells'
EPOCHS = 10


def _inputs(example):
    manifest = json.loads((example / 'source_manifest.json').read_text())
    files = {}
    for pair in manifest['pairs']:
        files[example / 'training' / pair['file']] = pair['image_sha256']
        files[example / 'training/masks' / pair['file']] = pair['mask_sha256']
    for row in manifest['apply_images']:
        files[example / 'apply' / row['file']] = row['image_sha256']
    if any(sha(path) != digest for path, digest in files.items()):
        raise ValueError('The extracted example differs from its manifest')
    return manifest, {str(path): digest for path, digest in files.items()}


def record(app, window, stage, captures, capture, settle, write_json, timeout, *, dry_run=False):
    import tifffile
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import (QAbstractButton, QDialogButtonBox, QFileDialog, QLineEdit,
                                   QMessageBox, QToolButton)
    from spacr.qt.widgets.fold_strip import FoldButton
    from spacr.qt.screens.train_cellpose import CellposeWorkbenchScreen
    from capture_geometry import capture_rect

    stage = Path(stage)
    example = stage / 'Cellpose_training_images_masks'
    training, apply_folder = example / 'training', example / 'apply'
    manifest, originals = _inputs(example)
    proof = dict(lesson='19_train_cellpose', accepted=False, dry_run=dry_run,
                 example_manifest=manifest, original_inputs=originals,
                 held_out_accuracy_validated=False, app_source_modified=False, rects={})
    write_json(captures / 'scientific_acceptance.json', proof)
    if not dry_run:
        import torch
        if not torch.cuda.is_available():
            raise ValueError('This recording trains on the available CUDA GPU')
        torch.cuda.set_per_process_memory_fraction(.5, 0)
        proof['gpu'] = torch.cuda.get_device_name(0)

    def click(widget):
        if not widget.isVisible() or not widget.isEnabled() or widget.visibleRegion().isEmpty():
            raise ValueError('Unavailable control: ' + (widget.objectName() or type(widget).__name__))
        QTest.mouseClick(widget, Qt.LeftButton, pos=widget.visibleRegion().boundingRect().center())
        settle(.25)

    def pick(opener, directory, name, frame):
        """Answer the real Qt file dialog that ``opener`` shows."""
        errors, accepted = [], []

        def answer():
            dialog = app.activeModalWidget()
            try:
                if not isinstance(dialog, QFileDialog):
                    raise ValueError('The real file dialog did not open')
                dialog.accepted.connect(lambda: accepted.append(True))
                dialog.resize(1700, 1050)
                dialog.setDirectory(str(directory)); settle(.6)
                field = dialog.findChild(QLineEdit, 'fileNameEdit')
                field.setFocus(); QTest.keyClick(field, Qt.Key_A, Qt.ControlModifier)
                QTest.keyClicks(field, name); settle(.4)
                if frame:
                    capture(frame)
                box = dialog.findChild(QDialogButtonBox)
                button = box.button(QDialogButtonBox.Open) or box.button(QDialogButtonBox.Save)
                QTest.mouseClick(button, Qt.LeftButton)
            except Exception as error:
                errors.append(str(error))
                if dialog is not None:
                    dialog.reject()

        watchdog = QTimer(window); watchdog.setSingleShot(True)
        watchdog.timeout.connect(lambda: app.activeModalWidget().reject() if app.activeModalWidget() else None)
        QTimer.singleShot(600, answer); watchdog.start(20000)
        opener(); watchdog.stop(); settle(.5)
        if errors or not accepted:
            raise ValueError('File dialog failed: ' + repr(errors))

    def rects(screen, keys):
        out = {}
        for key in keys:
            field = screen._settings_model._widgets[key]
            out[key] = capture_rect(field.parentWidget() or field, window)
        return out

    def open_fold(screen, folder, name):
        if not folder.shut:
            return
        from PySide6.QtWidgets import QLabel
        headings = [folder.heading] if folder.heading.isVisible() else [
            h for h in screen.findChildren(QLabel) if h.isVisible() and h.text().strip() == name]
        for heading in headings:
            QTest.mouseClick(heading, Qt.LeftButton); settle(.4)
            if not folder.shut:
                return
        raise ValueError(f'The {name} fold did not open')

    def foldable(widget):
        while widget is not None and not (hasattr(widget, 'is_expanded') and hasattr(widget, '_header')):
            widget = widget.parentWidget()
        return widget

    def tour(screen, frame, query, keys):
        bar = screen._settings_search
        bar._input.setFocus(); QTest.keyClick(bar._input, Qt.Key_A, Qt.ControlModifier)
        if query:
            QTest.keyClicks(bar._input, query)
        else:
            QTest.keyClick(bar._input, Qt.Key_Backspace)
        settle(.4)
        if not query:
            # An empty filter shows the category headings; open the ones that
            # hold these settings with their own header clicks, shut the rest.
            wanted = {id(foldable(bar.section_of(key))) for key in keys}
            for key in bar.indexed_keys():
                section = foldable(bar.section_of(key))
                if section is not None and section.is_expanded() != (id(section) in wanted):
                    click(section._header); settle(.2)
        settle(.3)
        visible = [key for key in bar.visible_keys()
                   if screen._settings_model._widgets.get(key) is not None
                   and screen._settings_model._widgets[key].isVisible()]
        missing = [key for key in keys if key not in visible]
        if missing:
            raise ValueError(f'{frame}: search {query!r} hides {missing}; shows {visible}')
        screen._settings_scroll.ensureWidgetVisible(screen._settings_model._widgets[keys[0]]); settle(.3)
        proof['rects'][frame] = dict(query=query, visible=visible, fields=rects(screen, keys))
        capture(frame)

    def console_text(screen):
        return '\n'.join(text for _, _, text in screen._console._pipeline_console_blocks())

    def reject_prompts():
        for box in app.topLevelWidgets():
            if isinstance(box, QMessageBox) and box.isVisible():
                proof.setdefault('unexpected_prompts', []).append(box.text()); box.reject()

    try:
        # Home -> Tools -> Make Masks.
        home = window._startup
        tabs = home._tabs
        tools = [i for i in range(tabs.count()) if tabs.tabText(i).replace('&&', '&').split('  (')[0] == 'Tools']
        if len(tools) != 1:
            raise ValueError('Home has no unique Tools tab')
        QTest.mouseClick(tabs.tabBar(), Qt.LeftButton, pos=tabs.tabBar().tabRect(tools[0]).center())
        settle(.6); capture('home_tools')
        tiles = [b for b in window.findChildren(QAbstractButton) if b.isVisible()
                 and (b.property('moduleAppKey') == 'make_masks' or b.property('navKey') == 'make_masks')]
        click(max(tiles, key=lambda b: b.width() * b.height()))
        deadline = time.monotonic() + 60
        while window._screens.get('make_masks') is None:
            if time.monotonic() > deadline:
                raise TimeoutError('Make Masks did not open')
            settle(.1)
        settle(1.5)
        host = window._screens['make_masks']
        capture('make_masks_host')

        # Inspect the training pairs in Make Masks' editor.
        pick(lambda: click(host._btn_open), example, 'training', 'make_masks_open_training')
        deadline = time.monotonic() + 60
        while host._loading or host._canvas.image is None:
            if time.monotonic() > deadline:
                raise TimeoutError('The training pair did not load')
            settle(.1)
        settle(.5)
        click(host._btn_next)
        deadline = time.monotonic() + 30
        expected = tifffile.imread(training / 'masks/cell_pair_02.tif')
        while host._loading or host._canvas.mask is None or not np.array_equal(host._canvas.mask, expected):
            if time.monotonic() > deadline:
                raise TimeoutError('The editor did not show cell_pair_02 and its mask')
            settle(.1)
        settle(.5)
        proof['editor_pair'] = dict(file='cell_pair_02.tif',
                                    objects=int(np.count_nonzero(np.unique(host._canvas.mask))),
                                    image_equal=bool(np.array_equal(np.squeeze(host._canvas.image),
                                                                    tifffile.imread(training / 'cell_pair_02.tif'))))
        capture('train_pairs')

        # Cellpose Workbench -> Train.
        folds = [b for b in host.findChildren(FoldButton) if b.isVisible() and b.app_key == 'train_cellpose']
        if len(folds) != 1:
            raise ValueError('The Cellpose Workbench fold is not unique')
        click(folds[0])
        panels = [p for p in window.findChildren(CellposeWorkbenchScreen) if p.isVisible()]
        if len(panels) != 1:
            raise ValueError('Cellpose Workbench did not open')
        panel = panels[0]; screen = panel.train_screen
        if panel.active_app_key() != 'train_cellpose':
            raise ValueError('The Train tab is not selected')
        if screen._ai_switch.isChecked():
            click(screen._ai_switch)
        bar = screen._settings_search
        if bar.modified_only():
            click(bar._modified)
        if bar.level() != 'all':
            click(bar._disclosure)
        if screen._console_folder.shut:
            click(screen._console_folder.heading)
        settle(.5)
        proof['rects']['home_make_masks'] = dict(fold=capture_rect(folds[0], window))
        capture('home_make_masks')

        # Source through its real folder button.
        source = screen._settings_model._widgets['src']
        bar._input.setFocus(); QTest.keyClicks(bar._input, 'src'); settle(.4)
        screen._settings_scroll.ensureWidgetVisible(source); settle(.3)
        proof['source_button_state'] = dict(field_visible=source.isVisible())
        actions = [a for a in source.actions() if a.toolTip().startswith('Choose folder')]
        buttons = [b for a in actions for b in a.associatedObjects()
                   if isinstance(b, QToolButton)]
        if len(buttons) != 1:
            raise ValueError(f'The Source folder button is not unique: {len(actions)} actions, '
                             f'{[type(b).__name__ for a in actions for b in a.associatedObjects()]}')
        proof['source_button_state']['button_visible'] = buttons[0].isVisible()
        pick(lambda: click(buttons[0]), example, 'training', 'train_source')
        if screen._settings_model.collect().get('src') != str(training):
            raise ValueError('Source did not take the training folder')
        for key, value in (('model_name', MODEL_NAME), ('n_epochs', EPOCHS)):
            if not screen._settings_model.set_value_for_key(key, value):
                raise ValueError('The Train form has no ' + key)
        settings = screen._settings_model.collect()
        expected_settings = dict(src=str(training), model_name=MODEL_NAME, n_epochs=EPOCHS,
                                 base_model='cpsam', batch_size=1, learning_rate=1e-5,
                                 normalize=True, percentiles=[1, 99], scale_range=.5,
                                 min_train_masks=5)
        wrong = {k: [v, settings.get(k)] for k, v in expected_settings.items() if settings.get(k) != v}
        empty = [k for k in ('mask_src', 'test_src', 'test_mask_src', 'save_path', 'channels', 'channel_axis')
                 if settings.get(k) not in (None, '', [])]
        if wrong or empty:
            raise ValueError(f'Unexpected training settings: {wrong} {empty}')
        proof['train_settings'] = settings
        write_json(captures / 'train_settings.json', settings)
        tour(screen, 'train_source_set', 'src', ['src', 'mask_src'])
        tour(screen, 'train_channels', 'channel', ['channels', 'channel_axis'])
        tour(screen, 'train_model_output', 'model', ['base_model', 'model_name', 'save_path'])
        tour(screen, 'train_save_path', 'save_path', ['save_path'])
        tour(screen, 'train_schedule', '', ['n_epochs', 'batch_size', 'learning_rate'])
        tour(screen, 'train_normalization', '', ['normalize', 'percentiles', 'scale_range', 'min_train_masks'])
        tour(screen, 'train_validation', 'test', ['test_src', 'test_mask_src'])
        tour(screen, 'train_form', '', ['src', 'model_name', 'n_epochs'])
        if screen._settings_model.collect() != settings:
            raise ValueError('The settings tour changed training values')
        if dry_run:
            proof['dry_run_stopped_before'] = 'Train Run'
            return

        QTimer.singleShot(1500, reject_prompts)
        started = time.monotonic(); click(screen._btn_run)
        worker = screen._worker
        if worker is None:
            raise ValueError('Run did not start training')
        outcome = dict(finished=False, ok=False, errors=[])
        worker.finished.connect(lambda ok: outcome.update(finished=True, ok=bool(ok)))
        worker.error.connect(lambda error: outcome['errors'].append(str(error)))
        seen = set()
        while not outcome['finished'] or screen._worker_thread_is_running():
            if time.monotonic() - started > timeout:
                screen._request_cooperative_stop()
                raise TimeoutError('Training exceeded its bound')
            text = console_text(screen)
            if 'Training model on' in text and not seen and screen._worker_thread_is_running():
                seen.add('started'); settle(.3); capture('train_run_started')
                open_fold(screen, screen._console_folder, 'Console')
                screen._console.jump_to_the_end(); settle(.3)
                if screen._worker_thread_is_running():
                    capture('train_run_console')
            settle(.1)
        settle(1)
        proof['train_elapsed_seconds'] = time.monotonic() - started
        text = console_text(screen)
        write_json(captures / 'training_console.json', text.splitlines())
        write_json(captures / 'training_outcome.json', outcome)
        if not outcome['ok'] or outcome['errors']:
            raise ValueError('Training failed: ' + repr(outcome['errors']) + text[-2000:])
        open_fold(screen, screen._console_folder, 'Console')
        screen._console.jump_to_the_end(); settle(.6); capture('train_checkpoint')
        folder = training / 'models/cellpose_model/models'
        checkpoints = sorted(p for p in folder.glob(MODEL_NAME + '_*') if p.is_file())
        if len(checkpoints) != 1:
            raise ValueError(f'Expected one checkpoint, found {checkpoints}')
        checkpoint = checkpoints[0]
        saved = re.findall(r'Model saved at: (\S+)', text)
        proof['checkpoint'] = dict(path=str(checkpoint), bytes=checkpoint.stat().st_size,
                                   sha256=sha(checkpoint), console_path=saved,
                                   console_matches=[str(checkpoint)] == saved)
        proof['training_losses_console'] = re.findall(r'train_loss=([0-9.]+)', text)
        if screen._figure_queue.count():
            screen._figure_queue.show_index(0); settle(.4)
            capture('train_pair_figure')
            screen._figure_queue.all_pixmaps()[0].save(str(captures / 'train_pair_figure.png'), 'PNG')

        # Apply the checkpoint to the separate images.
        QTest.mouseClick(panel._tabs.tabBar(), Qt.LeftButton, pos=panel._tabs.tabBar().tabRect(1).center())
        settle(.8)
        apply = panel.apply_screen
        if panel.active_app_key() != 'cellpose_masks':
            raise ValueError('The Apply tab is not selected')
        if apply._ai_switch.isChecked():
            click(apply._ai_switch)
        abar = apply._settings_search
        if abar.modified_only():
            click(abar._modified)
        if abar.level() != 'all':
            click(abar._disclosure)
        carried = apply._settings_model.collect().get('custom_model')
        if carried != str(checkpoint):
            raise ValueError(f'Apply did not pick up the checkpoint: {carried}')
        proof['carry_note'] = panel._carry_note.text()
        # One grayscale channel with explicit 2-99 percentile scaling (the
        # Apply lesson's recipe); save writes masks, verbose shows figures.
        # fill_in stays off: its hole filling relabels touching cells as one
        # object (ndimage.label on the label image), reported to the app owners.
        for key, value in (('src', str(apply_folder)), ('channels', [0]), ('percentiles', [2, 99]),
                           ('fill_in', False), ('save', True), ('verbose', True)):
            if not apply._settings_model.set_value_for_key(key, value):
                raise ValueError('The Apply form has no ' + key)
        apply_settings = apply._settings_model.collect()
        proof['apply_settings'] = apply_settings
        write_json(captures / 'apply_settings.json', apply_settings)
        if (apply_settings.get('custom_model') != str(checkpoint) or apply_settings.get('save') is not True
                or apply_settings.get('channels') != [0] or apply_settings.get('percentiles') != [2, 99]
                or apply_settings.get('fill_in') is not False):
            raise ValueError('Apply settings changed unexpectedly')
        tour(apply, 'apply_checkpoint', '', ['src', 'channels', 'percentiles', 'custom_model', 'fill_in', 'save', 'verbose'])
        tour(apply, 'apply_custom_model', 'model', ['model_name', 'custom_model'])
        tour(apply, 'apply_form', '', ['src'])

        # Live preview on one separate image.
        for folder_, heading in ((apply._usage_card.folder, apply._usage_card.title_label),
                                 (apply._console_folder, apply._console_header)):
            if not folder_.shut:
                QTest.mouseClick(heading, Qt.LeftButton); settle(.3)
        preview = apply._registry_preview
        QTest.mouseClick(preview.toggle, Qt.LeftButton); settle(.6)
        live = preview.panel
        if not live.isVisible():
            raise ValueError('The live preview is hidden')
        pick(lambda: QTest.mouseClick(live._pick_btn, Qt.LeftButton), apply_folder, 'cell_field_01.tif',
             'apply_preview_picker')
        deadline = time.monotonic() + 60
        while live._image is None:
            if time.monotonic() > deadline:
                raise TimeoutError('The preview image did not load')
            settle(.1)
        if not np.array_equal(np.squeeze(live._image), tifffile.imread(apply_folder / 'cell_field_01.tif')):
            raise ValueError('The preview loaded different pixels')
        live.open_live_settings(); settle(.4)
        dialog = live._live_settings_dialog
        field = live._cell_channel
        field.setFocus(); QTest.keyClick(field, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(field, '0'); QTest.keyClick(field, Qt.Key_Tab); settle(.3)
        for field, value in ((live._lo_pct, 2), (live._hi_pct, 99)):
            field.setFocus(); QTest.keyClick(field, Qt.Key_A, Qt.ControlModifier)
            QTest.keyClicks(field, str(value)); QTest.keyClick(field, Qt.Key_Tab); settle(.3)
        dialog.close(); settle(.3)
        proof['preview_params'] = live.current_params()
        failures = []
        QTest.mouseClick(live._run_btn, Qt.LeftButton)
        if live._worker is None:
            raise ValueError('Preview Run did not start')
        live._worker.finished_masks.connect(lambda masks, error, token: failures.append(error) if error else None)
        deadline = time.monotonic() + timeout
        while not live._raw_masks or live._worker.isRunning():
            if failures:
                raise ValueError('Preview failed: ' + repr(failures))
            if time.monotonic() > deadline:
                raise TimeoutError('Preview exceeded its bound')
            settle(.1)
        settle(.8)
        raw = live._raw_masks['cell']
        proof['preview'] = dict(status=live._status.text(), objects=int(np.count_nonzero(np.unique(raw))),
                                params=live.current_params())
        np.save(captures / 'preview_cell.npy', raw, allow_pickle=False)
        capture('apply_preview')

        # Run the whole folder.
        QTest.mouseClick(preview.toggle, Qt.LeftButton); settle(.5)
        open_fold(apply, apply._console_folder, 'Console')
        open_fold(apply, apply._actions_folder, 'Actions')
        QTimer.singleShot(1500, reject_prompts)
        started = time.monotonic(); click(apply._btn_run)
        worker = apply._worker
        if worker is None:
            raise ValueError('Apply Run did not start')
        outcome = dict(finished=False, ok=False, errors=[])
        worker.finished.connect(lambda ok: outcome.update(finished=True, ok=bool(ok)))
        worker.error.connect(lambda error: outcome['errors'].append(str(error)))
        while not outcome['finished'] or apply._worker_thread_is_running():
            if time.monotonic() - started > timeout:
                apply._request_cooperative_stop()
                raise TimeoutError('Apply exceeded its bound')
            settle(.1)
        settle(1)
        proof['apply_elapsed_seconds'] = time.monotonic() - started
        text = console_text(apply)
        write_json(captures / 'apply_console.json', text.splitlines())
        if not outcome['ok'] or outcome['errors']:
            raise ValueError('Apply failed: ' + repr(outcome['errors']) + text[-2000:])
        proof['apply_outputs'] = []
        for image in sorted(apply_folder.glob('*.tif')):
            mask = tifffile.imread(apply_folder / 'masks' / image.name)
            if mask.shape != (512, 512) or not np.issubdtype(mask.dtype, np.integer):
                raise ValueError('Unexpected Apply output for ' + image.name)
            proof['apply_outputs'].append(dict(image=image.name, sha256=sha(apply_folder / 'masks' / image.name),
                                               objects=int(np.count_nonzero(np.unique(mask)))))
        proof['preview']['batch_mask_equal'] = bool(np.array_equal(
            raw, tifffile.imread(apply_folder / 'masks/cell_field_01.tif')))
        apply._console.jump_to_the_end(); settle(.3)
        count = apply._figure_queue.count()
        proof['apply_figures'] = count
        if count:
            apply._figure_queue.show_index(0); settle(.5)
        capture('apply_result')
        if count > 1:
            apply._figure_queue.show_index(count - 1); settle(.5); capture('apply_result_last')
        proof['accepted'] = True
    finally:
        proof['original_inputs_preserved'] = all(sha(p) == h for p, h in originals.items())
        if not proof['original_inputs_preserved']:
            proof['accepted'] = False
        write_json(captures / 'scientific_acceptance.json', proof)
    if dry_run:
        proof['accepted'] = proof['original_inputs_preserved']
        write_json(captures / 'scientific_acceptance.json', proof)
