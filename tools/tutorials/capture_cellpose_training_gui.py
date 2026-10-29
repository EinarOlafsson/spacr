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
from contextlib import nullcontext

import numpy as np

import capture_ui
import capture_ui_drv_b

from build_evaluation_example import sha

MODEL_NAME = 'tutorial_cells'
EPOCHS = 10


def _inputs(example):
    manifest = json.loads((example / 'source_manifest.json').read_text(encoding="utf-8"))
    files = {}
    for pair in manifest['pairs']:
        files[example / 'training' / pair['file']] = pair['image_sha256']
        files[example / 'training/masks' / pair['file']] = pair['mask_sha256']
    for row in manifest['apply_images']:
        files[example / 'apply' / row['file']] = row['image_sha256']
    if any(sha(path) != digest for path, digest in files.items()):
        raise ValueError('The extracted example differs from its manifest')
    return manifest, {str(path): digest for path, digest in files.items()}


def _checkpoint_console_paths(checkpoint, text):
    """Require the actual saved model path in the genuine training console."""
    saved = re.findall(r'Model saved at: (\S+)', text)
    if saved != [str(checkpoint)]:
        raise ValueError(f'Training console must identify the actual saved checkpoint: {checkpoint}; found {saved}')
    return saved


def _tour_sections(capture, screen, keys, click, settle):
    """Use reachable headers once, retaining ancestors of every required field."""
    from spacr.qt.widgets.section import _logical_parent

    def section_of(widget):
        while widget is not None and not (hasattr(widget, 'is_expanded') and hasattr(widget, '_header')):
            widget = _logical_parent(widget)
        return widget

    def ancestors(section):
        chain = []
        while section is not None:
            chain.append(section)
            section = section_of(_logical_parent(section))
        return list(reversed(chain))

    bar = screen._settings_search
    wanted = {}
    for key in keys:
        section = section_of(bar.section_of(key))
        if section is None:
            raise ValueError(f'{key}: no reachable settings section')
        for parent in ancestors(section):
            wanted[id(parent)] = parent
    sections = {}
    for key in bar.indexed_keys():
        section = section_of(bar.section_of(key))
        if section is not None:
            sections[id(section)] = section
    # A collapsed ancestor makes an unused child header unreachable; leave
    # that hidden child alone instead of trying to click it. Close visible
    # children before their parents, then open required parents first.
    for section in sorted(sections.values(), key=lambda s: len(ancestors(s)), reverse=True):
        if id(section) not in wanted and section.is_expanded() and section._header.isVisible():
            capture_ui.scroll_to(capture, screen._settings_scroll, section._header)
            click(section._header); settle(.2)
            if section.is_expanded():
                raise ValueError('Unused settings section did not close: ' + section._header.text())
    for section in wanted.values():
        if not section._header.isVisible() or not section._header.isEnabled():
            raise ValueError('Required settings section is unavailable: ' + section._header.text())
        if not section.is_expanded():
            capture_ui.scroll_to(capture, screen._settings_scroll, section._header)
            click(section._header); settle(.2)
            if not section.is_expanded():
                raise ValueError('Required settings section did not open: ' + section._header.text())


def record(app, window, stage, captures, capture, settle, write_json, timeout, *, dry_run=False):
    import tifffile
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import (QDialogButtonBox, QFileDialog, QLineEdit,
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

    def pick(opener, directory, name, frame, *, scene=True, after=None):
        """Answer the real Qt file dialog that ``opener`` shows.

        Host: with ``scene`` the clip ``frame`` is recorded around the click that
        opens the dialog, the typed path and the answer (``after`` runs inside).
        """
        errors, accepted = [], []
        host = capture_ui.live(capture)

        def answer():
            dialog = app.activeModalWidget()
            try:
                if not isinstance(dialog, QFileDialog):
                    raise ValueError('The real file dialog did not open')
                dialog.accepted.connect(lambda: accepted.append(True))
                capture_ui.dialog_geometry(capture, dialog, (1700, 1050))
                capture_ui.file_dialog_dir(capture, dialog, directory); settle(.6)
                field = dialog.findChild(QLineEdit, 'fileNameEdit')
                typed = str(Path(directory) / name) if host else name
                if host:
                    capture_ui.set_text(capture, field, typed, commit=False)
                else:
                    field.setFocus(); QTest.keyClick(field, Qt.Key_A, Qt.ControlModifier)
                    QTest.keyClicks(field, name)
                settle(.4)
                if frame and not host:
                    capture(frame)
                box = dialog.findChild(QDialogButtonBox)
                button = box.button(QDialogButtonBox.Open) or box.button(QDialogButtonBox.Save)
                capture_ui.accept_file_dialog(capture, dialog, button)
            except Exception as error:
                errors.append(str(error))
                if dialog is not None:
                    dialog.reject()

        watchdog = QTimer(window); watchdog.setSingleShot(True)
        watchdog.timeout.connect(lambda: app.activeModalWidget().reject() if app.activeModalWidget() else None)
        with (capture_ui.clip(capture, frame) if (host and scene and frame) else nullcontext()):
            capture_ui.when_modal(capture, 600, answer); watchdog.start(90000 if host else 20000)
            opener(); watchdog.stop(); settle(.5)
            if errors or not accepted:
                raise ValueError('File dialog failed: ' + repr(errors))
            if after is not None:
                after()

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

    def tour(screen, frame, query, keys, act=None, pre=None):
        bar = screen._settings_search
        with capture_ui.clip(capture, frame):
            if pre is not None:
                pre()
            capture_ui.search_settings(capture, bar, query)
            settle(.4)
            if not query:
                _tour_sections(capture, screen, keys, click, settle)
            settle(.3)
            if act is not None:
                act()
            visible = [key for key in bar.visible_keys()
                       if screen._settings_model._widgets.get(key) is not None
                       and screen._settings_model._widgets[key].isVisible()]
            missing = [key for key in keys if key not in visible]
            if missing:
                raise ValueError(f'{frame}: search {query!r} hides {missing}; shows {visible}')
            capture_ui.scroll_to(capture, screen._settings_scroll, screen._settings_model._widgets[keys[0]]); settle(.3)
            proof['rects'][frame] = dict(query=query, visible=visible, fields=rects(screen, keys))
            capture_ui.point_at_settings(capture, screen, keys, dwell=1.0)

    def console_text(screen):
        return '\n'.join(text for _, _, text in screen._console._pipeline_console_blocks())

    def reject_prompts():
        for box in app.topLevelWidgets():
            if isinstance(box, QMessageBox) and box.isVisible():
                proof.setdefault('unexpected_prompts', []).append(box.text()); box.reject()

    try:
        from capture_external_masks import _enable_console_information
        _enable_console_information(app, window, capture)
        capture_ui.open_module(capture, window, 'make_masks', settle, prefer='sidebar')
        settle(1.5)
        host = window._screens['make_masks']
        capture('make_masks_host')

        # Inspect the training pairs in Make Masks' editor.
        with capture_ui.clip(capture, 'train_pairs'):
            pick(lambda: click(host._btn_open), example, 'training', 'make_masks_open_training', scene=False)
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

            capture_ui.point_at(capture, host._canvas, dwell=1.2)

        # Cellpose Workbench -> Train.
        folds = [b for b in host.findChildren(FoldButton) if b.isVisible() and b.app_key == 'train_cellpose']
        if len(folds) != 1:
            raise ValueError('The Cellpose Workbench fold is not unique')
        with capture_ui.clip(capture, 'home_make_masks'):
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
            capture_ui.point_at(capture, folds[0], dwell=1.2)

        # Source through its real folder button.
        source = screen._settings_model._widgets['src']
        with capture_ui.clip(capture, 'train_source'):
            capture_ui.search_settings(capture, bar, 'src'); settle(.4)
            capture_ui.scroll_to(capture, screen._settings_scroll, source); settle(.3)
            proof['source_button_state'] = dict(field_visible=source.isVisible())
            actions = [a for a in source.actions() if a.toolTip().startswith('Choose folder')]
            buttons = [b for a in actions for b in a.associatedObjects()
                       if isinstance(b, QToolButton)]
            if len(buttons) != 1:
                raise ValueError(f'The Source folder button is not unique: {len(actions)} actions, '
                                 f'{[type(b).__name__ for a in actions for b in a.associatedObjects()]}')
            proof['source_button_state']['button_visible'] = buttons[0].isVisible()
            pick(lambda: click(buttons[0]), example, 'training', 'train_source', scene=False)
            if screen._settings_model.collect().get('src') != str(training):
                raise ValueError('Source did not take the training folder')
            capture_ui.point_at(capture, source, dwell=1.2)
        def enter(key, value):
            def run():
                if not capture_ui_drv_b.fill_setting(capture, screen, key, value):
                    raise ValueError('The Train form has no ' + key)
            return run

        tour(screen, 'train_source_set', 'src', ['src', 'mask_src'])
        tour(screen, 'train_channels', 'channel', ['channels', 'channel_axis'])
        tour(screen, 'train_model_output', 'model', ['base_model', 'model_name', 'save_path'],
             act=enter('model_name', MODEL_NAME))
        tour(screen, 'train_save_path', 'save_path', ['save_path'])
        tour(screen, 'train_schedule', '', ['n_epochs', 'batch_size', 'learning_rate'],
             act=enter('n_epochs', EPOCHS))
        tour(screen, 'train_normalization', '', ['normalize', 'percentiles', 'scale_range', 'min_train_masks'])
        tour(screen, 'train_validation', 'test', ['test_src', 'test_mask_src'])
        tour(screen, 'train_form', '', ['src', 'model_name', 'n_epochs'])
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
        if dry_run:
            proof['dry_run_stopped_before'] = 'Train Run'
            return

        QTimer.singleShot(1500, reject_prompts)
        outcome = dict(finished=False, ok=False, errors=[])
        with capture_ui.clip(capture, 'train_run_console'):
            open_fold(screen, screen._console_folder, 'Console')
            if not screen._console.isVisible() or screen._console.visibleRegion().isEmpty():
                raise ValueError('The actual training console is not visible')
            started = time.monotonic(); click(screen._btn_run)
            # The run now starts in deferred parts, so the worker does not exist (the attribute is not
            # even set) at the moment of the click. Wait for it, with a bound that raises.
            start_deadline = started + 120
            worker = getattr(screen, '_worker', None)
            while worker is None and time.monotonic() < start_deadline:
                settle(.2)
                worker = getattr(screen, '_worker', None)
            if worker is None:
                raise ValueError('Run did not start training')
            worker.finished.connect(lambda ok: outcome.update(finished=True, ok=bool(ok)))
            worker.error.connect(lambda error: outcome['errors'].append(str(error)))
            while not outcome['finished'] or screen._worker_thread_is_running():
                if time.monotonic() - started > timeout:
                    screen._request_cooperative_stop()
                    raise TimeoutError('Training exceeded its bound')
                if 'Training model on' in console_text(screen) and screen._worker_thread_is_running():
                    settle(.3)
                    break
                settle(.1)
            if 'Training model on' not in console_text(screen):
                raise ValueError('The actual training progress did not appear in the console')
            capture_ui.console_to_end(capture, screen._console); settle(.3)
            capture_ui.point_at(capture, screen._console, dwell=1.5)
        while not outcome['finished'] or screen._worker_thread_is_running():
            if time.monotonic() - started > timeout:
                screen._request_cooperative_stop()
                raise TimeoutError('Training exceeded its bound')
            settle(.1)
        settle(1)
        proof['train_elapsed_seconds'] = time.monotonic() - started
        text = console_text(screen)
        write_json(captures / 'training_console.json', text.splitlines())
        write_json(captures / 'training_outcome.json', outcome)
        if not outcome['ok'] or outcome['errors']:
            raise ValueError('Training failed: ' + repr(outcome['errors']) + text[-2000:])
        with capture_ui.clip(capture, 'train_checkpoint'):
            open_fold(screen, screen._console_folder, 'Console')
            capture_ui.console_to_end(capture, screen._console); settle(.6)
            capture_ui.point_at(capture, screen._console, dwell=1.5)
        folder = training / 'models/cellpose_model/models'
        checkpoints = sorted(p for p in folder.glob(MODEL_NAME + '_*') if p.is_file())
        if len(checkpoints) != 1:
            raise ValueError(f'Expected one checkpoint, found {checkpoints}')
        checkpoint = checkpoints[0]
        saved = _checkpoint_console_paths(checkpoint, text)
        proof['checkpoint'] = dict(path=str(checkpoint), bytes=checkpoint.stat().st_size,
                                   sha256=sha(checkpoint), console_path=saved,
                                   console_matches=[str(checkpoint)] == saved)
        proof['training_losses_console'] = re.findall(r'train_loss=([0-9.]+)', text)
        if screen._figure_queue.count():
            with capture_ui.clip(capture, 'train_pair_figure'):
                capture_ui.show_figure(capture, screen._figure_queue, 0); settle(.4)
            screen._figure_queue.all_pixmaps()[0].save(str(captures / 'train_pair_figure.png'), 'PNG')

        # Apply the checkpoint to the separate images.
        apply = panel.apply_screen
        abar = apply._settings_search

        def apply_setup():
            capture_ui.select_tab(capture, panel._tabs, 1)
            settle(.8)
            if panel.active_app_key() != 'cellpose_masks':
                raise ValueError('The Apply tab is not selected')
            if apply._ai_switch.isChecked():
                click(apply._ai_switch)
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
                if not capture_ui_drv_b.fill_setting(capture, apply, key, value):
                    raise ValueError('The Apply form has no ' + key)

        tour(apply, 'apply_checkpoint', '', ['src', 'channels', 'percentiles', 'custom_model', 'fill_in', 'save', 'verbose'],
             pre=apply_setup)
        apply_settings = apply._settings_model.collect()
        proof['apply_settings'] = apply_settings
        write_json(captures / 'apply_settings.json', apply_settings)
        if (apply_settings.get('custom_model') != str(checkpoint) or apply_settings.get('save') is not True
                or apply_settings.get('channels') != [0] or apply_settings.get('percentiles') != [2, 99]
                or apply_settings.get('fill_in') is not False):
            raise ValueError('Apply settings changed unexpectedly')
        tour(apply, 'apply_custom_model', 'model', ['model_name', 'custom_model'])
        tour(apply, 'apply_form', '', ['src'])

        # Live preview on one separate image.
        preview = apply._registry_preview
        with capture_ui.clip(capture, 'apply_preview_picker'):
            for folder_, heading in ((apply._usage_card.folder, apply._usage_card.title_label),
                                     (apply._console_folder, apply._console_header)):
                if not folder_.shut:
                    QTest.mouseClick(heading, Qt.LeftButton); settle(.3)
            QTest.mouseClick(preview.toggle, Qt.LeftButton); settle(.6)
            live = preview.panel
            if not live.isVisible():
                raise ValueError('The live preview is hidden')
            pick(lambda: QTest.mouseClick(live._pick_btn, Qt.LeftButton), apply_folder, 'cell_field_01.tif',
                 'apply_preview_picker', scene=False)
            deadline = time.monotonic() + 60
            while live._image is None:
                if time.monotonic() > deadline:
                    raise TimeoutError('The preview image did not load')
                settle(.1)
            if not np.array_equal(np.squeeze(live._image), tifffile.imread(apply_folder / 'cell_field_01.tif')):
                raise ValueError('The preview loaded different pixels')
        failures = []
        with capture_ui.clip(capture, 'apply_preview'):
            capture_ui.open_live_settings(capture, live); settle(.4)
            dialog = live._live_settings_dialog
            capture_ui.set_number(capture, live._cell_channel, 0); settle(.3)
            for field, value in ((live._lo_pct, 2), (live._hi_pct, 99)):
                capture_ui.set_number(capture, field, value); settle(.3)
            capture_ui.close_dialog(capture, dialog, prefer=('Close', 'Done', 'OK')); settle(.3)
            proof['preview_params'] = live.current_params()
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
            capture_ui.point_at(capture, live, dwell=1.5)

        # Run the whole folder.
        outcome = dict(finished=False, ok=False, errors=[])
        with capture_ui.clip(capture, 'apply_run_started'):
            QTest.mouseClick(preview.toggle, Qt.LeftButton); settle(.5)
            open_fold(apply, apply._console_folder, 'Console')
            open_fold(apply, apply._actions_folder, 'Actions')
            QTimer.singleShot(1500, reject_prompts)
            started = time.monotonic(); click(apply._btn_run)
            start_deadline = started + 120
            worker = getattr(apply, '_worker', None)
            while worker is None and time.monotonic() < start_deadline:
                settle(.2)
                worker = getattr(apply, '_worker', None)
            if worker is None:
                raise ValueError('Apply Run did not start')
            worker.finished.connect(lambda ok: outcome.update(finished=True, ok=bool(ok)))
            worker.error.connect(lambda error: outcome['errors'].append(str(error)))
            settle(1.5)
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
        count = apply._figure_queue.count()
        proof['apply_figures'] = count
        with capture_ui.clip(capture, 'apply_result'):
            capture_ui.console_to_end(capture, apply._console); settle(.3)
            if count:
                capture_ui.show_figure(capture, apply._figure_queue, 0); settle(.5)
        if count > 1:
            with capture_ui.clip(capture, 'apply_result_last'):
                capture_ui.show_figure(capture, apply._figure_queue, count - 1); settle(.5)
        proof['accepted'] = True
    finally:
        proof['original_inputs_preserved'] = all(sha(p) == h for p, h in originals.items())
        if not proof['original_inputs_preserved']:
            proof['accepted'] = False
        write_json(captures / 'scientific_acceptance.json', proof)
    if dry_run:
        proof['accepted'] = proof['original_inputs_preserved']
        write_json(captures / 'scientific_acceptance.json', proof)
