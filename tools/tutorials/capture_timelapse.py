"""Record Mask's Timelapse fold as a practical walkthrough.

First the downloadable SYNTHETIC_timelapse_preview.zip in the real Track
preview (folder pickers, Run preview, overlap changes, explicit Re-link and
the frame scrubber), then a bounded full Mask run on a small synthetic
eight-frame sequence written by spaCR's own generator, imported through the
real Import settings picker. No application code is changed and no result is
injected; the checks below are independent of spaCR's tracking functions.
"""
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np

from capture_acceptance import assess_pipeline
from timelapse_evidence import verify_tracks, verify_worker_passes

REPO = Path(__file__).resolve().parents[2]
ARCHIVE = REPO / 'docs/source/_extra/tutorials/examples/SYNTHETIC_timelapse_preview.zip'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_example(stage):
    """Unpack the hosted archive exactly as a user would, into the stage."""
    root = Path(stage) / 'timelapse_example'
    if root.exists():
        raise FileExistsError('Use a fresh stage for the Timelapse example')
    root.mkdir(parents=True)
    with zipfile.ZipFile(ARCHIVE) as archive:
        for name in archive.namelist():
            if name.startswith('/') or '..' in Path(name).parts:
                raise ValueError('Unsafe archive member')
        archive.extractall(root)
    folder = root / 'SYNTHETIC_timelapse_preview'
    return folder, {'archive_sha256': digest(ARCHIVE), 'archive_bytes': ARCHIVE.stat().st_size,
                    'members': sorted(str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file())}


def record_timelapse(app, window, stage, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import (QAbstractButton, QDialogButtonBox, QFileDialog,
                                   QLabel, QLineEdit, QMessageBox)
    from spacr.qt.widgets.fold_strip import FoldButton
    from spacr.qt.widgets.card import Card
    from capture_geometry import capture_rect

    stage, captures = Path(stage), Path(captures)
    example, archive = extract_example(stage)
    images = example / 'image_sequences/field_A01'
    labels = example / 'label_sequences/field_A01'
    frames = [np.load(images / f'frame_{i:02}.npy') for i in range(8)]
    label_frames = [np.load(labels / f'frame_{i:02}.npy') for i in range(8)]
    proof = dict(lesson='17_timelapse', accepted=False, synthetic=True, example=archive,
                 app_source_modified=False, inputs_injected=False, published=False)
    write_json(captures / 'scientific_acceptance.json', proof)

    def click(widget):
        if not widget.isVisible() or not widget.isEnabled() or widget.visibleRegion().isEmpty():
            raise ValueError('A native Timelapse control is unavailable: %s visible=%s enabled=%s region=%s' % (
                widget.objectName(), widget.isVisible(), widget.isEnabled(),
                widget.visibleRegion().boundingRect()))
        QTest.mouseClick(widget, Qt.LeftButton, pos=widget.visibleRegion().boundingRect().center())
        settle(.3)

    def fill(widget, value):
        click(widget)
        QTest.keyClick(widget, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(widget, str(value))
        QTest.keyClick(widget, Qt.Key_Tab)
        settle(.3)

    def picker(button, path, name, *, accept_file=False, after=None):
        accepted, errors = [], []
        timer, watchdog = QTimer(window), QTimer(window)
        timer.setSingleShot(True)
        watchdog.setSingleShot(True)

        def handle():
            dialog = app.activeModalWidget()
            try:
                if not isinstance(dialog, QFileDialog):
                    raise ValueError('Expected the actual file picker')
                dialog.accepted.connect(lambda: accepted.append(True))
                dialog.resize(1400, 950)
                fill(dialog.findChild(QLineEdit, 'fileNameEdit'), path)
                if name:
                    capture(name)
                box = dialog.findChild(QDialogButtonBox)
                buttons = [b for b in box.buttons() if box.buttonRole(b) == QDialogButtonBox.AcceptRole]
                if len(buttons) != 1:
                    raise ValueError('No unique acceptance action')
                click(buttons[0])
            except Exception as error:
                errors.append(str(error))
                if dialog is not None:
                    dialog.reject()

        def abort():
            errors.append('The file picker timed out')
            if app.activeModalWidget() is not None:
                app.activeModalWidget().reject()
        timer.timeout.connect(handle)
        watchdog.timeout.connect(abort)
        timer.start(400)
        watchdog.start(90000)
        try:
            click(button)
        finally:
            timer.stop()
            watchdog.stop()
        if errors or not accepted:
            raise ValueError('; '.join(errors) or 'The file picker was not accepted')

    # Home -> Mask -> Time.
    home = [w for w in window.findChildren(QAbstractButton) if w.isVisible() and
            (w.property('moduleAppKey') == 'mask' or w.property('navKey') == 'mask')]
    if not home:
        raise ValueError('No native Home -> Mask route')
    click(max(home, key=lambda w: w.width() * w.height()))
    settle(1.5)
    screen = window._screens['mask']
    capture('01_mask_host')
    folds = [w for w in screen.findChildren(FoldButton) if w.isVisible() and w.app_key == 'timelapse']
    if len(folds) != 1:
        raise ValueError('No unique Mask -> Timelapse fold')
    if not folds[0].isChecked():
        click(folds[0])
    settle(1)
    if screen._settings_model.collect().get('timelapse') is not True:
        raise ValueError('The Timelapse fold did not enable tracking')
    proof['fold_rect'] = capture_rect(folds[0], window)
    capture('02_timelapse_fold')

    # A bounded full run on a synthetic sequence, set up through Import settings.
    from spacr.qt.synthetic import generate_timelapse_demo
    runs = stage / 'timelapse_runs'
    runs.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='SYNTHETIC-sequence-', dir=runs))
    layout = generate_timelapse_demo(root)
    inputs = sorted(root.glob('*.tif'))
    if len(inputs) != 16:
        raise ValueError('The synthetic sequence no longer has eight frames of two channels')
    original = {str(p): digest(p) for p in inputs}
    proof['full_run_inputs'] = dict(source=str(root), generator='spacr.qt.synthetic.generate_timelapse_demo',
                                    files=original, settings_csv=str(layout.settings_csv))
    if not screen._btn_import.isVisible():
        raise ValueError('Import settings is not visible')
    picker(screen._btn_import, layout.settings_csv, '05_import_settings')
    settle(1)
    screen = window._screens['mask']
    if not screen.isVisible():
        raise ValueError('The Mask form is not visible after importing settings')
    imported = screen._settings_model.collect()
    if imported.get('src') != str(root) or imported.get('timelapse') is not True:
        raise ValueError('Import settings did not fill Source and Timelapse: ' +
                         repr({k: imported.get(k) for k in ('src', 'timelapse')}))
    requested = dict(plot=True, batch_size=1, keep_original_images=True, timelapse_displacement=50)
    if 'n_jobs' in screen._settings_model._widgets:
        requested['n_jobs'] = 1
    for key, value in requested.items():
        if not screen._settings_model.set_value_for_key(key, value):
            raise ValueError('Missing setting: ' + key)
    configured = screen._settings_model.collect()
    if any(configured.get(k) != v for k, v in requested.items()):
        raise ValueError('A bounded/Plot setting did not stick')
    write_json(captures / 'configured_settings.json', configured)
    bar = screen._settings_search
    if bar.modified_only():
        click(bar._modified)
    if bar.level() != 'all':
        click(bar._disclosure)
    bar.set_query('')
    settle(.5)
    for key in ('src', 'channels', 'timelapse_frame_limits', 'timelapse_objects', 'timelapse_mode',
                'timelapse_displacement', 'keep_original_images', 'plot'):
        field = screen._settings_model._widgets.get(key)
        if field is None:
            continue
        parents, widget = [], field.parentWidget()
        while widget is not None and widget is not screen:
            if callable(getattr(widget, 'is_expanded', None)) and callable(getattr(widget, 'header', None)):
                parents.append(widget)
            widget = widget.parentWidget()
        for section in reversed(parents):
            if not section.is_expanded():
                screen._settings_scroll.ensureWidgetVisible(section.header())
                settle(.2)
                click(section.header())
        screen._settings_scroll.ensureWidgetVisible(field, 50, 200)
        settle(.3)
        if not field.isVisible() or field.visibleRegion().isEmpty():
            raise ValueError('The Timelapse setting is not visible: ' + key)
        proof.setdefault('setting_rectangles', {})[key] = capture_rect(field, window)
        capture('05_setting_' + key)
    if screen._settings_model.collect() != configured:
        raise ValueError('The settings tour changed the configuration')
    if screen._ai_switch.isChecked():
        click(screen._ai_switch)
    outcome = dict(finished=False, ok=False, errors=[])
    guard = QTimer(window)

    def reject_prompt():
        for box in app.topLevelWidgets():
            if isinstance(box, QMessageBox) and box.isVisible():
                outcome['errors'].append(box.windowTitle() + ': ' + box.text())
                capture('06_unexpected_pipeline_prompt')
                box.reject()
    guard.timeout.connect(reject_prompt)
    guard.start(400)
    try:
        click(screen._btn_run)
        worker = screen._worker
        if worker is None:
            raise ValueError('Run did not launch a worker: ' + repr(outcome['errors']))
        worker.finished.connect(lambda ok: outcome.update(finished=True, ok=bool(ok)))
        worker.error.connect(lambda message: outcome['errors'].append(str(message)))
        settle(3)
        capture('06_actual_running')
        deadline = time.monotonic() + timeout
        while not outcome['finished'] or screen._worker_thread_is_running():
            if time.monotonic() > deadline:
                screen._request_cooperative_stop()
                raise TimeoutError('The Timelapse run exceeded its time limit')
            settle(.2)
        settle(1)
    finally:
        guard.stop()
    blocks = [text for _, _, text in screen._console._pipeline_console_blocks()]
    write_json(captures / 'batch_console.json', blocks)
    proof['outcome'] = outcome
    proof['pipeline'] = assess_pipeline(outcome, blocks, screen._figure_queue.count())
    proof['figure_count'] = screen._figure_queue.count()
    proof['outputs'] = [dict(path=str(p), bytes=p.stat().st_size, sha256=digest(p))
                        for p in root.rglob('*') if p.is_file() and p.suffix != '.tif']
    def relocated(path):
        # The pipeline moves each source image into orig/ before stacking.
        path = Path(path)
        return path if path.exists() else path.parent / 'orig' / path.name
    proof['original_files_preserved'] = all(relocated(p).is_file() and digest(relocated(p)) == h
                                            for p, h in original.items())
    for block, _, _ in screen._console._pipeline_console_blocks():
        block.setFocus()
        QTest.keyClick(block, Qt.Key_End, Qt.ControlModifier)
    screen._console.jump_to_the_end()
    settle(.5)
    capture('07_actual_finished')
    write_json(captures / 'scientific_acceptance.json', proof)
    if not proof['pipeline']['accepted']:
        raise ValueError('The full Timelapse run did not complete: ' + repr(proof['pipeline']) +
                         ' ' + repr(outcome['errors'])[:2000])
    for i, pixmap in enumerate(screen._figure_queue.all_pixmaps()):
        pixmap.save(str(captures / f'figure_{i:02}.png'), 'PNG')
        screen._figure_queue.show_index(i)
        settle(.3)
        proof.setdefault('figure_rects', {})[f'08_actual_figure_{i:02}'] = capture_rect(
            screen._figures_card, window)
        capture(f'08_actual_figure_{i:02}')
    merged = sorted((root / 'merged').glob('*.npy'))
    tracks = sorted((root / 'tracks').glob('*.csv'))
    proof['merged_arrays'] = len(merged)
    proof['track_tables'] = [p.name for p in tracks]
    if not merged or not tracks or not proof['original_files_preserved']:
        raise ValueError('The run did not write merged arrays and track tables, or changed inputs')
    import pandas as pd
    table = pd.read_csv(tracks[0])
    proof['track_table_rows'] = len(table)
    proof['track_ids'] = int(table['track_id'].nunique()) if 'track_id' in table else None
    write_json(captures / 'scientific_acceptance.json', proof)

    # Track preview on the downloadable example.
    screen = window._screens['mask']
    clear = [b for b in screen.findChildren(QLabel) if b.isVisible() and
             b.objectName() == 'FigureQueueClear' and b.text() == 'Clear figures']
    if len(clear) == 1:
        click(clear[0])
    host = (getattr(screen, '_folded_previews', None) or {}).get('timelapse')
    if host is None:
        raise ValueError('The folded Timelapse preview is not attached')
    # Figures collapse the settings column and fold Console, System and the
    # actions row (which carries the Track preview switch). Reopen every
    # collapsed pane through its own heading or handle, as a user would.
    from spacr.qt.widgets.collapsible_splitter import EDGE, HEADER, _PaneHandle
    for split in (getattr(screen, '_body_splitter', None), getattr(screen, '_runtime_splitter', None)):
        if split is None or not hasattr(split, 'panes'):
            continue
        for pane in split.panes():
            if not pane.is_collapsed() or pane.name.strip().lower() == 'figures':
                continue
            if pane.mode == HEADER and pane.folder is not None and pane.folder.heading.isVisible():
                click(pane.folder.heading)
            elif pane.mode == EDGE:
                for index in range(1, split.count()):
                    handle = split.handle(index)
                    if isinstance(handle, _PaneHandle) and handle.edge_pane() is pane:
                        QTest.mouseClick(handle, Qt.LeftButton, pos=handle.rect().center())
            settle(.8)
            proof.setdefault('reopened_panes', []).append([pane.name, pane.is_collapsed()])
    if not host.toggle.isVisible():
        # After Import settings rebuilt the form, the Track preview switch
        # is not offered until the Time fold is switched off and on again.
        folds = [w for w in screen.findChildren(FoldButton) if w.isVisible() and w.app_key == 'timelapse']
        if len(folds) != 1 or not folds[0].isChecked():
            raise ValueError('No checked Time fold to re-offer the Track preview')
        click(folds[0])
        settle(.8)
        click(folds[0])
        settle(1)
        proof['preview_reoffered_by_time_fold'] = True
        if screen._settings_model.collect().get('timelapse') is not True:
            raise ValueError('Time did not switch back on')
    if not host.toggle.isChecked():
        click(host.toggle)
    settle(1)
    # Give the preview the room: fold the other runtime panes and the
    # settings column through their own headings and handle.
    for split in (getattr(screen, '_runtime_splitter', None), getattr(screen, '_body_splitter', None)):
        if split is None or not hasattr(split, 'panes'):
            continue
        for pane in split.panes():
            if pane.is_collapsed() or pane.widget is host.card or host.card.isAncestorOf(pane.widget) \
                    or pane.widget.isAncestorOf(host.card):
                continue
            if pane.mode == HEADER and pane.folder is not None and pane.folder.heading.isVisible():
                click(pane.folder.heading)
            elif pane.mode == EDGE:
                for index in range(1, split.count()):
                    handle = split.handle(index)
                    if isinstance(handle, _PaneHandle) and handle.edge_pane() is pane:
                        QTest.mouseClick(handle, Qt.LeftButton, pos=handle.rect().center())
            settle(.6)
            proof.setdefault('folded_for_preview', []).append([pane.name, pane.is_collapsed()])
    settle(1)
    panel = host.panel
    if not panel.isVisible():
        raise ValueError('The Track preview is hidden')
    proof['preview_rect'] = capture_rect(panel, window)
    capture('09_actual_track_preview')

    def loaded():
        deadline = time.monotonic() + timeout
        while panel._jobs.is_busy() or panel._jobs.active_jobs():
            if time.monotonic() > deadline:
                raise TimeoutError('The preview folder did not finish loading')
            settle(.1)
        settle(.4)

    fill(panel._max_frames, 8)
    fill(panel._channel, 0)
    fill(panel._movie_panel._fields_spin, 1)
    picker(panel._seq_btn, images, '10_actual_image_sequence_folder')
    loaded()
    picker(panel._mask_btn, labels, '11_actual_mask_sequence_folder')
    loaded()
    if panel._mode_box.currentText() != 'iou':
        index = panel._mode_box.findText('iou')
        panel._mode_box.setFocus()
        QTest.keyClick(panel._mode_box, Qt.Key_Home)
        for _ in range(index):
            QTest.keyClick(panel._mode_box, Qt.Key_Down)
        QTest.keyClick(panel._mode_box, Qt.Key_Tab)
        settle(.3)
    if panel._mode_box.currentText() != 'iou':
        raise ValueError('The preview must use the IoU linker')
    if panel._remove_transient.isChecked():
        click(panel._remove_transient)
    if panel._propagate_btn.isChecked():
        click(panel._propagate_btn)
    fill(panel._iou, .1)
    if panel._sequence is None or panel._mask_sequence is None:
        raise ValueError('The preview did not load both sequences')
    for i in range(8):
        if (not np.array_equal(panel._sequence.frame(i), frames[i]) or
                not np.array_equal(panel._mask_sequence.frame(i), label_frames[i])):
            raise ValueError('A preview input differs from the downloaded example file')

    def idle():
        deadline = time.monotonic() + timeout
        while panel._worker is not None or panel._movie_jobs.is_busy() or panel._movie_jobs.active_jobs():
            if time.monotonic() > deadline:
                raise TimeoutError('The tracking preview did not finish')
            settle(.1)
        settle(.5)
        if panel._tracks is None or panel._stats is None:
            raise ValueError('The preview returned no tracks')

    def inspect(name):
        idle()
        tracks = panel._tracks.to_dict('records')
        checked = verify_tracks(panel._masks, tracks, panel._iou.value(),
                                stats=asdict(panel._stats), minimum_length=panel._min_len.value(),
                                displacement_limit=panel._displacement.value())
        checked['status'] = panel._status.text()
        checked['indicator_text'] = panel._stats_label.text()
        write_json(captures / (name + '_tracks.json'), tracks)
        proof.setdefault('preview_rects', {})[name] = capture_rect(panel, window)
        capture(name)
        return checked

    raw_results = []

    def observe_worker(*_args):
        worker = panel._worker
        if worker is None:
            return
        threshold = worker._request.track['iou_threshold']
        worker.finished_result.connect(lambda result, error: raw_results.append(dict(
            segmented=None if result is None else result.get('segmented'), error=error,
            threshold=threshold)))
    panel._iou.valueChanged.connect(observe_worker)
    QTest.mouseClick(panel._run_btn, Qt.LeftButton)
    observe_worker()
    baseline = inspect('12_actual_linked_preview')
    raw, tracked = panel._masks.copy(), panel._tracked.copy()
    fill(panel._iou, 1)
    strict = inspect('13_actual_strict_overlap')
    if strict['independent_statistics']['n_tracks'] <= baseline['independent_statistics']['n_tracks']:
        raise ValueError('The strict overlap did not fragment the example')
    fill(panel._iou, .1)
    idle()
    QTest.mouseClick(panel._relink_btn, Qt.LeftButton)
    observe_worker()
    restored = inspect('14_actual_overlap_restored')
    if not np.array_equal(panel._masks, raw) or not np.array_equal(panel._tracked, tracked):
        raise ValueError('Restoring overlap did not restore the same tracks')
    write_json(captures / 'raw_preview_worker_results.json', raw_results)
    worker_checks = verify_worker_passes(raw_results)
    panel._frame_slider.setFocus()
    QTest.keyClick(panel._frame_slider, Qt.Key_End)
    settle(.5)
    if panel._frame_slider.value() != 7:
        raise ValueError('The scrubber did not reach the eighth frame')
    proof.setdefault('preview_rects', {})['15_actual_final_frame'] = capture_rect(panel, window)
    capture('15_actual_final_frame')
    if baseline['independent_statistics']['n_tracks'] != 16:
        raise ValueError('The narrated sixteen tracks are not what the example produced')
    proof['live_preview'] = dict(baseline=baseline, strict_overlap=strict, restored=restored,
                                 worker_checks=worker_checks, masks_loaded_not_segmented=True)
    write_json(captures / 'scientific_acceptance.json', proof)

    proof['accepted'] = True
    proof['scope'] = ('Downloadable preview example with independent track checks; bounded full run '
                      'on a disclosed synthetic sequence; not biological validation')
    write_json(captures / 'scientific_acceptance.json', proof)
