"""Drive the real folded track preview on the completed demo's existing masks."""
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import pandas as pd

from replication_demo import digest
from timelapse_evidence import verify_tracks, verify_worker_passes


def record_preview(app, window, screen, root, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QFileDialog, QLineEdit, QDialogButtonBox, QLabel

    merged = np.stack([np.load(root / f'merged/plate1_A01_1_{i}.npy') for i in range(1, 9)])
    import json
    layout = json.loads((root / 'merged/.spacr_plane_layout.json').read_text())
    if (merged.shape != (8, 256, 256, 4) or merged.dtype != np.uint16 or
            layout['intensity_channels'] != [0, 1] or
            layout['mask_dims'] != {'cell': 2, 'nucleus': 3}):
        raise ValueError('The actual merged demo has a different plane layout')
    csv = root / 'tracks/trackpy_tracks_cell_plate1_A01_1_norm_timelapse.csv'
    records = pd.read_csv(csv).to_dict('records')
    # Saved merged cell masks have already been relabelled by track_id.
    mapped = [dict(row, original_label=row['track_id']) for row in records]
    batch = verify_tracks(merged[..., 2], mapped, .1)
    figure = screen._figure_queue.figure_for(screen._figure_queue.count() - 1)
    plotted = [np.asarray(image.get_array()) for axis in figure.axes for image in axis.images] if figure else []
    figure_checks = dict(image_arrays=len(plotted), exact_merged_first_frame_planes=False)
    if len(plotted) == 4:
        figure_checks['exact_merged_first_frame_planes'] = all(
            np.array_equal(values, merged[0, ..., channel]) for channel, values in enumerate(plotted))
        if figure_checks['exact_merged_first_frame_planes']:
            figure_checks['pixels_checked'] = int(merged[0].size)
    original_files = {str(p): digest(p) for p in root.rglob('*') if p.is_file()}
    preview = captures / 'derived_preview_input'
    image_folder = preview / 'image_sequences/field_A01'
    mask_folder = preview / 'label_sequences/field_A01'
    image_folder.mkdir(parents=True)
    mask_folder.mkdir(parents=True)
    for i in range(8):
        # Exact planes, no normalization or manufactured segmentation.
        np.save(image_folder / f'frame_{i:02}.npy', merged[i, ..., 1])
        np.save(mask_folder / f'frame_{i:02}.npy', merged[i, ..., 2])

    def click(widget):
        if not widget.isVisible() or not widget.isEnabled() or widget.visibleRegion().isEmpty():
            raise ValueError('A real tracking-preview control is unavailable: ' + widget.objectName())
        QTest.mouseClick(widget, Qt.LeftButton, pos=widget.visibleRegion().boundingRect().center())
        settle(.3)

    def fill(widget, value):
        click(widget)
        QTest.keyClick(widget, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(widget, str(value))
        QTest.keyClick(widget, Qt.Key_Tab)
        settle(.3)

    # The scientific figures have already been captured. Clear only the
    # displayed queue through its ordinary button to give preview the room.
    clear = [b for b in screen.findChildren(QLabel) if b.isVisible() and
             b.objectName() == 'FigureQueueClear' and b.text() == 'Clear figures']
    if len(clear) != 1:
        raise ValueError('No unique visible Clear figures button')
    click(clear[0])
    for name in ('_console_folder', '_system_folder'):
        folder = getattr(screen, name, None)
        if folder is not None and not folder.shut:
            click(folder.heading)
    from spacr.qt.widgets.card import Card
    system = [card for card in screen.findChildren(Card) if card.title_label is not None
              and card.title_label.text().lstrip('▾▸▼▶ ') == 'System' and card.folder is not None]
    if len(system) != 1:
        raise ValueError('No unique actual foldable System card')
    if not system[0].folder.shut:
        click(system[0].folder.heading)
    host = screen._folded_previews.get('timelapse')
    if host is None:
        raise ValueError('The folded Timelapse preview is not attached')
    if not host.toggle.isChecked():
        click(host.toggle)
    panel = host.panel
    if not panel.isVisible():
        raise ValueError('The actual Track preview is hidden')
    capture('09_actual_track_preview')

    def picker(button, path, name):
        accepted, errors = [], []
        timer, watchdog = QTimer(window), QTimer(window)
        timer.setSingleShot(True)
        watchdog.setSingleShot(True)
        def handle():
            dialog = app.activeModalWidget()
            try:
                if not isinstance(dialog, QFileDialog):
                    raise ValueError('Expected the actual tracking folder picker')
                dialog.accepted.connect(lambda: accepted.append(True))
                dialog.resize(1400, 950)
                fill(dialog.findChild(QLineEdit, 'fileNameEdit'), path)
                capture(name)
                box = dialog.findChild(QDialogButtonBox)
                buttons = [b for b in box.buttons() if box.buttonRole(b) == QDialogButtonBox.AcceptRole]
                if len(buttons) != 1:
                    raise ValueError('No unique folder acceptance action')
                click(buttons[0])
            except Exception as error:
                errors.append(str(error))
                if dialog is not None:
                    dialog.reject()
        def abort():
            errors.append('The tracking folder picker timed out')
            if app.activeModalWidget() is not None:
                app.activeModalWidget().reject()
        timer.timeout.connect(handle)
        watchdog.timeout.connect(abort)
        timer.start(400)
        watchdog.start(20000)
        try:
            click(button)
        finally:
            timer.stop()
            watchdog.stop()
        if errors or not accepted:
            raise ValueError('; '.join(errors) or 'The real tracking folder was not accepted')
        deadline = time.monotonic() + timeout
        while panel._jobs.is_busy() or panel._jobs.active_jobs():
            if time.monotonic() > deadline:
                raise TimeoutError('The actual preview folder did not finish loading')
            settle(.1)
        settle(.4)

    fill(panel._max_frames, 8)
    fill(panel._channel, 0)
    fill(panel._movie_panel._fields_spin, 1)
    picker(panel._seq_btn, image_folder, '10_actual_image_sequence_folder')
    picker(panel._mask_btn, mask_folder, '11_actual_mask_sequence_folder')
    if panel._mode_box.currentText() != 'iou':
        panel._mode_box.setFocus()
        QTest.keyClick(panel._mode_box, Qt.Key_End)
        QTest.keyClick(panel._mode_box, Qt.Key_Tab)
        settle(.3)
    if panel._mode_box.currentText() != 'iou':
        raise ValueError('The actual preview must use the demonstrated IoU linker')
    if panel._remove_transient.isChecked():
        click(panel._remove_transient)
    if panel._propagate_btn.isChecked():
        click(panel._propagate_btn)
    fill(panel._iou, .1)
    if panel._sequence is None or panel._mask_sequence is None:
        raise ValueError('The native preview did not load both real sequences')
    for i in range(8):
        if (not np.array_equal(panel._sequence.frame(i), merged[i, ..., 1]) or
                not np.array_equal(panel._mask_sequence.frame(i), merged[i, ..., 2])):
            raise ValueError('A preview input differs from its actual saved result')

    def idle():
        deadline = time.monotonic() + timeout
        while panel._worker is not None or panel._movie_jobs.is_busy() or panel._movie_jobs.active_jobs():
            if time.monotonic() > deadline:
                raise TimeoutError('The native tracking preview did not finish')
            settle(.1)
        settle(.5)
        if panel._tracks is None or panel._stats is None:
            raise ValueError('The actual preview returned no tracks')

    def inspect(name):
        idle()
        tracks = panel._tracks.to_dict('records')
        checked = verify_tracks(panel._masks, tracks, panel._iou.value(),
            stats=asdict(panel._stats), minimum_length=panel._min_len.value(),
            displacement_limit=panel._displacement.value())
        checked['actual_status'] = panel._status.text()
        checked['actual_indicator_text'] = panel._stats_label.text()
        if (panel._iou.visibleRegion().boundingRect().height() < panel._iou.fontMetrics().height() + 2
                or panel._max_frames.visibleRegion().boundingRect().height() <
                   panel._max_frames.fontMetrics().height() + 2):
            raise ValueError('Preview controls are clipped; change the actual font preference before recording')
        write_json(captures / (name + '_tracks.json'), tracks)
        capture(name)
        return checked

    raw_results = []
    def observe_worker(*_args):
        worker = panel._worker
        if worker is None:
            raise ValueError('The actual preview did not start the expected worker')
        threshold = worker._request.track['iou_threshold']
        worker.finished_result.connect(lambda result, error: raw_results.append(dict(
            segmented=None if result is None else result.get('segmented'), error=error,
            threshold=threshold)))
    panel._iou.valueChanged.connect(observe_worker)
    QTest.mouseClick(panel._run_btn, Qt.LeftButton)
    observe_worker()
    baseline = inspect('12_actual_linked_preview')
    raw = panel._masks.copy()
    tracked = panel._tracked.copy()
    cache_keys = tuple(panel._mask_cache)
    fill(panel._iou, 1)
    filtered = inspect('13_actual_strict_overlap')
    if filtered['independent_statistics']['n_tracks'] <= baseline['independent_statistics']['n_tracks']:
        raise ValueError('The actual overlap change did not fragment this example')
    fill(panel._iou, .1)
    # Typing ".1" can first emit zero, and the auto-run for the final value
    # can be refused while that intermediate pass is busy. Let it finish,
    # then use the real explicit Re-link button for the displayed value.
    idle()
    QTest.mouseClick(panel._relink_btn, Qt.LeftButton)
    observe_worker()
    restored = inspect('14_actual_overlap_restored')
    if (not np.array_equal(panel._masks, raw) or not np.array_equal(panel._tracked, tracked) or
            tuple(panel._mask_cache) != cache_keys):
        raise ValueError('Restoring overlap did not restore the same cached masks and tracks')
    write_json(captures / 'raw_preview_worker_results.json', raw_results)
    worker_checks = verify_worker_passes(raw_results)
    panel._frame_slider.setFocus()
    QTest.keyClick(panel._frame_slider, Qt.Key_End)
    settle(.4)
    if panel._frame_slider.value() != 7:
        raise ValueError('The actual scrubber did not reach the eighth frame')
    capture('15_actual_final_frame')
    if any(digest(p) != sha for p, sha in original_files.items()):
        raise ValueError('The live preview changed a completed batch artifact')
    proof = dict(batch_tracks=batch, baseline=baseline, strict_overlap=filtered, restored=restored,
        actual_worker_results=raw_results, worker_checks=worker_checks, final_batch_figure=figure_checks,
        source_planes=dict(image=1, mask=2, recorded_layout=layout),
        derived_input_pixels_checked=1048576, input_source='Actual completed synthetic batch merged stacks',
        masks_loaded_not_segmented=True, exact_restoration=True, original_outputs_preserved=True,
        propagated_to_batch=False, biological_validation=False, published=False)
    write_json(captures / 'live_preview_checks.json', proof)
    return proof
