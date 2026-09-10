"""Exercise real Curate controls on exact synthetic Timelapse output copies."""
import csv
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import tifffile

from build_evaluation_example import sha
from curate_evidence import check_mask, check_tracks, painted_disk


def prepare(stage):
    source = stage / 'timelapse_runs/SYNTHETIC-current-demo-os07d4sa'
    mask_source = source / 'merged/plate1_A01_1_1.npy'
    track_source = source / 'tracks/trackpy_tracks_cell_plate1_A01_1_norm_timelapse.csv'
    original = {str(p): sha(p) for p in (mask_source, track_source)}
    raw = np.load(mask_source, allow_pickle=False)
    if raw.shape != (256, 256, 4) or raw.dtype != np.uint16:
        raise ValueError('Expected the exact synthetic tracked four-plane input')
    mask = raw[..., 2].copy()
    if np.unique(mask).tolist() != [0, *range(2, 18)] or np.any(mask[52:77,52:77]):
        raise ValueError('The sixteen teaching labels or empty practice area differ')
    rows = list(csv.DictReader(track_source.open()))
    check_tracks(rows, rows)
    if len(rows) != 128 or {int(r['track_id']) for r in rows} != set(range(2,18)):
        raise ValueError('Expected sixteen eight-frame synthetic tracks')
    parent = stage / 'curate_runs'; parent.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='SYNTHETIC-private-corrections-', dir=parent))
    # The native loader uses Pillow, even though its picker advertises NPY.
    # TIFF preserves this exact integer label plane and is actually readable.
    tifffile.imwrite(work / 'mask.tif', mask)
    shutil.copy2(track_source, work / 'tracks.csv')
    check_mask(tifffile.imread(work / 'mask.tif'), mask)
    if sha(work / 'tracks.csv') != original[str(track_source)]:
        raise ValueError('Private tracks copy differs')
    return work, mask, rows, original


def record_curate(app, window, stage, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import QPoint, Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton, QFileDialog, QLineEdit, QDialogButtonBox
    from spacr.qt.screens.curate import CurateScreen
    from spacr.qt.widgets.fold_strip import FoldButton

    stage = Path(stage)
    work, baseline, tracks, original = prepare(stage)
    mask_path, track_path = work / 'mask.tif', work / 'tracks.csv'
    # stack_from_paths deliberately promotes label layers to int64; the native
    # mask writer stores this small-label example as uint16. Check both exactly.
    baseline = baseline.astype(np.int64)
    mask_initial_sha, track_initial_sha = sha(mask_path), sha(track_path)
    proof = dict(lesson='42_curate', accepted=False, synthetic=True,
        private_folder=str(work), original_inputs=original, app_source_modified=False,
        synthetic_actions_not_biological_corrections=True, published=False, checks={})
    deadline = time.monotonic() + timeout

    def click(widget):
        if time.monotonic() > deadline:
            raise TimeoutError('Bounded Curate recording exceeded its limit')
        if not widget.isVisible() or not widget.isEnabled() or widget.visibleRegion().isEmpty():
            raise ValueError('Actual Curate control unavailable: ' + widget.objectName())
        QTest.mouseClick(widget, Qt.LeftButton, pos=widget.visibleRegion().boundingRect().center())
        settle(.25)

    def fill(widget, value):
        click(widget); QTest.keyClick(widget, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(widget, str(value)); QTest.keyClick(widget, Qt.Key_Tab); settle(.2)

    def picker(button, path, frame):
        errors, accepted = [], []
        timer, watchdog = QTimer(window), QTimer(window)
        timer.setSingleShot(True); watchdog.setSingleShot(True)
        def handle():
            dialog = app.activeModalWidget()
            try:
                if not isinstance(dialog, QFileDialog):
                    raise ValueError('The real Curate file picker did not open')
                dialog.accepted.connect(lambda: accepted.append(True))
                dialog.resize(1500,1000)
                fill(dialog.findChild(QLineEdit,'fileNameEdit'),path); capture(frame)
                box = dialog.findChild(QDialogButtonBox)
                buttons = [b for b in box.buttons() if box.buttonRole(b) == QDialogButtonBox.AcceptRole]
                if len(buttons) != 1: raise ValueError('No unique actual picker accept button')
                click(buttons[0])
            except Exception as error:
                errors.append(str(error))
                if dialog is not None: dialog.reject()
        def abort():
            errors.append('Actual file picker timed out')
            if app.activeModalWidget() is not None: app.activeModalWidget().reject()
        timer.timeout.connect(handle); watchdog.timeout.connect(abort)
        timer.start(300); watchdog.start(15000)
        try: click(button)
        finally: timer.stop(); watchdog.stop()
        if errors or not accepted: raise ValueError('; '.join(errors) or 'No accepted source file')

    def snapshot(name, **details):
        capture(name)
        proof['checks'][name] = details
        write_json(captures / 'scientific_acceptance.json',proof)

    try:
        options = [w for w in window.findChildren(QAbstractButton) if w.isVisible() and
            (w.property('moduleAppKey') == 'make_masks' or w.property('navKey') == 'make_masks')]
        if not options: raise ValueError('No actual Make Masks Home control')
        click(max(options,key=lambda w:w.width()*w.height())); host=window._screens['make_masks']
        capture('01_make_masks_host')
        folds=[w for w in host.findChildren(FoldButton) if w.isVisible() and w.app_key=='curate']
        if len(folds)!=1: raise ValueError('No unique actual Curate fold')
        click(folds[0]); candidates=[w for w in window.findChildren(CurateScreen) if w.isVisible()]
        if len(candidates)!=1: raise ValueError('No actual visible Curate page')
        panel=candidates[0]; capture('02_actual_curate_page')
        picker(panel._browse_mask,mask_path,'03_actual_mask_picker')
        if panel.brush is None:
            snapshot('03b_actual_mask_open_failure',path=panel._mask_edit.text(),
                status=panel.status.text(),viewer_status=panel.viewer.status.text())
            raise ValueError('Actual mask opening failed: '+panel.status.text()+'; '+panel.viewer.status.text())
        brush=panel.brush; layer=brush.session.layer
        snapshot('04_original_sixteen_label_mask',mask=check_mask(layer.data,baseline),
            status=panel.status.text(),ledger_entries=len(brush.session.log.edits))
        if tuple(layer.spacing.scale)!=(1.,1.) or tuple(layer.spacing.translate)!=(0.,0.):
            raise ValueError('The uncalibrated mask must use the unit-pixel oracle')
        fill(brush.radius_spin,8); click(brush.next_label_button)
        if brush.label_spin.value()!=18: raise ValueError('New did not select the next unused label')
        click(brush.paint_button)

        def paint_at(y,x):
            view=panel.viewer.canvas; canvas=view.canvas
            if tuple(canvas.axes)!=('y','x'):raise ValueError('Expected a genuine XY mask view')
            point=QPoint(round((x-canvas.origin[1])/canvas.step[1])+1,
                         round((y-canvas.origin[0])/canvas.step[0])+1)
            if not view.rect().contains(point):raise ValueError('Brush target lies outside the visible canvas')
            centre=(canvas.origin[0]+canvas.step[0]*(point.y()-1),
                    canvas.origin[1]+canvas.step[1]*(point.x()-1))
            expected=painted_disk(layer.data,centre,brush.radius_spin.value(),brush.label_spin.value())
            QTest.mouseClick(view,Qt.LeftButton,pos=point);settle(.3)
            check_mask(layer.data,expected)
            return expected,centre

        painted,centre=paint_at(64,64)
        changed=int(np.count_nonzero(painted!=baseline))
        if changed==0:raise ValueError('The visible brush changed no pixels')
        snapshot('05_practice_disk_not_a_real_object',mask=check_mask(layer.data,painted),
            changed_pixels=changed,actual_centre=list(centre),radius=8,label=18,
            disk_file_unchanged=sha(mask_path)==mask_initial_sha)
        click(brush.undo_button)
        snapshot('06_undo_restores_every_original_pixel',mask=check_mask(layer.data,baseline),
            actions=[e.kind for e in brush.session.log.edits])
        painted,centre=paint_at(64,64)
        saved=[];brush.saved.connect(saved.append);click(brush.save_mask_button)
        if saved!=[str(mask_path)]:raise ValueError('The real Save mask control did not save its private input')
        log_path=Path(str(mask_path)+'.curation.json')
        first_log=json.loads(log_path.read_text())
        snapshot('07_saved_mask_and_ledger',mask=check_mask(tifffile.imread(mask_path),painted.astype(np.uint16)),
            mask_sha256=sha(mask_path),ledger=first_log)
        # Reopening must be observed, not simulated by passing an old session.
        picker(panel._browse_mask,mask_path,'08_reopen_the_actual_saved_mask')
        brush=panel.brush;layer=brush.session.layer
        proof['mask_history_on_reopen']=dict(previous_entries=len(first_log['edits']),
            current_entries=len(brush.session.log.edits),status=panel.status.text())
        snapshot('09_reopened_mask_history',mask=check_mask(layer.data,painted),
            history=proof['mask_history_on_reopen'])
        fill(brush.radius_spin,4);click(brush.next_label_button);click(brush.paint_button)
        second,centre=paint_at(192,32);click(brush.save_mask_button)
        second_log=json.loads(log_path.read_text())
        proof['mask_prior_history_preserved_after_second_save']=(
            second_log['edits'][:len(first_log['edits'])]==first_log['edits'])
        snapshot('10_second_session_saved_history',mask=check_mask(tifffile.imread(mask_path),second.astype(np.uint16)),
            ledger=second_log,prior_history_preserved=proof['mask_prior_history_preserved_after_second_save'])
        click(brush.paint_button)
        picker(panel._browse_tracks,track_path,'11_actual_track_csv_picker')
        tr=panel.tracks
        def table():return tr.session.tracks.to_dict('records')
        def select(ids):
            for number,ident in enumerate(ids):
                items=[tr.track_list.item(i) for i in range(tr.track_list.count())
                    if tr.track_list.item(i).data(Qt.UserRole)==ident]
                if len(items)!=1:raise ValueError('Missing actual track-list identifier')
                item=items[0];tr.track_list.scrollToItem(item)
                QTest.mouseClick(tr.track_list.viewport(),Qt.LeftButton,
                    Qt.ControlModifier if number else Qt.NoModifier,
                    pos=tr.track_list.visualItemRect(item).center());settle(.15)
            if set(tr.selected_tracks())!=set(ids):raise ValueError('Native track selection differs')
        snapshot('12_sixteen_tracks_128_rows',tracks=check_tracks(table(),tracks),status=tr.status.text())
        select([2,3]);click(tr.join_button)
        if 'both present' not in tr.status.text() or len(tr.session.log.edits):
            raise ValueError('The overlapping-track join did not refuse cleanly')
        snapshot('13_overlapping_tracks_cannot_join',tracks=check_tracks(table(),tracks),status=tr.status.text())
        select([2]);fill(tr.frame_spin,4);click(tr.split_button)
        split=[dict(r,track_id=18) if int(r['track_id'])==2 and int(r['frame'])>=4 else dict(r) for r in tracks]
        snapshot('14_split_tail_starts_at_frame_four',tracks=check_tracks(table(),split),status=tr.status.text())
        select([2,18]);click(tr.join_button)
        snapshot('15_join_halves_restores_all_rows',tracks=check_tracks(table(),tracks),status=tr.status.text())
        if sha(track_path)!=track_initial_sha:raise ValueError('Unsaved track edits changed the disk CSV')
        saved=[];tr.saved.connect(saved.append);click(tr.save_button)
        if saved!=[str(track_path)]:raise ValueError('The real Save tracks button did not write')
        saved_log=json.loads(Path(str(track_path)+'.curation.json').read_text())
        snapshot('16_saved_tracks_and_history',tracks=check_tracks(list(csv.DictReader(track_path.open())),tracks),ledger=saved_log)
        select([3]);click(tr.delete_button)
        reduced=[dict(r) for r in tracks if int(r['track_id'])!=3]
        snapshot('17_unsaved_delete_removes_eight_rows',tracks=check_tracks(table(),reduced),status=tr.status.text())
        picker(panel._browse_tracks,track_path,'18_reload_discards_only_unsaved_deletion')
        snapshot('19_saved_tracks_and_two_actions_return',tracks=check_tracks(table(),tracks),
            actions=[e.kind for e in tr.session.log.edits],status=tr.status.text())
        proof['tracks_prior_history_preserved']=len(tr.session.log.edits)==2
        proof['accepted']=proof['mask_prior_history_preserved_after_second_save'] and proof['tracks_prior_history_preserved']
        if not proof['accepted']:
            proof['hold']='The native mask reopening/save path loses prior ledger entries; no tutorial claims complete edit provenance.'
    finally:
        proof['original_inputs_preserved']=all(sha(p)==value for p,value in original.items())
        write_json(captures/'scientific_acceptance.json',proof)
    if not proof['original_inputs_preserved']:raise ValueError('An original synthetic output was changed')
