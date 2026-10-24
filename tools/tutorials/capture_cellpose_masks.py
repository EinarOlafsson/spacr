"""Record the current Workbench Apply tab on preserved real-image copies.

The example is an inference demonstration, not segmentation ground truth or
held-out accuracy validation. No historic, incorrectly paired training model
is selected: the actual stock CPSAM checkpoint is used instead.
"""
import capture_ui
import capture_ui_drv_e as drv
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import tifffile

from build_evaluation_example import sha
from capture_acceptance import assess_pipeline
from cellpose_apply_evidence import minimum_area_labels,require_pixels


def prepare(stage):
    audit = json.loads((Path(__file__).parent / 'evidence' /
        '2026-09-09_model_zoo_provenance_and_assembly_audit.json').read_text(encoding="utf-8"))
    # SPACR_TUTORIAL_MODEL_FIELDS keeps the verified crops inside the stage.
    source = Path(os.environ.get('SPACR_TUTORIAL_MODEL_FIELDS', str(stage.parent / 'derived/cellpose_masks')))
    parent = stage / 'cellpose_apply_runs'
    parent.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='REAL-stock-model-', dir=parent))
    original = {}
    for row in audit['data_provenance']['fields']:
        path = source / row['file']
        expected = row['training_and_benchmark_tiff_sha256']
        if sha(path) != expected:
            raise ValueError('Previously verified real-image source changed')
        data = tifffile.imread(path)
        if data.shape != (512, 512) or data.dtype != np.uint16:
            raise ValueError('Expected the actual one-channel 512-square crop')
        shutil.copy2(path, work / path.name)
        if sha(work / path.name) != expected:
            raise ValueError('Private source copy differs')
        original[str(path)] = expected
    return work, original


@drv.guarded
def record_apply(app, window, stage, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton, QMessageBox
    from spacr.qt.widgets.fold_strip import FoldButton
    from spacr.qt.screens.train_cellpose import CellposeWorkbenchScreen

    stage = Path(stage)
    work, originals = prepare(stage)
    proof = dict(lesson='20_cellpose_masks', accepted=False,
        private_folder=str(work), original_inputs=originals, synthetic=False,
        accuracy_validated=False, ground_truth_used=False, app_source_modified=False,
        published=False, route=['make_masks', 'train_cellpose', 'Apply'])
    write_json(captures/'scientific_acceptance.json', proof)

    def click(widget):
        if not widget.isVisible() or not widget.isEnabled() or widget.visibleRegion().isEmpty():
            raise ValueError('The actual Apply control is unavailable: '+widget.objectName())
        QTest.mouseClick(widget, Qt.LeftButton, pos=widget.visibleRegion().boundingRect().center())
        settle(.25)

    try:
        live = capture_ui.live(capture)
        # Host: the clip opens before the first click on Home and ends once the
        # Apply tab is chosen, which is what its narration describes.
        rec = drv.scene(capture, settle, '02_actual_train_and_apply_tabs')
        buttons = [b for b in window.findChildren(QAbstractButton) if b.isVisible() and
            (b.property('moduleAppKey') == 'make_masks' or b.property('navKey') == 'make_masks')]
        if not buttons: raise ValueError('The actual Make Masks Home tile is absent')
        click(max(buttons, key=lambda b:b.width()*b.height()))
        host = window._screens['make_masks']; drv.snap(capture, '01_make_masks_host')
        folds = [b for b in host.findChildren(FoldButton) if b.isVisible() and b.app_key=='train_cellpose']
        if len(folds)!=1: raise ValueError('The actual Cellpose Workbench fold is not unique')
        click(folds[0])
        panels = [p for p in window.findChildren(CellposeWorkbenchScreen) if p.isVisible()]
        if len(panels)!=1: raise ValueError('The native Cellpose workbench did not open')
        panel=panels[0]
        if not live: capture('02_actual_train_and_apply_tabs')
        QTest.mouseClick(panel._tabs.tabBar(), Qt.LeftButton,
                         pos=panel._tabs.tabBar().tabRect(1).center()); settle(.5)
        screen=panel.apply_screen
        if panel.active_app_key()!='cellpose_masks' or not screen.isVisible():
            raise ValueError('The real Apply tab is not selected')
        if live: rec.stop()
        drv.snap(capture, '03_actual_apply_form')
        requested=dict(src=str(work),model_name='cpsam',custom_model=None,
            channels=[0],normalize=True,percentiles=[2,99],grayscale=False,
            invert=False,remove_background=False,background=100,Signal_to_noise=10,
            diameter=30,CP_prob=0,flow_threshold=.4,batch_size=1,
            save=True,verbose=True,resize=False,rescale=False,resample=False,fill_in=False)
        bar=screen._settings_search

        def fill_keys(keys):
            clear_search(screen._settings_model.collect())
            for key in keys:
                if not capture_ui.fill_setting(capture, screen, key, requested[key]):
                    raise ValueError('The native form has no '+key+' setting')

        def check_form():
            actual=screen._settings_model.collect()
            mismatch={key:[value,actual.get(key)] for key,value in requested.items() if actual.get(key)!=value}
            write_json(captures/'configured_settings.json',actual)
            if mismatch: raise ValueError('The actual form changed requested values: '+str(mismatch))
            return actual

        def tidy_form():
            if screen._console_folder.shut: click(screen._console_folder.heading)
            if screen._ai_switch.isChecked(): click(screen._ai_switch)
            if bar.modified_only(): click(bar._modified)
            if bar.level()!='all': click(bar._disclosure)

        def show_setting(key):
            capture_ui.focus(capture, bar._input); QTest.keyClick(bar._input,Qt.Key_A,Qt.ControlModifier)
            QTest.keyClicks(bar._input,key); settle(.3)
            field=screen._settings_model._widgets[key]
            capture_ui.scroll_to(capture, screen._settings_scroll, field); settle(.2)
            if key not in bar.visible_keys() or not field.isVisible():
                raise ValueError('The real setting is not exposed: '+key)

        def clear_search(actual):
            capture_ui.focus(capture, bar._input); QTest.keyClick(bar._input,Qt.Key_A,Qt.ControlModifier)
            QTest.keyClick(bar._input,Qt.Key_Backspace); settle(.3)
            if screen._settings_model.collect()!=actual:
                raise ValueError('The display-only settings tour changed inference')

        rec14 = drv.Rec(capture, '14_actual_inference_outcome', settle)
        if not live:
            fill_keys(requested)
            actual=check_form()
            tidy_form()
            for n,key in enumerate(('src','model_name','channels','normalize','percentiles','diameter','save','verbose'),4):
                show_setting(key)
                capture(f'{n:02}_setting_{key}')
            clear_search(actual)
            capture('12_ready_to_apply_stock_model')
        else:
            # Host: each setting is set where the narration names it, so the
            # clip of the scene shows the pointer filling it.
            tidy_form()
            rec = drv.scene(capture, settle, '04_setting_src')
            fill_keys(['src']); show_setting('src')
            rec.stop()
            rec = drv.scene(capture, settle, '05_setting_model_name')
            narrated = ['model_name','custom_model','channels','normalize','percentiles',
                        'diameter','flow_threshold','CP_prob']
            fill_keys(narrated); show_setting('model_name')
            rec.stop()
            for key in ('channels','normalize','percentiles','diameter'):
                show_setting(key)
            fill_keys([k for k in requested if k not in narrated and k not in ('src','batch_size','save','verbose')])
            rec14.start()
            fill_keys(['batch_size','save','verbose'])
            show_setting('save'); show_setting('verbose')
            clear_search(check_form())
        def reject_prompt():
            for box in app.topLevelWidgets():
                if isinstance(box,QMessageBox) and box.isVisible():
                    proof['unexpected_prompt']=box.text();drv.snap(capture, '13_unexpected_prompt');capture_ui.reject_dialog(capture, box)
        capture_ui.when_modal(capture, 1000, reject_prompt)
        QTest.mouseClick(screen._btn_run,Qt.LeftButton)
        start_deadline=time.monotonic()+60
        while getattr(screen, '_worker', None) is None and time.monotonic()<start_deadline:
            settle(.2)
        worker=getattr(screen, '_worker', None)
        if worker is None: raise ValueError('The actual Apply Run action did not start')
        outcome=dict(finished=False,ok=False,errors=[])
        worker.finished.connect(lambda ok:outcome.update(finished=True,ok=bool(ok)))
        worker.error.connect(lambda error:outcome['errors'].append(str(error)))
        settle(.3);drv.snap(capture, '13_actual_inference_running')
        deadline=time.monotonic()+timeout
        while not outcome['finished'] or screen._worker_thread_is_running():
            if time.monotonic()>deadline:
                QTest.mouseClick(screen._btn_stop,Qt.LeftButton);settle(3)
                raise TimeoutError('Bounded Apply run exceeded its deadline')
            settle(.1)
        settle(1)
        blocks=[text for _,_,text in screen._console._pipeline_console_blocks()]
        write_json(captures/'batch_console.json',blocks)
        write_json(captures/'batch_outcome.json',outcome)
        proof['pipeline']=assess_pipeline(outcome,blocks,screen._figure_queue.count(),requires_figure=True)
        capture_ui.console_end_all(capture, screen._console)
        if not capture_ui.live(capture): screen._console.jump_to_the_end()
        settle(.3);rec14.stop()
        proof['outputs']=[]
        for image in sorted(work.glob('*.tif')):
            target=work/'masks'/image.name
            if not target.is_file(): raise ValueError('No mask saved for '+image.name)
            mask=tifffile.imread(target)
            if mask.shape!=(512,512) or not np.issubdtype(mask.dtype,np.integer) or mask.min()<0:
                raise ValueError('The output is not a matching nonnegative instance-label image')
            labels,areas=np.unique(mask,return_counts=True)
            positive=labels>0
            proof['outputs'].append(dict(image=image.name,sha256=sha(target),
                dtype=str(mask.dtype),shape=list(mask.shape),objects=int(positive.sum()),
                foreground_pixels=int(areas[positive].sum()),
                label_areas={str(int(k)):int(v) for k,v in zip(labels[positive],areas[positive])}))
        proof['figures']=[]
        for i,pixmap in enumerate(screen._figure_queue.all_pixmaps()):
            path=captures/f'batch_figure_{i:02}.png'
            if not pixmap.save(str(path),'PNG'): raise ValueError('Cannot preserve the native figure')
            proof['figures'].append(dict(path=path.name,sha256=sha(path)))
            figure=screen._figure_queue.figure_for(i)
            if figure is None or len(figure.axes)!=3 or any(len(ax.images)!=1 for ax in figure.axes):
                raise ValueError('The original, overlay and flow figure arrays are unavailable')
            np.savez_compressed(captures/f'batch_figure_{i:02}_arrays.npz',
                original=np.asarray(figure.axes[0].images[0].get_array()),
                overlay=np.asarray(figure.axes[1].images[0].get_array()),
                flow=np.asarray(figure.axes[2].images[0].get_array()))
        if screen._figure_queue.count():
            rec=drv.scene(capture, settle, '15_actual_mask_and_flow_figures')
            capture_ui.show_figure(capture, screen._figure_queue, screen._figure_queue.count()-1)
            settle(.4);rec.stop()
        proof['hold']='Batch observed; preview, pixel-to-figure identity and narration are not yet verified.'
        write_json(captures/'scientific_acceptance.json',proof)
        record_preview(app,window,screen,work,captures,capture,settle,write_json,timeout,proof)
    finally:
        proof['original_inputs_preserved']=all(sha(p)==h for p,h in originals.items())
        proof['private_inputs_preserved']=all(sha(work/Path(p).name)==h for p,h in originals.items())
        write_json(captures/'scientific_acceptance.json',proof)
    if not proof['original_inputs_preserved'] or not proof['private_inputs_preserved']:
        raise ValueError('An input image changed during Apply')


@drv.guarded
def record_preview(app,window,screen,work,captures,capture,settle,write_json,timeout,proof):
    from PySide6.QtCore import Qt,QTimer,QPoint,QPointF
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QFileDialog,QLineEdit,QDialogButtonBox

    # Host: the picker clip opens before the folds close and ends at the picker;
    # the clip of the result opens inside the picker, before it is accepted.
    picker_rec=drv.scene(capture, settle, '17_actual_preview_image_picker')
    result_rec=drv.Rec(capture, '20_actual_preview_result', settle)
    # Use the native fold controls, not hidden widgets or changed size limits.
    # The batch figures stay listed; fold them so the preview has the height.
    for folder,heading in ((screen._usage_card.folder,screen._usage_card.title_label),
                           (screen._console_folder,screen._console_header),
                           (screen._figures_card.folder,screen._figures_card.title_label)):
        if not folder.shut:
            QTest.mouseClick(heading,Qt.LeftButton);settle(.3)
        if not folder.shut:raise ValueError('The real runtime fold did not close')
    # A run folds the Settings pane to give the figures room; open it again
    # with its own splitter handle, as a user clicks the chevron.
    body=screen._body_splitter
    if body.is_collapsed('Settings'):
        QTest.mouseClick(body.handle(1),Qt.LeftButton);settle(.6)
    if body.is_collapsed('Settings'):raise ValueError('The Settings pane did not reopen')
    host=screen._registry_preview
    QTest.mouseClick(host.toggle,Qt.LeftButton);settle(.5)
    panel=host.panel
    proof['preview_seed']=panel.current_params()
    drv.snap(capture, '16_native_preview_before_image')
    if not panel.isVisible():raise ValueError('The real Live preview card is hidden')
    errors,accepted=[],[]
    def pick():
        dialog=app.activeModalWidget()
        try:
            if not isinstance(dialog,QFileDialog):raise ValueError('The actual preview picker did not open')
            dialog.accepted.connect(lambda:accepted.append(True))
            capture_ui.dialog_geometry(capture, dialog, (1500,1000))
            # Browse to the folder, then type the file name: a typed absolute
            # path races the picker's completer.
            capture_ui.file_dialog_dir(capture, dialog, str(work));settle(.4)
            field=dialog.findChild(QLineEdit,'fileNameEdit');capture_ui.focus(capture, field)
            QTest.keyClick(field,Qt.Key_A,Qt.ControlModifier)
            wanted=str(work/'cell_pair_02.tif') if capture_ui.live(capture) else 'cell_pair_02.tif'
            QTest.keyClicks(field,wanted);settle(.2)
            if field.text()!=wanted:raise ValueError(f'The picker holds {field.text()!r}')
            picker_rec.stop();result_rec.start()
            capture_ui.accept_file_dialog(capture, dialog, dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Open))
        except Exception as error:
            errors.append(str(error))
            picker_rec.abort();result_rec.abort()
            if dialog is not None:capture_ui.reject_dialog(capture, dialog)
    timer=QTimer(window);timer.setSingleShot(True)
    timer.timeout.connect(lambda:app.activeModalWidget().reject() if app.activeModalWidget() else None)
    capture_ui.when_modal(capture, 300, pick);timer.start(120000 if capture_ui.live(capture) else 15000)
    QTest.mouseClick(panel._pick_btn,Qt.LeftButton);timer.stop()
    deadline=time.monotonic()+30
    while panel._image is None and time.monotonic()<deadline:settle(.1)
    if errors or not accepted or panel._image is None:raise ValueError('Preview image loading failed: '+str(errors))
    expected=tifffile.imread(work/'cell_pair_02.tif')
    if not np.array_equal(np.squeeze(panel._image),expected):raise ValueError('The native preview loaded different image pixels')
    drv.snap(capture, '18_real_preview_image')
    # The form's percentiles pair is not seeded by the current preview seam.
    # Set the real visible controls explicitly; do not claim automatic parity.
    capture_ui.open_live_settings(capture, panel);settle(.4)
    dialog=panel._live_settings_dialog;capture_ui.dialog_geometry(capture, dialog, (1850,1250))
    capture_ui.dialog_geometry(capture, dialog, None, window.geometry().center()-dialog.rect().center());settle(.3)
    def fill(widget,value):
        if not widget.isVisible() or not widget.isEnabled():raise ValueError('A preview field is hidden')
        capture_ui.focus(capture, widget);QTest.keyClick(widget,Qt.Key_A,Qt.ControlModifier)
        QTest.keyClicks(widget,str(value));QTest.keyClick(widget,Qt.Key_Tab);settle(.3)
    fill(panel._cell_channel,0);fill(panel._lo_pct,2);fill(panel._hi_pct,99)
    proof['preview_explicit_controls']=panel.current_params()
    drv.snap(capture, '19_explicit_preview_channel_and_percentiles')
    capture_ui.close_dialog(capture, dialog);settle(.3)
    preview_errors=[]
    QTest.mouseClick(panel._run_btn,Qt.LeftButton)
    if panel._worker is None:raise ValueError('The actual preview Run did not start')
    panel._worker.finished_masks.connect(lambda masks,error,token:preview_errors.append(error) if error else None)
    deadline=time.monotonic()+timeout
    while not panel._raw_masks or panel._worker.isRunning():
        if preview_errors:raise ValueError('Actual preview failed: '+str(preview_errors))
        if time.monotonic()>deadline:raise TimeoutError('Preview exceeded its bound')
        settle(.1)
    settle(.5);result_rec.stop()
    raw=panel._raw_masks['cell'].copy()
    np.save(captures/'preview_cell.npy',raw,allow_pickle=False)
    proof['preview']=dict(status=panel._status.text(),shape=list(raw.shape),
        objects=int(np.count_nonzero(np.unique(raw))),params=panel.current_params(),
        batch_mask_equal=np.array_equal(raw,tifffile.imread(work/'masks/cell_pair_02.tif')))
    write_json(captures/'scientific_acceptance.json',proof)
    def zoom_detail(name, rec=None):
        if rec is None:rec=drv.scene(capture, settle, name)
        # Deliver actual wheel gestures through the viewport event path.
        # The narrow native card remains narrow; zooming is not a layout fix.
        view=panel._mask_view;viewport=view.viewport()
        point=viewport.rect().center()
        if capture_ui.live(capture):
            capture_ui.wheel_at(capture, viewport, -4, point)     # four real wheel notches up
        else:
          for _ in range(4):
            event=QWheelEvent(QPointF(point),QPointF(viewport.mapToGlobal(point)),
                QPoint(),QPoint(0,120),Qt.NoButton,Qt.NoModifier,Qt.NoScrollPhase,False)
            app.sendEvent(viewport,event)
        settle(.3)
        # Four synthetic 120-unit events give exactly 1.2**4; four real wheel
        # notches zoom by what the compositor delivers -- require a zoom-in
        # that both canvases share.
        expected_ok = (view.scale_factor() > 1.05) if capture_ui.live(capture) else np.isclose(view.scale_factor(), 1.2**4)
        if not expected_ok or not np.isclose(
                panel._src_view.scale_factor(),view.scale_factor()):
            raise ValueError('Actual wheel zoom did not synchronise the two canvases')
        proof.setdefault('native_zoom',[]).append(dict(scene=name,
            factor=view.scale_factor(),viewport_height=viewport.height(),
            viewport_width=viewport.width(),both_canvases_match=True))
        rec.stop()
    zoom_detail('20a_actual_zoomed_preview')
    filter_rec=drv.scene(capture, settle, '22_native_filter_reduces_objects')
    capture_ui.open_live_settings(capture, panel);settle(.3)
    dialog=panel._live_settings_dialog;capture_ui.dialog_geometry(capture, dialog, (1850,1250));settle(.3)
    minimum=panel._compartment_widgets['cell']['min_area']
    original=minimum.value();baseline=panel._masks['cell'].copy();worker=panel._worker
    labels,areas=np.unique(raw[raw>0],return_counts=True)
    cutoff=int(np.median(areas))+1
    drv.snap(capture, '21_native_filter_before');fill(minimum,cutoff)
    expected=minimum_area_labels(baseline,cutoff)
    require_pixels(panel._masks['cell'],expected)
    if panel._worker is not worker or not np.array_equal(panel._raw_masks['cell'],raw):raise ValueError('Filtering reran or changed the raw segmentation')
    proof['filter']=dict(before=int(np.count_nonzero(np.unique(baseline))),
        cutoff=cutoff,after=int(np.count_nonzero(np.unique(expected))),
        checked_pixels=int(raw.size),raw_unchanged=True)
    filter_rec.stop()
    np.save(captures/'preview_cell_filtered.npy',panel._masks['cell'],allow_pickle=False)
    zoomed_rec=drv.scene(capture, settle, '22b_actual_zoomed_filtered_preview')
    capture_ui.close_dialog(capture, dialog);settle(.3);drv.snap(capture, '22a_filtered_preview_dialog_closed')
    zoom_detail('22b_actual_zoomed_filtered_preview', zoomed_rec)
    restored_rec=drv.scene(capture, settle, '23b_actual_zoomed_restored_preview')
    capture_ui.open_live_settings(capture, panel);settle(.3)
    # Opening again constructs a new dialog. Closing the old reference leaves
    # the real current dialog covering the restored image.
    dialog=panel._live_settings_dialog
    capture_ui.dialog_geometry(capture, dialog, (1850,1250));settle(.2)
    fill(minimum,original)
    if not np.array_equal(panel._masks['cell'],baseline):raise ValueError('Restoring the filter did not restore every pixel')
    proof['filter']['restored']=True
    drv.snap(capture, '23_native_filter_restored');capture_ui.close_dialog(capture, dialog);settle(.3)
    if panel._live_settings_dialog is not None and panel._live_settings_dialog.isVisible():
        raise ValueError('The current Live settings dialog did not close')
    np.save(captures/'preview_cell_restored.npy',panel._masks['cell'],allow_pickle=False)
    drv.snap(capture, '23a_restored_preview_dialog_closed')
    zoom_detail('23b_actual_zoomed_restored_preview', restored_rec)
    proof['hold']='Native batch and preview recorded; independent reference/figure and narration checks remain.'
