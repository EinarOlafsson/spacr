"""Record bounded native training on verified cell-compartment copies.

Two epochs cross Cellpose's zero-rate first warm-up epoch. Neither training
loss nor an exported checkpoint is presented as held-out segmentation accuracy.
"""
import csv
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

from build_evaluation_example import sha
from capture_acceptance import assess_pipeline


def record_training(app, window, stage, captures, capture, settle, write_json, timeout,
                    *, settings_override=None, source_check_only=False):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton, QMessageBox, QFileDialog, QDialogButtonBox, QLineEdit
    from spacr.qt.widgets.fold_strip import FoldButton
    from spacr.qt.screens.train_cellpose import CellposeWorkbenchScreen
    import torch

    stage=Path(stage);source=stage/'derived/train_cellpose_corrected'
    manifest=json.loads((source/'source_manifest.json').read_text())
    if manifest['image_channel']!=1 or manifest['mask_plane']!=4:
        raise ValueError('The verified cell-compartment preparation is required')
    parent=stage/'cellpose_training_runs';parent.mkdir(exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='REAL-corrected-cell-pairs-',dir=parent))
    originals={}
    for role in ('images','masks'):
        destination=work/'train'/role;destination.mkdir(parents=True)
        for row in manifest['pairs']:
            path=source/'train'/role/row['file'];expected=row['image_sha256' if role=='images' else 'mask_sha256']
            if sha(path)!=expected:raise ValueError('Verified training pair changed')
            shutil.copy2(path,destination/path.name)
            if sha(destination/path.name)!=expected:raise ValueError('Private training copy differs')
            originals[str(path)]=expected
    proof=dict(lesson='19_train_cellpose',accepted=False,private_folder=str(work),
        original_inputs=originals,compartment_manifest=manifest,
        independent_annotation_review=False,held_out_accuracy_validated=False,
        application_modified=False,published=False,requested_epochs=2,
        first_warmup_epoch_learning_rate=0,gpu_memory_fraction_limit=.4)
    write_json(captures/'scientific_acceptance.json',proof)
    if not torch.cuda.is_available():raise ValueError('This bounded recording requires the available CUDA GPU')
    torch.cuda.set_per_process_memory_fraction(.4,0)
    proof['gpu']=torch.cuda.get_device_name(0)

    def click(widget):
        if not widget.isVisible() or not widget.isEnabled() or widget.visibleRegion().isEmpty():
            raise ValueError('The actual Train control is unavailable: '+widget.objectName())
        QTest.mouseClick(widget,Qt.LeftButton,pos=widget.visibleRegion().boundingRect().center());settle(.2)

    try:
        buttons=[b for b in window.findChildren(QAbstractButton) if b.isVisible() and
            (b.property('moduleAppKey')=='make_masks' or b.property('navKey')=='make_masks')]
        click(max(buttons,key=lambda b:b.width()*b.height()))
        host=window._screens['make_masks'];capture('01_make_masks_host')
        folds=[b for b in host.findChildren(FoldButton) if b.isVisible() and b.app_key=='train_cellpose']
        if len(folds)!=1:raise ValueError('The actual Cellpose Workbench fold is not unique')
        click(folds[0])
        panels=[p for p in window.findChildren(CellposeWorkbenchScreen) if p.isVisible()]
        if len(panels)!=1:raise ValueError('The actual Cellpose Workbench did not open')
        panel=panels[0];screen=panel.train_screen;capture('02_actual_train_tab')
        requested=json.loads((source/'tutorial_settings.json').read_text())
        requested.update(src=str(work),model_name='tutorial_cells_two_epoch_demo',n_epochs=2)
        if settings_override:
            requested.update({k:v for k,v in settings_override.items() if k!='src'})
        settings_csv=work/'tutorial_settings.csv'
        with settings_csv.open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['Key','Value'])
            writer.writerows(requested.items())
        errors=[];accepted=[]
        def import_settings():
            dialog=app.activeModalWidget()
            try:
                if not isinstance(dialog,QFileDialog):raise ValueError('The real Import settings picker did not open')
                dialog.accepted.connect(lambda:accepted.append(True))
                dialog.resize(1700,1000)
                field=dialog.findChild(QLineEdit,'fileNameEdit');field.setFocus()
                QTest.keyClick(field,Qt.Key_A,Qt.ControlModifier)
                QTest.keyClicks(field,str(settings_csv));settle(.2)
                capture('02a_real_settings_import_picker')
                click(dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Open))
            except Exception as exc:
                errors.append(str(exc))
                if dialog is not None:dialog.reject()
        watchdog=QTimer(window);watchdog.setSingleShot(True)
        watchdog.timeout.connect(lambda:app.activeModalWidget().reject() if app.activeModalWidget() else None)
        QTimer.singleShot(300,import_settings);watchdog.start(15000)
        click(screen._btn_import);watchdog.stop();settle(.5)
        if errors or not accepted:raise ValueError('Actual settings import failed: '+str(errors))
        imported=screen._settings_model.collect()
        proof['actual_import']={'requested':requested,'collected':imported,
            'rendered_keys':sorted(screen._settings_model._widgets),
            'console':[text for _,_,text in screen._console._pipeline_console_blocks()]}
        capture('02b_actual_imported_values')
        if imported.get('src')!=str(work):
            if any(imported.get(key)!=requested[key] for key in
                   ('n_epochs','batch_size','learning_rate','target_size')):
                raise ValueError('The positive counterpart settings did not import either')
            proof['hold']='Native CSV import applies epochs, batch size, learning rate and target size, but drops required src; no source widget/default exists. The requested output model name also remains new_model. No training started.'
            return
        if source_check_only:
            raise ValueError('The native source now imports; review the route instead of reusing the source-defect workaround')
        for key,value in requested.items():
            if not screen._settings_model.set_value_for_key(key,value):
                raise ValueError('The actual Train form has no '+key)
        actual=screen._settings_model.collect()
        differences={k:[v,actual.get(k)] for k,v in requested.items() if actual.get(k)!=v}
        if differences:raise ValueError('Requested training controls changed: '+str(differences))
        write_json(captures/'configured_settings.json',actual)
        if screen._console_folder.shut:click(screen._console_folder.heading)
        if screen._ai_switch.isChecked():click(screen._ai_switch)
        bar=screen._settings_search
        if bar.modified_only():click(bar._modified)
        if bar.level()!='all':click(bar._disclosure)
        for number,key in enumerate(('src','model_name','n_epochs','batch_size','target_size','learning_rate','augment'),3):
            bar._input.setFocus();QTest.keyClick(bar._input,Qt.Key_A,Qt.ControlModifier)
            QTest.keyClicks(bar._input,key);settle(.2)
            field=screen._settings_model._widgets[key]
            screen._settings_scroll.ensureWidgetVisible(field);settle(.2)
            if key not in bar.visible_keys() or not field.isVisible():
                raise ValueError('The actual training setting is not visible: '+key)
            capture(f'{number:02}_setting_{key}')
        bar._input.setFocus();QTest.keyClick(bar._input,Qt.Key_A,Qt.ControlModifier)
        QTest.keyClick(bar._input,Qt.Key_Backspace);settle(.3)
        if screen._settings_model.collect()!=actual:
            raise ValueError('The settings tour changed training values')
        capture('10_ready_two_epoch_workflow_demo')
        def reject_prompt():
            for box in app.topLevelWidgets():
                if isinstance(box,QMessageBox) and box.isVisible():
                    proof['unexpected_prompt']=box.text();box.reject()
        QTimer.singleShot(1000,reject_prompt)
        started=time.monotonic();click(screen._btn_run);worker=screen._worker
        if worker is None:raise ValueError('The native Run button did not start training')
        outcome=dict(finished=False,ok=False,errors=[])
        worker.finished.connect(lambda ok:outcome.update(finished=True,ok=bool(ok)))
        worker.error.connect(lambda error:outcome['errors'].append(str(error)))
        capture('11_actual_training_running')
        while not outcome['finished'] or screen._worker_thread_is_running():
            if time.monotonic()-started>timeout:
                click(screen._btn_stop);settle(2)
                raise TimeoutError('Bounded native training deadline exceeded')
            settle(.1)
        settle(.5);proof['elapsed_seconds']=time.monotonic()-started
        blocks=[text for _,_,text in screen._console._pipeline_console_blocks()]
        write_json(captures/'training_console.json',blocks)
        write_json(captures/'training_outcome.json',outcome)
        proof['pipeline']=assess_pipeline(outcome,blocks,screen._figure_queue.count(),requires_figure=True)
        screen._console.jump_to_the_end();settle(.3);capture('12_actual_training_outcome')
        proof['checkpoints']=[dict(path=str(p.relative_to(work)),size=p.stat().st_size,sha256=sha(p))
            for p in sorted((work/'models').rglob('*')) if p.is_file()]
        proof['figures']=[]
        for index,pixmap in enumerate(screen._figure_queue.all_pixmaps()):
            path=captures/f'training_figure_{index:02}.png';pixmap.save(str(path),'PNG')
            figure=screen._figure_queue.figure_for(index)
            arrays={f'axis_{a}_image_{i}':np.asarray(im.get_array())
                for a,axis in enumerate(figure.axes if figure is not None else [])
                for i,im in enumerate(axis.images)}
            np.savez_compressed(captures/f'training_figure_{index:02}_arrays.npz',**arrays)
            proof['figures'].append(dict(path=path.name,sha256=sha(path),arrays=len(arrays)))
        if screen._figure_queue.count():
            screen._figure_queue.show_index(0);settle(.3);capture('13_actual_training_pair_figure')
        if outcome['ok'] and proof['checkpoints']:
            QTest.mouseClick(panel._tabs.tabBar(),Qt.LeftButton,pos=panel._tabs.tabBar().tabRect(1).center());settle(.5)
            apply_settings=panel.apply_screen._settings_model.collect()
            proof['apply_handoff']={k:apply_settings.get(k) for k in ('custom_model','src','model_name')}
            capture('14_actual_checkpoint_handoff_to_apply')
        proof['next_gate']='Checkpoint weight-change and saved figure/input verification; not held-out accuracy'
    finally:
        proof['original_inputs_preserved']=all(sha(p)==h for p,h in originals.items())
        proof['private_inputs_preserved']=all(sha(work/'train'/Path(p).parent.name/Path(p).name)==h for p,h in originals.items())
        write_json(captures/'scientific_acceptance.json',proof)
