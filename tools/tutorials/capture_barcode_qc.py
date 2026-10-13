"""Record the actual Map Barcodes -> Barcode QC route on published counts."""
from pathlib import Path
import hashlib
import shutil
import tempfile
import time

from barcode_qc_evidence import read_counts, verify_outputs, check_native_run


def record_barcode_qc(app, window, stage, captures, capture, settle, write_json, timeout, *, saved_plots=False):
    from PySide6.QtCore import Qt, QTimer, QPoint
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import (QAbstractButton, QLineEdit, QComboBox,
                                  QCheckBox, QMessageBox, QFileDialog, QDialogButtonBox)
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widgets.fold_strip import FoldButton
    from spacr.qt.preferences import set_figure_format

    set_figure_format('png')  # Private recording preference, not an application patch.
    original=stage/'example_data/plate_1_unique_combinations.csv'
    parent=stage/'barcode_qc_runs';parent.mkdir(exist_ok=True)
    project=Path(tempfile.mkdtemp(prefix='real-plate1-',dir=parent))
    source=project/original.name;shutil.copy2(original,source)
    def snapshot(path):
        stat=path.stat()
        return dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    before={str(p):snapshot(p) for p in (original,source)}
    oracle=read_counts(source)
    proof=dict(lesson='47_barcode_qc',accepted=False,project=str(project),
        inputs=before,input_rows=oracle['source_rows'],app_source_modified=False,
        target_is_demonstration_not_experiment_design=True,
        omitted_optional_inputs=['qc_data','grna_csv','row_csv','column_csv'],
        provider_contacted=False,published=False)
    write_json(captures/'barcode-qc-workflow.json',proof)
    deadline=time.monotonic()+timeout;screen=None

    def tick():
        if time.monotonic()>deadline:raise TimeoutError('Bounded Barcode QC recording expired')

    def click(w):
        tick()
        if not w.isVisible() or not w.isEnabled() or w.visibleRegion().isEmpty():
            raise ValueError('Actual Barcode QC control unavailable: '+w.objectName())
        pos=QPoint(12,w.rect().center().y()) if isinstance(w,QCheckBox) else w.visibleRegion().boundingRect().center()
        QTest.mouseClick(w,Qt.LeftButton,pos=pos);settle(.2)

    def fill(field,value):
        if field is None:raise ValueError('No visible text field')
        click(field);QTest.keyClick(field,Qt.Key_A,Qt.ControlModifier)
        if str(value):QTest.keyClicks(field,str(value))
        else:QTest.keyClick(field,Qt.Key_Backspace)
        QTest.keyClick(field,Qt.Key_Tab);settle(.2)

    def expose(key):
        w=screen._settings_model._widgets.get(key)
        if w is None:raise ValueError('No actual bound setting: '+key)
        parents=[];p=w.parentWidget();sections={id(s) for s in screen._settings_sections}
        while p is not None and p is not screen:
            if id(p) in sections:parents.append(p)
            p=p.parentWidget()
        for section in reversed(parents):
            if not section.is_expanded():
                screen._settings_scroll.ensureWidgetVisible(section.header());settle(.2)
                click(section.header())
        screen._settings_scroll.ensureWidgetVisible(w);settle(.2)
        if not w.isVisible():raise ValueError('Setting remains hidden: '+key)
        return w

    def set_value(key,value):
        w=expose(key)
        if screen._settings_model.collect().get(key)==value:return
        if isinstance(w,QComboBox):
            index=w.findData(value)
            if index<0:index=w.findText(str(value))
            if index<0:raise ValueError('Actual selector lacks '+str(value))
            click(w);QTest.keyClick(w.view(),Qt.Key_Home)
            for _ in range(index):QTest.keyClick(w.view(),Qt.Key_Down)
            QTest.keyClick(w.view(),Qt.Key_Return);settle(.2)
        elif isinstance(w,QCheckBox):
            if w.isChecked()!=value:click(w)
        else:fill(w if isinstance(w,QLineEdit) else w.findChild(QLineEdit),value)
        got=screen._settings_model.collect().get(key)
        if got!=value:raise ValueError(f'Visible setting {key}: {got!r} != {value!r}')

    def choose_counts():
        w=expose('count_data')
        if w.paths():click(w._clear_button)
        accepted=[];errors=[];timer=QTimer(window);watch=QTimer(window)
        timer.setSingleShot(True);watch.setSingleShot(True)
        def handle():
            dialog=app.activeModalWidget()
            try:
                if not isinstance(dialog,QFileDialog):raise ValueError('Expected genuine count-file picker')
                dialog.accepted.connect(lambda:accepted.append(True));dialog.resize(1500,1000)
                fill(dialog.findChild(QLineEdit,'fileNameEdit'),source);capture('03_choose_published_plate1')
                box=dialog.findChild(QDialogButtonBox)
                buttons=[b for b in box.buttons() if box.buttonRole(b)==QDialogButtonBox.AcceptRole]
                if len(buttons)!=1:raise ValueError('No unique file-picker accept action')
                click(buttons[0])
            except Exception as exc:
                errors.append(str(exc))
                if dialog is not None:dialog.reject()
        def abort():
            errors.append('Count-file picker timed out');dialog=app.activeModalWidget()
            if dialog is not None:dialog.reject()
        timer.timeout.connect(handle);watch.timeout.connect(abort);timer.start(300);watch.start(15000)
        try:click(w._add_files_button)
        finally:timer.stop();watch.stop();timer.deleteLater();watch.deleteLater()
        if errors or not accepted:raise ValueError('; '.join(errors) or 'No file was accepted')
        if screen._settings_model.collect()['count_data']!=[str(source)]:
            raise ValueError('Actual count-data setting does not identify exactly the copied example')
        capture('04_actual_count_input')

    try:
        tiles=[w for w in window.findChildren(QAbstractButton) if w.isVisible() and
               (w.property('moduleAppKey')=='map_barcodes' or w.property('navKey')=='map_barcodes')]
        if not tiles:raise ValueError('No actual Home Map Barcodes route')
        click(max(tiles,key=lambda w:w.width()*w.height()));settle(.8)
        host=window._screens['map_barcodes'];capture('01_map_barcodes_host')
        folds=[w for w in host.findChildren(FoldButton) if w.isVisible() and w.app_key=='barcode_qc']
        if len(folds)!=1:raise ValueError('No unique Barcode QC fold')
        click(folds[0]);settle(.8)
        screens=[w for w in app.allWidgets() if isinstance(w,AppScreen) and w.app_key=='barcode_qc' and w.isVisible()]
        if len(screens)!=1:raise ValueError('Actual Barcode QC form did not open')
        screen=screens[0];capture('02_barcode_qc_fold');proof['defaults']=screen._settings_model.collect()
        handle=screen._body_splitter.handle(1);start=handle.rect().center()
        end=start+QPoint(1950-handle.mapTo(window,start).x(),0)
        QTest.mousePress(handle,Qt.LeftButton,pos=start);QTest.mouseMove(handle,end,delay=150)
        QTest.mouseRelease(handle,Qt.LeftButton,pos=end);settle(.3)
        choose_counts()
        for index,(key,value) in enumerate([
            ('dst',str(project/'results')),('target_grnas_per_well',5),('target_statistic','median'),
            ('min_reads_per_well',0),('starved_read_fraction',.1),('exclude_starved_wells',True),
            ('position_effect_ratio',2.),('sweep_span',4.),('sweep_points',25),
            ('plot',True),('save',True),('verbose',True)]):
            set_value(key,value)
            if key in ('dst','target_grnas_per_well','starved_read_fraction','plot'):
                capture(f'05_{index:02d}_{key}')
        settings=screen._settings_model.collect();proof['settings']=settings
        for key in proof['omitted_optional_inputs']:
            if settings.get(key):raise ValueError('Unvalidated optional input is not empty: '+key)
        proof['live_preview_control_present']=getattr(screen,'_preview_switch',None) is not None
        outcome=dict(finished=False,ok=False,errors=[]);lines=[]
        proof['run']=dict(outcome=outcome,console=lines)
        def reject_prompt():
            for box in app.topLevelWidgets():
                if isinstance(box,QMessageBox) and box.isVisible():
                    outcome['errors'].append(box.windowTitle()+': '+box.text())
                    capture('06_prompt');box.reject()
        timer=QTimer(window);timer.timeout.connect(reject_prompt);timer.start(300)
        try:
            if not screen._btn_run.isEnabled():raise ValueError('Actual Run is disabled')
            QTest.mouseClick(screen._btn_run,Qt.LeftButton);worker=getattr(screen,'_worker',None)
            if worker is None:raise ValueError('Actual Run did not start a worker')
            worker.finished.connect(lambda ok:outcome.update(finished=True,ok=bool(ok)))
            worker.error.connect(lambda text:outcome['errors'].append(str(text)))
            worker.line_ready.connect(lambda text:lines.append(str(text)))
            settle(.5);capture('06_running')
            while not outcome['finished'] or screen._worker_thread_is_running():tick();settle(.1)
            settle(1)
        finally:timer.stop();timer.deleteLater()
        for block,_,_ in screen._console._pipeline_console_blocks():
            block.setFocus();QTest.keyClick(block,Qt.Key_End,Qt.ControlModifier)
        screen._console.jump_to_the_end();settle(.3);capture('07_finished_console')
        queue=screen._figure_queue;figures=[]
        for index,pixmap in enumerate(queue.all_pixmaps()):
            path=captures/f'actual_gui_figure_{index}.png'
            if not pixmap.save(str(path),'PNG'):raise ValueError('Could not preserve actual figure')
            figures.append(path.name)
        if queue.count():queue.show_index(queue.count()-1);settle(.3)
        capture('08_actual_figures_state')
        proof['run'].update(gui_figure_count=queue.count(),figures_card_visible=screen._figures_card.isVisible(),
            figures=figures,settings_errors=[line for line in lines if '[settings] ERROR' in line])
        proof['independent_csv_check']=verify_outputs(project/'results',oracle)
        proof['saved_files']={str(p.relative_to(project)):snapshot(p) for p in sorted(project.rglob('*')) if p.is_file() and p!=source}
        if saved_plots:
            from capture_saved_plots import check_saved_file_run, show_saved_plots
            check_saved_file_run(proof['run'])
            proof['saved_plot_viewer'] = show_saved_plots(app, window, stage, capture, settle,
                [project/'results/barcode_qc.png', project/'results/threshold_sweep.png'])
            proof['scope'] = 'Verified CSV outputs and real external plot viewer; GUI defects remain'
            proof['gui_figure_workflow_accepted'] = False
            proof['settings_preflight_fixed'] = False
            if {str(p.relative_to(project)):snapshot(p) for p in sorted(project.rglob('*'))
                    if p.is_file() and p != source} != proof['saved_files']:
                raise ValueError('Viewing unexpectedly changed saved output files')
        else:
            check_native_run(proof['run'])
        proof['accepted']=True
    finally:
        if screen is not None and screen._worker_thread_is_running():
            QTest.mouseClick(screen._btn_stop,Qt.LeftButton);until=time.monotonic()+20
            while screen._worker_thread_is_running() and time.monotonic()<until:settle(.1)
        for child in window.findChildren(AppScreen):child.close()
        window.close();settle(.5)
        after={str(p):snapshot(p) for p in (original,source)}
        proof['inputs_unchanged']=after==before
        proof['active_jobs_after_close']=screen.active_jobs() if screen is not None else 0
        if not proof['inputs_unchanged'] or proof['active_jobs_after_close']:
            proof['accepted']=False
        write_json(captures/'barcode-qc-workflow.json',proof)
        write_json(captures/'scientific_acceptance.json',proof)
    if not proof['inputs_unchanged'] or proof['active_jobs_after_close']:
        raise ValueError('Barcode QC recording changed its input or left an active job')
