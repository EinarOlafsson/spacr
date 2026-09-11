"""Record genuine Measure -> AnnData exports on a private downloaded database."""
from pathlib import Path
import sys
import tempfile
import time
from capture_database import prepare_database_copy, require_unchanged_source, _digest
from anndata_evidence import read_source, verify_file


def record_anndata(app, window, stage, captures, capture, settle, write_json, timeout, *, route_only=False):
    from PySide6.QtCore import Qt, QTimer, QPoint
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton, QLineEdit, QComboBox, QCheckBox, QMessageBox
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widgets.fold_strip import FoldButton
    import anndata

    parent=Path(stage)/'anndata_runs';parent.mkdir(exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='native-export-',dir=parent));project=work/'plate1'
    (project/'measurements').mkdir(parents=True)
    source=Path(stage)/'annotate_fresh/example_data/plate1/measurements/measurements.db'
    database=project/'measurements/measurements.db'
    original=prepare_database_copy(source,database,
        expected_sha256='7b18161f0161d39b3ecedf92cfb0ccf9fee2328980da8e43167555a8f6fd27cd')
    reference_source=read_source(database)
    proof=dict(lesson='59_anndata_export',accepted=False,source=original,project=str(project),
               synthetic_measurements=False,app_source_modified=False,published=False,exports=[])
    write_json(Path(captures)/'scientific_acceptance.json',proof)
    deadline=time.monotonic()+timeout;screen=None

    def tick():
        if time.monotonic()>deadline:raise TimeoutError('Bounded AnnData recording expired')

    def click(w):
        tick()
        if not w.isVisible() or not w.isEnabled() or w.visibleRegion().isEmpty():
            raise ValueError('Actual AnnData control unavailable: '+w.objectName())
        pos=QPoint(12,w.rect().center().y()) if isinstance(w,QCheckBox) else w.visibleRegion().boundingRect().center()
        QTest.mouseClick(w,Qt.LeftButton,pos=pos);settle(.2)

    def fill(w,value):
        click(w);QTest.keyClick(w,Qt.Key_A,Qt.ControlModifier)
        if str(value):QTest.keyClicks(w,str(value))
        else:QTest.keyClick(w,Qt.Key_Backspace)
        QTest.keyClick(w,Qt.Key_Tab);settle(.2)

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
        if isinstance(w,QComboBox):
            idx=w.findData(value)
            if idx<0:idx=w.findText(str(value))
            if idx<0:raise ValueError('Actual selector lacks '+str(value))
            click(w);QTest.keyClick(w.view(),Qt.Key_Home)
            for _ in range(idx):QTest.keyClick(w.view(),Qt.Key_Down)
            QTest.keyClick(w.view(),Qt.Key_Return);settle(.2)
        elif isinstance(w,QCheckBox):
            if w.isChecked()!=value:click(w)
        else:
            field=w if isinstance(w,QLineEdit) else w.findChild(QLineEdit)
            if field is None:raise ValueError('No text entry for '+key+' '+type(w).__name__)
            fill(field,value)
        got=screen._settings_model.collect().get(key)
        # Empty optional text is collected as None; run_anndata_export
        # deliberately normalises both spellings to the empty-string default.
        if got!=value and not (value=='' and got is None):
            raise ValueError(f'Visible setting {key}: {got!r} != {value!r}')

    def run(name,single,policy,expected_shape):
        set_value('anndata_single_table',single);set_value('anndata_nan_policy',policy)
        path=project/'results'/f'{name}.h5ad'
        if path.exists():raise FileExistsError(path)
        set_value('anndata_out',str(path));capture(name+'_settings')
        settings=screen._settings_model.collect()
        outcome=dict(finished=False,ok=False,errors=[]);lines=[]
        def reject_prompt():
            for box in app.topLevelWidgets():
                if isinstance(box,QMessageBox) and box.isVisible():
                    outcome['errors'].append(box.windowTitle()+': '+box.text());capture(name+'_prompt');box.reject()
        timer=QTimer(window);timer.timeout.connect(reject_prompt);timer.start(300)
        try:
            if not screen._btn_run.isEnabled():raise ValueError('Run is disabled')
            QTest.mouseClick(screen._btn_run,Qt.LeftButton)
            worker=getattr(screen,'_worker',None)
            if worker is None:raise ValueError('Run did not start a worker')
            worker.finished.connect(lambda ok:outcome.update(finished=True,ok=bool(ok)))
            worker.error.connect(lambda text:outcome['errors'].append(str(text)))
            worker.line_ready.connect(lambda text:lines.append(str(text)))
            settle(.5);capture(name+'_running')
            while not outcome['finished'] or screen._worker_thread_is_running():
                tick();settle(.1)
            settle(1)
        finally:
            timer.stop();timer.deleteLater()
            write_json(Path(captures)/(name+'_worker.json'),dict(outcome=outcome,lines=lines,settings=settings))
        for block,_,_ in screen._console._pipeline_console_blocks():
            block.setFocus();QTest.keyClick(block,Qt.Key_End,Qt.ControlModifier)
        screen._console.jump_to_the_end();settle(.3);capture(name+'_finished')
        if not outcome['finished'] or not outcome['ok'] or outcome['errors']:
            raise ValueError('AnnData GUI export failed: '+str(outcome))
        if not path.is_file():raise ValueError('Successful worker wrote no requested file')
        a=anndata.read_h5ad(path)
        if tuple(a.shape)!=expected_shape:raise ValueError('Unexpected exported shape '+str(a.shape))
        record=dict(name=name,path=str(path),sha256=_digest(path),bytes=path.stat().st_size,
                    shape=list(a.shape),single_table=single,nan_policy=policy,settings=settings,
                    obs_columns=list(a.obs),features=list(a.var_names),obsm=list(a.obsm),layers=list(a.layers),
                    n_missing=int(__import__('numpy').isnan(a.X).sum()),
                    outcome=outcome,independent_matrix_validation=False)
        record['independent_check']=verify_file(path,reference_source,record,database)
        record['independent_matrix_validation']=True
        proof['exports'].append(record)
        write_json(Path(captures)/'anndata_acceptance.json',proof)

    try:
        tiles=[w for w in window.findChildren(QAbstractButton) if w.isVisible() and
               (w.property('moduleAppKey')=='measure' or w.property('navKey')=='measure')]
        if not tiles:raise ValueError('No Home Measure route')
        click(max(tiles,key=lambda w:w.width()*w.height()));settle(.8)
        host=window._screens['measure'];capture('01_measure_host')
        folds=[w for w in host.findChildren(FoldButton) if w.isVisible() and w.app_key=='anndata_export']
        if len(folds)!=1:raise ValueError('No unique Measure AnnData fold')
        click(folds[0]);settle(1)
        screens=[w for w in window.findChildren(AppScreen) if w.app_key=='anndata_export' and w.isVisible()]
        if len(screens)!=1:raise ValueError('Actual AnnData form did not open')
        screen=screens[0]
        splitter=screen._body_splitter;handle=splitter.handle(1);start=handle.rect().center()
        end=start+QPoint(1950-handle.mapTo(window,start).x(),0)
        QTest.mousePress(handle,Qt.LeftButton,pos=start);QTest.mouseMove(handle,end,delay=150)
        QTest.mouseRelease(handle,Qt.LeftButton,pos=end);settle(.3)
        capture('02_current_anndata_form')
        set_value('src',str(project));capture('03_private_downloaded_project')
        for key,value in [('anndata_row_limit',0),('anndata_compute_umap',False),
                          ('anndata_dtype','float32'),('anndata_compression','gzip'),
                          ('anndata_register_artifact',True)]:
            set_value(key,value);capture('04_setting_'+key)
        proof['default_tables']=screen._settings_model.collect().get('anndata_tables')
        if proof['default_tables']!=['cell','cytoplasm','nucleus','pathogen','png_list']:
            raise ValueError('Unexpected requested default tables')
        if route_only:
            proof.update(accepted=True, gui_exports_completed=False,
                         gui_defect_fixed=False, scope='Navigation/settings only; no GUI export requested')
        else:
            run('05_joined_keep','','keep',(2341,1136))
            run('06_cell_keep','cell','keep',(2341,261))
            run('07_cell_mean','cell','mean',(2341,261))
            run('08_cell_drop_features','cell','drop_features',(2341,257))
            run('09_cell_drop_objects','cell','drop_objects',(2334,261))
            run('10_nucleus_keep','nucleus','keep',(2682,341))
            proof['gui_exports_completed']=True
            proof['accepted']=len(proof['exports'])==6 and all(
                r['independent_matrix_validation'] for r in proof['exports'])
    finally:
        if screen is not None and screen._worker_thread_is_running():
            QTest.mouseClick(screen._btn_stop,Qt.LeftButton)
            until=time.monotonic()+20
            while screen._worker_thread_is_running() and time.monotonic()<until:settle(.1)
        # Normal close drains the form's usage/background runners as well as
        # its pipeline. A failure must not destroy a still-running QThread.
        for child in window.findChildren(AppScreen):child.close()
        settle(.5)
        require_unchanged_source(source,original['source_bundle']);proof['original_unchanged']=True
        proof['private_measurements_unchanged']=_digest(database)==original['database_sha256']
        proof['remaining_workers']=bool(screen is not None and (screen._worker_thread_is_running() or screen.active_jobs()))
        write_json(Path(captures)/'anndata_acceptance.json',proof)
        write_json(Path(captures)/'scientific_acceptance.json',proof)
        if sys.exc_info()[0] is not None:
            # capture_refresh's normal window cleanup is after its recorder
            # dispatch, so an exception must close the shell here as well.
            window.close();settle(.5)
    if proof['remaining_workers'] or not proof['private_measurements_unchanged']:
        raise ValueError('AnnData workflow left a worker or changed measurements')
