"""Record two genuine Classify -> Activation Maps runs on four real crops."""
from pathlib import Path
import time

from activation_evidence import prepare, reference, verify_saved, preserved, check_native_runs, check_saved_runs


def record_activation(app, window, stage, captures, capture, settle, write_json, timeout, *, saved_plots=False):
    from PySide6.QtCore import Qt, QTimer, QPoint
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton, QLineEdit, QComboBox, QCheckBox, QMessageBox
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.screens.classify import HOST_KEY
    from spacr.qt.widgets.fold_strip import FoldButton
    from spacr.qt.preferences import set_figure_format

    set_figure_format('png')  # Private recording preference; actual save_figure path.
    prepared = prepare(stage)
    proof = dict(lesson='16_activation', accepted=False, inputs=prepared, runs=[],
                 app_source_modified=False, provider_contacted=False, published=False)
    write_json(captures/'activation-workflow.json', proof)
    deadline = time.monotonic()+timeout; screen = None

    def tick():
        if time.monotonic()>deadline: raise TimeoutError('Bounded activation recording expired')

    def click(w):
        tick()
        if not w.isVisible() or not w.isEnabled() or w.visibleRegion().isEmpty():
            raise ValueError('Actual activation control unavailable: '+w.objectName())
        pos = QPoint(12,w.rect().center().y()) if isinstance(w,QCheckBox) else w.visibleRegion().boundingRect().center()
        QTest.mouseClick(w,Qt.LeftButton,pos=pos); settle(.2)

    def expose(key):
        w = screen._settings_model._widgets.get(key)
        if w is None: raise ValueError('No actual bound setting: '+key)
        parents=[]; p=w.parentWidget(); sections={id(s) for s in screen._settings_sections}
        while p is not None and p is not screen:
            if id(p) in sections: parents.append(p)
            p=p.parentWidget()
        for section in reversed(parents):
            if not section.is_expanded():
                screen._settings_scroll.ensureWidgetVisible(section.header()); settle(.2)
                click(section.header())
        screen._settings_scroll.ensureWidgetVisible(w); settle(.2)
        if not w.isVisible(): raise ValueError('Setting remains hidden: '+key)
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
        else:
            field=w if isinstance(w,QLineEdit) else w.findChild(QLineEdit)
            if field is None:raise ValueError('No text entry for '+key+' '+type(w).__name__)
            click(field);QTest.keyClick(field,Qt.Key_A,Qt.ControlModifier)
            if str(value):QTest.keyClicks(field,str(value))
            else:QTest.keyClick(field,Qt.Key_Backspace)
            QTest.keyClick(field,Qt.Key_Tab);settle(.2)
        got=screen._settings_model.collect().get(key)
        if got!=value:raise ValueError(f'Visible setting {key}: {got!r} != {value!r}')

    def run(method,overlay,prefix):
        set_value('cam_type',method);set_value('overlay',overlay)
        capture(prefix+'_method_and_overlay')
        settings=screen._settings_model.collect()
        outcome=dict(finished=False,ok=False,errors=[]);lines=[]
        def reject_prompt():
            for box in app.topLevelWidgets():
                if isinstance(box,QMessageBox) and box.isVisible():
                    outcome['errors'].append(box.windowTitle()+': '+box.text())
                    capture(prefix+'_prompt');box.reject()
        timer=QTimer(window);timer.timeout.connect(reject_prompt);timer.start(300)
        try:
            if not screen._btn_run.isEnabled():raise ValueError('Actual Run is disabled')
            QTest.mouseClick(screen._btn_run,Qt.LeftButton)
            worker=getattr(screen,'_worker',None)
            if worker is None:raise ValueError('Actual Run did not start a worker')
            worker.finished.connect(lambda ok:outcome.update(finished=True,ok=bool(ok)))
            worker.error.connect(lambda text:outcome['errors'].append(str(text)))
            worker.line_ready.connect(lambda text:lines.append(str(text)))
            settle(.5);capture(prefix+'_running')
            while not outcome['finished'] or screen._worker_thread_is_running():tick();settle(.1)
            settle(1)
        finally:
            timer.stop();timer.deleteLater()
            write_json(captures/(prefix+'_worker.json'),dict(outcome=outcome,lines=lines,settings=settings))
        for block,_,_ in screen._console._pipeline_console_blocks():
            block.setFocus();QTest.keyClick(block,Qt.Key_End,Qt.ControlModifier)
        screen._console.jump_to_the_end();settle(.3);capture(prefix+'_finished_console')
        if not outcome['finished'] or not outcome['ok'] or outcome['errors']:
            raise ValueError('Activation run failed: '+str(outcome))
        queue=screen._figure_queue
        figures=[]
        for index,pixmap in enumerate(queue.all_pixmaps()):
            path=captures/f'{prefix}_actual_figure_{index}.png'
            if not pixmap.save(str(path),'PNG'):raise ValueError('Could not preserve actual figure')
            figures.append(path.name)
        if queue.count():
            queue.show_index(queue.count()-1);settle(.3);capture(prefix+'_actual_plot')
        else:
            capture(prefix+'_missing_gui_plot')
        evidence=verify_saved(prepared,method,oracle)
        evidence.update(settings=settings,outcome=outcome,console=lines,figures=figures,
                        gui_figure_count=queue.count(), figures_card_visible=screen._figures_card.isVisible(),
                        settings_errors=[line for line in lines if '[settings] ERROR' in line])
        proof['runs'].append(evidence);write_json(captures/'activation-workflow.json',proof)
        if saved_plots:
            from capture_saved_plots import check_saved_file_run, show_saved_plots
            check_saved_file_run(evidence)
            archive = Path(prepared['archive'])
            grid = archive.parent / archive.stem / method / 'batch_grids' / 'batch_0_grid.png'
            evidence['external_viewer'] = show_saved_plots(
                app, window, stage,
                lambda name, **kwargs: capture(prefix + '_' + name, **kwargs), settle, [grid])
            # Recheck the raw maps and complete source identities after viewing.
            if verify_saved(prepared, method, oracle) != {
                    key: evidence[key] for key in ('method', 'maps', 'database',
                        'database_rows', 'independent_saved_pixels_checked')}:
                raise ValueError('Viewing changed the independently verified Activation outputs')
            write_json(captures/'activation-workflow.json',proof)

    try:
        from spacr.deep_spacr import pick_device
        device,note=pick_device(what='tutorial reference')
        proof['reference_device_resolution']=dict(device=str(device),note=note)
        oracle = reference(prepared,device=str(device)); proof['independent_reference'] = oracle['proof']
        tiles=[w for w in window.findChildren(QAbstractButton) if w.isVisible() and
               (w.property('moduleAppKey')==HOST_KEY or w.property('navKey')==HOST_KEY)]
        if not tiles:raise ValueError('No actual Home Classify route')
        click(max(tiles,key=lambda w:w.width()*w.height()));settle(.8)
        host=window._screens[HOST_KEY];capture('01_classify_host')
        folds=[w for w in host.findChildren(FoldButton) if w.isVisible() and w.app_key=='activation']
        if len(folds)!=1:raise ValueError('No unique Classify Activation Maps fold')
        click(folds[0]);settle(.8)
        screens=[w for w in app.allWidgets() if isinstance(w,AppScreen) and w.app_key=='activation' and w.isVisible()]
        if len(screens)!=1:raise ValueError('Actual Activation Maps form did not open')
        screen=screens[0];capture('02_activation_fold')
        proof['defaults']=screen._settings_model.collect()
        handle=screen._body_splitter.handle(1);start=handle.rect().center()
        end=start+QPoint(1950-handle.mapTo(window,start).x(),0)
        QTest.mousePress(handle,Qt.LeftButton,pos=start);QTest.mouseMove(handle,end,delay=150)
        QTest.mouseRelease(handle,Qt.LeftButton,pos=end);settle(.3)
        bar=getattr(screen,'_settings_search',None)
        if bar is not None and bar.level()!='all':click(bar._disclosure)
        values=[('dataset',prepared['archive']),('model_path',prepared['model']),
                ('model_type','resnet18'),('image_size',128),('batch_size',4),
                ('channels',[1,2,3]),('normalize_input',True),('normalize',True),
                ('plot',True),('save',True),('shuffle',False),('correlation',False),('n_jobs',1)]
        for index,(key,value) in enumerate(values):
            set_value(key,value)
            if key in ('dataset','model_path','image_size','channels','normalize_input','plot','correlation','n_jobs'):
                capture(f'03_{index:02d}_{key}')
        proof['live_preview_control_present']=bool(getattr(screen,'_preview_switch',None) is not None)
        run('saliency_channel',True,'04_channel_saliency')
        run('saliency_image',False,'05_image_saliency')
        if saved_plots:
            check_saved_runs(proof['runs'])
            proof.update(scope='Verified saved maps and real external viewer; GUI defects remain',
                         application_figures_fixed=False, src_preflight_fixed=False)
        else:
            check_native_runs(proof['runs'])
        proof['accepted']=True
    finally:
        if screen is not None and screen._worker_thread_is_running():
            QTest.mouseClick(screen._btn_stop,Qt.LeftButton);until=time.monotonic()+20
            while screen._worker_thread_is_running() and time.monotonic()<until:settle(.1)
        for child in window.findChildren(AppScreen):child.close()
        window.close();settle(.5)
        proof['preservation']=preserved(prepared)
        proof['active_jobs_after_close']=screen.active_jobs() if screen is not None else 0
        write_json(captures/'activation-workflow.json',proof)
        write_json(captures/'scientific_acceptance.json',proof)
    if proof['active_jobs_after_close']:raise ValueError('Activation workflow left an active job')
    if not proof['accepted']:raise ValueError('Activation outputs exist but native figures or settings validation failed')
