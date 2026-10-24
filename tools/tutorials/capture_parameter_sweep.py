"""Record at most two genuine regression sweep trials on private CSV copies."""
import csv
import json
from pathlib import Path
import shutil
import tempfile
import time

import capture_ui
import capture_ui_drv_d as drv
from build_evaluation_example import sha


def support_geometry(table, window):
    """Record unclipped actual header/cell bounds in captured window coordinates."""
    from PySide6.QtCore import QRect
    from PySide6.QtGui import QRegion
    from sweep_evidence import SUPPORT_COLUMNS

    def bounds(widget, rect):
        """Map the whole rectangle without clipping it to the captured window."""
        origin = widget.mapTo(window, rect.topLeft())
        return [origin.x(), origin.y(), rect.width(), rect.height()]

    headers = [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())]
    header = table.horizontalHeader()
    viewport = table.viewport()
    evidence = dict(frame="12_saved_guide_family_restored",
        window=[0, 0, window.width(), window.height()],
        viewport=bounds(viewport, viewport.rect()),
        header_viewport=bounds(header.viewport(), header.viewport().rect()),
        horizontal_scroll=table.horizontalScrollBar().value(), headers=[], cells=[])
    for name in SUPPORT_COLUMNS:
        if headers.count(name) != 1:
            raise ValueError("Missing or repeated measured support column: " + name)
        column = headers.index(name)
        rect = QRect(header.sectionViewportPosition(column), 0,
                     header.sectionSize(column), header.viewport().height())
        evidence['headers'].append(dict(name=name, rect=bounds(header.viewport(), rect),
            visible=header.isVisible() and not table.isColumnHidden(column)
            and QRegion(rect).subtracted(header.viewport().visibleRegion()).isEmpty()))
        for row in range(table.rowCount()):
            item = table.item(row, column)
            rect = table.visualItemRect(item)
            evidence['cells'].append(dict(trial_id=table.item(row, headers.index('trial_id')).text(),
                column=name, value=item.text(), rect=bounds(viewport, rect),
                visible=table.isVisible() and not table.isRowHidden(row)
                and not table.isColumnHidden(column)
                and QRegion(rect).subtracted(viewport.visibleRegion()).isEmpty()))
    return evidence


def reveal_support(table, settle, capture=None):
    """Use the real horizontal scrollbar to expose the measured support columns.

    In the capture host (``capture`` given) the scroll bar handle is dragged."""
    from PySide6.QtWidgets import QAbstractItemView
    from sweep_evidence import SUPPORT_COLUMNS

    headers = [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())]
    first = headers.index(SUPPORT_COLUMNS[0])
    table.scrollToItem(table.item(0, first), QAbstractItemView.PositionAtTop)
    header = table.horizontalHeader()
    scroll = table.horizontalScrollBar()
    if table.horizontalScrollMode() == QAbstractItemView.ScrollPerPixel:
        goal = scroll.value() + header.sectionViewportPosition(first)
    else:
        goal = header.visualIndex(first)
    if capture is not None and capture_ui.live(capture):
        drv.drag_scroll_bar_horizontal(capture, scroll, goal)
    else:
        scroll.setValue(goal)
    settle(.3)


def prepare(stage,existing=None,*,input_root=None):
    stage=Path(stage).resolve()
    settings=json.loads((stage/'captures/regression_release/batch_settings.json').read_text(encoding="utf-8"))
    if len(settings['paired_data'])!=4:
        raise ValueError('Expected the four already-verified Regression input pairs')
    names=set();sources={}
    if input_root is not None:
        input_root=Path(input_root).resolve()
        if not input_root.is_dir() or not input_root.is_relative_to(stage):
            raise ValueError('Sweep input root must be inside the private stage')
        original_run=Path(settings['src'])
        if not original_run.is_absolute() or original_run.parent.name!='regression_runs':
            raise ValueError('The original settings do not identify their Regression workspace')
        original_root=original_run.parent.parent.resolve()
    for row in settings['paired_data']:
        for key in ('score','count'):
            source=Path(row[key])
            if source.name in names:
                raise ValueError('Sweep input filenames collide; every source needs its own private copy')
            names.add(source.name)
            copied_source=source
            if input_root is not None:
                if not source.is_absolute() or not source.resolve().is_relative_to(original_root):
                    raise ValueError('Sweep input source leaves the original workspace')
                copied_source=(input_root/source.resolve().relative_to(original_root)).resolve()
                if not copied_source.is_relative_to(input_root) or not copied_source.is_file():
                    raise ValueError('Sweep input copy is missing or leaves the private input root')
                if sha(copied_source)!=sha(source):
                    raise ValueError('Relocated sweep input differs from its original source')
            sources[str(source)]=copied_source
    parent=stage/'sweep_runs';parent.mkdir(exist_ok=True)
    work=Path(existing).resolve() if existing else Path(tempfile.mkdtemp(prefix='REAL-two-ridge-trials-',dir=parent))
    if existing and (work.parent!=parent.resolve() or not (work/'trials/sweep_results.csv').is_file()):
        raise ValueError('Replay must name an existing private tutorial sweep')
    inputs=work/'inputs'
    if not existing:inputs.mkdir()
    originals={};pairs=[]
    for row in settings['paired_data']:
        pair=dict(row)
        for key in ('score','count'):
            source=Path(row[key]);originals[str(source)]=sha(source)
            copied=inputs/source.name
            if not existing:shutil.copy2(sources[str(source)],copied)
            if sha(copied)!=originals[str(source)]:raise ValueError('Private sweep input differs')
            pair[key]=str(copied)
        pairs.append(pair)
    if len(pairs)!=4:raise ValueError('Expected the four already-verified Regression input pairs')
    settings.update(src=str(work),paired_data=pairs)
    return work,settings,originals


def record_sweep(app,window,stage,captures,capture,settle,write_json,timeout,*,existing=None,input_root=None):
    from PySide6.QtCore import Qt,QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton,QScrollArea,QFileDialog,QLineEdit,QDialogButtonBox,QPushButton

    stage=Path(stage);work,settings,originals=prepare(stage,existing,input_root=input_root)
    proof=dict(lesson='73_parameter_sweep',accepted=False,private_folder=str(work),
        original_inputs=originals,published=False,synthetic=False,app_source_modified=False,
        maximum_trials=2,requested_workers=1,biological_hits_validated=False)
    proof['new_sweep_requested']=existing is None
    proof['retained_input_root']=str(Path(input_root).resolve()) if input_root else None
    proof['settings_manifest_sha256']=sha(stage/'captures/regression_release/batch_settings.json')
    proof['settings_manifest_rewritten']=False
    write_json(captures/'scientific_acceptance.json',proof)

    def reveal(widget):
        parent=widget.parentWidget()
        while parent is not None:
            if isinstance(parent,QScrollArea):capture_ui.scroll_to(capture,parent,widget)
            # A file dialog's parent is the screen: walking on past the dialog (a window) tried to scroll the SCREEN
            # to the dialog's file-name field ('Could not scroll fileNameEdit into view').
            if parent.isWindow():break
            parent=parent.parentWidget()
        settle(.15)

    def click(widget):
        reveal(widget)
        if not widget.isVisible() or not widget.isEnabled() or widget.visibleRegion().isEmpty():
            raise ValueError('Actual sweep control unavailable: '+widget.objectName())
        QTest.mouseClick(widget,Qt.LeftButton,pos=widget.visibleRegion().boundingRect().center());settle(.15)

    def fill(widget,value):
        if capture_ui.live(capture):
            reveal(widget);capture_ui.set_text(capture,widget,str(value));settle(.15)
            return
        click(widget);QTest.keyClick(widget,Qt.Key_A,Qt.ControlModifier)
        QTest.keyClicks(widget,str(value));QTest.keyClick(widget,Qt.Key_Tab);settle(.15)

    def open_multi(button,paths,name,own_clip=True):
        """Press ``button``; answer its multi-file picker with all ``paths`` typed."""
        errors,accepted=[],[]
        def select():
            dialog=app.activeModalWidget()
            try:
                if not isinstance(dialog,QFileDialog):raise ValueError('Native multi-file picker absent')
                dialog.accepted.connect(lambda:accepted.append(True));capture_ui.dialog_geometry(capture,dialog,(1600,1000))
                fill(dialog.findChild(QLineEdit,'fileNameEdit'),' '.join('"'+p+'"' for p in paths))
                drv.snap(capture,name);capture_ui.wait(capture,1.5,settle)
                click(dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Open))
            except Exception as error:
                errors.append(str(error))
                if dialog is not None:dialog.reject()
        watch=QTimer(window);watch.setSingleShot(True)
        watch.timeout.connect(lambda:app.activeModalWidget().reject() if app.activeModalWidget() else None)
        started=drv.begin(capture,settle,name) if own_clip else False
        capture_ui.when_modal(capture,300,select);watch.start(15000)
        click(button);watch.stop()
        drv.finish(capture,started)
        if errors or not accepted:
            raise ValueError('The multi-file picker failed: '+str(errors))

    def picker(widget,paths,name):
        open_multi(widget._add_files_button,paths,name)
        if widget.get_value()!=paths:
            raise ValueError('The real picker did not preserve ordered input paths')

    def parent_pairs(screen):
        """Fill the parent's paired_data table through its two Add buttons (a user does the same)."""
        widget=screen._settings_model._widgets['paired_data']
        for key,label in (('score','Add score tables…'),('count','Add count tables…')):
            button=next(b for b in widget.findChildren(QPushButton) if b.text().replace('&','')==label)
            open_multi(button,[row[key] for row in settings['paired_data']],'01_pairs_'+key,own_clip=False)
        wanted=[{k:str(v) for k,v in row.items() if k in ('score','count')} for row in settings['paired_data']]
        got=[{k:str(v) for k,v in row.items() if k in ('score','count')}
             for row in (screen._settings_model.collect().get('paired_data') or [])]
        if got!=wanted:
            capture_ui.code_action(capture,'paired_data proposal differs from the staged pairs; set by code')
            screen._settings_model.set_value_for_key('paired_data',settings['paired_data'])

    try:
        buttons=[b for b in window.findChildren(QAbstractButton) if b.isVisible() and
            (b.property('moduleAppKey')=='regression' or b.property('navKey')=='regression')]
        if not buttons:raise ValueError('Actual Regression Home control unavailable')
        scene1=drv.begin(capture,settle,'01_regression_with_four_existing_pairs')
        click(max(buttons,key=lambda b:b.width()*b.height()));screen=window._screens['regression']
        if capture_ui.live(capture):
            capture_ui.fill_setting(capture,screen,'src',settings['src'])
            parent_pairs(screen)
            capture_ui.fill_setting(capture,screen,'dependent_variable',settings['dependent_variable'])
            for key in ('src','dependent_variable'):
                if screen._settings_model.collect().get(key)!=settings[key]:
                    raise ValueError('The actual parent form cannot accept '+key)
        else:
            for key in ('src','paired_data','dependent_variable'):
                if not screen._settings_model.set_value_for_key(key,settings[key]):
                    raise ValueError('The actual parent form cannot accept '+key)
        capture_ui.point_at_settings(capture,screen,['src','paired_data','dependent_variable'])
        drv.snap(capture,'01_regression_with_four_existing_pairs');drv.finish(capture,scene1)
        scene2=drv.begin(capture,settle,'02_actual_nested_sweep')
        click(screen._sweep_switch);panel=screen._sweep.panel();settle(.5)
        # Resize only the actual user-adjustable splitters; no fake visibility.
        width=sum(screen._body_splitter.sizes())
        capture_ui.splitter_sizes(capture,screen._body_splitter,[600,max(1,width-600)])
        capture_ui.splitter_sizes(capture,screen._runtime_splitter,[1,1600,160])
        if not screen._console_folder.shut:click(screen._console_folder.heading)
        settle(.3)
        proof['initial_inherited_inputs']=dict(score=panel.score_data.get_value(),count=panel.count_data.get_value())
        capture_ui.point_at(capture,panel.score_data,panel.count_data)
        drv.snap(capture,'02_actual_nested_sweep');drv.finish(capture,scene2)
        # The present sweep seam reads legacy score_data/count_data, not
        # paired_data. Use the real pickers when the parent's pairs are absent.
        for key,widget in [('score',panel.score_data),('count',panel.count_data)]:
            paths=[row[key] for row in settings['paired_data']]
            if widget.get_value():click(widget._clear_button)
            picker(widget,paths,'03_'+key+'_file_picker')
        with capture_ui.clip(capture,'04_ordered_pairs_and_private_destination'):
            fill(panel.destination,work/'trials');fill(panel.dependent_variable,'pred')
        fixed=dict(regression_type='ridge',alpha='0.01, 0.1',inference='parametric',
            analysis_unit='well',agg_type='mean',transform='None',random_row_column_effects=False,
            batch_correction='none',multiple_testing_method='fdr_bh',fdr_alpha=.05,
            threshold_method='std',threshold_multiplier=3,fraction_threshold=.02,
            min_cells_per_well=100,min_observations_per_hit=0,outlier_detection=False)
        if set(fixed)!=set(panel._axis_rows):raise ValueError('Sweep axes changed; recheck the bounded design')
        scene5=drv.begin(capture,settle,'05_two_trials_one_worker_budget')
        for key,value in fixed.items():
            include,editor=panel._axis_rows[key]
            if include.isChecked()!=(key=='alpha'):click(include)
            fill(editor,value)
        fill(panel.max_trials,2);fill(panel.workers,1);fill(panel.seed,0)
        if capture_ui.live(capture):
            capture_ui.select_combo(capture,panel.mode,panel.mode.count()-1);settle(.2)
        else:
            click(panel.mode);QTest.keyClick(panel.mode.view(),Qt.Key_End);QTest.keyClick(panel.mode.view(),Qt.Key_Return);settle(.2)
        if panel.mode.currentText()!='grid':raise ValueError('Actual sampling mode is not grid')
        space=panel.space()
        if space.axes!={'alpha':[.01,.1]} or space.size()!=2:raise ValueError('The real form is not a two-trial alpha sweep')
        proof['space']=dict(axes=space.axes,fixed=space.fixed,mode=panel.mode.currentText(),seed=panel.seed.value())
        proof['containment']=panel.containment.text()
        proof['base_settings']=panel.base_settings()
        capture_ui.point_at(capture,panel.max_trials,panel.workers)
        drv.snap(capture,'05_two_trials_one_worker_budget');drv.finish(capture,scene5)
        with capture_ui.clip(capture,'06_actual_estimate'):
            click(panel.estimate_button);proof['estimate']=panel.status.text()
            capture_ui.point_at(capture,panel.status)
        if '2 valid trials' not in panel.status.text():raise ValueError('Estimate does not confirm exactly two trials')
        write_json(captures/'scientific_acceptance.json',proof)
        if existing:
            with capture_ui.clip(capture,'07_load_existing_sweep_results'):
                click(panel.refresh_button)
        else:
            scene7=drv.begin(capture,settle,'07_actual_sweep_running')
            click(panel.start_button)
            if panel.start_button.isEnabled():raise ValueError('The actual sweep did not start')
            drv.snap(capture,'07_actual_sweep_running')
            deadline=time.monotonic()+timeout
            while not panel.start_button.isEnabled():
                if time.monotonic()>deadline:
                    raise TimeoutError('The bounded sweep exceeded its deadline; retain its partial trials')
                if scene7 and capture_ui.clip_due(capture):
                    capture.end();scene7=False       # a long sweep: the plan ends the clip
                settle(.2)
            if scene7:
                capture_ui.point_at(capture,panel.status)
                drv.finish(capture,scene7)
        settle(1);proof['status']=panel.status.text()
        scene8=drv.begin(capture,settle,'08_actual_sweep_finished')
        panel.table.resizeColumnsToContents();settle(.3)
        proof['trial_table_geometry']=dict(height=panel.table.height(),
            viewport_height=panel.table.viewport().height(),row_height=panel.table.rowHeight(0),
            both_rows_visible=panel.table.viewport().height()>=sum(panel.table.rowHeight(i) for i in range(2)))
        capture_ui.point_at(capture,panel.table)
        drv.snap(capture,'08_actual_sweep_finished');drv.finish(capture,scene8)
        path=work/'trials/sweep_results.csv'
        if not path.is_file():raise ValueError('No sweep results table was saved')
        rows=list(csv.DictReader(path.open()));proof['saved_rows']=rows
        proof['displayed_headers']=[panel.table.horizontalHeaderItem(c).text() for c in range(panel.table.columnCount())]
        proof['displayed_rows']=[[panel.table.item(r,c).text() for c in range(panel.table.columnCount())]
                                for r in range(panel.table.rowCount())]
        if len(rows)!=2:raise ValueError('The saved sweep does not contain exactly two trials')
        # Showing a saved trial must not silently start another fit.
        trial_files={str(p):sha(p) for p in (work/'trials').glob('trial_*/results/ridge/results.csv')}
        if len(trial_files)!=2:raise ValueError('The two trials did not save their ridge result tables')
        with capture_ui.clip(capture,'09_actual_saved_trial_results'):
            if capture_ui.live(capture):
                capture_ui._wheel_item_into_view(capture,panel.table,panel.table.item(0,0))
            else:
                panel.table.scrollToItem(panel.table.item(0,0))
            QTest.mouseClick(panel.table.viewport(),Qt.LeftButton,pos=panel.table.visualItemRect(panel.table.item(0,0)).center())
            settle(.2);click(panel.show_button);settle(.5)
            proof['selected_trial_status']=panel.trial_status.text()
            proof['selected_trial_results_status']=panel.results._status
        def result_snapshot(name,act=None):
            started=drv.begin(capture,settle,name)
            if act is not None:act()
            result=panel.results;table=result.table.table;plot=result.volcano
            snapshot=dict(level=result.level(),status=result._status,
                headers=[table.horizontalHeaderItem(c).text() for c in range(table.columnCount())],
                rows=[[table.item(r,c).text() for c in range(table.columnCount())]
                      for r in range(table.rowCount())],
                p_axis=plot.p_axis(),caption=plot._caption,
                keys=list(plot._keys),points=[])
            for item in plot._scatter_items():
                snapshot['points'].extend([[float(p.pos().x()),float(p.pos().y()),int(p.data())]
                                           for p in item.points()])
            proof.setdefault('result_snapshots',{})[name]=snapshot
            capture_ui.point_at(capture,plot if hasattr(plot,'isVisible') else result)
            drv.snap(capture,name);drv.finish(capture,started)
            write_json(captures/'scientific_acceptance.json',proof)

        result_snapshot('10_saved_guide_family')
        box=panel.results._level_box
        def choose_level(level,keys=False):
            index=box.findData(level)
            if index<0:raise ValueError('Actual saved-result '+level+' selector unavailable')
            if capture_ui.live(capture):
                capture_ui.select_combo(capture,box,index);settle(.6)
            else:
                click(box);QTest.keyClick(box.view(),Qt.Key_Home)
                for _ in range(index):QTest.keyClick(box.view(),Qt.Key_Down)
                QTest.keyClick(box.view(),Qt.Key_Return);settle(.6)
            if panel.results.level()!=level:raise ValueError('Actual selector did not choose '+level)
        result_snapshot('11_saved_gene_family',lambda:choose_level('gene'))
        from sweep_evidence import check_support_geometry
        def restore_guides():
            choose_level('grna')
            reveal_support(panel.table, settle, capture)
            proof['support_geometry'] = support_geometry(panel.table, window)
            check_support_geometry(proof['support_geometry'], rows)
        result_snapshot('12_saved_guide_family_restored',restore_guides)
        tabs=panel.results.tabs;bar=tabs.tabBar()
        index=tabs.indexOf(panel.results._summary)
        if index<0:raise ValueError('Actual model Summary tab unavailable')
        with capture_ui.clip(capture,'13_actual_saved_model_summary'):
            if capture_ui.live(capture):
                capture_ui.select_tab(capture,tabs,index);settle(.4)
            else:
                QTest.mouseClick(bar,Qt.LeftButton,pos=bar.tabRect(index).center());settle(.4)
            proof['model_summary_text']=panel.results._summary.toPlainText()
            capture_ui.point_at(capture,panel.results._summary)
        if 'Nothing was re-fitted' not in panel.trial_status.text():
            raise ValueError('The selected saved trial did not use its existing results')
        if any(sha(p)!=h for p,h in trial_files.items()):raise ValueError('Opening saved results rewrote a trial')
        proof['hold']='Two-trial execution recorded; saved summary and displayed-table checks remain.'
        write_json(captures/'scientific_acceptance.json',proof)
    finally:
        proof['original_inputs_preserved']=all(sha(p)==h for p,h in originals.items())
        proof['private_inputs_preserved']=all(sha(work/'inputs'/Path(p).name)==h for p,h in originals.items())
        write_json(captures/'scientific_acceptance.json',proof)
