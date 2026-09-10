"""Record at most two genuine regression sweep trials on private CSV copies."""
import csv
import json
from pathlib import Path
import shutil
import tempfile
import time

from build_evaluation_example import sha


def prepare(stage,existing=None):
    settings=json.loads((stage/'captures/regression_release/batch_settings.json').read_text())
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
            if not existing:shutil.copy2(source,copied)
            if sha(copied)!=originals[str(source)]:raise ValueError('Private sweep input differs')
            pair[key]=str(copied)
        pairs.append(pair)
    if len(pairs)!=4:raise ValueError('Expected the four already-verified Regression input pairs')
    settings.update(src=str(work),paired_data=pairs)
    return work,settings,originals


def record_sweep(app,window,stage,captures,capture,settle,write_json,timeout,*,existing=None):
    from PySide6.QtCore import Qt,QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton,QScrollArea,QFileDialog,QLineEdit,QDialogButtonBox

    stage=Path(stage);work,settings,originals=prepare(stage,existing)
    proof=dict(lesson='73_parameter_sweep',accepted=False,private_folder=str(work),
        original_inputs=originals,published=False,synthetic=False,app_source_modified=False,
        maximum_trials=2,requested_workers=1,biological_hits_validated=False)
    proof['new_sweep_requested']=existing is None
    write_json(captures/'scientific_acceptance.json',proof)

    def reveal(widget):
        parent=widget.parentWidget()
        while parent is not None:
            if isinstance(parent,QScrollArea):parent.ensureWidgetVisible(widget)
            parent=parent.parentWidget()
        settle(.15)

    def click(widget):
        reveal(widget)
        if not widget.isVisible() or not widget.isEnabled() or widget.visibleRegion().isEmpty():
            raise ValueError('Actual sweep control unavailable: '+widget.objectName())
        QTest.mouseClick(widget,Qt.LeftButton,pos=widget.visibleRegion().boundingRect().center());settle(.15)

    def fill(widget,value):
        click(widget);QTest.keyClick(widget,Qt.Key_A,Qt.ControlModifier)
        QTest.keyClicks(widget,str(value));QTest.keyClick(widget,Qt.Key_Tab);settle(.15)

    def picker(widget,paths,name):
        errors,accepted=[],[]
        def select():
            dialog=app.activeModalWidget()
            try:
                if not isinstance(dialog,QFileDialog):raise ValueError('Native multi-file picker absent')
                dialog.accepted.connect(lambda:accepted.append(True));dialog.resize(1600,1000)
                fill(dialog.findChild(QLineEdit,'fileNameEdit'),' '.join('"'+p+'"' for p in paths))
                capture(name)
                click(dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Open))
            except Exception as error:
                errors.append(str(error))
                if dialog is not None:dialog.reject()
        watch=QTimer(window);watch.setSingleShot(True)
        watch.timeout.connect(lambda:app.activeModalWidget().reject() if app.activeModalWidget() else None)
        QTimer.singleShot(300,select);watch.start(15000)
        click(widget._add_files_button);watch.stop()
        if errors or not accepted or widget.get_value()!=paths:
            raise ValueError('The real picker did not preserve ordered input paths: '+str(errors))

    try:
        buttons=[b for b in window.findChildren(QAbstractButton) if b.isVisible() and
            (b.property('moduleAppKey')=='regression' or b.property('navKey')=='regression')]
        if not buttons:raise ValueError('Actual Regression Home control unavailable')
        click(max(buttons,key=lambda b:b.width()*b.height()));screen=window._screens['regression']
        for key in ('src','paired_data','dependent_variable'):
            if not screen._settings_model.set_value_for_key(key,settings[key]):
                raise ValueError('The actual parent form cannot accept '+key)
        capture('01_regression_with_four_existing_pairs')
        click(screen._sweep_switch);panel=screen._sweep.panel();settle(.5)
        # Resize only the actual user-adjustable splitters; no fake visibility.
        width=sum(screen._body_splitter.sizes())
        screen._body_splitter.setSizes([600,max(1,width-600)])
        screen._runtime_splitter.setSizes([1,1600,160])
        if not screen._console_folder.shut:click(screen._console_folder.heading)
        settle(.3)
        proof['initial_inherited_inputs']=dict(score=panel.score_data.get_value(),count=panel.count_data.get_value())
        capture('02_actual_nested_sweep')
        # The present sweep seam reads legacy score_data/count_data, not
        # paired_data. Use the real pickers when the parent's pairs are absent.
        for key,widget in [('score',panel.score_data),('count',panel.count_data)]:
            paths=[row[key] for row in settings['paired_data']]
            if widget.get_value():click(widget._clear_button)
            picker(widget,paths,'03_'+key+'_file_picker')
        fill(panel.destination,work/'trials');fill(panel.dependent_variable,'pred')
        capture('04_ordered_pairs_and_private_destination')
        fixed=dict(regression_type='ridge',alpha='0.01, 0.1',inference='parametric',
            analysis_unit='well',agg_type='mean',transform='None',random_row_column_effects=False,
            batch_correction='none',multiple_testing_method='fdr_bh',fdr_alpha=.05,
            threshold_method='std',threshold_multiplier=3,fraction_threshold=.02,
            min_cells_per_well=100,min_observations_per_hit=0,outlier_detection=False)
        if set(fixed)!=set(panel._axis_rows):raise ValueError('Sweep axes changed; recheck the bounded design')
        for key,value in fixed.items():
            include,editor=panel._axis_rows[key]
            if include.isChecked()!=(key=='alpha'):click(include)
            fill(editor,value)
        fill(panel.max_trials,2);fill(panel.workers,1);fill(panel.seed,0)
        click(panel.mode);QTest.keyClick(panel.mode.view(),Qt.Key_End);QTest.keyClick(panel.mode.view(),Qt.Key_Return);settle(.2)
        if panel.mode.currentText()!='grid':raise ValueError('Actual sampling mode is not grid')
        space=panel.space()
        if space.axes!={'alpha':[.01,.1]} or space.size()!=2:raise ValueError('The real form is not a two-trial alpha sweep')
        proof['space']=dict(axes=space.axes,fixed=space.fixed,mode=panel.mode.currentText(),seed=panel.seed.value())
        proof['containment']=panel.containment.text()
        proof['base_settings']=panel.base_settings()
        capture('05_two_trials_one_worker_budget')
        click(panel.estimate_button);proof['estimate']=panel.status.text()
        capture('06_actual_estimate')
        if '2 valid trials' not in panel.status.text():raise ValueError('Estimate does not confirm exactly two trials')
        write_json(captures/'scientific_acceptance.json',proof)
        if existing:
            click(panel.refresh_button);capture('07_load_existing_sweep_results')
        else:
            click(panel.start_button)
            if panel.start_button.isEnabled():raise ValueError('The actual sweep did not start')
            capture('07_actual_sweep_running')
            deadline=time.monotonic()+timeout
            while not panel.start_button.isEnabled():
                if time.monotonic()>deadline:
                    raise TimeoutError('The bounded sweep exceeded its deadline; retain its partial trials')
                settle(.2)
        settle(1);proof['status']=panel.status.text()
        panel.table.resizeColumnsToContents();settle(.3)
        proof['trial_table_geometry']=dict(height=panel.table.height(),
            viewport_height=panel.table.viewport().height(),row_height=panel.table.rowHeight(0),
            both_rows_visible=panel.table.viewport().height()>=sum(panel.table.rowHeight(i) for i in range(2)))
        capture('08_actual_sweep_finished')
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
        panel.table.scrollToItem(panel.table.item(0,0))
        QTest.mouseClick(panel.table.viewport(),Qt.LeftButton,pos=panel.table.visualItemRect(panel.table.item(0,0)).center())
        settle(.2);click(panel.show_button);settle(.5)
        proof['selected_trial_status']=panel.trial_status.text()
        proof['selected_trial_results_status']=panel.results._status
        capture('09_actual_saved_trial_results')
        def result_snapshot(name):
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
            capture(name);write_json(captures/'scientific_acceptance.json',proof)

        result_snapshot('10_saved_guide_family')
        box=panel.results._level_box;index=box.findData('gene')
        if index<0:raise ValueError('Actual saved-result gene selector unavailable')
        click(box);QTest.keyClick(box.view(),Qt.Key_Home)
        for _ in range(index):QTest.keyClick(box.view(),Qt.Key_Down)
        QTest.keyClick(box.view(),Qt.Key_Return);settle(.6)
        if panel.results.level()!='gene':raise ValueError('Actual selector did not choose genes')
        result_snapshot('11_saved_gene_family')
        index=box.findData('grna');click(box);QTest.keyClick(box.view(),Qt.Key_Home)
        for _ in range(index):QTest.keyClick(box.view(),Qt.Key_Down)
        QTest.keyClick(box.view(),Qt.Key_Return);settle(.6)
        if panel.results.level()!='grna':raise ValueError('Actual selector did not restore guides')
        result_snapshot('12_saved_guide_family_restored')
        if 'Nothing was re-fitted' not in panel.trial_status.text():
            raise ValueError('The selected saved trial did not use its existing results')
        if any(sha(p)!=h for p,h in trial_files.items()):raise ValueError('Opening saved results rewrote a trial')
        proof['hold']='Two-trial execution recorded; saved summary and displayed-table checks remain.'
        write_json(captures/'scientific_acceptance.json',proof)
    finally:
        proof['original_inputs_preserved']=all(sha(p)==h for p,h in originals.items())
        proof['private_inputs_preserved']=all(sha(work/'inputs'/Path(p).name)==h for p,h in originals.items())
        write_json(captures/'scientific_acceptance.json',proof)
