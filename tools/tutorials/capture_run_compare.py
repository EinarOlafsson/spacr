"""Record real Run Compare controls over two explicitly relocated snapshots."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import time

from build_evaluation_example import sha
from run_compare_example import SOURCES,read_identity,require_same_identity
from stage_lesson import read


def record(app,window,stage,captures,capture,settle,write_json,timeout):
    from PySide6.QtCore import Qt,QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton,QFileDialog,QLineEdit,QDialogButtonBox
    root=Path(stage)/'run_compare_verified_snapshots_v1';prepared=read(root/'preparation.json')
    proof=dict(lesson='50_run_compare',accepted=False,preparation=prepared,
        synthetic_runs=False,new_analysis_started=False,app_source_modified=False,published=False)
    deadline=time.monotonic()+timeout
    def click(widget):
        if time.monotonic()>deadline:raise TimeoutError('Run Compare recording exceeded its bound')
        if not widget.isVisible() or not widget.isEnabled() or widget.visibleRegion().isEmpty():
            raise ValueError('The actual comparison control is unavailable')
        QTest.mouseClick(widget,Qt.LeftButton,pos=widget.visibleRegion().boundingRect().center());settle(.25)
    identities=[read_identity(root/'snapshots'/run/'measurements.db') for _,run in SOURCES]
    require_same_identity(*identities)
    before={str(root/'snapshots'/run/'measurements.db'):sha(root/'snapshots'/run/'measurements.db') for _,run in SOURCES}
    tabs=window._startup._tabs
    data=[i for i in range(tabs.count()) if tabs.tabText(i).split(' (',1)[0].strip().lower()=='data']
    if len(data)!=1:raise ValueError('The real Home Data category is absent')
    QTest.mouseClick(tabs.tabBar(),Qt.LeftButton,pos=tabs.tabBar().tabRect(data[0]).center());settle(.4)
    capture('01_home_data')
    buttons=[b for b in window.findChildren(QAbstractButton) if b.isVisible() and
        (b.property('moduleAppKey')=='run_compare' or b.property('navKey')=='run_compare')]
    if not buttons:raise ValueError('The actual Run Compare Home tile is absent')
    click(max(buttons,key=lambda b:b.width()*b.height()))
    screen=window._screens['run_compare'];capture('02_actual_empty_compare')
    accepted=[];errors=[]
    def pick():
        dialog=app.activeModalWidget()
        try:
            if not isinstance(dialog,QFileDialog):raise ValueError('The real directory picker did not open')
            dialog.accepted.connect(lambda:accepted.append(True));dialog.resize(1500,1000)
            field=dialog.findChild(QLineEdit,'fileNameEdit');field.setFocus()
            QTest.keyClick(field,Qt.Key_A,Qt.ControlModifier);QTest.keyClicks(field,str(root));settle(.2)
            capture('03_actual_snapshot_project_picker')
            click(dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Open))
        except Exception as error:
            errors.append(str(error))
            if dialog is not None:dialog.reject()
    timer=QTimer(window);timer.setSingleShot(True)
    timer.timeout.connect(lambda:app.activeModalWidget().reject() if app.activeModalWidget() else None)
    QTimer.singleShot(300,pick);timer.start(15000);click(screen._browse_button);timer.stop()
    if errors or not accepted or screen._project_edit.text()!=str(root):
        raise ValueError('Native project selection failed: '+str(errors))
    if {r.run_id for r in screen.runs()}!={r for _,r in SOURCES}:
        raise ValueError('The two actual historical run IDs were not loaded')
    capture('04_two_actual_runs_no_setting_change')
    def choose(combo,run):
        index=combo.findData(run)
        if index<0:raise ValueError('A real run is missing from the dropdown')
        click(combo);view=combo.view();QTest.keyClick(view,Qt.Key_Home)
        for _ in range(index):QTest.keyClick(view,Qt.Key_Down)
        QTest.keyClick(view,Qt.Key_Return);settle(.2)
        if combo.currentData()!=run:raise ValueError('The actual run dropdown selected the wrong item')
    choose(screen._a_combo,SOURCES[0][1]);choose(screen._b_combo,SOURCES[1][1]);click(screen._compare_button)
    comparison=screen._comparison
    expected=dict(prepared['counts'],plates=1,wells=4,fields=16)
    wanted={(scope,key):(value,value) for scope in ('overall','plate1') for key,value in expected.items()
            if scope=='overall' or key!='plates'}
    observed={(r.scope,r.metric):(r.a,r.b) for r in comparison.counts.rows}
    if (not comparison.comparable or comparison.forced or observed!=wanted
            or not comparison.settings.identical or comparison.hits.available):
        raise ValueError('The actual comparison differs from independent SQL identities/counts')
    if any(r.delta!=0 or r.pct!=0 for r in comparison.counts.rows):
        raise ValueError('The actual unchanged-count arithmetic differs')
    for run in screen.runs():
        source=next(row['original'] for row in prepared['records'] if row['original']['run_id']==run.run_id)
        if run.spacr_version!=source['spacr_version'] or run.created_ns!=source['created_ns'] or run.settings!=json.loads(source['settings_json']):
            raise ValueError('The visible run data lost historical version, time or settings')
    click(screen._show_all);screen._settings_tree.expandAll();settle(.4)
    if not screen._comparison.settings.include_same or not screen._comparison.settings.identical:
        raise ValueError('Show unchanged settings did not retain an identical comparison')
    capture('05_actual_unchanged_settings')
    tabbar=screen._tabs.tabBar()
    QTest.mouseClick(tabbar,Qt.LeftButton,pos=tabbar.tabRect(1).center());settle(.3)
    tree=screen._counts_tree;visible={}
    for i in range(tree.topLevelItemCount()):
        group=tree.topLevelItem(i)
        scope='overall' if group.text(0)=='Overall' else group.text(0).removeprefix('Plate ')
        for j in range(group.childCount()):
            row=group.child(j);visible[(scope,row.text(0))]=tuple(row.text(k) for k in range(1,5))
    wanted_text={key:(str(a),str(b),'+0','+0.0%') for key,(a,b) in wanted.items()}
    if visible!=wanted_text:raise ValueError('The rendered count table differs from independent expected cells')
    capture('06_actual_equal_object_counts')
    QTest.mouseClick(tabbar,Qt.LeftButton,pos=tabbar.tabRect(2).center());settle(.3)
    if screen._hits_tree.topLevelItemCount()!=1 or 'no regression results' not in screen._hits_tree.topLevelItem(0).text(0):
        raise ValueError('The actual Hits tab does not disclose missing regression results')
    capture('07_actual_no_hit_list')
    choose(screen._b_combo,SOURCES[0][1]);settle(.2)
    if not any(f.code=='same-run' for f in screen._comparison.comparability.findings):
        raise ValueError('Selecting the same real run did not produce its warning')
    capture('08_actual_same_run_warning')
    choose(screen._b_combo,SOURCES[1][1]);click(screen._compare_button)
    QTest.mouseClick(tabbar,Qt.LeftButton,pos=tabbar.tabRect(1).center());settle(.3)
    if any(f.code=='same-run' for f in screen._comparison.comparability.findings):
        raise ValueError('The warning did not clear after restoring distinct runs')
    capture('09_actual_distinct_runs_restored')
    if any(sha(path)!=digest for path,digest in {**before,**prepared['source_hashes']}.items()):
        raise ValueError('Comparison changed an original database or snapshot')
    with closing(sqlite3.connect((root/'artifacts.db').as_uri()+'?mode=ro',uri=True)) as con:
        con.row_factory=sqlite3.Row
        actual=[dict(r) for r in con.execute('select * from artifacts order by run_id')]
    if actual!=sorted([r['transported'] for r in prepared['records']],key=lambda r:r['run_id']):
        raise ValueError('The actual GUI changed the transported historical registry rows')
    proof.update(accepted=True,count_rows_checked=len(wanted),rendered_count_cells_checked=len(wanted)*4,
        historical_metadata_preserved=True,unchanged_setting_values_checked=True,
        missing_hits_disclosed=True,same_run_warning_observed_and_cleared=True,
        original_and_snapshot_bytes_unchanged=True,registry_rows_unchanged=True,
        scientific_accuracy_validated=False)
    write_json(captures/'scientific_acceptance.json',proof)
