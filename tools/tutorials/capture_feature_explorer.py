"""Record current Classify fold, real rankings, reversible filters and exports."""
from dataclasses import asdict
from collections import Counter
import json
from pathlib import Path
import tempfile
import time

from capture_database import prepare_database_copy, require_unchanged_source, _digest
from feature_explorer_evidence import read_measurements, verify_ranking


def record_explorer(app, window, stage, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import Qt, QTimer, QPoint
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import (QAbstractButton, QFileDialog, QDialogButtonBox,
                                   QLineEdit, QPushButton, QTabWidget)
    from spacr.qt.widgets.fold_strip import FoldButton
    from spacr.qt.screens.feature_explorer import FeatureExplorerScreen
    import pandas as pd

    parent = Path(stage)/'feature_explorer_runs'
    parent.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='real-measurements-', dir=parent))
    source = Path(stage)/'annotate_fresh/example_data/plate1/measurements/measurements.db'
    original = prepare_database_copy(source,work/'measurements.db')
    rows, features = read_measurements(work/'measurements.db')
    if len(rows)!=2341 or {r['rowID'] for r in rows} != {'r5','r12'}:
        raise ValueError('Expected the actual downloaded two-row cell example')
    deadline=time.monotonic()+timeout
    proof={'accepted':False,'lesson':'65_feature_explorer','source':original,
           'app_source_modified':False,'synthetic_measurements':False,'published':False}
    screen=None
    outcomes=[]

    def wait_for(predicate):
        while not predicate():
            if time.monotonic()>deadline:
                raise TimeoutError('Bounded Feature Explorer capture timed out')
            settle(.05)

    def click(widget):
        if not widget.isVisible() or not widget.isEnabled():
            raise ValueError('Actual Feature Explorer control is unavailable')
        QTest.mouseClick(widget,Qt.LeftButton,pos=widget.visibleRegion().boundingRect().center())
        settle(.25)

    def idle():
        wait_for(lambda: not screen.is_busy() and not screen.active_jobs() and
                 not screen.explorer._debounce.isActive() and not screen.filters._debounce.isActive())
        settle(.4)

    def fill(edit,text):
        click(edit)
        QTest.keyClick(edit,Qt.Key_A,Qt.ControlModifier)
        QTest.keyClicks(edit,str(text))
        QTest.keyClick(edit,Qt.Key_Tab)
        settle(.3)

    def combo(box,value,*,data=False):
        index=box.findData(value) if data else box.findText(value)
        if index<0:
            raise ValueError(f'The real selector does not offer {value}')
        if index==box.currentIndex():
            return
        click(box)
        view=box.view()
        QTest.keyClick(view,Qt.Key_Home)
        for _ in range(index):QTest.keyClick(view,Qt.Key_Down)
        QTest.keyClick(view,Qt.Key_Return)
        idle()
        if (box.currentData() if data else box.currentText())!=value:
            raise ValueError('The real selector did not reach the requested value')

    def picker(button,path,frame):
        accepted,errors=[],[]
        timer,watchdog=QTimer(window),QTimer(window)
        timer.setSingleShot(True);watchdog.setSingleShot(True)
        def handle():
            dialog=app.activeModalWidget()
            try:
                if not isinstance(dialog,QFileDialog):raise ValueError('Expected genuine file picker')
                dialog.accepted.connect(lambda:accepted.append(True))
                fill(dialog.findChild(QLineEdit,'fileNameEdit'),path)
                capture(frame)
                box=dialog.findChild(QDialogButtonBox)
                buttons=[b for b in box.buttons() if box.buttonRole(b)==QDialogButtonBox.AcceptRole]
                if len(buttons)!=1:raise ValueError('Expected unique picker accept action')
                click(buttons[0])
            except Exception as error:
                errors.append(str(error))
                if dialog is not None:dialog.reject()
        def abort():
            errors.append('Actual file picker timed out')
            dialog=app.activeModalWidget()
            if dialog is not None:dialog.reject()
        timer.timeout.connect(handle);watchdog.timeout.connect(abort)
        timer.start(300);watchdog.start(15000)
        try:click(button)
        finally:
            timer.stop();watchdog.stop();timer.deleteLater();watchdog.deleteLater()
        if errors or not accepted:raise ValueError('; '.join(errors) or 'File not accepted')
        idle()

    def verify(name,*,selected_rows=None,statistic='auc',top=20,extra=False):
        actual_rows=rows if selected_rows is None else selected_rows
        actual_features=features
        if extra:
            actual_rows=[dict(r,area_k=r['cell_area']/1000) for r in actual_rows]
            actual_features=features+['area_k']
        check=verify_ranking(screen.explorer.result,actual_rows,actual_features,statistic=statistic,top=top)
        proof.setdefault('ranking_checks',{})[name]=check
        proof.setdefault('summaries',{})[name]=screen.explorer.summary()
        return check

    def export(name):
        buttons=[b for b in screen.findChildren(QPushButton) if b.text()=='Export ranking…']
        if len(buttons)!=1:raise ValueError('No unique Export ranking control')
        target=work/(name+'.csv')
        expected=screen.ranking_frame().copy()
        picker(buttons[0],target,name+'_picker')
        actual=pd.read_csv(target)
        pd.testing.assert_frame_equal(actual,expected,check_dtype=False,rtol=1e-12,atol=1e-12)
        proof.setdefault('exports',{})[name]={'path':str(target),'rows':len(actual),'columns':list(actual.columns),
                                            'sha256':_digest(target),'all_cells_match_current_ranking':True}
        capture(name+'_complete')

    try:
        tiles=[w for w in window.findChildren(QAbstractButton) if w.isVisible() and
               (w.property('moduleAppKey')=='classify_merged' or w.property('navKey')=='classify_merged')]
        if not tiles:raise ValueError('No actual Home Classify tile')
        click(max(tiles,key=lambda w:w.width()*w.height()))
        wait_for(lambda:window._screens.get('classify_merged') is not None)
        host=window._screens['classify_merged']
        folds=[w for w in host.findChildren(FoldButton) if w.isVisible() and w.app_key=='feature_explorer']
        if len(folds)!=1:raise ValueError('No unique Classify -> Feature Explorer fold')
        capture('01_classify_host')
        click(folds[0])
        panels=[w for w in window.findChildren(FeatureExplorerScreen) if w.isVisible()]
        if len(panels)!=1:raise ValueError('Actual Feature Explorer fold did not open')
        screen=panels[0]
        screen.explorer.ranked.connect(lambda r:outcomes.append({'rows':r.n_rows,'statistic':r.spec.statistic,'top':r.spec.top}))
        capture('02_current_feature_explorer')
        loads=[b for b in screen.findChildren(QPushButton) if b.text()=='Load table…']
        picker(loads[0],work/'measurements.db','03_choose_real_measurements')
        combo(screen._table_picker,'cell')
        if len(screen._frame)!=2341:raise ValueError('Expected full downloaded cell table')
        combo(screen.explorer._label,'rowID')
        verify('initial')
        header=screen.explorer.table.horizontalHeader()
        for section in range(1,6):
            QTest.mouseDClick(header.viewport(),Qt.LeftButton,
                pos=QPoint(header.sectionViewportPosition(section)+header.sectionSize(section)-1,
                           header.height()//2))
        settle(.3)
        capture('04_real_row_comparison')
        fill(screen.explorer._top.lineEdit(),4);idle();verify('top4',top=4)
        capture('05_four_distributions')
        export('06_top_four_ranking')
        fill(screen.explorer._top.lineEdit(),500);idle();verify('all',top=500)
        capture('07_all_ranked_features')
        export('08_full_ranking')
        combo(screen.explorer._statistic,'ks',data=True);verify('ks',statistic='ks',top=500)
        capture('09_ks_comparison')
        combo(screen.explorer._statistic,'auc',data=True);verify('restored_auc',top=500)
        combo(screen.filters._picker,'cell_area')
        click(screen.filters.findChild(QPushButton,'FilterAddButton'));idle()
        range_row=screen.filters._rows['cell_area']
        threshold=sorted(r['cell_area'] for r in rows)[len(rows)//2]
        fill(range_row._low.lineEdit(),threshold);idle()
        filtered=[r for r in rows if r['cell_area']>=threshold]
        verify('area_filtered',selected_rows=filtered,top=500)
        proof['area_threshold']=threshold
        capture('10_live_area_filter')
        click(screen.filters._clear);idle();verify('cleared_area',top=500)
        capture('11_restore_all_cells')
        combo(screen.filters._picker,'rowID')
        click(screen.filters.findChild(QPushButton,'FilterAddButton'));idle()
        category=screen.filters._rows['rowID']
        toggle=next(b for b in category._boxes if b.text()=='r5')
        click(toggle);idle()
        actual_label=screen.explorer._label.currentText()
        proof['one_class_filter_observation']={'selected_label':actual_label,
            'summary':screen.explorer.summary(),'result_label':screen.explorer.result.label if screen.explorer.result else None}
        capture('12_one_row_changes_comparison')
        if actual_label!='cell_channel_0_maxima_count':
            raise ValueError('Investigate the observed post-filter comparison before narrating it')
        groups=Counter(str(r[actual_label]) for r in rows if r['rowID']=='r12' and r[actual_label] is not None)
        changed=screen.explorer.result
        if (changed is None or changed.n_rows!=1201 or set(changed.classes)!=set(groups) or
                len(groups)!=8 or min(groups.values())!=1):
            raise ValueError('Automatic comparison no longer matches the actual remaining rows')
        proof['one_class_filter_observation'].update(
            independently_counted_classes=dict(groups),
            multiclass_scores_not_independently_validated=True,
            not_presented_as_evidence_of_biological_separation=True)
        click(screen.filters._clear);idle()
        proof['label_after_clear']=screen.explorer._label.currentText()
        combo(screen.explorer._label,'rowID');verify('restored_classes',top=500)
        capture('13_restore_two_classes')
        tabs=screen.formulas.parentWidget().parentWidget()
        if not isinstance(tabs,QTabWidget):raise ValueError('Expected actual Filter/Columns tabs')
        QTest.mouseClick(tabs.tabBar(),Qt.LeftButton,pos=tabs.tabBar().tabRect(tabs.indexOf(screen.formulas)).center())
        settle(.3)
        fill(screen.formulas._name,'area_k')
        fill(screen.formulas._expression,'cell_area/1e3')
        wait_for(lambda:screen.formulas._add.isEnabled())
        capture('14_scaled_area_formula')
        click(screen.formulas._add);idle()
        verify('derived_area',top=500,extra=True)
        computed=screen.formulas.computed_frame()
        if any(a!=b/1000 for a,b in zip(computed['area_k'],computed['cell_area'])):
            raise ValueError('Actual formula differs from independent scaling')
        original_score=screen.explorer.result.score_for('cell_area')
        derived=screen.explorer.result.score_for('area_k')
        if original_score.auc!=derived.auc or original_score.ks!=derived.ks:
            raise ValueError('Positive scaling did not preserve actual rank/CDF separation')
        capture('15_derived_feature_ranked')
        export('16_ranking_with_derived_area')
        items=screen.formulas._list
        item=items.item(0)
        QTest.mouseClick(items.viewport(),Qt.LeftButton,pos=items.visualItemRect(item).center())
        remove=next(b for b in screen.formulas.findChildren(QPushButton) if b.text()=='Remove')
        click(remove);idle();verify('formula_removed',top=500)
        capture('17_original_features_restored')
        require_unchanged_source(source,original['source_bundle'])
        if _digest(work/'measurements.db')!=original['database_sha256']:
            raise ValueError('The private source database was modified')
        proof.update(accepted=True,ranking_events=outcomes,remaining_workers=screen.active_jobs(),
                     original_and_private_database_bytes_unchanged=True,shuffle_requested=False,
                     analysis_claim='Descriptive spatial-row separation only; cells are not independent experiments')
    finally:
        write_json(Path(captures)/'feature_explorer_acceptance.json',proof)
        if screen is not None:screen.close()
