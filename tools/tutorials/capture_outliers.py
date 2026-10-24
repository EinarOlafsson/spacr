"""Record QC's actual Outliers fold on a private copy of downloaded cells."""
from pathlib import Path
import csv
import math
from numbers import Real
import tempfile
import time

import capture_ui
import capture_ui_drv_d as drv
from capture_database import prepare_database_copy, require_unchanged_source, _digest
from feature_explorer_evidence import read_measurements
from outliers_evidence import verify_scan, verify_unscored_wells


def record_outliers(app, window, stage, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import Qt, QTimer, QPoint
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton, QFileDialog, QDialogButtonBox, QLineEdit, QPushButton
    from spacr.qt.widgets.fold_strip import FoldButton
    from spacr.qt.screens.outliers import OutliersScreen
    import pandas as pd

    parent=Path(stage)/'outliers_runs';parent.mkdir(exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='real-measurements-',dir=parent))
    source=Path(stage)/'annotate_fresh/example_data/plate1/measurements/measurements.db'
    original=prepare_database_copy(source,work/'measurements.db')
    rows,_=read_measurements(work/'measurements.db')
    if len(rows)!=2341:raise ValueError('Expected full real downloaded cell table')
    proof=dict(lesson='66_outliers',accepted=False,source=original,
               synthetic_measurements=False,app_source_modified=False,published=False)
    panel=None;events=[];failures=[];deadline=time.monotonic()+timeout

    def tick():
        if time.monotonic()>deadline:raise TimeoutError('Bounded Outliers capture timed out')

    def click(widget):
        tick()
        if not widget.isVisible() or not widget.isEnabled() or widget.visibleRegion().isEmpty():
            raise ValueError('Actual Outliers control is not available')
        QTest.mouseClick(widget,Qt.LeftButton,pos=widget.visibleRegion().boundingRect().center())
        settle(.15)

    def idle():
        settle(.2)
        while panel.is_busy() or panel.active_jobs():
            tick();settle(.1)
        settle(.3)

    def shot(name,act=None,*targets):
        """Clip ``name``: the action performed with the pointer, then the result read."""
        started=drv.begin(capture,settle,name)
        if act is not None:act()
        capture_ui.point_at(capture,*(targets or (panel.object_table,)))
        drv.snap(capture,name);drv.finish(capture,started)

    def fill(widget,value):
        if capture_ui.live(capture):
            if str(value):
                capture_ui.set_text(capture,widget,str(value))
            else:
                h=capture_ui._h();h.click(widget);h.key(widget,Qt.Key_A,Qt.ControlModifier)
                h.key(widget,Qt.Key_Backspace);h.key(widget,Qt.Key_Tab)
            settle(.2)
            return
        click(widget);QTest.keyClick(widget,Qt.Key_A,Qt.ControlModifier)
        if str(value):QTest.keyClicks(widget,str(value))
        else:QTest.keyClick(widget,Qt.Key_Backspace)
        QTest.keyClick(widget,Qt.Key_Tab);settle(.2)

    def combo(widget,value,data=True):
        index=widget.findData(value) if data else widget.findText(value)
        if index<0:raise ValueError('Actual selector has no requested option')
        if index!=widget.currentIndex():
            if capture_ui.live(capture):
                capture_ui.select_combo(capture,widget,index);idle()
            else:
                click(widget);view=widget.view()
                if widget.objectName()=='OutlierMethod':capture('method_choices_'+value,desktop=True)
                QTest.keyClick(view,Qt.Key_Home)
                for _ in range(index):QTest.keyClick(view,Qt.Key_Down)
                QTest.keyClick(view,Qt.Key_Return);idle()
        if widget.currentIndex()!=index:raise ValueError('Actual selector picked wrong item')

    def tab(index):
        if capture_ui.live(capture):
            capture_ui.select_tab(capture,panel.tabs,index);settle(.3)
            return
        QTest.mouseClick(panel.tabs.tabBar(),Qt.LeftButton,
                         pos=panel.tabs.tabBar().tabRect(index).center());settle(.3)

    def picker(button,path,frame,pre=None):
        accepted,errors=[],[]
        timer,watchdog=QTimer(window),QTimer(window)
        timer.setSingleShot(True);watchdog.setSingleShot(True)
        def handle():
            dialog=app.activeModalWidget()
            try:
                if not isinstance(dialog,QFileDialog):raise ValueError('Expected actual file picker')
                dialog.accepted.connect(lambda:accepted.append(True))
                fill(dialog.findChild(QLineEdit,'fileNameEdit'),path);drv.snap(capture,frame);capture_ui.wait(capture,1.5,settle)
                box=dialog.findChild(QDialogButtonBox)
                buttons=[b for b in box.buttons() if box.buttonRole(b)==QDialogButtonBox.AcceptRole]
                if len(buttons)!=1:raise ValueError('Expected unique picker accept button')
                click(buttons[0])
            except Exception as error:
                errors.append(str(error))
                if dialog is not None:dialog.reject()
        def abort():
            errors.append('Actual file picker timed out')
            dialog=app.activeModalWidget()
            if dialog is not None:dialog.reject()
        watchdog.timeout.connect(abort)
        watchdog.start(120000 if getattr(capture, 'host', None) is not None else 15000)
        started=drv.begin(capture,settle,frame)
        if pre is not None:pre()
        if callable(button):button=button()
        capture_ui.when_modal(capture,300,handle)
        try:click(button)
        finally:
            drv.finish(capture,started)
            timer.stop();watchdog.stop();timer.deleteLater();watchdog.deleteLater()
        if errors or not accepted:raise ValueError('; '.join(errors) or 'File not accepted')
        idle()

    def feature(name):
        previous=panel.result
        fill(panel.features._search,'')
        buttons=[b for b in panel.features.findChildren(QPushButton) if b.text()=='None']
        click(buttons[0])
        if panel.features.selected():raise ValueError('None did not clear actual selection')
        fill(panel.features._search,name)
        items=[panel.features._list.item(i) for i in range(panel.features._list.count())
               if panel.features._list.item(i).data(Qt.UserRole)==name]
        if len(items)!=1:raise ValueError('Feature is not offered by actual picker')
        rect=panel.features._list.visualItemRect(items[0])
        QTest.mouseClick(panel.features._list.viewport(),Qt.LeftButton,
                         pos=QPoint(rect.left()+10,rect.center().y()));settle(.3)
        if panel.features.selected()!=(name,):raise ValueError('Actual checkbox did not select exact feature')
        fill(panel.features._search,'')
        if panel.result is not previous:raise ValueError('Feature change unexpectedly rescanned')

    def scan(name,selected='cell_area',act=None):
        started=drv.begin(capture,settle,name)
        if act is not None:act()
        before=len(events);click(panel.scan_button);idle()
        if len(events)!=before+1 or panel.result is None:raise ValueError('Expected one completed real Scan')
        spec=panel.spec();r=panel.result
        result=verify_scan(r,rows,selected,method=spec.method,transform=spec.transform,
                           threshold=spec.k if spec.method=='mad' else spec.c)
        if spec.per_well:result['wells']=verify_unscored_wells(r,rows,selected)
        elif r.has_wells:raise ValueError('Per-well switch did not disable comparison')
        pd.testing.assert_frame_equal(panel.objects_frame()[panel.frame.columns],panel.frame)
        result.update(table_rows=panel.object_table.rowCount(),headline=panel._source.text(),
                      report=panel.report.toPlainText(),per_well=spec.per_well)
        if result['table_rows']!=500:raise ValueError('Expected bounded 500-row object display')
        proof.setdefault('scans',{})[name]=result
        tab(0);capture_ui.point_at(capture,panel.object_table)
        drv.snap(capture,name);drv.finish(capture,started)

    def export(name):
        target=work/(name+'.csv');before=panel.objects_frame().copy()
        picker(panel._export,target,name+'_picker')
        expected={'objects':before,'flagged':before.loc[panel.result.flags],
                  'wells':panel.result.well_frame()}
        files={}
        for suffix,frame in expected.items():
            path=work/(name+'_'+suffix+'.csv')
            with path.open(newline='') as stream:
                reader=csv.DictReader(stream);actual=list(reader)
                if reader.fieldnames!=list(frame.columns) or len(actual)!=len(frame):
                    raise ValueError('CSV columns or full population differs')
            for written,values in zip(actual,frame.itertuples(index=False,name=None)):
                for key,value in zip(frame.columns,values):
                    cell=written[key]
                    if pd.isna(value):equal=cell==''
                    elif isinstance(value,bool):equal=cell==str(value)
                    elif isinstance(value,Real):equal=math.isclose(float(cell),float(value),rel_tol=1e-12,abs_tol=1e-12)
                    else:equal=cell==str(value)
                    if not equal:raise ValueError('Exported CSV cell differs from actual scan')
            files[suffix]=dict(path=str(path),rows=len(actual),columns=len(frame.columns),sha256=_digest(path))
        report=work/(name+'_report.txt')
        if report.read_text(encoding="utf-8")!=panel.result.report()+'\n':raise ValueError('Exported report differs')
        files['report']=dict(path=str(report),sha256=_digest(report))
        proof.setdefault('exports',{})[name]=files
        with capture_ui.clip(capture,name+'_complete'):
            capture_ui.point_at(capture,panel._export)

    try:
        home=window._startup;tiles=[]
        scene1=drv.begin(capture,settle,'01_qc_home_route')
        for i in range(home._tabs.count()):
            if capture_ui.live(capture):
                capture_ui.select_tab(capture,home._tabs,i)
            else:
                QTest.mouseClick(home._tabs.tabBar(),Qt.LeftButton,pos=home._tabs.tabBar().tabRect(i).center())
            settle(.3)
            tiles=[w for w in window.findChildren(QAbstractButton) if w.isVisible() and
                   (w.property('moduleAppKey')=='qc_dashboard' or w.property('navKey')=='qc_dashboard')]
            if tiles:break
        if not tiles:raise ValueError('No visible actual QC Home tile')
        tile=max(tiles,key=lambda w:w.width()*w.height())
        capture_ui.point_at(capture,tile)
        drv.snap(capture,'01_qc_home_route');drv.finish(capture,scene1)
        scene2=drv.begin(capture,settle,'02_qc_host')
        click(tile)
        settle(1)
        host=window._screens['qc_dashboard']
        folds=[w for w in host.findChildren(FoldButton) if w.isVisible() and w.app_key=='outliers']
        if len(folds)!=1:raise ValueError('No unique QC Outliers fold')
        capture_ui.point_at(capture,folds[0])
        drv.snap(capture,'02_qc_host');drv.finish(capture,scene2)
        def open_fold():
            click(folds[0])
        def load_button():
            nonlocal panel
            panels=[w for w in window.findChildren(OutliersScreen) if w.isVisible()]
            if len(panels)!=1:raise ValueError('Actual QC Outliers fold did not open')
            panel=panels[0]
            panel.scanned.connect(lambda r:events.append(dict(rows=r.n_rows_in,features=len(r.features),flagged=r.n_flagged)))
            panel.failed.connect(failures.append)
            loads=[b for b in panel.findChildren(QPushButton) if b.text()=='Load table…']
            return loads[0]
        load_started=time.monotonic()
        picker(load_button,work/'measurements.db','03_choose_real_database',pre=open_fold)
        shot('04_automatic_all_features',lambda:combo(panel._table_picker,'cell',False))
        proof['initial_load_and_auto_scan_seconds']=time.monotonic()-load_started
        if len(panel.frame)!=2341:raise ValueError('Actual cell table population differs')
        proof['automatic_initial_scan']=events[-1]
        shot('05_choose_one_feature_before_scan',lambda:feature('cell_area'),panel.features)
        scan('06_area_mad_3_5');shot('07_four_wells_not_scored',lambda:tab(1))
        shot('08_area_report',lambda:tab(2),panel.report)
        scan('09_area_mad_2',act=lambda:fill(panel.threshold,2))
        scan('10_broad_rule_not_quality_cutoff',act=lambda:fill(panel.threshold,.5));export('11_all_rows_not_display_cap')
        scan('12_restore_mad',act=lambda:fill(panel.threshold,3.5))
        scan('13_area_iqr',act=lambda:combo(panel.method,'iqr'))
        shot('14_iqr_report',lambda:tab(2),panel.report)
        scan('15_log_area_iqr',act=lambda:combo(panel.transform,'log10'))
        scan('16_log_area_mad',act=lambda:combo(panel.method,'mad'))
        scan('17_real_intensity_raw',selected='cell_channel_0_min_intensity',
             act=lambda:(combo(panel.transform,'none'),feature('cell_channel_0_min_intensity')))
        before=len(failures)
        def refuse():
            combo(panel.transform,'log10');click(panel.scan_button);idle()
            tab(2)
        shot('18_real_zero_refusal',refuse,panel.report)
        if (len(failures)!=before+1 or panel.result is not None or panel._export.isEnabled()
                or '1 value(s) are zero or negative' not in failures[-1]
                or 'cell_channel_0_min_intensity (1)' not in failures[-1]):
            raise ValueError('Expected explicit refusal of the actual zero, with export disabled')
        proof['log_refusal']=dict(message=failures[-1],real_nonpositive_values=sum(r['cell_channel_0_min_intensity']<=0 for r in rows),
            old_objects_tab=panel.tabs.tabText(0),old_table_rows=panel.object_table.rowCount(),export_enabled=False)
        scan('19_restored_after_refusal',act=lambda:(combo(panel.transform,'none'),feature('cell_area')))
        scan('20_high_minimum_not_clean',act=lambda:fill(panel.min_well,700));shot('21_low_count_reasons',lambda:tab(1))
        scan('22_objects_only',act=lambda:(fill(panel.min_well,20),click(panel.per_well)))
        scan('23_original_rule_and_wells',act=lambda:click(panel.per_well));export('24_final_flags_and_report')
        proof['accepted']=True
    finally:
        require_unchanged_source(source,original['source_bundle'])
        proof['original_unchanged']=True
        proof['private_database_unchanged']=_digest(work/'measurements.db')==original['database_sha256']
        proof['scan_events']=events;proof['failures']=failures
        if panel is not None:
            idle();proof['remaining_workers']=panel.active_jobs()
        write_json(Path(captures)/'outliers_acceptance.json',proof)
    if not proof['private_database_unchanged']:raise ValueError('Private measurement database changed')
