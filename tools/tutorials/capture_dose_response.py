"""Record the genuine Dose Response screen on explicitly synthetic curves."""
import dataclasses
import hashlib
import math
import os
from pathlib import Path
import time

from dose_response_evidence import (prepare,TRUTH,REVERSAL,synthetic_rows,
    check_fit,check_groups,check_profile,check_wald,check_drawn_curves)


def record_dose_response(app,window,stage,captures,capture,settle,write_json,timeout):
    from PySide6.QtCore import Qt,QTimer,QPoint
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton,QPushButton,QLineEdit,QFileDialog,QDialogButtonBox,QSplitter
    from spacr.qt.widgets.dose_response import PROFILE_TOLERANCE
    if PROFILE_TOLERANCE!=.001:raise ValueError('Revisit the independent interval precision before recording')
    inputs=prepare(stage);source=Path(inputs['path']);deadline=time.monotonic()+timeout
    proof=dict(lesson='69_dose_response',accepted=False,inputs=inputs,runs=[],
        app_source_modified=False,provider_contacted=False,published=False,synthetic=True,
        gui_export_action_present=False,job_errors=[])
    reproduce_sort_refit=os.environ.get('SPACR_TUTORIAL_DOSE_SORT_BEFORE_REFIT')=='1'
    proof['sort_before_refit_reproduction']=reproduce_sort_refit
    screen=None

    def clean(value):
        if hasattr(value,'tolist'):return clean(value.tolist())
        if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [clean(v) for v in value]
        if isinstance(value,float) and not math.isfinite(value):return None
        return value

    def tick():
        if time.monotonic()>deadline:raise TimeoutError('Bounded Dose Response recording expired')

    def click(w):
        tick()
        if not w.isVisible() or not w.isEnabled() or w.visibleRegion().isEmpty():
            raise ValueError('Actual Dose Response control unavailable: '+w.objectName())
        QTest.mouseClick(w,Qt.LeftButton,pos=w.visibleRegion().boundingRect().center());settle(.2)

    def idle():
        settle(.2)
        while screen.is_busy() or screen.active_jobs():tick();settle(.05)
        settle(.3)
        if proof['job_errors']:raise ValueError('Dose Response job failed: '+str(proof['job_errors']))

    def fill(field,value):
        if field is None:raise ValueError('No actual file-name field')
        click(field);QTest.keyClick(field,Qt.Key_A,Qt.ControlModifier)
        QTest.keyClicks(field,str(value));QTest.keyClick(field,Qt.Key_Tab);settle(.2)

    def combo(box,text):
        index=box.findText(text)
        if index<0:raise ValueError('Actual selector lacks '+text)
        click(box);QTest.keyClick(box.view(),Qt.Key_Home)
        for _ in range(index):QTest.keyClick(box.view(),Qt.Key_Down)
        QTest.keyClick(box.view(),Qt.Key_Return);settle(.2)
        if box.currentText()!=text:raise ValueError('Actual selector did not change')

    def load():
        buttons=[w for w in screen.findChildren(QPushButton) if w.isVisible() and w.text()=='Load table…']
        if len(buttons)!=1:raise ValueError('No unique actual Load table button')
        accepted=[];errors=[];timer=QTimer(window);watch=QTimer(window)
        timer.setSingleShot(True);watch.setSingleShot(True)
        def handle():
            dialog=app.activeModalWidget()
            try:
                if not isinstance(dialog,QFileDialog):raise ValueError('Expected actual CSV picker')
                dialog.accepted.connect(lambda:accepted.append(True));dialog.resize(1500,1000)
                fill(dialog.findChild(QLineEdit,'fileNameEdit'),source);capture('03_choose_disclosed_synthetic_csv')
                box=dialog.findChild(QDialogButtonBox)
                buttons=[b for b in box.buttons() if box.buttonRole(b)==QDialogButtonBox.AcceptRole]
                if len(buttons)!=1:raise ValueError('No unique actual file accept action')
                click(buttons[0])
            except Exception as exc:
                errors.append(str(exc))
                if dialog is not None:dialog.reject()
        def abort():
            errors.append('File picker timed out');dialog=app.activeModalWidget()
            if dialog is not None:dialog.reject()
        timer.timeout.connect(handle);watch.timeout.connect(abort);timer.start(300);watch.start(15000)
        try:click(buttons[0])
        finally:timer.stop();watch.stop();timer.deleteLater();watch.deleteLater()
        if errors or not accepted:raise ValueError('; '.join(errors) or 'File not accepted')
        idle()
        if screen._path!=str(source) or len(screen._frame)!=120:raise ValueError('Actual loaded source differs')
        import pandas as pd
        pd.testing.assert_frame_equal(screen._frame.reset_index(drop=True),pd.DataFrame(synthetic_rows()),check_exact=False,rtol=1e-12,atol=1e-12)
        capture('04_loaded_120_synthetic_rows')

    def verify_run(name):
        result=screen.result_set()
        if result is None:raise ValueError('No actual fitted result')
        fits={f.group:dict(result=None if f.result is None else clean(dataclasses.asdict(f.result)),error=f.error) for f in result}
        check_groups(fits);run=dict(name=name,spec=screen.spec().to_dict(),groups={},table=clean(result.table().to_dict('records')))
        for f in result:
            if f.result is None:run['groups'][f.group]=dict(status='refused',error=f.error);continue
            record=clean(dataclasses.asdict(f.result));record['status']=f.result.status
            checks=dict(planted=check_fit(f.group,record))
            if f.result.ec50_bounded:
                checks['interval']=check_profile(f.group,record) if record['ci_method']=='profile' else check_wald(f.group,record)
            run['groups'][f.group]=dict(status=f.result.status,record=record,checks=checks)
        if screen.table.rowCount()!=4:raise ValueError('GUI omitted a refused or unbounded row')
        from spacr.qt.screens.dose_response import TABLE_COLUMNS
        expected={str(row['group']):row for row in run['table']}
        proof['pending_run']=run
        proof['last_native_table']=[[None if screen.table.item(r,c) is None else screen.table.item(r,c).text()
            for c in range(screen.table.columnCount())] for r in range(screen.table.rowCount())]
        if {screen.table.item(row,0).text() for row in range(4)}!=set(expected):
            raise ValueError('GUI duplicates or omits a result identity')
        for row in range(4):
            name=screen.table.item(row,0).text()
            if name not in expected:raise ValueError('GUI group identity differs')
            for column,(key,_) in enumerate(TABLE_COLUMNS):
                value=expected[name][key]
                text='—' if value is None else f'{value:.4g}' if isinstance(value,float) else str(value)
                if key=='note' and len(text)>90:text=text[:90].rstrip()+'…'
                if screen.table.item(row,column).text()!=text:raise ValueError('GUI table differs: '+name+' '+key)
        run['drawn']=check_drawn_curves(screen._figure.axes[0]);proof['runs'].append(run)
        proof.pop('pending_run',None)
        return result

    def select(group,frame):
        rows=[r for r in range(screen.table.rowCount()) if screen.table.item(r,0).text()==group]
        if len(rows)!=1:raise ValueError('No unique actual results row')
        item=screen.table.item(rows[0],0);screen.table.scrollToItem(item);settle(.1)
        QTest.mouseClick(screen.table.viewport(),Qt.LeftButton,pos=screen.table.visualItemRect(item).center());settle(.3)
        fit=screen.result_set().get(group)
        expected=fit.result.report() if fit.result is not None else f'{group}: REFUSED\n\n{fit.error}'
        if screen.report.toPlainText()!=expected:raise ValueError('Selected row shows another group report')
        check_drawn_curves(screen._figure.axes[0])
        proof.setdefault('selected_reports',{})[frame]=dict(group=group,report=expected)
        capture(frame)

    def sort_and_copy():
        header=screen.table.horizontalHeader()
        QTest.mouseClick(header.viewport(),Qt.LeftButton,pos=QPoint(header.sectionViewportPosition(0)+30,header.height()//2));settle(.3)
        select('SYNTHETIC inhibition','16_select_after_sorting')
        click(screen.report);QTest.keyClick(screen.report,Qt.Key_A,Qt.ControlModifier);QTest.keyClick(screen.report,Qt.Key_C,Qt.ControlModifier)
        if app.clipboard().text()!=screen.report.toPlainText():raise ValueError('Native report copy differs')
        proof['report_copied']=app.clipboard().text();capture('17_copy_actual_report')
        QTest.keyClick(screen.report,Qt.Key_Right);settle(.2)

    try:
        # Data is a visible band of Home tiles, not a Home tab.
        capture('01_home_data')
        tiles=[w for w in window.findChildren(QAbstractButton) if w.isVisible() and
               (w.property('moduleAppKey')=='dose_response' or w.property('navKey')=='dose_response')]
        if not tiles:raise ValueError('No actual Dose Response Home tile')
        click(max(tiles,key=lambda w:w.width()*w.height()));settle(.6)
        screen=window._screens['dose_response'];screen._jobs.job_failed.connect(lambda text:proof['job_errors'].append(str(text)))
        capture('02_empty_dose_response');load()
        splitters=[w for w in screen.findChildren(QSplitter) if w.orientation()==Qt.Horizontal and w.isVisible()]
        if len(splitters)!=1:raise ValueError('No unique native plot/results splitter')
        handle=splitters[0].handle(1);start=handle.rect().center()
        end=start+QPoint(2000-handle.mapTo(window,start).x(),0)
        QTest.mousePress(handle,Qt.LeftButton,pos=start);QTest.mouseMove(handle,end,delay=150)
        QTest.mouseRelease(handle,Qt.LeftButton,pos=end);settle(.3)
        combo(screen.concentration_picker,'concentration');combo(screen.response_picker,'response')
        combo(screen.group_picker,'series');fill(screen.unit_edit,'uM')
        if screen.force_check.isChecked() or screen.ci_picker.currentData()!='profile':raise ValueError('Unexpected initial fit policy')
        capture('05_explicit_columns_and_profile')
        click(screen.fit_button);idle();verify_run('profile');capture('06_profile_results')
        select('SYNTHETIC inhibition','07_inhibition_midpoint')
        select('SYNTHETIC activation','08_activation_midpoint')
        select('SYNTHETIC beyond range','09_unbounded_not_a_point_estimate')
        select(REVERSAL,'10_refused_reversing_series')
        if reproduce_sort_refit:sort_and_copy()
        combo(screen.ci_picker,'Wald (symmetric, always finite)');capture('13_choose_wald')
        click(screen.fit_button);idle();verify_run('wald');select('SYNTHETIC inhibition','14_wald_results')
        combo(screen.ci_picker,'Profile likelihood (can decline to close)')
        click(screen.fit_button);idle();verify_run('profile_restored');select('SYNTHETIC inhibition','15_profile_restored')
        if not reproduce_sort_refit:sort_and_copy()
        proof['accepted']=True
    except Exception as exc:
        proof['failure']=str(exc)
        if screen is not None:capture('99_failure_state')
        raise
    finally:
        if screen is not None:screen.close()
        window.close();settle(.3)
        proof['source_unchanged']=hashlib.sha256(source.read_bytes()).hexdigest()==inputs['sha256']
        proof['active_jobs_after_close']=screen.active_jobs() if screen is not None else 0
        if not proof['source_unchanged'] or proof['active_jobs_after_close']:proof['accepted']=False
        write_json(captures/'dose-response-workflow.json',proof)
        write_json(captures/'scientific_acceptance.json',proof)
    if not proof['accepted']:raise ValueError('Dose Response did not pass native acceptance')
