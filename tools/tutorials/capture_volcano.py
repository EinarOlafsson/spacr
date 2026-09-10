"""Record the real Regression fold, existing permutation results and exports."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import tempfile
import time
from capture_database import _digest
from volcano_evidence import read_family,check_plot,check_frame


def record_volcano(app,window,stage,captures,capture,settle,write_json,timeout):
    from PySide6.QtCore import Qt,QTimer,QPoint
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton,QPushButton,QFileDialog,QLineEdit,QDialogButtonBox,QScrollArea,QSplitter,QCheckBox
    from spacr.qt.widgets.fold_strip import FoldButton
    from spacr.qt.widgets.volcano_explorer import VolcanoExplorer

    source=Path(stage)/'regression_runs/example-9wut6lcv/results/guide_permutation'
    parent=Path(stage)/'volcano_runs';parent.mkdir(exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='real-guide-family-',dir=parent));data=work/'results';data.mkdir()
    snapshots={str(p):_digest(p) for p in source.rglob('*') if p.is_file()}
    copies={}
    for name in ('guide_permutation_results_long.csv','results_grna.csv','results.csv','model_summary.txt'):
        shutil.copy2(source/name,data/name);copies[name]=_digest(data/name)
        if copies[name]!=snapshots[str(source/name)]:raise ValueError('Volcano source changed during copy')
    rows,family=read_family(data/'guide_permutation_results_long.csv')
    if len(rows)!=434 or copies['guide_permutation_results_long.csv']!='5a72ae6789482fe3f44a8916486abb1ba5d6d0633c24e25d94f1f670feef22ff':
        raise ValueError('Expected the checked real Regression tutorial guide family')
    proof=dict(lesson='72_volcano_explorer',accepted=False,source_folder=str(source),source_files=snapshots,
               private_folder=str(work),private_input_hashes=copies,independent_family=family,
               app_source_modified=False,synthetic_results=False,published=False)
    deadline=time.monotonic()+timeout;explorer=None;screen=None;events=[];selections=[]

    def tick():
        if time.monotonic()>deadline:raise TimeoutError('Bounded Volcano capture timed out')

    def click(w):
        tick()
        if not w.isVisible() or not w.isEnabled() or w.visibleRegion().isEmpty():
            raise ValueError('Actual Volcano control unavailable: '+w.objectName())
        pos=QPoint(12,w.rect().center().y()) if isinstance(w,QCheckBox) else w.visibleRegion().boundingRect().center()
        QTest.mouseClick(w,Qt.LeftButton,pos=pos);settle(.3)

    def fill(w,value):
        click(w);QTest.keyClick(w,Qt.Key_A,Qt.ControlModifier)
        if str(value):QTest.keyClicks(w,str(value))
        else:QTest.keyClick(w,Qt.Key_Backspace)
        QTest.keyClick(w,Qt.Key_Tab);settle(.35)

    def reveal(w):
        # Real keyboard scrolling, no direct scroll-position or state injection.
        if w.isVisible() and w.visibleRegion().boundingRect().height()>=w.height()-2:return
        bar=scroll.verticalScrollBar();bar.setFocus();QTest.keyClick(bar,Qt.Key_Home);settle(.1)
        while w.visibleRegion().boundingRect().height()<w.height()-2:
            tick();before=bar.value();QTest.keyClick(bar,Qt.Key_Down);settle(.035)
            if before==bar.value():raise ValueError('Actual scrollbar cannot reveal '+w.objectName())

    def control(key):
        section=explorer.section_for(key)
        if section is not None and not section.is_expanded():
            reveal(section.header());click(section.header())
        w=explorer._controls[key];reveal(w);return w

    def combo(key,value,frame=None):
        box=control(key);idx=box.findData(value)
        if idx<0:raise ValueError('Actual Volcano selector has no '+str(value))
        click(box)
        if frame:capture(frame,desktop=True)
        QTest.keyClick(box.view(),Qt.Key_Home)
        for _ in range(idx):QTest.keyClick(box.view(),Qt.Key_Down)
        QTest.keyClick(box.view(),Qt.Key_Return);settle(.5)
        if box.currentIndex()!=idx:raise ValueError('Actual selector did not reach requested value')

    def picker(button,path,name):
        accepted=[];errors=[];timer=QTimer(window);watch=QTimer(window)
        timer.setSingleShot(True);watch.setSingleShot(True)
        def handle():
            dialog=app.activeModalWidget()
            try:
                if not isinstance(dialog,QFileDialog):raise ValueError('Expected genuine file picker')
                dialog.accepted.connect(lambda:accepted.append(True));dialog.resize(1500,1000)
                fill(dialog.findChild(QLineEdit,'fileNameEdit'),path);capture(name)
                box=dialog.findChild(QDialogButtonBox)
                buttons=[b for b in box.buttons() if box.buttonRole(b)==QDialogButtonBox.AcceptRole]
                if len(buttons)!=1:raise ValueError('No unique file-picker accept button')
                click(buttons[0])
            except Exception as error:
                errors.append(str(error))
                if dialog is not None:dialog.reject()
        def abort():
            errors.append('File picker timed out');dialog=app.activeModalWidget()
            if dialog is not None:dialog.reject()
        timer.timeout.connect(handle);watch.timeout.connect(abort);timer.start(300);watch.start(15000)
        try:click(button)
        finally:timer.stop();watch.stop();timer.deleteLater();watch.deleteLater()
        if errors or not accepted:raise ValueError('; '.join(errors) or 'No accepted file')
        settle(.6)

    def action(text):
        buttons=[b for b in explorer.findChildren(QPushButton) if b.text()==text]
        if len(buttons)!=1:raise ValueError('No unique action '+text)
        reveal(buttons[0]);return buttons[0]

    def check(name,**opts):
        settle(.4);p=check_plot(explorer,rows,**opts)
        p['path_label']=screen._path_label.text();p['style']=asdict(explorer.style())
        proof.setdefault('checks',{})[name]=p;capture(name);return p

    try:
        tiles=[w for w in window.findChildren(QAbstractButton) if w.isVisible() and
               (w.property('moduleAppKey')=='regression' or w.property('navKey')=='regression')]
        if not tiles:raise ValueError('No Home Regression tile')
        click(max(tiles,key=lambda w:w.width()*w.height()));settle(.9)
        host=window._screens['regression'];folds=[w for w in host.findChildren(FoldButton) if w.isVisible() and w.app_key=='volcano_explorer']
        if len(folds)!=1:raise ValueError('No unique Regression Volcano Explorer fold')
        capture('01_regression_host');click(folds[0]);settle(.9)
        found=[w for w in window.findChildren(VolcanoExplorer) if w.isVisible()]
        if len(found)!=1:raise ValueError('Actual Volcano Explorer not visible')
        explorer=found[0];screen=explorer.parentWidget()
        explorer.style_changed.connect(lambda:events.append(asdict(explorer.style())))
        explorer.point_selected.connect(lambda d:selections.append({k:str(v) for k,v in d.items()}))
        # This explorer has one settings scroll area; reject ambiguity.
        scrolls=explorer.findChildren(QScrollArea)
        if len(scrolls)!=1:raise ValueError('Expected one settings scrollbar')
        scroll=scrolls[0]
        splitter=scroll.parentWidget()
        if not isinstance(splitter,QSplitter):raise ValueError('Expected actual figure/settings divider')
        handle=splitter.handle(1);start=handle.rect().center();end=start+QPoint(2300-handle.mapTo(window,start).x(),0)
        QTest.mousePress(handle,Qt.LeftButton,pos=start);QTest.mouseMove(handle,end,delay=150)
        QTest.mouseRelease(handle,Qt.LeftButton,pos=end);settle(.4)
        picker(next(b for b in screen.findChildren(QPushButton) if b.text()=='Open results…'),data,'02_choose_completed_results_folder')
        check('03_real_guide_family_no_adjusted_hits')
        proof['initial_control_disagreement']=dict(
            rendered_alpha=explorer.style().alpha,displayed_alpha=explorer._controls['alpha'].value(),
            rendered_multiplier=explorer.style().threshold_multiplier,
            displayed_multiplier=explorer._controls['threshold_multiplier'].value(),
            rendered_log_transform=explorer.style().y_neg_log10,
            displayed_log_transform=explorer._controls['y_neg_log10'].isChecked())
        baseline=work/'baseline_style.json'
        picker(action('Save style'),baseline,'03a_save_rendered_baseline_style')
        picker(action('Load style'),baseline,'03b_load_baseline_to_sync_controls')
        if (explorer._controls['alpha'].value()!=explorer.style().alpha or
            explorer._controls['threshold_multiplier'].value()!=explorer.style().threshold_multiplier or
            explorer._controls['y_neg_log10'].isChecked()!=explorer.style().y_neg_log10):
            raise ValueError('Actual Save then Load style did not synchronise controls')
        check('03c_rendered_baseline_and_controls_agree')
        control('y_column');capture('04_actual_effect_and_adjusted_p_columns')
        # Native exact-coordinate click selects an existing guide, not a made-up point.
        index=max(range(len(rows)),key=lambda i:abs(float(rows[i]['standardized_marginal_effect'])))
        r=rows[index];axis=explorer._panels[0]
        x,y=axis.transData.transform((float(r['standardized_marginal_effect']),-math.log10(float(r['adjusted_p_value']))))
        QTest.mouseClick(explorer._canvas,Qt.LeftButton,pos=QPoint(round(x),round(explorer._canvas.height()-y)));settle(.4)
        if explorer.selected_index()!=index or not selections or selections[-1]['guide']!=r['guide']:
            raise ValueError('Native point click resolved wrong guide identity')
        proof['selected_guide']=dict(position=index,source=r,readout=selections[-1])
        capture('05_selected_real_guide')
        table=explorer._detail_table
        table.verticalScrollBar().setFocus();QTest.keyClick(table.verticalScrollBar(),Qt.Key_End);settle(.3)
        capture('06_selected_plotted_coordinates')
        combo('y_column','permutation_p_value');check('07_raw_p_values_not_new_hits',y_column='permutation_p_value')
        combo('y_column','adjusted_p_value');check('08_adjusted_p_restored')
        click(control('y_neg_log10'));check('09_untransformed_adjusted_p',neglog=False)
        click(control('y_neg_log10'));check('10_log_transform_restored')
        fill(control('alpha').lineEdit(),.5);p=check('11_reference_line_does_not_retest',alpha=.5)
        proof['alpha_semantics']=dict(below_new_cut=p['below_current_y_cut'],still_source_flagged=p['significant_by_source_flags'],new_statistical_test=False)
        fill(control('alpha').lineEdit(),.05);check('12_original_alpha_restored')
        effect=control('effect_threshold');click(effect._auto);fill(effect._spins[0].lineEdit(),.1)
        check('13_explicit_effect_cut_times_multiplier',cut=.1)
        click(control('effect_threshold')._auto);check('14_no_effect_cut_restored')
        combo('color_by','wells_with_guide');check('15_colour_by_actual_support',colour='wells_with_guide')
        combo('color_by',None);check('16_original_significance_colours')
        for key,value in [('figure_width',7),('figure_height',5),('dpi',150),('font_size',12),('label_font_size',12),('tick_font_size',10)]:
            fill(control(key).lineEdit(),value)
        # Close sections through their actual headers before displaying export actions.
        for section in explorer.sections():
            if section.is_expanded():reveal(section.header());click(section.header())
        check('17_publication_size_settings')
        saved=work/'volcano_style.json';picker(action('Save style'),saved,'18_save_actual_style')
        wanted=json.loads(json.dumps(asdict(explorer.style())))
        if json.loads(saved.read_text())!=wanted:raise ValueError('Saved JSON differs from actual Volcano settings')
        fill(control('marker_size').lineEdit(),40);capture('19_temporary_marker_change')
        picker(action('Load style'),saved,'20_load_saved_style')
        if json.loads(json.dumps(asdict(explorer.style())))!=wanted:raise ValueError('Native Load style failed to restore exact style')
        check('21_saved_style_restored')
        output=work/'output/pdf';output.mkdir(parents=True)
        picker(action('Export PDF…'),output/'real_guide_volcano.pdf','22_vector_pdf_export')
        picker(action('Export PNG…'),work/'real_guide_volcano.png','23_png_export')
        proof['exports']={str(p):dict(bytes=p.stat().st_size,sha256=_digest(p)) for p in [saved,output/'real_guide_volcano.pdf',work/'real_guide_volcano.png']}
        check('24_export_finished_source_preserved');proof['accepted']=True
    finally:
        after={str(p):_digest(p) for p in source.rglob('*') if p.is_file()}
        proof['all_source_files_unchanged']=after==snapshots
        proof['all_private_inputs_unchanged']=all(_digest(data/n)==h for n,h in copies.items())
        proof['style_events']=events;proof['point_selections']=selections
        write_json(Path(captures)/'volcano_acceptance.json',proof)
    if not proof['all_source_files_unchanged'] or not proof['all_private_inputs_unchanged']:
        raise ValueError('Original or private Volcano input changed')
