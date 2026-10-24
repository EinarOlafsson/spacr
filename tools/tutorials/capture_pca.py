"""Drive genuine PCA controls on a private copy of the downloadable cell table."""
from pathlib import Path
import tempfile
import time
import numpy as np
import capture_ui
import capture_ui_drv_d as drv
from capture_database import prepare_database_copy, require_unchanged_source, _digest
from feature_explorer_evidence import read_measurements
from pca_evidence import verify, verify_csv, close


def capture_pca_host(window, capture, settle):
    """Record the real PCA entry point without loading data or fitting PCA."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton
    from spacr.qt.widgets.fold_strip import FoldButton

    tiles = [w for w in window.findChildren(QAbstractButton)
             if w.isVisible() and w.isEnabled()
             and (w.property('moduleAppKey') == 'umap' or w.property('navKey') == 'umap')]
    if not tiles:
        raise ValueError('No actual Home Image UMAP tile')
    tile = max(tiles, key=lambda w: w.width() * w.height())
    if tile.visibleRegion().isEmpty():
        raise ValueError('Image UMAP tile is not exposed')
    with capture_ui.clip(capture, '01_image_umap_host'):
        QTest.mouseClick(tile, Qt.LeftButton, pos=tile.visibleRegion().boundingRect().center())
        settle(.8)
        host = window._screens['umap']
        folds = [w for w in host.findChildren(FoldButton)
                 if w.isVisible() and w.isEnabled() and w.app_key == 'pca']
        if len(folds) != 1 or folds[0].visibleRegion().isEmpty():
            raise ValueError('No unique exposed Image UMAP PCA fold')
        capture_ui.point_at(capture, folds[0])
    return folds[0]


def record_pca(app,window,stage,captures,capture,settle,write_json,timeout):
    from PySide6.QtCore import Qt,QTimer,QPoint
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QPushButton,QFileDialog,QLineEdit,QDialogButtonBox
    from spacr.qt.screens.pca import PCAScreen

    parent=Path(stage)/'pca_runs';parent.mkdir(exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='real-measurements-',dir=parent))
    source=Path(stage)/'annotate_fresh/example_data/plate1/measurements/measurements.db'
    original=prepare_database_copy(source,work/'measurements.db',
        expected_sha256='7b18161f0161d39b3ecedf92cfb0ccf9fee2328980da8e43167555a8f6fd27cd')
    rows,_=read_measurements(work/'measurements.db')
    if len(rows)!=2341:raise ValueError('Expected genuine full cell table')
    proof=dict(lesson='60_pca',accepted=False,source=original,synthetic_measurements=False,
               app_source_modified=False,published=False)
    deadline=time.monotonic()+timeout;screen=None;events=[];failures=[]
    chosen=('cell_area','cell_channel_1_mean_intensity','cell_perimeter')
    missing='cell_channel_0_homogeneity_distance_32'

    def tick():
        if time.monotonic()>deadline:raise TimeoutError('Bounded PCA recording timed out')

    def click(w):
        tick()
        if not w.isVisible() or not w.isEnabled() or w.visibleRegion().isEmpty():
            raise ValueError('Actual PCA control unavailable: '+w.objectName())
        QTest.mouseClick(w,Qt.LeftButton,pos=w.visibleRegion().boundingRect().center());settle(.2)

    def idle():
        settle(.3)
        while screen.is_busy() or screen.active_jobs() or screen._refilter.isActive() or screen.filters._debounce.isActive() or screen.pca.canvas._debounce.isActive():
            tick();settle(.1)
        settle(.4)

    def fill(w,value):
        click(w);QTest.keyClick(w,Qt.Key_A,Qt.ControlModifier)
        if str(value):QTest.keyClicks(w,str(value))
        else:QTest.keyClick(w,Qt.Key_Backspace)
        QTest.keyClick(w,Qt.Key_Tab);settle(.25)

    def open_popup(box,frame):
        """Clip ``frame``: the selector's real popup opened and read."""
        started=drv.begin(capture,settle,frame)
        click(box);capture_ui.wait(capture,1.5,settle)
        if not capture_ui.live(capture):capture(frame,desktop=True)
        drv.finish(capture,started)

    def combo(box,value,*,data=True,popup_open=False):
        idx=box.findData(value) if data else box.findText(value)
        if idx<0:raise ValueError('Actual PCA selector has no '+str(value))
        if not popup_open:click(box)
        QTest.keyClick(box.view(),Qt.Key_Home)
        for _ in range(idx):QTest.keyClick(box.view(),Qt.Key_Down)
        QTest.keyClick(box.view(),Qt.Key_Return);idle()
        if box.currentIndex()!=idx:raise ValueError('Actual selector did not reach requested value')

    def picker(button,path,name):
        accepted=[];errors=[];timer=QTimer(window);watch=QTimer(window)
        timer.setSingleShot(True);watch.setSingleShot(True)
        def handle():
            dialog=app.activeModalWidget()
            try:
                if not isinstance(dialog,QFileDialog):raise ValueError('Expected genuine file picker')
                dialog.accepted.connect(lambda:accepted.append(True));capture_ui.dialog_geometry(capture,dialog,(1500,1000))
                fill(dialog.findChild(QLineEdit,'fileNameEdit'),path);drv.snap(capture,name);capture_ui.wait(capture,1.5,settle)
                box=dialog.findChild(QDialogButtonBox)
                buttons=[b for b in box.buttons() if box.buttonRole(b)==QDialogButtonBox.AcceptRole]
                if len(buttons)!=1:raise ValueError('No unique file-picker accept control')
                click(buttons[0])
            except Exception as error:
                errors.append(str(error))
                if dialog is not None:dialog.reject()
        def abort():
            errors.append('File picker timed out');dialog=app.activeModalWidget()
            if dialog is not None:dialog.reject()
        watch.timeout.connect(abort);watch.start(120000 if getattr(capture, 'host', None) is not None else 15000)
        started=drv.begin(capture,settle,name);capture_ui.when_modal(capture,300,handle)
        try:click(button)
        finally:drv.finish(capture,started);timer.stop();watch.stop();timer.deleteLater();watch.deleteLater()
        if errors or not accepted:raise ValueError('; '.join(errors) or 'No accepted file')
        idle()

    def feature(name,on):
        fill(panel.features._search,name);listing=panel.features._list
        matches=[listing.item(i) for i in range(listing.count()) if listing.item(i).data(Qt.UserRole)==name and not listing.item(i).isHidden()]
        if len(matches)!=1:raise ValueError('No unique actual feature '+name)
        item=matches[0];listing.scrollToItem(item);settle(.1)
        rect=listing.visualItemRect(item)
        QTest.mouseClick(listing.viewport(),Qt.LeftButton,pos=QPoint(rect.left()+12,rect.center().y()));settle(.2)
        if (item.checkState()==Qt.Checked)!=on:raise ValueError('Actual feature checkbox did not change')

    def check(name,population=None,features=None,scaling='zscore',policy='auto',components=3,act=None):
        started=drv.begin(capture,settle,name)
        if act is not None:act()
        idle();data=rows if population is None else population;fs=chosen if features is None else features
        if panel.result is None:raise ValueError('PCA has no result: '+panel.report.text())
        p=verify(panel.result,data,fs,scaling=scaling,policy=policy,components=components)
        axes=panel.canvas.panel_axes()
        if len(axes)!=1:raise ValueError('Expected one genuine PCA score panel')
        ax=next(iter(axes.values()));plane=panel.canvas.plane()
        points=np.concatenate([np.asarray(c.get_offsets(),float) for c in ax.collections if len(c.get_offsets())])
        wanted=panel.result.scores[:,list(plane)]
        # Categorical colour legitimately reorders the drawing into groups.
        ordered=lambda a:a[np.lexsort((a[:,1],a[:,0]))]
        p['painted_score_error']=close(ordered(points),ordered(wanted),'actual drawn scores')
        p.update(plane=list(plane),drawn_points=len(points),source_label=screen._source.text(),
                 report=panel.report.text(),arrow_scale=panel.canvas.arrow_scale,
                 selected_features=list(panel.features.selected()))
        proof.setdefault('checks',{})[name]=p;capture_ui.point_at(capture,panel.canvas);drv.snap(capture,name);drv.finish(capture,started);return p

    try:
        fold = capture_pca_host(window, capture, settle)
        scene2=drv.begin(capture,settle,'02_current_pca')
        click(fold);settle(.8)
        screens=[w for w in window.findChildren(PCAScreen) if w.isVisible()]
        if len(screens)!=1:raise ValueError('Actual PCA fold not visible')
        screen=screens[0];panel=screen.pca
        panel.computed.connect(lambda r:events.append(dict(rows=len(r),features=list(r.features),policy=r.nan_policy,scaling=r.scaling)))
        panel.failed.connect(lambda text:failures.append(text))
        splitter=panel.splitter;handle=splitter.handle(1)
        start=handle.rect().center();end=start+QPoint(1400-handle.mapTo(window,start).x(),0)
        QTest.mousePress(handle,Qt.LeftButton,pos=start);QTest.mouseMove(handle,end,delay=150)
        QTest.mouseRelease(handle,Qt.LeftButton,pos=end);settle(.3)
        capture_ui.point_at(capture,panel.features);drv.snap(capture,'02_current_pca');drv.finish(capture,scene2)
        picker(next(b for b in screen.findChildren(QPushButton) if b.text()=='Load table…'),work/'measurements.db','03_real_measurement_picker')
        proof['initial_auto_fit']=dict(rows=len(panel.result) if panel.result else 0,report=panel.report.text(),independently_validated=False)
        def choose_features():
            fill(panel.features._search,'')
            click(next(b for b in panel.features.findChildren(QPushButton) if b.text()=='None'))
            for name in chosen:feature(name,True)
            fill(panel.features._search,'')
        with capture_ui.clip(capture,'04_three_features_before_run'):
            choose_features();capture_ui.point_at(capture,panel.features)
        def run_three():
            fill(panel._components.lineEdit(),3);idle();click(panel._run)
        base=check('05_standardised_three_features',act=run_three)
        old=panel.result
        check('06_spatial_row_colours',act=lambda:combo(panel._colour,'rowID'))
        if panel.result is not old:raise ValueError('Colour unexpectedly refitted PCA')
        check('07_pc3_against_pc1',act=lambda:(combo(panel._pc_x,'PC3'),combo(panel._pc_y,'PC1')))
        if panel.result is not old:raise ValueError('Axis choice unexpectedly refitted PCA')
        check('08_arrows_hidden',act=lambda:(combo(panel._pc_x,'PC1'),combo(panel._pc_y,'PC2'),click(panel._biplot)))
        if panel.canvas.arrow_scale!=0:raise ValueError('Hidden loadings retained display scale')
        check('09_biplot_restored',act=lambda:click(panel._biplot))
        open_popup(panel._scaling,'10_actual_scaling_menu')
        check('11_raw_units_pca',scaling='none',act=lambda:combo(panel._scaling,'none',popup_open=True))
        check('12_standardisation_restored',scaling='zscore',act=lambda:combo(panel._scaling,'zscore'))
        def add_filter():
            combo(screen.filters._picker,'cell_area',data=False);click(screen.filters.findChild(QPushButton,'FilterAddButton'))
            fill(screen.filters._rows['cell_area']._low.lineEdit(),21925)
        selected=[r for r in rows if r['cell_area']>=21925]
        p=check('13_live_filter_refits_basis',selected,act=add_filter)
        if p['analysed_rows']!=1171 or p['centre']==base['centre']:raise ValueError('Filter did not genuinely refit population')
        p=check('14_clear_restores_full_basis',act=lambda:click(screen.filters._clear))
        if p['centre']!=base['centre']:raise ValueError('Clear did not restore original PCA basis')
        with capture_ui.clip(capture,'15_missing_feature_ticked_not_run'):
            feature(missing,True)
            capture_ui.point_at(capture,panel.features)
        if panel.result.n_features!=3:raise ValueError('Feature ticks unexpectedly recomputed before Run')
        fs=(*chosen,missing);check('16_auto_seven_rows_removed',features=fs,act=lambda:click(panel._run))
        open_popup(panel._nan,'17_missing_value_choices')
        check('18_drop_feature_keeps_objects',features=fs,policy='drop_features',act=lambda:combo(panel._nan,'drop_features',popup_open=True))
        check('19_mean_imputes_seven_values',features=fs,policy='mean',act=lambda:combo(panel._nan,'mean'))
        check('20_complete_cases',features=fs,policy='complete',act=lambda:combo(panel._nan,'complete'))
        def restore():
            combo(panel._nan,'auto');feature(missing,False);fill(panel.features._search,'');click(panel._run)
            combo(panel._colour,'')
        check('21_original_three_features_restored',act=restore)
        stem=work/'pca_example';picker(screen._export,stem.with_suffix('.csv'),'22_actual_three_file_export')
        proof['exports']=verify_csv(stem,rows,panel.result)
        for p in proof['exports'].values():p['sha256']=_digest(p['path'])
        with capture_ui.clip(capture,'23_export_confirmation'):
            capture_ui.point_at(capture,screen._export)
        proof['accepted']=True
    finally:
        require_unchanged_source(source,original['source_bundle']);proof['original_unchanged']=True
        proof['private_database_unchanged']=_digest(work/'measurements.db')==original['database_sha256']
        proof['compute_events']=events;proof['failures']=failures
        if screen is not None:idle();proof['remaining_workers']=screen.active_jobs()
        write_json(Path(captures)/'pca_acceptance.json',proof)
    if not proof['private_database_unchanged']:raise ValueError('Private PCA database changed')
