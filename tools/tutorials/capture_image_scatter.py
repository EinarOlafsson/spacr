"""Native Image UMAP -> Image Scatter: real cell crops, no invented results."""
import json
from pathlib import Path
import time
import numpy as np
from capture_database import _readonly, _digest
from lineage_data import verify_preserved
from feature_explorer_evidence import read_measurements
from image_scatter_evidence import identity, verify, verify_preview


def record_scatter(app,window,stage,captures,capture,settle,write_json,timeout):
    from PySide6.QtCore import Qt,QTimer,QPoint
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton,QFileDialog,QLineEdit,QDialogButtonBox
    from spacr.qt.screens.image_scatter import ImageScatterScreen
    from spacr.qt.widgets.fold_strip import FoldButton
    from spacr.qt.linked_selection import linked_selection
    prepared=json.loads((Path(stage)/'image_scatter_state'/f'{Path(captures).name}.json').read_text())
    database=Path(prepared['source']['database']);rows,_=read_measurements(database)
    with _readonly(database) as con:
        cols=[d[0] for d in con.execute('SELECT * FROM png_list LIMIT 0').description]
        crop_rows=[dict(zip(cols,r)) for r in con.execute('SELECT * FROM png_list')]
    paths={identity(dict(r,object_label=int(str(r['cell_id']).removeprefix('o')))):r['png_path'] for r in crop_rows}
    if len(rows)!=2341 or len(paths)!=2341:raise ValueError('Wrong genuine crop population')
    proof=dict(lesson='55_image_scatter',accepted=False,source=prepared['source'],private_cache=prepared['cache'],
               synthetic_measurements=False,published=False,checks={},hover_checks=[],selection_events=[],job_errors=[])
    write_json(Path(captures)/'scientific_acceptance.json',proof)
    deadline=time.monotonic()+timeout;screen=None;annotate=None

    def tick():
        if time.monotonic()>deadline:raise TimeoutError('Bounded Image Scatter recording expired')

    def click(w):
        tick()
        if not w.isVisible() or not w.isEnabled() or w.visibleRegion().isEmpty():raise ValueError('Actual Image Scatter control unavailable')
        QTest.mouseClick(w,Qt.LeftButton,pos=w.visibleRegion().boundingRect().center());settle(.2)

    def fill(w,value):
        click(w);QTest.keyClick(w,Qt.Key_A,Qt.ControlModifier);QTest.keyClicks(w,str(value));QTest.keyClick(w,Qt.Key_Tab);settle(.2)

    def idle():
        settle(.3)
        while screen is not None and (screen._jobs.is_busy() or screen._jobs.active_jobs() or screen._hover_timer.isActive()):tick();settle(.1)
        settle(.3)
        if proof['job_errors']:raise ValueError('Image Scatter worker failed: '+str(proof['job_errors']))

    def picker(button,path,name):
        accepted=[];errors=[];timer=QTimer(window);watch=QTimer(window)
        timer.setSingleShot(True);watch.setSingleShot(True)
        def handle():
            dialog=app.activeModalWidget()
            try:
                if not isinstance(dialog,QFileDialog):raise ValueError('Expected real file picker')
                dialog.accepted.connect(lambda:accepted.append(True));dialog.resize(1500,1000)
                fill(dialog.findChild(QLineEdit,'fileNameEdit'),path);capture(name)
                box=dialog.findChild(QDialogButtonBox)
                buttons=[b for b in box.buttons() if box.buttonRole(b)==QDialogButtonBox.AcceptRole]
                if len(buttons)!=1:raise ValueError('No unique actual picker accept action')
                click(buttons[0])
            except Exception as exc:
                errors.append(str(exc))
                if dialog is not None:dialog.reject()
        def abort():
            errors.append('File picker timed out');dialog=app.activeModalWidget()
            if dialog is not None:dialog.reject()
        timer.timeout.connect(handle);watch.timeout.connect(abort);timer.start(300);watch.start(15000)
        try:click(button)
        finally:timer.stop();watch.stop();timer.deleteLater();watch.deleteLater()
        if errors or not accepted:raise ValueError('; '.join(errors) or 'File not accepted')
        idle()

    def combo(box,text):
        idx=box.findText(text)
        if idx<0:raise ValueError('Actual selector lacks '+text)
        click(box);QTest.keyClick(box.view(),Qt.Key_Home)
        for _ in range(idx):QTest.keyClick(box.view(),Qt.Key_Down)
        QTest.keyClick(box.view(),Qt.Key_Return);idle()
        if box.currentText()!=text:raise ValueError('Actual axis/table choice differs')

    def check(name):
        idle();p=verify(screen,rows,paths);proof['checks'][name]=p;capture(name);return p

    def hover(index,name):
        xy=screen.canvas.point_position(index)
        if xy is None:raise ValueError('Intended real point is not plottable')
        pos=QPoint(round(xy[0]),round(xy[1]))
        if screen.canvas.index_at(pos.x(),pos.y())!=index:raise ValueError('Recorded gesture would hit a different nearby object')
        QTest.mouseMove(screen.canvas,pos,delay=100);idle()
        if screen.canvas.hovered!=index:raise ValueError('Actual pointer did not hover the intended point')
        key=identity(rows[index]);path=paths[key]
        p=verify_preview(screen.preview.pixmap(),path,screen.caption.text(),key)
        p.update(source_sha256=_digest(path),index=index,x=screen._x_choice.currentText(),y=screen._y_choice.currentText(),
                 decodes=screen._thumbs.decodes,cache_hits=screen._thumbs.hits,open_button_enabled=screen._open_button.isEnabled())
        proof['hover_checks'].append(p);capture(name);return pos

    def nav(key):
        buttons=[w for w in window.findChildren(QAbstractButton) if w.isVisible() and w.property('navKey')==key]
        if len(buttons)!=1:raise ValueError('No unique actual navigation '+key)
        click(buttons[0]);settle(.8)

    def return_scatter():
        # UMAP's Data navigation group may be collapsed. Home always exposes
        # the real tile; do not invoke an invisible sidebar action.
        nav('__home__')
        tiles=[w for w in window.findChildren(QAbstractButton) if w.isVisible() and
               w.property('moduleAppKey')=='umap']
        if not tiles:raise ValueError('Home has no visible Image UMAP tile')
        click(max(tiles,key=lambda w:w.width()*w.height()));settle(.5)
        if not screen.isVisible():
            folds=[w for w in window._screens['umap'].findChildren(FoldButton) if w.isVisible() and w.app_key=='image_scatter']
            if len(folds)!=1:raise ValueError('Cannot return to existing Image Scatter')
            click(folds[0]);idle()

    try:
        tiles=[w for w in window.findChildren(QAbstractButton) if w.isVisible() and
               (w.property('moduleAppKey')=='umap' or w.property('navKey')=='umap')]
        click(max(tiles,key=lambda w:w.width()*w.height()));settle(.8);capture('01_image_umap_host')
        folds=[w for w in window._screens['umap'].findChildren(FoldButton) if w.isVisible() and w.app_key=='image_scatter']
        if len(folds)!=1:raise ValueError('No Image UMAP Image Scatter fold')
        click(folds[0]);settle(.8)
        screens=[w for w in window.findChildren(ImageScatterScreen) if w.isVisible()]
        if len(screens)!=1:raise ValueError('No visible actual Image Scatter')
        screen=screens[0];screen._jobs.job_failed.connect(lambda t:proof['job_errors'].append(str(t)))
        bus=linked_selection();bus.selection_changed.connect(lambda:proof['selection_events'].append(
            dict(source=bus.selection.source,keys=[] if bus.selection.keys is None else list(bus.selection.keys))))
        idle();capture('02_current_image_scatter')
        picker(screen._browse,database,'03_choose_real_measurements')
        combo(screen._table,'cell');capture('04_choose_cell_not_children');click(screen._load);idle()
        combo(screen._x_choice,'cell_area');combo(screen._y_choice,'cell_channel_1_mean_intensity')
        check('05_real_measurement_scatter')
        high=int(np.argmax([r['cell_area'] for r in rows]));hover(high,'06_genuine_large_cell_preview')
        before=screen._thumbs.decodes
        QTest.mouseMove(screen._db,screen._db.rect().center());idle();capture('07_leaving_clears_hover')
        if screen.preview.text()!='Hover a point' or screen.caption.text() or screen._open_button.isEnabled():
            raise ValueError('Leaving canvas did not clear temporary hover')
        hover(high,'08_cached_same_cell')
        if screen._thumbs.decodes!=before:raise ValueError('Same unchanged crop was decoded twice')
        combo(screen._x_choice,'cell_channel_1_mean_intensity');combo(screen._y_choice,'cell_area');check('09_swapped_raw_axes')
        hover(high,'10_same_cell_after_axis_swap')
        combo(screen._x_choice,'cell_area');combo(screen._y_choice,'cell_channel_0_homogeneity_distance_32')
        p=check('11_missing_texture_not_plotted')
        if p['finite_points']!=2334:raise ValueError('Expected seven missing coordinate pairs')
        combo(screen._y_choice,'measurement_ndim');check('12_constant_axis_is_centred')
        combo(screen._y_choice,'cell_channel_1_mean_intensity');check('13_original_axes_restored')
        nav('annotate');annotate=window._screens['annotate'];capture('14_annotate_before_project')
        picker(annotate._btn_open,prepared['project'],'15_annotate_private_project_picker')
        def crops_ready():
            while (annotate._page_worker is not None or annotate._pending_page_load is not None
                   or not annotate._page_paths or len(annotate._raw_thumb_images)<len(annotate._page_paths)
                   or any(i is None for i in annotate._raw_thumb_images[:len(annotate._page_paths)])):
                tick();settle(.1)
            settle(.5)
        crops_ready();capture('16_real_annotate_grid')
        return_scatter();pos=hover(high,'17_cell_ready_for_click')
        before=screen._open_button.isEnabled()
        QTest.mouseMove(screen._open_button,screen._open_button.rect().center(),delay=100);idle()
        proof['open_button_pointer_exit']=dict(enabled_before=before,enabled_after=screen._open_button.isEnabled(),
                                              hover_after=screen.canvas.hovered,caption_after=screen.caption.text())
        capture('17b_actual_open_button_pointer_exit')
        pos=hover(high,'17c_native_point_click_target')
        QTest.mouseClick(screen.canvas,Qt.LeftButton,pos=pos);settle(.4)
        proof['selection_after_click']=dict(keys=[] if bus.selection.keys is None else list(bus.selection.keys),
             selected_rings=int(screen.canvas._selected.sum()),annotate_visible=annotate.isVisible())
        if bus.selection.keys is None or set(bus.selection.keys)!={identity(rows[high])}:
            raise ValueError('Actual click published a different object')
        nav('annotate');crops_ready();request=annotate._object_request
        if set(request.keys)!={identity(rows[high])} or annotate._total!=1 or len(annotate._page_paths)!=1 or annotate._page_paths[0][0]!=paths[identity(rows[high])]:
            raise ValueError('Actual annotation handoff does not show exactly the clicked cell')
        proof['handoff']=dict(key=identity(rows[high]),paths=list(annotate._page_paths),note=annotate._request_note,
                             sha256=_digest(annotate._page_paths[0][0]),shown=1,labels_edited=False)
        capture('18_verified_clicked_cell_in_annotate');return_scatter();check('19_returned_measurement_scatter')
        proof.update(accepted=True,pipeline_requested=False,own_export_control=False,
                     no_nucleus_or_pathogen_crop_mapping_claim=True)
    finally:
        if annotate is not None:annotate.close();settle(.5)
        if screen is not None:idle();screen.close();settle(.3)
        proof['preservation']=verify_preserved(prepared)
        proof['remaining_workers']=screen._jobs.active_jobs() if screen is not None else 0
        write_json(Path(captures)/'scientific_acceptance.json',proof)
        if __import__('sys').exc_info()[0] is not None:window.close();settle(.5)
