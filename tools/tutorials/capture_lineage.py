"""Drive real containment trees and cell-only crop handoff on isolated data."""
import csv
import json
from pathlib import Path
import time

import capture_ui
import capture_ui_drv_d as drv
from capture_database import _digest
from lineage_data import verify_preserved
from lineage_evidence import read_expected, verify_forest, verify_tree, verify_selected, verify_unavailable_crop


def record_lineage(app,window,stage,captures,capture,settle,write_json,timeout):
    from PySide6.QtCore import Qt,QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QFileDialog,QDialogButtonBox,QLineEdit,QAbstractButton,QAbstractItemView
    from spacr.qt.widgets.fold_strip import FoldButton
    from spacr.qt.screens.lineage import LineageScreen
    from spacr.qt.linked_selection import linked_selection,has_object_opener
    from spacr import lineage as lin

    prepared=json.loads((Path(stage)/'lineage_state'/f'{Path(captures).name}.json').read_text(encoding="utf-8"))
    database=Path(prepared['source']['database']);work=Path(prepared['work'])
    expected=read_expected(database)
    if expected['source_counts']!={'cell':2341,'nucleus':2682,'pathogen':2178}:
        raise ValueError('Wrong real Lineage source population')
    proof=dict(lesson='56_lineage',accepted=False,source=prepared['source'],private_cache=prepared['cache'],
               synthetic_measurements=False,app_source_modified=False,published=False)
    screen=None;annotate=None;events=[];deadline=time.monotonic()+timeout

    def tick():
        if time.monotonic()>deadline:raise TimeoutError('Bounded real Lineage recording timed out')

    def click(widget):
        tick()
        if not widget.isVisible() or not widget.isEnabled() or widget.visibleRegion().isEmpty():
            raise ValueError('Actual Lineage control unavailable: '+widget.objectName())
        QTest.mouseClick(widget,Qt.LeftButton,pos=widget.visibleRegion().boundingRect().center());settle(.3)

    def fill(widget,value):
        click(widget);QTest.keyClick(widget,Qt.Key_A,Qt.ControlModifier)
        QTest.keyClicks(widget,str(value));QTest.keyClick(widget,Qt.Key_Tab);settle(.2)

    def idle():
        settle(.3)
        while screen._jobs.is_busy() or screen._jobs.active_jobs():tick();settle(.1)
        settle(.3)

    def picker(button,path,name):
        accepted=[];errors=[];timer=QTimer(window);watchdog=QTimer(window)
        timer.setSingleShot(True);watchdog.setSingleShot(True)
        def handle():
            dialog=app.activeModalWidget()
            try:
                if not isinstance(dialog,QFileDialog):raise ValueError('Expected real file/folder picker')
                dialog.accepted.connect(lambda:accepted.append(True));capture_ui.dialog_geometry(capture,dialog,(1400,950))
                fill(dialog.findChild(QLineEdit,'fileNameEdit'),path);drv.snap(capture,name);capture_ui.wait(capture,1.5,settle)
                box=dialog.findChild(QDialogButtonBox)
                choices=[b for b in box.buttons() if box.buttonRole(b)==QDialogButtonBox.AcceptRole]
                if len(choices)!=1:raise ValueError('No unique picker accept action')
                click(choices[0])
            except Exception as exc:
                errors.append(str(exc))
                if dialog is not None:dialog.reject()
        def abort():
            errors.append('File/folder picker timed out');dialog=app.activeModalWidget()
            if dialog is not None:dialog.reject()
        watchdog.timeout.connect(abort);watchdog.start(120000 if getattr(capture, 'host', None) is not None else 15000)
        started=drv.begin(capture,settle,name);capture_ui.when_modal(capture,300,handle)
        try:click(button)
        finally:drv.finish(capture,started);timer.stop();watchdog.stop();timer.deleteLater();watchdog.deleteLater()
        if errors or not accepted:raise ValueError('; '.join(errors) or 'Picker not accepted')

    def database_route(name,follow=None):
        """Open Help, show the menu (clip ``name``), then click Database browser (clip ``follow`` starts)."""
        actions=[a for a in window.menuBar().actions() if a.text().replace('&','')=='Help']
        if len(actions)!=1:raise ValueError('No actual Help menu')
        menu=actions[0].menu();choices=[a for a in menu.actions() if a.text().replace('&','').casefold()=='database browser']
        if len(choices)!=1:raise ValueError('No actual Database browser route')
        started=drv.begin(capture,settle,name)
        QTest.mouseClick(window.menuBar(),Qt.LeftButton,pos=window.menuBar().actionGeometry(actions[0]).center())
        settle(.2);capture_ui.wait(capture,1.5,settle);drv.snap(capture,name);drv.finish(capture,started)
        following=drv.begin(capture,settle,follow) if follow else False
        QTest.mouseClick(menu,Qt.LeftButton,pos=menu.actionGeometry(choices[0]).center());settle(.8)
        return following

    def node_rows():
        out=[]
        def visit(node,parent='',depth=0):
            out.append(dict(key=node.key,table=node.table,label=node.label,field=node.field,
                            parent_key=parent,depth=depth,n_children=len(node.children)))
            for child in node.children:visit(child,node.key,depth+1)
        for root in screen._forest:visit(root)
        return out

    def tree_items():
        out=[]
        def visit(item,parent='',depth=0):
            key=str(item.data(0,Qt.UserRole));identity=str(item.data(0,Qt.UserRole+1))
            table,rest=identity.split(':',1)
            if rest!=key:raise ValueError('Tree qualified id differs from its shared key')
            number=int(item.text(0).split()[-1]);field=key.rsplit('_',1)[0]
            out.append((item,dict(key=key,table=table,label=number,field=field,
                                 parent_key=parent,depth=depth,n_children=item.childCount())))
            for i in range(item.childCount()):visit(item.child(i),key,depth+1)
        for i in range(screen.tree.topLevelItemCount()):visit(screen.tree.topLevelItem(i))
        return out

    def check(name,act=None,started=None):
        if started is None:started=drv.begin(capture,settle,name)
        if act is not None:act()
        idle();orph=[]
        for row in screen._orphans.to_dict('records'):
            field='_'.join(str(row[k]) for k in ('plateID','rowID','columnID','fieldID'))
            orph.append(dict(key=f"{field}_{row['table']}{int(row['object_label'])}",
                             table=row['table'],label=int(row['object_label']),parent_id=str(row['parent_id'])))
        p=verify_forest(node_rows(),orph,expected);p.update(verify_tree([r for _,r in tree_items()],expected))
        if screen.orphan_list.count()!=1 or screen.orphan_list.item(0).text()!='(none — every child names a parent that exists)':
            raise ValueError('Actual real-data empty-orphan message differs')
        p.update(summary=screen.summary.text(),status=screen.status.text(),key_column_hidden=screen.tree.isColumnHidden(2))
        proof.setdefault('checks',{})[name]=p;capture_ui.point_at(capture,screen.tree,screen.summary);drv.snap(capture,name);drv.finish(capture,started);return p

    def select(key,control=False,expand=False):
        items=[w for w,r in tree_items() if r['key']==key]
        if len(items)!=1:raise ValueError('Requested source object not in actual capped tree')
        item=items[0]
        if item.parent() is not None:
            parent=item.parent();screen.tree.scrollToItem(parent);settle(.2)
            if not parent.isExpanded():
                rect=screen.tree.visualItemRect(parent)
                QTest.mouseClick(screen.tree.viewport(),Qt.LeftButton,pos=QPoint(rect.left()+55,rect.center().y()))
                QTest.keyClick(screen.tree,Qt.Key_Right);settle(.2)
        screen.tree.scrollToItem(item);settle(.2)
        rect=screen.tree.visualItemRect(item)
        QTest.mouseClick(screen.tree.viewport(),Qt.LeftButton,Qt.ControlModifier if control else Qt.NoModifier,
                         pos=QPoint(rect.left()+55,rect.center().y()));settle(.3)
        if screen.tree.currentItem() is not item:raise ValueError('Actual tree click did not select requested row')
        if not control:verify_selected(screen.selected_keys(),[key])
        if expand and not item.isExpanded():QTest.keyClick(screen.tree,Qt.Key_Right);settle(.3)
        if expand and not item.isExpanded():raise ValueError('Actual first-column Right key did not expand the family')
        if expand:
            screen.tree.scrollToItem(item,QAbstractItemView.PositionAtTop);settle(.3)
        return item

    def annotate_route():
        buttons=[w for w in window.findChildren(QAbstractButton) if w.isVisible() and w.property('navKey')=='annotate']
        if len(buttons)!=1:raise ValueError('No unique visible Annotate navigation')
        click(buttons[0]);settle(.8)
        return window._screens['annotate']

    def crops_ready():
        settle(.3)
        while (annotate._page_worker is not None or annotate._pending_page_load is not None
               or not annotate._page_paths or len(annotate._raw_thumb_images)<len(annotate._page_paths)
               or any(i is None for i in annotate._raw_thumb_images[:len(annotate._page_paths)])):
            tick();settle(.1)
        settle(.5)

    try:
        from PySide6.QtCore import QPoint
        scene2=database_route('01_help_database_route','02_database_host');host=window._screens['db_browser']
        folds=[w for w in host.findChildren(FoldButton) if w.isVisible() and w.app_key=='lineage']
        if len(folds)!=1:raise ValueError('No actual Lineage fold')
        capture_ui.point_at(capture,folds[0]);drv.snap(capture,'02_database_host');drv.finish(capture,scene2)
        scene3=drv.begin(capture,settle,'03_empty_containment_view')
        click(folds[0]);settle(.8)
        panels=[w for w in window.findChildren(LineageScreen) if w.isVisible()]
        if len(panels)!=1:raise ValueError('Actual Lineage screen not visible')
        screen=panels[0];bus=linked_selection()
        # Resize actual native headers/dividers, not application defaults or
        # painted replacements. The initial100px Object column clips labels.
        header=screen.tree.header();start=QPoint(header.sectionSize(0)-1,header.height()//2)
        end=start+QPoint(420-header.sectionSize(0),0)
        QTest.mousePress(header.viewport(),Qt.LeftButton,pos=start)
        QTest.mouseMove(header.viewport(),end,delay=150)
        QTest.mouseRelease(header.viewport(),Qt.LeftButton,pos=end);settle(.3)
        if header.sectionSize(0)<250:raise ValueError('Actual Object header remains too narrow to read')
        splitter=screen._body;handle=splitter.handle(1)
        start=handle.rect().center();end=start+QPoint(2900-handle.mapTo(window,start).x(),0)
        QTest.mousePress(handle,Qt.LeftButton,pos=start);QTest.mouseMove(handle,end,delay=150)
        QTest.mouseRelease(handle,Qt.LeftButton,pos=end);settle(.3)
        bus.selection_changed.connect(lambda:events.append(dict(source=bus.selection.source,
            keys=[] if bus.selection.keys is None else list(bus.selection.keys))))
        capture_ui.point_at(capture,screen._browse);drv.snap(capture,'03_empty_containment_view');drv.finish(capture,scene3)
        picker(screen._browse,database,'04_choose_real_measurements')
        check('05_full_counts_capped_tree_and_zero_orphans')
        single='plate1_r12_c1_f1_cell1';family='plate1_r12_c1_f1_cell3';child='plate1_r12_c1_f1_nucleus5'
        def act06():
            select(single,expand=True);verify_selected(bus.selection.keys,[single])
        check('06_one_nucleus_no_pathogen',act=act06)
        def act07():
            select(family,expand=True);verify_selected(bus.selection.keys,[family])
        check('07_cell_with_nucleus_and_pathogen',act=act07)
        def act08():
            select(child);verify_selected(bus.selection.keys,[child])
        check('08_one_child_is_not_its_family',act=act08)
        def act09():
            select(family);click(screen._publish_family)
            proof['family_selection']=verify_selected(bus.selection.keys,expected['families'][family])
        check('09_publish_three_typed_objects',act=act09)
        def act10():
            select(family);select(child,control=True);click(screen._publish_family)
            proof['deduplicated_family_selection']=verify_selected(bus.selection.keys,expected['families'][family])
        check('10_parent_plus_child_no_duplicate',act=act10)
        largest=max(expected['roots'][:2000],key=lambda key:len(expected['families'][key]))
        def act11():
            select(largest,expand=True);proof['largest_family']=dict(key=largest,keys=expected['families'][largest])
        check('11_large_family_not_biological_validation',act=act11)
        def act12():
            header=screen.tree.header();x=header.sectionViewportPosition(1)+header.sectionSize(1)//2
            QTest.mouseClick(header.viewport(),Qt.LeftButton,pos=QPoint(x,header.height()//2));settle(.3)
        check('12_actual_header_sort_preserves_containment',act=act12)
        # Annotate registers its crop opener when its SCREEN IS BUILT, not when it is opened. A host that already
        # built the screen in the background (the live recording did) has an opener registered although the user has
        # not opened Annotate, and the message 'Open the Annotate screen first' is then never shown. Setting the
        # session up as a user who has not opened Annotate: take the opener out for this scene and put it back after.
        from spacr.qt.linked_selection import register_object_opener, unregister_object_opener
        def _no_opener(*_a, **_k): raise RuntimeError('placeholder opener')
        held=register_object_opener('annotate',_no_opener)
        unregister_object_opener('annotate',_no_opener)
        proof['annotate_opener_registered_before_scene']=held is not None
        try:
            with capture_ui.clip(capture,'13_real_open_annotate_prerequisite'):
                select(family);click(screen._open_button)
                proof['crop_precondition']=verify_unavailable_crop(screen.status.text(),has_object_opener('annotate'))
                capture_ui.point_at(capture,screen.status)
        finally:
            if held is not None: register_object_opener('annotate',held)
        # Opening Annotate and its source are actual UI actions. The cache
        # bind resolves the original PNG paths to byte-identical private files.
        with capture_ui.clip(capture,'14_annotate_before_source'):
            annotate=annotate_route()
        picker(annotate._btn_open,prepared['project'],'15_choose_private_crop_project')
        with capture_ui.clip(capture,'16_real_downloaded_cell_crops'):
            crops_ready();capture_ui.point_at(capture,annotate)
        if not has_object_opener('annotate'):raise ValueError('Actual Annotate opener was not registered')
        scene18=database_route('17_return_to_lineage','18_actual_family_crop_request');settle(.5)
        if not screen.isVisible():
            visiblefolds=[w for w in host.findChildren(FoldButton) if w.isVisible() and w.app_key=='lineage']
            if len(visiblefolds)!=1:raise ValueError('Cannot return to existing Lineage fold')
            click(visiblefolds[0]);settle(.4)
        select(family);click(screen._open_button)
        capture_ui.point_at(capture,screen.status)
        drv.snap(capture,'18_actual_family_crop_request');drv.finish(capture,scene18)
        scene19=drv.begin(capture,settle,'19_actual_crop_request_result_before_acceptance')
        annotate=annotate_route();crops_ready()
        request=annotate._object_request
        verify_selected(request.keys,expected['families'][family])
        proof['observed_crop_request']=dict(keys=list(request.keys),total=annotate._total,
            rows=len(annotate._object_rows),paths=list(annotate._page_paths),note=annotate._request_note)
        capture_ui.point_at(capture,annotate)
        drv.snap(capture,'19_actual_crop_request_result_before_acceptance');drv.finish(capture,scene19)
        import sqlite3
        with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True) as db:
            source_columns=[r[1] for r in db.execute('PRAGMA table_info(png_list)')]
            crop_rows=dict(db.execute("SELECT cell_id,png_path FROM png_list WHERE plateID='plate1' AND rowID='r12' AND columnID='c1' AND fieldID='f1' AND cell_id IN ('o3','o5','o1')").fetchall())
        if any(c in source_columns for c in ('nucleus_id','pathogen_id')) or len(crop_rows)!=3:
            raise ValueError('This demonstration requires genuine cell-only source crop identities')
        observed_paths=[r[0] for r in annotate._page_paths]
        if len(annotate._object_rows)!=1 or annotate._total!=1 or observed_paths!=[crop_rows['o3']]:
            raise ValueError('Family crop handoff must show only the matching parent, never unrelated cell crops')
        if '2 of them are not in this database' not in annotate._request_note:
            raise ValueError('Annotate must report both missing child crops')
        proof['family_crop_handoff']=dict(requested=list(request.keys),shown_paths=observed_paths,
            source_object_types=['cell'],source_labels=[3],note=annotate._request_note,
            matching_crops=1,missing_crops=2,unrelated_crops_substituted=False,
            child_crops_in_source=False,labels_edited=False)
        with capture_ui.clip(capture,'20_family_crop_identity_preserved'):
            capture_ui.point_at(capture,annotate)
        # A genuine double-click sends ONLY this parent, unlike Open crops,
        # which requests its family and reports unavailable child crops.
        scene22=database_route('21_return_for_parent_only','22_verified_parent_only_real_cell_crop');settle(.4)
        item=select(family)
        rect=screen.tree.visualItemRect(item)
        QTest.mouseDClick(screen.tree.viewport(),Qt.LeftButton,pos=QPoint(rect.left()+55,rect.center().y()));settle(.4)
        annotate=annotate_route();crops_ready();request=annotate._object_request
        verify_selected(request.keys,[family])
        if len(annotate._object_rows)!=1 or annotate._total!=1 or [r[0] for r in annotate._page_paths]!=[crop_rows['o3']]:
            raise ValueError('Actual Annotate page does not show exactly the requested parent crop')
        actual=Path(annotate._page_paths[0][0]);relative=actual.relative_to('/home/olafsson/.cache/spacr/example_data')
        actual=Path.home()/'.cache/spacr/example_data'/relative   # the stage holds the same crops under this home
        record=next(r for r in prepared['crops'] if r['relative']==str(relative))
        if _digest(actual)!=record['sha256']:raise ValueError('Displayed real crop differs from original downloaded bytes')
        proof['crop_handoff']=dict(requested=list(request.keys),shown_paths=list(annotate._page_paths),
            actual_crop_sha256=record['sha256'],note=annotate._request_note,shown=1,missing=0,
            cell_crop_loaded=True,child_crops_in_source=False,labels_edited=False,gesture='double-click parent only')
        capture_ui.point_at(capture,annotate)
        drv.snap(capture,'22_verified_parent_only_real_cell_crop');drv.finish(capture,scene22)
        # Native inspection gesture, not a plain click that would assign a class.
        # Preserve the actual thumbnail and check the overlay displays that slot.
        original_pixmap = annotate._thumb_pixmaps[0].cacheKey()
        scene22a=drv.begin(capture,settle,'22a_zoom_parent_without_labelling')
        QTest.mouseClick(annotate._thumbs[0],Qt.LeftButton,Qt.ShiftModifier)
        settle(.4)
        if not annotate._zoom_is_open() or annotate._zoom_overlay.slot != 0:
            raise ValueError('Native Shift-click did not open the verified parent crop')
        if annotate._thumb_pixmaps[0].cacheKey() != original_pixmap:
            raise ValueError('Zoom unexpectedly replaced the verified thumbnail')
        capture_ui.wait(capture,1.5,settle)
        drv.snap(capture,'22a_zoom_parent_without_labelling');drv.finish(capture,scene22a)
        scene22b=drv.begin(capture,settle,'22b_parent_grid_restored')
        QTest.keyClick(annotate._zoom_overlay,Qt.Key_Escape);settle(.4)
        if annotate._zoom_is_open():
            raise ValueError('Native Escape did not dismiss the crop zoom')
        proof['crop_zoom'] = dict(native_shift_click=True,slot=0,
            same_thumbnail_preserved=True,escape_restored_grid=True,
            new_image_resolution_created=False)
        capture_ui.wait(capture,1.0,settle)
        drv.snap(capture,'22b_parent_grid_restored');drv.finish(capture,scene22b)
        scene24=database_route('23_return_for_reload','24_rebuild_preserves_full_containment')
        check('24_rebuild_preserves_full_containment',act=lambda:click(screen._reload),started=scene24)
        # Explicit API-only export, never described as a nonexistent GUI button.
        path=work/'lineage_api_export.csv';lin.lineage_frame(screen._forest).to_csv(path,index=False)
        with path.open(newline='') as stream:exported=list(csv.DictReader(stream))
        normalized=[{k:(int(v) if k in ('label','depth','n_children') else v) for k,v in r.items()} for r in exported]
        verify_forest(normalized,[],expected)
        proof['api_export']=dict(path=str(path),sha256=_digest(path),rows=len(exported),
            function='spacr.lineage.lineage_frame',written_by_gui=False,all_rows_independently_checked=True)
        proof.update(accepted=True,selection_events=events,pipeline_run_requested=False)
    except Exception as exc:
        proof['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if annotate is not None:annotate.close();settle(.6)
        if screen is not None:screen._jobs.cancel();idle()
        proof['preservation']=verify_preserved(prepared)
        proof['remaining_lineage_workers']=screen._jobs.active_jobs() if screen is not None else 0
        write_json(Path(captures)/'lineage_acceptance.json',proof)
