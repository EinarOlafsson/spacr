"""Drive real containment trees and cell-only crop handoff on isolated data."""
import csv
import json
from pathlib import Path
import time

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

    prepared=json.loads((Path(stage)/'lineage_state'/f'{Path(captures).name}.json').read_text())
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
                dialog.accepted.connect(lambda:accepted.append(True));dialog.resize(1400,950)
                fill(dialog.findChild(QLineEdit,'fileNameEdit'),path);capture(name)
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
        timer.timeout.connect(handle);watchdog.timeout.connect(abort);timer.start(300);watchdog.start(15000)
        try:click(button)
        finally:timer.stop();watchdog.stop();timer.deleteLater();watchdog.deleteLater()
        if errors or not accepted:raise ValueError('; '.join(errors) or 'Picker not accepted')

    def database_route(name):
        actions=[a for a in window.menuBar().actions() if a.text().replace('&','')=='Help']
        if len(actions)!=1:raise ValueError('No actual Help menu')
        menu=actions[0].menu();choices=[a for a in menu.actions() if a.text().replace('&','')=='Database browser']
        if len(choices)!=1:raise ValueError('No actual Database browser route')
        QTest.mouseClick(window.menuBar(),Qt.LeftButton,pos=window.menuBar().actionGeometry(actions[0]).center())
        settle(.2);capture(name)
        QTest.mouseClick(menu,Qt.LeftButton,pos=menu.actionGeometry(choices[0]).center());settle(.8)

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

    def check(name):
        idle();orph=[]
        for row in screen._orphans.to_dict('records'):
            field='_'.join(str(row[k]) for k in ('plateID','rowID','columnID','fieldID'))
            orph.append(dict(key=f"{field}_{row['table']}{int(row['object_label'])}",
                             table=row['table'],label=int(row['object_label']),parent_id=str(row['parent_id'])))
        p=verify_forest(node_rows(),orph,expected);p.update(verify_tree([r for _,r in tree_items()],expected))
        if screen.orphan_list.count()!=1 or screen.orphan_list.item(0).text()!='(none — every child names a parent that exists)':
            raise ValueError('Actual real-data empty-orphan message differs')
        p.update(summary=screen.summary.text(),status=screen.status.text(),key_column_hidden=screen.tree.isColumnHidden(2))
        proof.setdefault('checks',{})[name]=p;capture(name);return p

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
        database_route('01_help_database_route');host=window._screens['db_browser']
        folds=[w for w in host.findChildren(FoldButton) if w.isVisible() and w.app_key=='lineage']
        if len(folds)!=1:raise ValueError('No actual Lineage fold')
        capture('02_database_host');click(folds[0]);settle(.8)
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
        splitter=screen.tree.parentWidget().parentWidget();handle=splitter.handle(1)
        start=handle.rect().center();end=start+QPoint(2900-handle.mapTo(window,start).x(),0)
        QTest.mousePress(handle,Qt.LeftButton,pos=start);QTest.mouseMove(handle,end,delay=150)
        QTest.mouseRelease(handle,Qt.LeftButton,pos=end);settle(.3)
        bus.selection_changed.connect(lambda:events.append(dict(source=bus.selection.source,
            keys=[] if bus.selection.keys is None else list(bus.selection.keys))))
        capture('03_empty_containment_view');picker(screen._browse,database,'04_choose_real_measurements')
        check('05_full_counts_capped_tree_and_zero_orphans')
        single='plate1_r12_c1_f1_cell1';family='plate1_r12_c1_f1_cell3';child='plate1_r12_c1_f1_nucleus5'
        select(single,expand=True);verify_selected(bus.selection.keys,[single]);check('06_one_nucleus_no_pathogen')
        select(family,expand=True);verify_selected(bus.selection.keys,[family]);check('07_cell_with_nucleus_and_pathogen')
        select(child);verify_selected(bus.selection.keys,[child]);check('08_one_child_is_not_its_family')
        select(family);click(screen._publish_family)
        proof['family_selection']=verify_selected(bus.selection.keys,expected['families'][family]);check('09_publish_three_typed_objects')
        select(family);select(child,control=True);click(screen._publish_family)
        proof['deduplicated_family_selection']=verify_selected(bus.selection.keys,expected['families'][family]);check('10_parent_plus_child_no_duplicate')
        largest=max(expected['roots'][:2000],key=lambda key:len(expected['families'][key]))
        select(largest,expand=True);proof['largest_family']=dict(key=largest,keys=expected['families'][largest])
        check('11_large_family_not_biological_validation')
        header=screen.tree.header();x=header.sectionViewportPosition(1)+header.sectionSize(1)//2
        QTest.mouseClick(header.viewport(),Qt.LeftButton,pos=QPoint(x,header.height()//2));settle(.3)
        check('12_actual_header_sort_preserves_containment')
        select(family);click(screen._open_button)
        proof['crop_precondition']=verify_unavailable_crop(screen.status.text(),has_object_opener('annotate'))
        capture('13_real_open_annotate_prerequisite')
        # Opening Annotate and its source are actual UI actions. The cache
        # bind resolves the original PNG paths to byte-identical private files.
        annotate=annotate_route();capture('14_annotate_before_source')
        picker(annotate._btn_open,prepared['project'],'15_choose_private_crop_project');crops_ready()
        capture('16_real_downloaded_cell_crops')
        if not has_object_opener('annotate'):raise ValueError('Actual Annotate opener was not registered')
        database_route('17_return_to_lineage');settle(.5)
        if not screen.isVisible():
            visiblefolds=[w for w in host.findChildren(FoldButton) if w.isVisible() and w.app_key=='lineage']
            if len(visiblefolds)!=1:raise ValueError('Cannot return to existing Lineage fold')
            click(visiblefolds[0]);settle(.4)
        select(family);click(screen._open_button);capture('18_actual_family_crop_request')
        annotate=annotate_route();crops_ready()
        request=annotate._object_request
        verify_selected(request.keys,expected['families'][family])
        proof['observed_crop_request']=dict(keys=list(request.keys),total=annotate._total,
            rows=len(annotate._object_rows),paths=list(annotate._page_paths),note=annotate._request_note)
        capture('19_actual_crop_request_result_before_acceptance')
        import sqlite3
        with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True) as db:
            source_columns=[r[1] for r in db.execute('PRAGMA table_info(png_list)')]
            crop_rows=dict(db.execute("SELECT cell_id,png_path FROM png_list WHERE plateID='plate1' AND rowID='r12' AND columnID='c1' AND fieldID='f1' AND cell_id IN ('o3','o5','o1')").fetchall())
        if any(c in source_columns for c in ('nucleus_id','pathogen_id')) or len(crop_rows)!=3:
            raise ValueError('This demonstration requires genuine cell-only source crop identities')
        observed_paths=[r[0] for r in annotate._page_paths]
        wrong_family_paths=[crop_rows[k] for k in ('o3','o5','o1')]
        if len(annotate._object_rows)!=3 or annotate._total!=3 or observed_paths!=wrong_family_paths:
            raise ValueError('The known unsafe family-crop fallback changed; re-evaluate this warning')
        proof['unsafe_family_crop_handoff']=dict(requested=list(request.keys),shown_paths=observed_paths,
            source_object_types=['cell','cell','cell'],source_labels=[3,5,1],note=annotate._request_note,
            scientifically_valid=False,recommended=False,
            cause='Typed missing nucleus/pathogen keys fall back to untyped cell labels despite known cell-only identity columns')
        capture('20_wrong_typed_children_are_unrelated_cell_crops')
        # A genuine double-click sends ONLY this parent, unlike Open crops,
        # which expands its family. Verify the actual safe counterpart.
        database_route('21_return_for_parent_only');settle(.4)
        item=select(family)
        rect=screen.tree.visualItemRect(item)
        QTest.mouseDClick(screen.tree.viewport(),Qt.LeftButton,pos=QPoint(rect.left()+55,rect.center().y()));settle(.4)
        annotate=annotate_route();crops_ready();request=annotate._object_request
        verify_selected(request.keys,[family])
        if len(annotate._object_rows)!=1 or annotate._total!=1 or [r[0] for r in annotate._page_paths]!=[crop_rows['o3']]:
            raise ValueError('Actual Annotate page does not show exactly the requested parent crop')
        actual=Path(annotate._page_paths[0][0]);relative=actual.relative_to('/home/olafsson/.cache/spacr/example_data')
        record=next(r for r in prepared['crops'] if r['relative']==str(relative))
        if _digest(actual)!=record['sha256']:raise ValueError('Displayed real crop differs from original downloaded bytes')
        proof['crop_handoff']=dict(requested=list(request.keys),shown_paths=list(annotate._page_paths),
            actual_crop_sha256=record['sha256'],note=annotate._request_note,shown=1,missing=0,
            cell_crop_loaded=True,child_crops_in_source=False,labels_edited=False,gesture='double-click parent only')
        capture('22_verified_parent_only_real_cell_crop')
        database_route('23_return_for_reload');click(screen._reload);check('24_rebuild_preserves_full_containment')
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
