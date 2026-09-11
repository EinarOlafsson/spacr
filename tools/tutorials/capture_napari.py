"""Record the real optional napari bridge using private synthetic mask copies."""
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import tifffile

from build_evaluation_example import sha
from curate_evidence import check_mask


def record_napari(app,window,stage,captures,capture,settle,write_json,timeout):
    from PySide6.QtCore import Qt,QTimer,QPoint
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton,QPushButton,QFileDialog,QLineEdit,QDialogButtonBox
    from spacr.qt.screens.make_masks import NapariBridgeScreen
    from spacr.qt.widgets.fold_strip import FoldButton
    import napari

    stage=Path(stage)
    source=stage/'timelapse_runs/SYNTHETIC-current-demo-os07d4sa/merged/plate1_A01_1_1.npy'
    original_sha=sha(source);raw=np.load(source,allow_pickle=False)
    if raw.shape!=(256,256,4) or raw.dtype!=np.uint16:
        raise ValueError('Expected the checked synthetic Timelapse four-plane array')
    before=raw[...,2].copy();image=raw[...,1].copy()
    if np.unique(before).tolist()!=[0,*range(2,18)]:
        raise ValueError('Expected the sixteen synthetic practice objects')
    parent=stage/'napari_runs';parent.mkdir(exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='SYNTHETIC-private-round-trip-',dir=parent))
    mask_path=work/'mask.tif';image_path=work/'image.tif';log_path=work/'mask.tif.curation.json'
    tifffile.imwrite(mask_path,before);tifffile.imwrite(image_path,image)
    original_mask_sha=sha(mask_path);original_image_sha=sha(image_path)
    proof=dict(lesson='46_napari_bridge',accepted=False,synthetic=True,
        original_source=str(source),original_sha256=original_sha,private_folder=str(work),
        napari_version=napari.__version__,app_source_modified=False,published=False,
        biological_corrections_validated=False,checks={})
    write_json(captures/'scientific_acceptance.json',proof)
    deadline=time.monotonic()+timeout;panel=None;viewer=None

    def click(widget):
        if time.monotonic()>deadline:raise TimeoutError('Bounded napari recording timed out')
        if not widget.isVisible() or not widget.isEnabled() or widget.visibleRegion().isEmpty():
            raise ValueError('Actual napari tutorial control unavailable: '+widget.objectName())
        QTest.mouseClick(widget,Qt.LeftButton,pos=widget.visibleRegion().boundingRect().center());settle(.3)

    def fill(widget,value):
        click(widget);QTest.keyClick(widget,Qt.Key_A,Qt.ControlModifier)
        QTest.keyClicks(widget,str(value));QTest.keyClick(widget,Qt.Key_Tab);settle(.25)

    def picker(button,path,name):
        errors=[];accepted=[]
        def choose():
            dialog=app.activeModalWidget()
            try:
                if not isinstance(dialog,QFileDialog):raise ValueError('The actual file dialog did not open')
                dialog.accepted.connect(lambda:accepted.append(True));dialog.resize(1500,1000)
                fill(dialog.findChild(QLineEdit,'fileNameEdit'),path);capture(name)
                box=dialog.findChild(QDialogButtonBox)
                accept=[b for b in box.buttons() if box.buttonRole(b)==QDialogButtonBox.AcceptRole]
                if len(accept)!=1:raise ValueError('No unique real file acceptance button')
                click(accept[0])
            except Exception as e:
                errors.append(str(e))
                if dialog is not None:dialog.reject()
        watch=QTimer(window);watch.setSingleShot(True)
        watch.timeout.connect(lambda:app.activeModalWidget().reject() if app.activeModalWidget() else None)
        QTimer.singleShot(300,choose);watch.start(15000);click(button);watch.stop()
        if errors or not accepted:raise ValueError('; '.join(errors) or 'Actual file selection was not accepted')

    def snapshot(name,*,desktop=False,**facts):
        capture(name,desktop=desktop);proof['checks'][name]=facts
        write_json(captures/'scientific_acceptance.json',proof)

    def front_spacr():
        window.raise_();window.activateWindow();settle(.5)

    def front_napari():
        # Xvfb has no window manager to honour showMaximized. Resize the real
        # viewer window explicitly; do not enlarge a tiny screenshot afterwards.
        top=viewer.window._qt_window
        top.showNormal();top.resize(3840,2160);top.move(0,0)
        top.raise_();top.activateWindow();settle(.7)

    def layers():
        labels=[l for l in viewer.layers if l._type_string=='labels']
        images=[l for l in viewer.layers if l._type_string=='image']
        if len(labels)!=1 or len(images)!=1:raise ValueError('Actual viewer does not contain one image and one labels layer')
        check_mask(images[0].data,image)
        return labels[0]

    def fill_object(old,new,expected,name):
        front_napari();layer=layers();qt=viewer.window._qt_viewer
        controls=qt._controls.widgets[layer]
        fill(controls._label_control.selection_spinbox,new)
        click(controls.fill_button)
        if layer.selected_label!=new or str(layer.mode)!='fill':
            raise ValueError('The actual label selector/fill button did not set the requested edit')
        locations=np.argwhere(expected==old)
        if not len(locations):raise ValueError('The practice object is absent')
        y,x=locations[len(locations)//2]
        node=qt.layer_to_visual[layer].node
        position=node.get_transform(map_from='visual',map_to='canvas').map([float(x),float(y),0,1])
        canvas=qt.canvas.native;point=QPoint(round(float(position[0])),round(float(position[1])))
        if not canvas.rect().contains(point):raise ValueError('Projected practice pixel is outside the real canvas')
        QTest.mouseMove(canvas,point);settle(.1)
        QTest.mouseClick(canvas,Qt.LeftButton,pos=point);settle(.8)
        after=expected.copy();after[expected==old]=new
        result=check_mask(layer.data,after)
        # A real edit is visible, but disk and the saved baseline must not move yet.
        snapshot(name,desktop=True,mask=result,old_label=old,new_label=new,
            changed_pixels=int(np.count_nonzero(after!=expected)),
            native_click=[point.x(),point.y()],data_pixel=[int(y),int(x)])
        return after

    try:
        buttons=[b for b in window.findChildren(QAbstractButton) if b.isVisible() and
            (b.property('moduleAppKey')=='make_masks' or b.property('navKey')=='make_masks')]
        if not buttons:raise ValueError('Actual Make Masks Home control missing')
        click(max(buttons,key=lambda b:b.width()*b.height()));host=window._screens['make_masks']
        folds=[b for b in host.findChildren(FoldButton) if b.isVisible() and b.app_key=='napari_bridge']
        if len(folds)!=1:raise ValueError('Actual Make Masks Napari Bridge fold is not unique')
        snapshot('01_make_masks_host');click(folds[0])
        found=[p for p in window.findChildren(NapariBridgeScreen) if p.isVisible()]
        if len(found)!=1:raise ValueError('Actual Napari Bridge did not open')
        panel=found[0];snapshot('02_actual_bridge')
        browse=sorted([b for b in panel.findChildren(QPushButton) if b.text()=='Browse…'],key=lambda b:b.y())
        if len(browse)!=2:raise ValueError('Expected actual mask/image Browse buttons')
        picker(browse[0],mask_path,'03_private_mask_picker')
        picker(browse[1],image_path,'04_private_image_picker')
        if panel.mask_path()!=str(mask_path) or panel.image_path()!=str(image_path):
            raise ValueError('Native file selection differs from private teaching copies')
        snapshot('05_ready_to_open',status=panel.status.toPlainText())
        opened=[];corrected=[];panel.opened.connect(opened.append);panel.corrected.connect(corrected.append)
        click(panel.open_button);viewer=panel._viewer
        if viewer is None:raise ValueError('Real napari viewer did not open: '+panel.status.toPlainText())
        if opened!=[str(mask_path)]:raise ValueError('Actual bridge opened a different mask')
        front_napari();layer=layers();check_mask(layer.data,before)
        if np.shares_memory(layer.data,panel._handoff.mask):raise ValueError('Editable labels alias the saved baseline')
        snapshot('06_actual_napari_layers',desktop=True,mask=check_mask(layer.data,before),names=[l.name for l in viewer.layers])
        front_spacr();click(panel.take_button)
        if sha(mask_path)!=original_mask_sha or log_path.exists() or corrected:
            raise ValueError('An unchanged round trip wrote a file or correction')
        snapshot('07_unchanged_round_trip',status=panel.status.toPlainText(),no_writes=True)
        after=fill_object(2,18,before,'08_fill_one_synthetic_object')
        if sha(mask_path)!=original_mask_sha:raise ValueError('Editing in napari prematurely wrote the original mask')
        canvas=viewer.window._qt_viewer.canvas.native;canvas.setFocus()
        QTest.keyClick(canvas,Qt.Key_Z,Qt.ControlModifier);settle(.7)
        snapshot('09_actual_napari_undo',desktop=True,mask=check_mask(layers().data,before))
        after=fill_object(2,18,before,'10_refill_before_import')
        front_spacr();click(panel.take_button)
        check_mask(tifffile.imread(mask_path),after)
        first=json.loads(log_path.read_text())
        if len(first['edits'])!=1 or corrected!=[str(mask_path)]:raise ValueError('One import did not record exactly one correction')
        entry=first['edits'][0]
        if (entry['n_changed']!=int(np.count_nonzero(after!=before))
                or entry['detail']!={'added':[18],'altered':[],'removed':[2],'via':'napari'}
                or entry['kind']!='napari' or entry['target']!=[18]):
            raise ValueError('First correction ledger differs from the actual pixel edit')
        snapshot('11_imported_and_recorded',status=panel.status.toPlainText(),mask=check_mask(tifffile.imread(mask_path),after),ledger=first)
        saved_sha=sha(mask_path);log_sha=sha(log_path);click(panel.take_button)
        if sha(mask_path)!=saved_sha or sha(log_path)!=log_sha or len(corrected)!=1:
            raise ValueError('A duplicate import rewrote data or padded the ledger')
        snapshot('12_second_import_is_noop',status=panel.status.toPlainText(),no_writes=True)
        # The first import must not turn the saved baseline into the editable
        # napari array. Exercise another real edit before closing that viewer.
        proof['baseline_alias_after_first_import']=np.shares_memory(layers().data,panel._handoff.mask)
        same_session=fill_object(3,19,after,'12b_same_viewer_new_edit')
        front_spacr();click(panel.take_button)
        disk=tifffile.imread(mask_path)
        saved_second=np.array_equal(disk,same_session)
        if not saved_second:check_mask(disk,after)
        proof['same_session_second_edit_saved']=saved_second
        snapshot('12c_same_viewer_import_result',status=panel.status.toPlainText(),
            requested_pixels=int(np.count_nonzero(same_session!=after)),
            pixels_not_saved=int(np.count_nonzero(disk!=same_session)),
            saved_second_edit=saved_second,ledger=json.loads(log_path.read_text()))
        if saved_second:after=same_session
        click(panel.close_button)
        if panel._viewer is not None or panel.take_button.isEnabled():raise ValueError('Close viewer did not release the bridge')
        click(panel.open_button);viewer=panel._viewer
        if viewer is None:raise ValueError('The saved corrected field did not reopen')
        check_mask(layers().data,after)
        again=fill_object(4,20,after,'13_reopened_second_correction')
        front_spacr();click(panel.take_button)
        second=json.loads(log_path.read_text());check_mask(tifffile.imread(mask_path),again)
        if len(second['edits'])!=(3 if saved_second else 2) or second['edits'][0]!=first['edits'][0]:
            raise ValueError('Reopening and saving failed to preserve the previous correction history')
        entry=second['edits'][-1]
        if (entry['n_changed']!=int(np.count_nonzero(again!=after))
                or entry['detail']!={'added':[20],'altered':[],'removed':[4],'via':'napari'}
                or entry['kind']!='napari' or entry['target']!=[20]):
            raise ValueError('Reopened correction ledger differs from the actual pixel edit')
        snapshot('14_history_preserved_after_reopen',status=panel.status.toPlainText(),mask=check_mask(tifffile.imread(mask_path),again),ledger=second)
        click(panel.close_button);snapshot('15_viewer_closed',status=panel.status.toPlainText())
        proof.update(accepted=bool(saved_second),acceptance_scope='synthetic editing, exact pixels and correction history only',
            corrected_signals=corrected,original_image_unchanged=sha(image_path)==original_image_sha,
            originals_preserved=sha(source)==original_sha,final_mask_sha256=sha(mask_path),final_ledger_sha256=sha(log_path))
        if not saved_second:
            proof['hold']='The first import aliases the saved baseline to the editable labels. A second edit in the same viewer is reported unchanged and is not saved; close/reopen avoids this, but the unrestricted workflow is not accepted.'
    finally:
        proof['originals_preserved']=sha(source)==original_sha
        write_json(captures/'scientific_acceptance.json',proof)
        if panel is not None and panel._viewer is not None:panel.close_viewer()
