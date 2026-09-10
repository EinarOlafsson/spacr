"""Record the genuine QC Layer Viewer over preserved real image/mask pixels."""
import ctypes
import json
import math
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import tifffile

from capture_database import _digest, file_bundle, require_unchanged_source
from capture_diagnostics import PrivateDesktop
from layer_viewer_evidence import verify_layer_state, verify_pixels


def record_layers(app, window, stage, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import Qt, QTimer, QPoint
    from PySide6.QtGui import QImage
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton, QFileDialog, QDialogButtonBox, QLineEdit
    from spacr.qt.widgets.fold_strip import FoldButton
    from spacr.qt.layer_viewer import LayerViewer

    stage = Path(stage); parent = stage/'layer_viewer_runs'; parent.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='real-field-',dir=parent))
    manifest = stage/'captures/foreign_release_v2/inputs.json'
    source = next(r for r in json.loads(manifest.read_text())['records'] if r['neutral_stem']=='fov01')
    original = {str(manifest):file_bundle(manifest)}
    for key in ('source','image','mask'):
        path = Path(source[key]); original[str(path)] = file_bundle(path)
        if _digest(path) != source[key+'_sha256']: raise ValueError('Accepted real source changed')
    for key in ('image','mask'):
        destination = work/Path(source[key]).name
        if destination.exists(): raise FileExistsError(destination)
        shutil.copy2(source[key],destination)
    image = tifffile.imread(work/Path(source['image']).name)
    mask = tifffile.imread(work/Path(source['mask']).name)
    acquired = np.load(source['source'],mmap_mode='r')
    if (image.shape != (1994,1994) or image.dtype != np.uint16 or len(np.unique(mask))-1 != 44
            or not np.array_equal(image,acquired[...,1]) or not np.array_equal(mask,acquired[...,4])):
        raise ValueError('Real image/label TIFFs differ from preserved acquired planes')
    del acquired
    proof = dict(lesson='57_layer_viewer',accepted=False,source=source,private_folder=str(work),
                 original_bundles=original,source_planes_independently_checked=[1,4],
                 generated_measurements=False,app_source_modified=False,published=False)
    expected = []; screen = None; events = []; picks = []; keys = []
    deadline = time.monotonic()+timeout
    desktop = PrivateDesktop(stage); xtest = ctypes.CDLL('libXtst.so.6')
    xtest.XTestFakeMotionEvent.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_ulong]
    xtest.XTestFakeButtonEvent.argtypes=[ctypes.c_void_p,ctypes.c_uint,ctypes.c_int,ctypes.c_ulong]

    def tick():
        if time.monotonic()>deadline: raise TimeoutError('Bounded Layer Viewer capture timed out')

    def click(widget):
        tick()
        if not widget.isVisible() or not widget.isEnabled() or widget.visibleRegion().isEmpty():
            raise ValueError('Actual Layer Viewer control unavailable: '+widget.objectName())
        QTest.mouseClick(widget,Qt.LeftButton,pos=widget.visibleRegion().boundingRect().center());settle(.25)

    def fill(widget,value):
        click(widget);QTest.keyClick(widget,Qt.Key_A,Qt.ControlModifier)
        QTest.keyClicks(widget,str(value));QTest.keyClick(widget,Qt.Key_Tab);settle(.2)

    def select(name):
        listing=screen.layer_list
        items=[listing.item(i) for i in range(listing.count()) if listing.item(i).text()==name]
        if len(items)!=1:raise ValueError('No unique actual layer-list entry')
        QTest.mouseClick(listing.viewport(),Qt.LeftButton,pos=listing.visualItemRect(items[0]).center());settle(.25)
        if screen.stack.selected.name!=name:raise ValueError('Actual layer selection did not change')

    def visible(name,value):
        select(name);listing=screen.layer_list;item=listing.currentItem()
        rect=listing.visualItemRect(item)
        QTest.mouseClick(listing.viewport(),Qt.LeftButton,pos=QPoint(rect.left()+12,rect.center().y()));settle(.3)
        next(x for x in expected if x['name']==name)['visible']=value

    def opacity(value):
        bar=screen.opacity_slider;click(bar);QTest.keyClick(bar,Qt.Key_Home)
        for _ in range(value):QTest.keyClick(bar,Qt.Key_Right)
        settle(.3)
        if bar.value()!=value:raise ValueError('Actual opacity slider did not reach requested value')
        next(x for x in expected if x['name']==screen.stack.selected.name)['opacity']=value/100

    def combo(box,value,frame=None):
        index=box.findText(value)
        if index<0:raise ValueError('Actual selector has no '+value)
        click(box)
        if frame:capture(frame,desktop=True)
        QTest.keyClick(box.view(),Qt.Key_Home)
        for _ in range(index):QTest.keyClick(box.view(),Qt.Key_Down)
        QTest.keyClick(box.view(),Qt.Key_Return);settle(.3)
        if box.currentText()!=value:raise ValueError('Actual selector did not change')

    def check(name):
        tick();settle(.3)
        p=verify_layer_state(screen.stack,image,mask,expected)
        canvas=screen.canvas.canvas
        painted=screen.canvas.grab().toImage().convertToFormat(QImage.Format_RGB888)
        if painted.devicePixelRatio()!=1:raise ValueError('This native pixel proof requires DPR1')
        rgb=np.frombuffer(painted.constBits(),dtype=np.uint8).reshape(painted.height(),painted.bytesPerLine())
        rgb=rgb[:,:painted.width()*3].reshape(painted.height(),painted.width(),3)[1:-1,1:-1]
        p.update(verify_pixels(rgb,canvas,image,mask,expected))
        p.update(canvas=dict(origin=list(canvas.origin),step=list(canvas.step),shape=list(canvas.shape)),
                 status=screen.status.text(),expected_layers=[dict(x) for x in expected])
        proof.setdefault('checks',{})[name]=p;capture(name)
        return p

    def picker(pressed,path,name):
        accepted,errors=[],[];timer,watchdog=QTimer(window),QTimer(window)
        timer.setSingleShot(True);watchdog.setSingleShot(True)
        def handle():
            dialog=app.activeModalWidget()
            try:
                if not isinstance(dialog,QFileDialog):raise ValueError('Expected actual file picker')
                dialog.accepted.connect(lambda:accepted.append(True));dialog.resize(1400,950)
                fill(dialog.findChild(QLineEdit,'fileNameEdit'),path);capture(name)
                box=dialog.findChild(QDialogButtonBox)
                choices=[b for b in box.buttons() if box.buttonRole(b)==QDialogButtonBox.AcceptRole]
                if len(choices)!=1:raise ValueError('No unique picker accept action')
                click(choices[0])
            except Exception as error:
                errors.append(str(error))
                if dialog is not None:dialog.reject()
        def abort():
            errors.append('Actual picker timed out');dialog=app.activeModalWidget()
            if dialog is not None:dialog.reject()
        timer.timeout.connect(handle);watchdog.timeout.connect(abort);timer.start(300);watchdog.start(20000)
        try:click(pressed)
        finally:timer.stop();watchdog.stop();timer.deleteLater();watchdog.deleteLater()
        if errors or not accepted:raise ValueError('; '.join(errors) or 'No accepted file')
        settle(.4)

    def native_motion(point):
        xtest.XTestFakeMotionEvent(desktop.display,-1,point.x(),point.y(),0);desktop.x.XFlush(desktop.display)

    try:
        tiles=[w for w in window.findChildren(QAbstractButton) if w.isVisible() and
               (w.property('moduleAppKey')=='qc_dashboard' or w.property('navKey')=='qc_dashboard')]
        if not tiles:raise ValueError('No actual Home QC tile')
        click(max(tiles,key=lambda w:w.width()*w.height()));settle(.8)
        host=window._screens['qc_dashboard']
        folds=[w for w in host.findChildren(FoldButton) if w.isVisible() and w.app_key=='layer_viewer']
        if len(folds)!=1:raise ValueError('No unique QC Layer Viewer fold')
        capture('01_qc_host');click(folds[0]);settle(.8)
        viewers=[w for w in window.findChildren(LayerViewer) if w.isVisible()]
        if len(viewers)!=1:raise ValueError('Actual Layer Viewer did not open')
        screen=viewers[0];screen.stack.subscribe(lambda e:events.append(dict(kind=e.kind,detail=e.detail)))
        screen.canvas.picked.connect(lambda l,w,v:picks.append(dict(layer=l.name if l else None,world=dict(w),value=v)))
        screen.object_picked.connect(lambda key:keys.append(key));capture('02_empty_layer_viewer')
        picker(screen.add_image_button,work/Path(source['image']).name,'03_choose_real_image')
        expected.append(dict(name='image',kind='image',visible=True,opacity=1.,blending='translucent',colormap='gray'))
        wide=check('04_raw_image_in_wide_pane')
        proof['initial_axis_step_ratio']=wide['canvas']['step'][1]/wide['canvas']['step'][0]
        # A real splitter drag makes the canvas square; it does not change
        # a model array or forge physical spacing to improve the picture.
        splitter=screen.canvas.parentWidget();handle=splitter.handle(1);start=handle.rect().center()
        target=screen.canvas.mapTo(window,QPoint(0,0)).x()+screen.canvas.height()
        end=start+QPoint(target-handle.mapTo(window,start).x(),0)
        QTest.mousePress(handle,Qt.LeftButton,pos=start);QTest.mouseMove(handle,end,delay=150)
        QTest.mouseRelease(handle,Qt.LeftButton,pos=end);settle(.4);click(screen.reset_button)
        check('05_fit_after_actual_splitter_resize')
        picker(screen.add_mask_button,work/Path(source['mask']).name,'06_choose_matching_cell_mask')
        expected.append(dict(name='mask',kind='labels',visible=True,opacity=.5,blending='translucent'))
        baseline=check('07_real_44_label_overlay')
        visible('mask',False);check('08_mask_hidden_source_visible')
        visible('mask',True);check('09_mask_visible_again')
        opacity(0);check('10_opacity_zero_not_deleted');opacity(100);check('11_full_label_colours')
        opacity(50);check('12_half_opacity_restored')
        for mode,num in [('additive','13'),('multiply','14'),('minimum','15'),('opaque','16')]:
            combo(screen.blending_combo,mode,frame=num+'_actual_blend_menu' if num=='13' else None)
            expected[1]['blending']=mode;check(num+'_blend_'+mode)
        combo(screen.blending_combo,'translucent');expected[1]['blending']='translucent'
        select('image');combo(screen.colormap_combo,'cyan',frame='17_image_colormap_choices')
        expected[0]['colormap']='cyan';check('18_cyan_image_not_changed_measurements')
        combo(screen.colormap_combo,'gray');expected[0]['colormap']='gray'
        select('mask');click(screen.lower_button);expected.reverse();check('19_mask_under_image')
        click(screen.raise_button);expected.reverse();check('20_mask_back_on_top')
        # Select a real source-label interior; verify what the actual pointer
        # maps to after integer widget positioning, without injecting a pick.
        label=int(np.argmax(np.bincount(mask.ravel())[1:])+1)
        ys,xs=np.nonzero(mask==label);canvas=screen.canvas.canvas;point=None
        for i in range(len(ys)//2,len(ys),max(1,len(ys)//40)):
            row=round((int(ys[i])-canvas.origin[0])/canvas.step[0]);col=round((int(xs[i])-canvas.origin[1])/canvas.step[1])
            y=round(canvas.origin[0]+canvas.step[0]*row);x=round(canvas.origin[1]+canvas.step[1]*col)
            if 0<=row<canvas.shape[0] and 0<=col<canvas.shape[1] and int(mask[y,x])==label:
                point=QPoint(col+1,row+1);break
        if point is None:raise ValueError('No visible real label interior')
        QTest.mouseClick(screen.canvas,Qt.LeftButton,pos=point);settle(.3)
        if not picks or picks[-1]['value']!=label or picks[-1]['layer']!='mask' or f'label {label}' not in screen.status.text():
            raise ValueError('Actual label pick differs from independent source pixel')
        proof['label_pick']=picks[-1];check('21_actual_label_no_field_identity')
        before=screen.canvas.canvas;anchor=(point.y()-1,point.x()-1)
        world=[before.origin[i]+before.step[i]*anchor[i] for i in (0,1)]
        native_motion(screen.canvas.mapToGlobal(point));settle(.1)
        for _ in range(3):
            xtest.XTestFakeButtonEvent(desktop.display,4,1,0);xtest.XTestFakeButtonEvent(desktop.display,4,0,0)
            desktop.x.XFlush(desktop.display);settle(.3)
        after=screen.canvas.canvas
        if any(not math.isclose(after.step[i],before.step[i]/1.2**3,abs_tol=1e-10) or
               not math.isclose(after.origin[i]+after.step[i]*anchor[i],world[i],abs_tol=1e-10) for i in (0,1)):
            raise ValueError('Actual wheel zoom did not preserve cursor world position')
        check('22_native_wheel_zoom_keeps_world_anchor')
        start=screen.canvas.rect().center();end=start+QPoint(90,45);before=screen.canvas.canvas
        QTest.mousePress(screen.canvas,Qt.LeftButton,Qt.ShiftModifier,pos=start)
        QTest.mouseMove(screen.canvas,end,delay=180);QTest.mouseRelease(screen.canvas,Qt.LeftButton,Qt.ShiftModifier,pos=end);settle(.4)
        after=screen.canvas.canvas
        if any(not math.isclose(after.origin[i],before.origin[i]-before.step[i]*delta,abs_tol=1e-10)
               for i,delta in enumerate((45,90))):raise ValueError('Actual shift-drag pan differs')
        check('23_native_pan');click(screen.reset_button);restored=check('24_fit_restores_original_view')
        if restored['canvas']!=baseline['canvas']:raise ValueError('Fit did not restore actual pre-zoom geometry')
        for kind,button in [('points',screen.add_points_button),('shapes',screen.add_shapes_button)]:
            click(button);expected.append(dict(name=kind,kind=kind,visible=True,opacity=1.,blending='translucent'))
            check('25_empty_'+kind+'_layer')
            QTest.mouseClick(screen.canvas,Qt.LeftButton,pos=point);settle(.25)
            check('26_click_does_not_draw_'+kind)
            select(kind);click(screen.remove_button);expected.pop();check('27_remove_empty_'+kind)
        select('mask');click(screen.remove_button);expected.pop();check('28_remove_mask_only_from_view')
        if keys:raise ValueError('Bare TIFF unexpectedly claimed globally keyed objects')
        proof.update(accepted=True,object_keys_published=keys,layer_events=events,pipeline_run_requested=False,
                     mask_objects=44,source_pixel_count=int(image.size),image_dtype=str(image.dtype))
    finally:
        desktop.close()
        for path,before in original.items():require_unchanged_source(Path(path),before)
        proof['original_files_unchanged']=True
        proof['private_files_unchanged']=all(_digest(work/Path(source[k]).name)==source[k+'_sha256'] for k in ('image','mask'))
        write_json(Path(captures)/'layer_viewer_acceptance.json',proof)
    if not proof['private_files_unchanged']:raise ValueError('Private image or mask file changed')
