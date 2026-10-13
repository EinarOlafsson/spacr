"""Independent source identities, data-to-pixel coordinates and RGB crops."""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image


def identity(row):
    return '_'.join(str(row[k]) for k in ('plateID','rowID','columnID','fieldID'))+f"_cell{int(row['object_label'])}"


def close(actual,expected,name):
    a=np.asarray(actual,dtype=float);b=np.asarray(expected,dtype=float)
    if a.shape!=b.shape or not np.allclose(a,b,rtol=0,atol=1e-9,equal_nan=True):
        raise ValueError('Image Scatter numeric discrepancy: '+name)
    finite=np.isfinite(a)&np.isfinite(b)
    return float(np.max(np.abs(a[finite]-b[finite]),initial=0))


def positions(rows,x,y,width,height):
    xx=np.array([np.nan if r[x] is None else r[x] for r in rows],float)
    yy=np.array([np.nan if r[y] is None else r[y] for r in rows],float)
    good=np.isfinite(xx)&np.isfinite(yy);px=np.full(len(rows),np.nan);py=px.copy()
    if good.any():
        left,right=float(min(xx[good])),float(max(xx[good]));bottom,top=float(min(yy[good])),float(max(yy[good]))
        w=max(1,width-36);h=max(1,height-36)
        for i in np.flatnonzero(good):
            px[i]=18+(float(xx[i])-left)/(right-left)*w if right>left else 18+w/2
            py[i]=18+h-(float(yy[i])-bottom)/(top-bottom)*h if top>bottom else 18+h/2
    return xx,yy,px,py,good


def verify(screen,rows,crops):
    keys=[identity(r) for r in rows]
    if list(screen._keys)!=keys or len(screen._frame)!=len(rows):raise ValueError('Image Scatter object identities differ')
    if dict(screen._paths)!=crops:raise ValueError('Image Scatter crop identity mapping differs')
    for column in rows[0]:
        actual=list(screen._frame[column]);wanted=[r[column] for r in rows]
        for a,b in zip(actual,wanted):
            if b is None:
                if not __import__('pandas').isna(a):raise ValueError('Image Scatter source missing value changed')
            elif a!=b:raise ValueError('Image Scatter source table value changed: '+column)
    x=screen._x_choice.currentText();y=screen._y_choice.currentText();canvas=screen.canvas
    xx,yy,px,py,good=positions(rows,x,y,canvas.width(),canvas.height())
    errors=[close(canvas._x,xx,'source X'),close(canvas._y,yy,'source Y'),
            close(canvas._px,px,'painted X'),close(canvas._py,py,'painted Y')]
    if not np.array_equal(canvas.plottable,good):raise ValueError('Image Scatter finite mask differs')
    if canvas._x_label!=x or canvas._y_label!=y:raise ValueError('Image Scatter axis labels differ')
    return dict(rows=len(rows),x=x,y=y,finite_points=int(good.sum()),missing_coordinate_pairs=int((~good).sum()),
                every_source_cell_checked=len(rows)*len(rows[0]),all_crop_paths_checked=len(crops),
                maximum_pixel_coordinate_error=max(errors),status=screen.status.text())


def verify_preview(pixmap,path,caption,wanted_key):
    from PySide6.QtGui import QImage
    if caption!=wanted_key:raise ValueError('Image Scatter hover caption differs from object')
    if pixmap is None or pixmap.isNull():raise ValueError('Image Scatter hover image missing')
    marker=json.loads((Path(path).parent/'.spacr_crop_format.json').read_text())
    if marker.get('spacr_crop_format')!=3 or marker.get('channel_order')!='declared_rgb':
        raise ValueError('Tutorial crop oracle requires declared RGB format3')
    with Image.open(path) as src:
        if src.mode!='RGB':raise ValueError('Tutorial crop oracle requires uint8 RGB PNG')
        expected=src.copy();expected.thumbnail((192,192));wanted=np.array(expected)
    image=pixmap.toImage().convertToFormat(QImage.Format_RGB888)
    actual=np.frombuffer(image.constBits(),dtype=np.uint8).reshape(image.height(),image.bytesPerLine())[:,:image.width()*3].reshape(image.height(),image.width(),3)
    if actual.shape!=wanted.shape or not np.array_equal(actual,wanted):
        raise ValueError('Image Scatter preview pixels differ from declared RGB source')
    return dict(key=wanted_key,path=str(path),shape=list(actual.shape),maximum_rgb_error=0,pixels_checked=int(image.width()*image.height()))
