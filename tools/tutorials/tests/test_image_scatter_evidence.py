"""Small synthetic oracle tests; real tutorial data are never fabricated."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from image_scatter_evidence import identity, verify, verify_preview
from spacr.qt.screens.image_scatter import ImageScatterScreen


@pytest.fixture
def view():
    app=QApplication.instance() or QApplication([])
    rows=[dict(plateID='plate1',rowID='r1',columnID='c1',fieldID='f1',object_label=i+1,
               cell_area=float(i+2),cell_signal=None if i==2 else float(i*i+1),measurement_ndim=2)
          for i in range(5)]
    paths={identity(r):f'/example/{i}.png' for i,r in enumerate(rows)}
    screen=ImageScatterScreen(threaded=False);screen.resize(1200,700)
    screen.set_frame(pd.DataFrame(rows),keys=list(paths),paths=paths,x='cell_area',y='cell_signal')
    screen.show();app.processEvents()
    yield app,screen,rows,paths
    screen.close();app.processEvents()


def test_actual_canvas_positions_preserve_missing_row_numbers_and_constant_axes(view):
    app,s,rows,paths=view
    assert verify(s,rows,paths)['finite_points']==4
    assert s.canvas.point_position(2) is None
    s._y_choice.setCurrentText('measurement_ndim');app.processEvents()
    assert verify(s,rows,paths)['finite_points']==5
    assert np.ptp(s.canvas._py)==0


@pytest.mark.parametrize('kind',['key','path','source','missing','x','y','pixel_x','pixel_y','label'])
def test_corrupt_identities_source_values_and_projection_fail_after_positive(view,kind):
    _,s,rows,paths=view;verify(s,rows,paths)
    if kind=='key':s._keys[0]='wrong'
    elif kind=='path':s._paths[list(paths)[0]]='/wrong.png'
    elif kind=='source':s._frame.loc[0,'cell_area']=99
    elif kind=='missing':
        # The canvas can share NumPy storage with the frame. Isolate this
        # corruption so only the source-NULL guard, not finite plotting,
        # can make the negative counterpart pass.
        s._frame=s._frame.copy(deep=True);s._frame.loc[2,'cell_signal']=99
    elif kind=='x':s.canvas._x[0]+=1
    elif kind=='y':s.canvas._y[0]+=1
    elif kind=='pixel_x':s.canvas._px[0]+=1
    elif kind=='pixel_y':s.canvas._py[0]+=1
    elif kind=='label':s.canvas._x_label='wrong'
    with pytest.raises(ValueError):verify(s,rows,paths)


@pytest.fixture
def crop(tmp_path):
    app=QApplication.instance() or QApplication([])
    grid=np.arange(300*240*3,dtype=np.uint8).reshape(300,240,3)
    path=tmp_path/'crop.png';Image.fromarray(grid).save(path)
    marker=tmp_path/'.spacr_crop_format.json'
    marker.write_text(json.dumps(dict(spacr_crop_format=3,channel_order='declared_rgb')))
    preview=Image.fromarray(grid);preview.thumbnail((192,192))
    from PIL.ImageQt import ImageQt
    pixmap=QPixmap.fromImage(ImageQt(preview).copy())
    return app,path,marker,pixmap


@pytest.mark.parametrize('kind',['caption','missing','pixels','format'])
def test_preview_identity_pixels_and_format_fail_after_positive(crop,kind):
    _,path,marker,pixmap=crop;verify_preview(pixmap,path,'cell1','cell1')
    caption='cell1'
    if kind=='caption':caption='cell2'
    elif kind=='missing':pixmap=None
    elif kind=='pixels':pixmap.fill(__import__('PySide6.QtGui',fromlist=['QColor']).QColor('red'))
    elif kind=='format':marker.write_text(json.dumps(dict(spacr_crop_format=2,channel_order='rgb')))
    with pytest.raises(ValueError):verify_preview(pixmap,path,caption,'cell1')
