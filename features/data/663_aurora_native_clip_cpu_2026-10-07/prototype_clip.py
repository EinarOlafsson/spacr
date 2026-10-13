import hashlib
import inspect
import json
import statistics
import textwrap
import time
import types
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import QApplication
from spacr.qt.widgets import ambient

old='''            surge = QBrush(self._surge(curtain, peak))
            surge.setTransform(QTransform(
                (right - left) / pulse_w, 0.0, 0.0, band / pulse_h,
                left, top - ray * AURORA_PULSE_PAD))
            painter.setBrush(surge)
            painter.drawPath(self._sheet(
                columns, zero - ray * (AURORA_PULSE_HEIGHT
                                       + AURORA_PULSE_PAD)))'''
new='''            surge = self._surge(curtain, peak)
            painter.save()
            painter.setClipPath(self._sheet(
                columns, zero - ray * (AURORA_PULSE_HEIGHT
                                       + AURORA_PULSE_PAD)))
            painter.drawImage(QRectF(left, top - ray * AURORA_PULSE_PAD,
                                     right - left, band), surge)
            painter.restore()'''
source=textwrap.dedent(inspect.getsource(ambient.AuroraEngine._paint_field))
start=source.index('        surge = QBrush(self._surge(curtain, peak))')
end=source.index('        roles = self.ramp_colors(',start)
old=source[start:end]
new=textwrap.indent(textwrap.dedent(new),'        ')+'\n'
namespace=dict(ambient.__dict__)
exec(source.replace(old,new),namespace)
replacement=namespace['_paint_field']
QApplication.instance() or QApplication([])
rows=[]
for width,height,density,size,resolution in ((1280,720,1,1,1),(3840,2160,1,1,1),(3840,2160,3,3,2)):
    timings={}
    frames={}
    for mode in ('baseline','candidate'):
        engine=ambient.make_engine('aurora','spacr','#101418',seed=7,density=density,size=size,resolution=resolution)
        engine.set_max_pixels(width*height)
        if mode=='candidate':
            engine._paint_field=types.MethodType(replacement,engine)
        durations=[]
        for index in range(9):
            engine.set_time(9+index/12)
            start=time.perf_counter()
            frame=engine.shade(width,height)
            durations.append((time.perf_counter()-start)*1000)
            if index==6:
                frames[mode]=np.frombuffer(frame.bits(),dtype=np.uint8).copy()
        timings[mode]={'warm_median_ms':round(statistics.median(durations[2:]),2),'warm_max_ms':round(max(durations[2:]),2),'all_ms':[round(x,2) for x in durations]}
    a,b=frames['baseline'],frames['candidate']
    assert a.size==b.size
    changed=int(np.count_nonzero(np.any(a.reshape(-1,4)!=b.reshape(-1,4),axis=1)))
    rows.append({'size':[width,height],'density':density,'element_size':size,'resolution':resolution,'timings':timings,'changed_pixels':changed,'changed_fraction':round(changed/(width*height),6),'max_byte_diff':int(np.abs(a.astype(np.int16)-b.astype(np.int16)).max())})
print(json.dumps({'source':ambient.__file__,'source_sha256':hashlib.sha256(Path(ambient.__file__).read_bytes()).hexdigest(),'candidate':'same painter code except transformed 16x16 surge brush path replaced by antialiased path clip and drawImage mapped to the same rect','rows':rows},indent=2))
