import hashlib
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from PySide6.QtWidgets import QApplication
from spacr.qt.widgets import ambient

QApplication.instance() or QApplication([])
rows=[]
for width,height,density,size,resolution in ((1280,720,1,1,1),(3840,2160,1,1,1),(3840,2160,3,3,2)):
    engine=ambient.make_engine('aurora','spacr','#101418',seed=7,density=density,size=size,resolution=resolution)
    engine.set_max_pixels(width*height)
    times=defaultdict(list)
    for name in ('geometry','_tile','_surge','_sheet','_soften'):
        original=getattr(engine,name)
        def timed(*args,_original=original,_name=name,**kwargs):
            start=time.perf_counter()
            answer=_original(*args,**kwargs)
            times[_name].append(time.perf_counter()-start)
            return answer
        setattr(engine,name,timed)
    field=engine._paint_field
    class Meter:
        def __init__(self,painter):
            self.painter=painter
        def __getattr__(self,name):
            attr=getattr(self.painter,name)
            if name != 'drawPath':
                return attr
            def timed(*args,**kwargs):
                start=time.perf_counter()
                result=attr(*args,**kwargs)
                times[name].append(time.perf_counter()-start)
                return result
            return timed
    def measured_field(painter,w,h):
        start=time.perf_counter()
        result=field(Meter(painter),w,h)
        times['field'].append(time.perf_counter()-start)
        return result
    engine._paint_field=measured_field
    elapsed=[]
    cache=[]
    for index in range(10):
        engine.set_time(9+index/12)
        start=time.perf_counter()
        frame=engine.shade(width,height)
        if index>=3:
            elapsed.append(time.perf_counter()-start)
            cache.append((len(engine._tiles),len(engine._surges)))
        assert frame.width()==width and frame.height()==height
    summary={name:{'calls':len(values),'sum_ms':round(sum(values)*1000,2),'median_ms':round(statistics.median(values)*1000,3)} for name,values in times.items()}
    rows.append({'width':width,'height':height,'density':density,'size':size,'resolution':resolution,'warm_median_ms':round(statistics.median(elapsed)*1000,2),'warm_max_ms':round(max(elapsed)*1000,2),'cache_after_warm':cache,'instrumented_components_all_10_frames':summary})
print(json.dumps({'source':ambient.__file__,'source_sha256':hashlib.sha256(Path(ambient.__file__).read_bytes()).hexdigest(),'rows':rows},indent=2))
