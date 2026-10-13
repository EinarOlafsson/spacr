"""Short causal GUI blit/worker contention diagnostic; production unmodified."""
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time

import numpy as np
from PySide6.QtCore import QEvent, QEventLoop, QTimer
from PySide6.QtWidgets import QApplication, QWidget

ROOT=Path(sys.argv[1]).resolve();sys.path.insert(0,str(ROOT))
from spacr.qt.widgets import ambient
assert Path(ambient.__file__).resolve()==ROOT/'spacr/qt/widgets/ambient.py'
OUT=Path(__file__).parent
app=QApplication([]);ambient.screen_pixels=lambda _:3840*2160
host=QWidget();host.resize(3840,2160)
widget=ambient.install_ambient(host,theme='data_art_tissue_facets',palette='spacr',background='#101418',seed=42,blur=0,speed=1,size=1,resolution=2,density=3,gravity_radius=.5,fps=24)
widget._data_art_pointer_for_tick=lambda:(.5,.5)
engine=widget.engine
samples=[];phase='cold';gui=threading.get_ident();original_shade=engine.shade;original_paint=widget._paint_ambient;original_blit=engine.blit


def measured(name,function):
 def call(*args,**kwargs):
  wall=time.perf_counter();cpu=time.thread_time();result=function(*args,**kwargs)
  samples.append((phase,name,threading.get_ident()==gui,(time.perf_counter()-wall)*1000,(time.thread_time()-cpu)*1000))
  return result
 return call


engine.shade=measured('shade',original_shade)
widget._paint_ambient=measured('paint',original_paint)
engine.blit=measured('blit',original_blit)


def wait(milliseconds):
 loop=QEventLoop();QTimer.singleShot(milliseconds,loop.quit);loop.exec()


host.show();app.processEvents();wait(500)
producer=widget._producer_box[0];assert producer.size==(3840,2160)
rows=[]
for label in ('normal_A','diagnostic_no_blit','normal_B'):
 phase=label
 if label=='diagnostic_no_blit':engine.blit=lambda *_args:None
 else:engine.blit=measured('blit',original_blit)
 started=time.perf_counter();produced=producer.frames_shaded;displayed=widget.frames_painted;repeated=widget.repeated_frames;wait(1500);elapsed=time.perf_counter()-started
 selected=[row for row in samples if row[0]==label];groups={}
 for name in ('shade','paint','blit'):
  values=[r for r in selected if r[1]==name]
  if values:groups[name]={'count':len(values),'wall_median_ms':float(np.median([r[3] for r in values])),'wall_p95_ms':float(np.percentile([r[3] for r in values],95)),'thread_cpu_median_ms':float(np.median([r[4] for r in values])),'wall_total_ms':sum(r[3] for r in values),'thread_cpu_total_ms':sum(r[4] for r in values)}
 rows.append({'phase':label,'duration_seconds':elapsed,'producer_fps':(producer.frames_shaded-produced)/elapsed,'paint_fps':(widget.frames_painted-displayed)/elapsed,'published_slot_paint_fps':((widget.frames_painted-displayed)-(widget.repeated_frames-repeated))/elapsed,'groups':groups})
engine.blit=original_blit
with widget._engine_lock:
 tiles=next(v for k,v in engine._material_cache.items() if k[0]=='tissue_facets');angles=next(v[1] for k,v in engine._material_cache.items() if k[0]=='tissue_rotation');population=len(tiles);spinning=sum(a!=0 for a in angles)
image=producer.latest();assert image.width()==3840 and image.height()==2160;assert np.all(np.frombuffer(image.constBits(),np.uint32)>>24==255)
host.close();assert not producer.is_alive();host.deleteLater();app.sendPostedEvents(None,QEvent.DeferredDelete);app.processEvents()
result={'source_sha256':hashlib.sha256(Path(ambient.__file__).read_bytes()).hexdigest(),'source_commit':'254f3dd39f','controls':{'width':3840,'height':2160,'density':3,'detail':2,'size':1,'radius':.5,'fps':24},'population':population,'spinning':spinning,'phases':rows,'load_average':os.getloadavg(),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'worker_retired':not producer.is_alive(),'widgets_after_delete':len(app.allWidgets()),'samples':samples,'limits':'Instrumented short diagnostic, not optimization/performance acceptance. Only middle scratch phase suppresses GUI blit to isolate contention; worker retains native pixels/count/clock/input. Product untouched; normal phases restored.'}
(OUT/'live_overhead.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2))
