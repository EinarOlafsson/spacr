"""Balanced real native widget/producer probe at fixed requested controls."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
from PySide6.QtCore import QEvent, QEventLoop, QTimer
from PySide6.QtWidgets import QApplication, QWidget

ROOT=Path(sys.argv[1]).resolve();sys.path.insert(0,str(ROOT))
from spacr.qt.widgets import ambient
OUT=Path(__file__).parent
spec=importlib.util.spec_from_file_location('spacr.qt.widgets._spinn_live_owned',OUT/'owned_candidate.py');candidate=importlib.util.module_from_spec(spec);sys.modules[spec.name]=candidate;spec.loader.exec_module(candidate)
app=QApplication([])
for owner in (ambient,candidate):owner.screen_pixels=lambda _:3840*2160


def wait(milliseconds):
 loop=QEventLoop();QTimer.singleShot(milliseconds,loop.quit);loop.exec()


def run(owner,label,active):
 host=QWidget();host.resize(3840,2160)
 widget=owner.install_ambient(host,theme='data_art_tissue_facets',palette='spacr',background='#101418',seed=42,blur=0,speed=1,size=1,resolution=2,density=3,gravity_radius=.5 if active else 0,fps=24)
 if active:widget._data_art_pointer_for_tick=lambda:(.5,.5)
 start=time.perf_counter();host.show();app.processEvents();cold=(time.perf_counter()-start)*1000
 wait(500)
 producer=widget._producer_box[0];assert producer is not None and producer.size==(3840,2160)
 first=producer.latest();assert first.size().width()==3840 and first.size().height()==2160
 retained=bytes(first.constBits());assert np.all(np.frombuffer(first.constBits(),np.uint32)>>24==255)
 assert widget.engine.effective_density()==3
 beats=[];timer=QTimer();timer.setInterval(10);timer.timeout.connect(lambda:beats.append(time.perf_counter()));timer.start()
 produced=producer.frames_shaded;painted=widget.frames_painted;repeat=widget.repeated_frames;started=time.perf_counter()
 wait(2000)
 elapsed=time.perf_counter()-started;frames=producer.frames_shaded-produced;displayed=widget.frames_painted-painted;repeats=widget.repeated_frames-repeat
 timer.stop();gaps=np.diff(beats)*1000
 with widget._engine_lock:
  material=next(v for k,v in widget.engine._material_cache.items() if k[0]=='tissue_facets')
  angles=next(v[1] for k,v in widget.engine._material_cache.items() if k[0]=='tissue_rotation')
  state={'count':len(material),'rotating':sum(a!=0 for a in angles),'tile_bytes':sum(c[5].sizeInBytes() for c in material),'rest_markers':sum(k[0]=='tissue_resting' for k in widget.engine._material_cache)}
 assert bytes(first.constBits())==retained
 host.close();assert not producer.is_alive();assert not widget.shading_thread_alive()
 row={'variant':label,'active':active,'elapsed_seconds':elapsed,'cold_show_ms':cold,'producer_fps':frames/elapsed,'paint_fps':displayed/elapsed,'distinct_slot_paint_fps':(displayed-repeats)/elapsed,'heartbeat_p95_ms':float(np.percentile(gaps,95)),'heartbeat_max_ms':float(gaps.max()),'worker_retired':True,'retained_frame_unchanged':True,**state}
 del first,retained,widget,producer,material,angles
 host.deleteLater();app.sendPostedEvents(None,QEvent.DeferredDelete);app.processEvents();del host
 row['widgets_after_delete']=len(app.allWidgets())
 return row


rows=[]
for active in (False,True):
 for label,owner in [('before',ambient),('after',candidate),('after',candidate),('before',ambient)]:
  row=run(owner,label,active);rows.append(row);print(json.dumps(row),flush=True)
result={'before_sha256':hashlib.sha256(Path(ambient.__file__).read_bytes()).hexdigest(),'after_sha256':hashlib.sha256((OUT/'owned_candidate.py').read_bytes()).hexdigest(),'source_commit':'368e2cdf1d','native_size':[3840,2160],'controls':{'density':3,'detail':2,'size':1,'fps':24,'palette':'spacr','background':'#101418','active_radius':.5},'records':rows,'load_average':os.getloadavg(),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'limits':'Offscreen actual worker+GUI with fixed normalized pointer supplied on each active tick. Rest has intentionally identical pixels; slot FPS is not scene-change FPS. No all-theme hard24/native display/aesthetic acceptance.'}
(OUT/'live_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
