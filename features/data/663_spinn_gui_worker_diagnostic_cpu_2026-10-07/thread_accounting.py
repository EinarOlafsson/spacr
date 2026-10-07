"""Short own-process CPU accounting to distinguish raster work from idle wait."""
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

from PySide6.QtCore import QEvent, QEventLoop, QTimer
from PySide6.QtWidgets import QApplication, QWidget

ROOT=Path(sys.argv[1]).resolve();sys.path.insert(0,str(ROOT))
from spacr.qt.widgets import ambient
OUT=Path(__file__).parent;app=QApplication([]);ambient.screen_pixels=lambda _:3840*2160
host=QWidget();host.resize(3840,2160);widget=ambient.install_ambient(host,theme='data_art_tissue_facets',palette='spacr',background='#101418',seed=42,blur=0,speed=1,size=1,resolution=2,density=3,gravity_radius=.5,fps=24);widget._data_art_pointer_for_tick=lambda:(.5,.5)


def wait(ms):
 loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()


def snapshot():
 rows={}
 for p in Path('/proc/self/task').iterdir():
  try:
   s=(p/'stat').read_text().rsplit(') ',1)[1].split();rows[int(p.name)]={'name':(p/'comm').read_text().strip(),'cpu_ticks':int(s[11])+int(s[12])}
  except FileNotFoundError:pass
 return rows


host.show();app.processEvents();wait(300);producer=widget._producer_box[0]
first=snapshot();start=time.perf_counter();cpu=time.process_time();wait(900);elapsed=time.perf_counter()-start;cpu=time.process_time()-cpu;last=snapshot();ticks=os.sysconf('SC_CLK_TCK')
rows=[]
for tid,row in last.items():
 delta=(row['cpu_ticks']-first.get(tid,{'cpu_ticks':0})['cpu_ticks'])/ticks
 rows.append({'tid':tid,'name':row['name'],'cpu_seconds':delta,'role':'producer' if tid==producer._thread.native_id else 'gui' if tid==threading.get_native_id() else 'other'})
rows.sort(key=lambda r:-r['cpu_seconds'])
cgroup=Path('/proc/self/cgroup').read_text().strip().split('::')[-1];group=Path('/sys/fs/cgroup')/cgroup.lstrip('/');limits={}
for name in ('cpu.max','cpu.stat','cpuset.cpus.effective'):
 p=group/name
 if p.exists():limits[name]=p.read_text()
host.close();assert not producer.is_alive();host.deleteLater();app.sendPostedEvents(None,QEvent.DeferredDelete);app.processEvents()
result={'source_sha256':hashlib.sha256(Path(ambient.__file__).read_bytes()).hexdigest(),'elapsed_seconds':elapsed,'process_cpu_seconds':cpu,'threads':rows,'own_cgroup_cpu':limits,'worker_retired':not producer.is_alive(),'widgets_after_delete':len(app.allWidgets()),'scope':'900ms own-process accounting only, native3840x2160 Density3/Detail2/Radius.5; not a native cadence/performance candidate.'};(OUT/'thread_accounting.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
