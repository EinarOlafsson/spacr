import hashlib,importlib.util,json,statistics,sys,time
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import QApplication
import spacr.qt.widgets
root=Path(__file__).resolve().parent
app=QApplication([]);mods={}
for name in ['before','candidate']:
 spec=importlib.util.spec_from_file_location('spacr.qt.widgets._stroker_'+name,root/(name+'.py'))
 m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);mods[name]=m
rows=[]
for palette,size,resolution,density in [('spacr',2.5,2,3),('random',2.5,2,3),('spacr',2.0,1,3),('spacr',1.0,2,3)]:
 engines={k:m.make_engine('data_art_fungal_growth',palette,'#101418',seed=42,resolution=resolution,density=density,size=size,blur=0) for k,m in mods.items()}
 for e in engines.values():e.set_max_pixels(3840*2160);e.set_time(95)
 for e in engines.values():
  for _ in range(5):e.shade(3840,2160)
 failed=False
 for clock in [95,98,3600.33]:
  for e in engines.values():e.set_time(clock)
  frames={k:e.shade(3840,2160) for k,e in engines.items()}
  planes={k:np.frombuffer(f.constBits(),dtype=np.uint32) for k,f in frames.items()}
  delta=np.flatnonzero(planes['before']!=planes['candidate'])
  row={'palette':palette,'size':size,'resolution':resolution,'density':density,'clock':clock,'different_pixels':int(len(delta)),'strokes':len(engines['candidate']._fungal_strokes),'hashes':{k:hashlib.sha256(f.constBits()).hexdigest() for k,f in frames.items()}}
  rows.append(row);print('PARITY',json.dumps(row),flush=True)
  if len(delta):failed=True;break
 if failed:break
 for e in engines.values():e.set_time(95)
 times={k:[] for k in engines}
 for index in range(16):
  for k in (list(engines) if index%2==0 else list(engines)[::-1]):
   start=time.perf_counter();engines[k].shade(3840,2160);times[k].append((time.perf_counter()-start)*1000)
 row={'palette':palette,'size':size,'resolution':resolution,'density':density,'shader_median_ms':{k:statistics.median(v) for k,v in times.items()}}
 rows.append(row);print('TIMING',json.dumps(row),flush=True)
(root/'cached-receipt.json').write_text(json.dumps({'source_sha256':{k:hashlib.sha256((root/(k+'.py')).read_bytes()).hexdigest() for k in mods},'records':rows,'scope':'Scratch source only; full native4K parity and shader timing, not live widget cadence.'},indent=2)+'\n')
