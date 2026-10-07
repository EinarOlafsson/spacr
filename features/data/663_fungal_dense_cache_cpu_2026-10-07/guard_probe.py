"""Private density-domain cache gate with exact original native pixels."""
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QApplication

root=Path(__file__).resolve().parent
sys.path.insert(0,sys.argv[1])
import spacr.qt.widgets
app=QApplication([])
mods={}
for name in ['before','candidate']:
 spec=importlib.util.spec_from_file_location('spacr.qt.widgets._density_gate_'+name,root/(name+'.py'))
 module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);mods[name]=module

def engine(name,palette,density):
 e=mods[name].make_engine('data_art_fungal_growth','spacr' if palette in ['white','rgb16'] else palette,'#101418',seed=42,resolution=1,density=density,size=2.5,blur=0)
 if palette=='white':e.set_colors(['white'])
 if palette=='rgb16':e.set_colors(['#fffe00017777','#0303fffefefe'])
 e.set_max_pixels(3840*2160);e.set_time(95)
 e._hits=0
 original=e._reuse_fungal_raster
 def reuse(*args):
  result=original(*args);e._hits+=bool(result);return result
 e._reuse_fungal_raster=reuse
 return e

rows=[];timings=[];rejected=False
for palette,density in [('spacr',2),('spacr',3),('random',2),('random',3),('white',3),('rgb16',3)]:
 engines={name:engine(name,palette,density) for name in mods}
 e=engines['candidate']
 for _ in range(28):e.shade(3840,2160)
 for clock in [95,98]:
  for engine_ in engines.values():engine_.set_time(clock)
  before_hits=e._hits
  frames={name:engine_.shade(3840,2160) for name,engine_ in engines.items()}
  planes={name:np.frombuffer(image.constBits(),dtype=np.uint32) for name,image in frames.items()}
  delta=np.flatnonzero(planes['before']!=planes['candidate'])
  assert all(np.all(plane>>24==255) for plane in planes.values())
  retained=sum(entry[1].nbytes+entry[2].nbytes for entry in e._fungal_rasters.values());assert retained<=8*1024**2 and len(e._fungal_rasters)<=64
  row={'palette':palette,'density':density,'resolution':1,'effective_density':e.effective_density(),'clock':clock,'differing_pixels':len(delta),'first_offsets':delta[:10].tolist(),'cache_hits':e._hits-before_hits,'entries':len(e._fungal_rasters),'owned_array_bytes':retained,'hashes':{name:hashlib.sha256(frame.constBits()).hexdigest() for name,frame in frames.items()}}
  rows.append(row);print('PARITY',json.dumps(row),flush=True)
  if len(delta):rejected=True;break
 if rejected:break
 if palette in ['spacr','random']:
  for engine_ in engines.values():engine_.set_time(95)
  durations={name:[] for name in mods}
  for repeat in range(20):
   for name in (['before','candidate'] if repeat%2==0 else ['candidate','before']):
    start=time.perf_counter();frame=engines[name].shade(3840,2160);durations[name].append((time.perf_counter()-start)*1000)
  item={'palette':palette,'density':density,'shader_balanced':{name:{'median_ms':statistics.median(v),'p95_ms':sorted(v)[math.ceil(.95*len(v))-1]} for name,v in durations.items()}}
  timings.append(item);print('TIMING',json.dumps(item),flush=True)
receipt={'source_sha256':{name:hashlib.sha256((root/(name+'.py')).read_bytes()).hexdigest() for name in mods},'scope':'scratch-only widened cache domain, full native3840x2160 CPU; no app edit, cache still64/8MiB','all_pairs_exact':not rejected,'records':rows,'timings':timings}
(root/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
