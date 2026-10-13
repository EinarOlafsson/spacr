"""Attribute retained production costs and test exact scalar Qt path overloads."""
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time
from pathlib import Path

from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, sys.argv[1])
import spacr.qt.widgets
app = QApplication([])
mods = {}
for label in ['before', 'scalar']:
    spec = importlib.util.spec_from_file_location('spacr.qt.widgets._scalar_probe_' + label, ROOT / (label+'.py'))
    a = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = a
    spec.loader.exec_module(a)
    mods[label] = a


def summary(values):
    return {'median_ms': statistics.median(values), 'p95_ms': sorted(values)[math.ceil(.95*len(values))-1], 'samples': len(values)}


def engine(label, palette='spacr', background='#101418', resolution=2):
    e=mods[label].make_engine('data_art_fungal_growth', palette, background, seed=42,
                              resolution=resolution, density=3, size=2.5, blur=0)
    e.set_max_pixels(3840*2160)
    e.set_time(95)
    return e

stages, calls = {}, {}


def timed(name, function, *args):
    start=time.perf_counter()
    result=function(*args)
    stages[name]=stages.get(name,0)+(time.perf_counter()-start)*1000
    calls[name]=calls.get(name,0)+1
    return result


class Painter(QPainter):
    def drawPath(self,path):
        return timed('Qt.drawPath',super().drawPath,path)

    def setPen(self,pen):
        return timed('Qt.setPen',super().setPen,pen)

    def drawEllipse(self,*args):
        return timed('Qt.drawEllipse',super().drawEllipse,*args)


mods['before'].QPainter=Painter
profile=[]
for palette in ['spacr','random']:
    e=engine('before',palette)
    for _ in range(48):e.shade(3840,2160)
    for name in ['geometry','_fungal_paths','_paint_cached_fungal_paths','_reuse_fungal_raster','_warm_fungal_raster','_paint_fungal_tips']:
        original=getattr(e,name)
        setattr(e,name,lambda *args,n=name,f=original: timed(n,f,*args))
    rows=[]
    for index in range(8):
        stages.clear();calls.clear();e.set_time(95+index/24)
        start=time.perf_counter();image=e.shade(3840,2160)
        rows.append({'total_ms':(time.perf_counter()-start)*1000,'stages':dict(stages),'calls':dict(calls)})
    profile.append({'palette':palette,'scope':'instrumented inclusive wrappers; not additive or actualFPS',
                    'shade':summary([r['total_ms'] for r in rows]),
                    'stages':{name:summary([r['stages'].get(name,0) for r in rows]) for name in stages},'last_calls':dict(calls),'frames':rows})
mods['before'].QPainter=QPainter
builder=[]
for palette in ['spacr','random']:
    engines={label:engine(label,palette) for label in mods}
    for e in engines.values():e._fungal_paths(3840,2160)
    durations={label:[] for label in mods}
    for index in range(80):
        clock=95+index/24
        paths={}
        for label in (['before','scalar'] if index%2==0 else ['scalar','before']):
            e=engines[label];e.set_time(clock);start=time.perf_counter();paths[label]=e._fungal_paths(3840,2160);durations[label].append((time.perf_counter()-start)*1000)
        a,b=paths['before'],paths['scalar']
        assert list(a[0])==list(b[0]) and a[1:]==b[1:]
        assert all(a[0][key]==b[0][key] for key in a[0])
    builder.append({'palette':palette,'pairs':80,'all_QPainterPaths_maturity_tips_exact':True,'timing':{label:summary(times) for label,times in durations.items()}})
frames=[]
for palette in ['spacr','random']:
    engines={label:engine(label,palette) for label in mods}
    for e in engines.values():
        for _ in range(48):e.shade(3840,2160)
    durations={label:[] for label in mods};hashes=[]
    for index in range(24):
        clock=95+index/24;images={}
        for label in (['before','scalar'] if index%2==0 else ['scalar','before']):
            e=engines[label];e.set_time(clock);start=time.perf_counter();images[label]=e.shade(3840,2160);durations[label].append((time.perf_counter()-start)*1000)
        h={label:hashlib.sha256(im.constBits()).hexdigest() for label,im in images.items()};assert len(set(h.values()))==1;hashes.append(h['before'])
    frames.append({'palette':palette,'pairs':24,'all_native_frames_exact':True,'timing':{label:summary(times) for label,times in durations.items()},'hashes':hashes})
receipt={'source_sha256':{label:hashlib.sha256((ROOT/(label+'.py')).read_bytes()).hexdigest() for label in mods},
         'scope':'scratch-only native3840x2160 CPU; maxdensity3/detail2/size2.5; no sourceedit/liveFPS/integration',
         'current_production_profile':profile,'balanced_builder':builder,'balanced_native_frames':frames}
(ROOT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
for item in profile:print('PROFILE',item['palette'],item['shade'],{key:value['median_ms'] for key,value in item['stages'].items()},flush=True)
for item in builder:print('BUILDER',item['palette'],item['timing'],flush=True)
for item in frames:print('FRAME',item['palette'],item['timing'],flush=True)
