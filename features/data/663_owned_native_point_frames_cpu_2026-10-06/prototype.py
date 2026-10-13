import ast,hashlib,json,statistics,time,types
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import QApplication
from spacr.qt.widgets import ambient
app=QApplication([])
path=Path(ambient.__file__)
expected='c6652f61bf'
source=path.read_text()
root=ast.parse(source)
cls=next(n for n in root.body if isinstance(n,ast.ClassDef) and n.name=='_DataArtEngine')
functions={}
for family in ('point_atlas','impulse_lens','genetic_advection'):
 method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_paint_'+family)
 method.name='_direct_'+family
 method.args.args.pop(1)
 call=method.body[-1].value
 assert isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='drawImage'
 assert len(call.args)==3
 method.body[-1]=ast.Return(value=call.args[-1])
 namespace=dict(ambient.__dict__)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(path),'exec'),namespace)
 functions[family]=namespace[method.name]
records=[]
for family,key in [('point_atlas','data_art_point_atlas'),('impulse_lens','data_art_impulse_lens'),('genetic_advection','data_art_genetic_advection')]:
 for background in ('#101418','#ffffff'):
  e=ambient.make_engine(key,palette='spacr',background=background,seed=42,resolution=2.,blur=0.)
  e.max_pixels=3840*2160
  e.set_gravity_radius(.65);e.set_pointer((.3,.6))
  for t in (2.,11.,23.):
   e.set_time(t)
   old=e.shade(3840,2160)
   new=functions[family](e,3840,2160)
   left=np.frombuffer(old.constBits(),dtype=np.uint8).copy()
   right=np.frombuffer(new.constBits(),dtype=np.uint8).copy()
   assert np.array_equal(left,right),(family,background,t)
  before=[];after=[]
  for i in range(16):
   e.set_time(30.+i*.04)
   operations=(('before',e.shade),('after',lambda w,h:functions[family](e,w,h)))
   if i%2:operations=operations[::-1]
   for label,operation in operations:
    start=time.perf_counter();image=operation(3840,2160);duration=(time.perf_counter()-start)*1000
    (before if label=='before' else after).append(duration)
  records.append({'family':family,'background':background,'native_pairs':3,'before_median_ms':statistics.median(before),'direct_median_ms':statistics.median(after)})
print(json.dumps({'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'scope':'scratch-only direct owned point image prototype; shader timing, not widget FPS','records':records},indent=2),flush=True)
