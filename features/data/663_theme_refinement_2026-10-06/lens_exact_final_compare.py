import hashlib,importlib.util,json,math,statistics,sys,time
from pathlib import Path
sys.path.insert(0,'/mnt/wd4tb/spacr-worktrees/codex-theme-clock-20261006')
import spacr.qt.widgets
from PySide6.QtGui import QImage
base=Path('/mnt/wd4tb/scratch/theme-refinement-20261006')
modules=[]
for kind,path in (('before',base/'lens_sparse_before/ambient.py'),('cached',base/'lens_sparse_safe/ambient.py')):
    spec=importlib.util.spec_from_file_location('spacr.qt.widgets._lens_'+kind,path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);modules.append(module)
results=[]
for width,height,size,density in ((320,200,1.5,.25),(1920,1080,1.,1.),(3840,2160,1.,1.),(3840,2160,.5,3.)):
    engines=[]
    for module in modules:
        engine=module.make_engine('data_art_impulse_lens','spacr','#101418',seed=42,resolution=2,blur=0,speed=1,size=size,density=density)
        engine.set_max_pixels(width*height)
        engine.set_time(8)
        engine._gravity_impulses=[(8-i*.1,(.5+.12*math.sin(i*.07),.5+.09*math.cos(i*.1)),.24 if i else 1.) for i in range(24)]
        engines.append(engine)
    pairs=[]
    for age in (0.,.06,.25,.8,2.4,5.2):
        frames=[]
        for engine in engines:
            engine.set_time(8+age)
            engine.set_pointer((.38+age*.01,.57))
            image=engine.shade(width,height)
            frames.append(image.bits().tobytes())
        pairs.append({'age':age,'byte_identical':frames[0]==frames[1]})
    times=[[],[]]
    if False:
        for engine in engines:
            engine._gravity_impulses=[(9-i*.1,(.5+.12*math.sin(i*.07),.5+.09*math.cos(i*.1)),.24 if i else 1.) for i in range(24)]
        for index in range(20):
            for slot in (index%2,1-index%2):
                engine=engines[slot];engine.set_time(9+index/24)
                started=time.perf_counter();engine.shade(width,height);times[slot].append((time.perf_counter()-started)*1000)
    result={'display':[width,height],'size':size,'density':density,'frame_parity':pairs,'shade_median_ms':[statistics.median(v) if v else None for v in times],'shade_p95_ms':[sorted(v)[math.ceil(.95*len(v))-1] if v else None for v in times]}
    results.append(result);print(json.dumps(result),flush=True)
(base/'lens_exact_final_parity.json').write_text(json.dumps(results,indent=2))
assert all(pair['byte_identical'] for result in results for pair in result['frame_parity'])
