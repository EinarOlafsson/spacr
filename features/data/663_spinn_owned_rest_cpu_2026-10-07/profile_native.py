"""Profile exact current high-density native spinn without GUI scaling."""
import cProfile
import hashlib
import io
import json
import os
from pathlib import Path
import pstats
import resource
import statistics
import sys
import time

import numpy as np
from PySide6.QtWidgets import QApplication

ROOT=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(ROOT))
from spacr.qt.widgets import ambient
assert Path(ambient.__file__).resolve() == ROOT/'spacr/qt/widgets/ambient.py'
OUT=Path(__file__).parent
app=QApplication([])
ambient._begin_ambient_startup()
rows=[]
for background in ('#101418','#f4f4f0'):
 engine=ambient.make_engine('data_art_tissue_facets','spacr',background,seed=42,
                             resolution=2,density=3,size=1,blur=0)
 engine.set_max_pixels(3840*2160)
 engine.set_gravity_radius(.5)
 engine.set_pointer((.5,.5))
 start=time.perf_counter();first=engine.shade(3840,2160);cold=(time.perf_counter()-start)*1000
 assert engine.effective_density()==3
 assert (first.width(),first.height())==(3840,2160)
 samples=[]
 for index in range(8):
  engine.advance(1/24);start=time.perf_counter(); frame=engine.shade(3840,2160);samples.append((time.perf_counter()-start)*1000)
 assert np.all(np.frombuffer(frame.constBits(),np.uint32)>>24==255)
 profiler=cProfile.Profile();profiler.enable()
 for _ in range(4):
  engine.advance(1/24);frame=engine.shade(3840,2160)
 profiler.disable()
 stream=io.StringIO();pstats.Stats(profiler,stream=stream).sort_stats('cumtime').print_stats(24)
 label='dark' if background=='#101418' else 'light'
 (OUT/(label+'-profile.txt')).write_text(stream.getvalue())
 tiles=next(v for k,v in engine._material_cache.items() if k[0]=='tissue_facets')
 angles=next(v[1] for k,v in engine._material_cache.items() if k[0]=='tissue_rotation')
 rows.append({'background':background,'cold_ms':cold,'samples_ms':samples,'median_ms':statistics.median(samples),'p95_ms':float(np.percentile(samples,95)),'count':len(tiles),'rotating':sum(a!=0 for a in angles),'tile_bytes':sum(c[5].sizeInBytes() for c in tiles),'profile':label+'-profile.txt','frame_sha256':hashlib.sha256(frame.constBits()).hexdigest()})
 frame.save(str(OUT/(label+'-native.png')))
 del engine,first,frame
receipt={'source_sha256':hashlib.sha256(Path(ambient.__file__).read_bytes()).hexdigest(),'source_commit':'368e2cdf1d','import':ambient.__file__,'size':[3840,2160],'controls':{'density':3,'detail':2,'size':1,'radius':.5},'records':rows,'load_average':os.getloadavg(),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'scope':'Direct shader/profile only, not live cadence or hard24FPS acceptance.'}
(OUT/'profile_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
