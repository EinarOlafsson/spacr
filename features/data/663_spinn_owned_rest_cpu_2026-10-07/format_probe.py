"""Bounded exact native tile-format-only diagnostic, no production edits."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from PySide6.QtWidgets import QApplication

ROOT=Path(sys.argv[1]).resolve();sys.path.insert(0,str(ROOT))
from spacr.qt.widgets import ambient
OUT=Path(__file__).parent;SOURCE=Path(ambient.__file__).read_text()
node=next(n for cls in ast.parse(SOURCE).body if isinstance(cls,ast.ClassDef) and cls.name=='_DataArtEngine' for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_paint_tissue_facets')
method='\n'.join(SOURCE.splitlines()[node.lineno-1:node.end_lineno]);modified=method.replace('                    inner.end()\n                    cells.append(', '                    inner.end()\n                    tile = tile.convertToFormat(QImage.Format_ARGB32_Premultiplied)\n                    cells.append(')
assert modified!=method
candidate=SOURCE.replace(method,modified);(OUT/'format_candidate.py').write_text(candidate)
spec=importlib.util.spec_from_file_location('spacr.qt.widgets._spinn_format',OUT/'format_candidate.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
app=QApplication([]);rows=[]
for label in ('dark','light'):
 background='#101418' if label=='dark' else '#f4f4f0'
 engines=[]
 for owner in (ambient,module):
  e=owner.make_engine('data_art_tissue_facets','spacr',background,seed=42,resolution=2,density=3,size=1,blur=0);e.set_max_pixels(3840*2160);e.set_gravity_radius(.5);e.set_pointer((.5,.5));e.shade(3840,2160);engines.append(e)
 samples=[[],[]];mismatches=[]
 for index in range(12):
  images=[None,None]
  for which in ([0,1] if index%2==0 else [1,0]):
   e=engines[which];e.advance(1/24);start=time.perf_counter();images[which]=e.shade(3840,2160);samples[which].append((time.perf_counter()-start)*1000)
  count=int(np.count_nonzero(np.frombuffer(images[0].constBits(),np.uint32)!=np.frombuffer(images[1].constBits(),np.uint32)));mismatches.append(count)
 rows.append({'background':background,'samples_ms':samples,'median_ms':[statistics.median(s) for s in samples],'differing_pixels_each_frame':mismatches})
result={'before_sha256':hashlib.sha256(SOURCE.encode()).hexdigest(),'candidate_sha256':hashlib.sha256(candidate.encode()).hexdigest(),'load_average':os.getloadavg(),'records':rows,'scope':'12 alternating native shaders per variant/background; no production or live FPS acceptance.'}
(OUT/'format_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
