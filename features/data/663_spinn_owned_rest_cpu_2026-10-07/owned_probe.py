"""Exact native spinn owned/rest-buffer candidate, before production adoption."""
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
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QApplication

ROOT=Path(sys.argv[1]).resolve();sys.path.insert(0,str(ROOT))
from spacr.qt import preferences
from spacr.qt.widgets import ambient
OUT=Path(__file__).parent;SOURCE=Path(ambient.__file__).read_text()
replacement='''    def _shade(self, width: int, height: int) -> QImage:
        """Publish owned native points or a copy-on-write resting paper frame.

        Paper has no autonomous motion at zero pointer influence. Reusing its
        exact resting image does not skip clock/input acknowledgement: the
        rotation timestamp follows each shade. Active paper is always drawn
        into a fresh image, so previously published frames remain untouched.
        The resting marker adds no raster storage and existing material-cache
        invalidations discard it with palette, size, density or resolution.
        """
        if self.family not in ("point_atlas", "impulse_lens", "genetic_advection",
                               "tissue_facets"):
            return super()._shade(width, height)
        bw, bh = self.buffer_size(width, height)
        previous = self._buffer
        if previous is None or previous.width() != bw or previous.height() != bh:
            self._material_cache.clear()
        if self.family == "tissue_facets":
            resting = self.pointer is None or self.gravity_radius == 0.0
            key = ("tissue_resting", bw, bh, self.size, self.density)
            if resting and key in self._material_cache:
                rotation_key = ("tissue_rotation", bw, bh, self.size, self.density)
                _, angles = self._material_cache[rotation_key]
                self._material_cache[rotation_key] = (self.time, angles)
                return self._soften(QImage(previous), width, height)
            self._material_cache.pop(key, None)
            image = QImage(bw, bh, QImage.Format_RGB32)
            inner = QPainter(image)
            try:
                inner.fillRect(image.rect(), self.identity)
                inner.setCompositionMode(self.mode)
                inner.setPen(Qt.NoPen)
                self._paint_field(inner, bw, bh)
            finally:
                inner.end()
            self._buffer = image
            if resting:
                self._material_cache[key] = True
            return self._soften(QImage(image), width, height)
        image = getattr(self, f"_frame_{self.family}")(bw, bh)
        self._buffer = image
        return self._soften(image, width, height)
'''
tree=ast.parse(SOURCE)
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='_DataArtEngine')
node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_shade')
method='\n'.join(SOURCE.splitlines()[node.lineno-1:node.end_lineno]);candidate=SOURCE.replace(method,replacement.rstrip())
node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='shade')
method='\n'.join(SOURCE.splitlines()[node.lineno-1:node.end_lineno])
modified=method.replace('"genetic_advection"):', '"genetic_advection",\n                               "tissue_facets"):')
candidate=candidate.replace(method,modified)
(OUT/'owned_candidate.py').write_text(candidate)
spec=importlib.util.spec_from_file_location('spacr.qt.widgets._spinn_owned',OUT/'owned_candidate.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
app=QApplication([]);preferences._ambient_custom_colors=lambda:('#fbe631','#21abf8');rows=[]
for label in ('dark','light'):
 for palette in ('spacr','random','custom'):
  background='#101418' if label=='dark' else '#f4f4f0';engines=[]
  for owner in (ambient,module):
   e=owner.make_engine('data_art_tissue_facets',palette,background,seed=42,resolution=2,density=3,size=1,blur=0);e.set_max_pixels(3840*2160);engines.append(e)
  records=[]
  for state in ('idle','idle_later','entry','moving','leave','resize','density','size','palette','background'):
   images=[]
   for e in engines:
    if state=='idle_later':e.advance(100)
    elif state=='entry':e.set_gravity_radius(.5);e.set_pointer((.5,.5));e.advance(1/24)
    elif state=='moving':e.set_pointer((.3,.7));e.advance(1/24)
    elif state=='leave':e.set_pointer(None)
    elif state=='resize':e.set_resolution(.75)
    elif state=='density':e.set_density(2)
    elif state=='size':e.set_size(2)
    elif state=='palette':e.set_colors(('#fc7c19','#18cbfd'))
    elif state=='background':e.set_background(QColor('#eff1f4'))
    images.append(e.shade(3840,2160))
   a,b=images;assert a.size()==b.size();pa=np.frombuffer(a.constBits(),np.uint32);pb=np.frombuffer(b.constBits(),np.uint32);count=int(np.count_nonzero(pa!=pb));assert count==0,(label,palette,state,count)
   assert np.all(pb>>24==255)
   records.append({'state':state,'size':[b.width(),b.height()],'differing_pixels':count,'sha256':hashlib.sha256(b.constBits()).hexdigest()})
  rows.append({'background':background,'palette':palette,'parity':records})
  del engines,images,a,b,pa,pb
samples={}
for active in (False,True):
 engines=[]
 for owner in (ambient,module):
  e=owner.make_engine('data_art_tissue_facets','spacr','#101418',seed=42,resolution=2,density=3,size=1,blur=0);e.set_max_pixels(3840*2160)
  if active:e.set_gravity_radius(.5);e.set_pointer((.5,.5))
  e.shade(3840,2160);engines.append(e)
 values=[[],[]]
 for index in range(12):
  images=[None,None]
  for which in ([0,1] if index%2==0 else [1,0]):
   e=engines[which];e.advance(1/24);start=time.perf_counter();images[which]=e.shade(3840,2160);values[which].append((time.perf_counter()-start)*1000)
  assert bytes(images[0].constBits())==bytes(images[1].constBits())
 samples[str(active)]={'samples_ms':values,'median_ms':[statistics.median(v) for v in values]}
result={'before_sha256':hashlib.sha256(SOURCE.encode()).hexdigest(),'candidate_sha256':hashlib.sha256(candidate.encode()).hexdigest(),'load_average':os.getloadavg(),'records':rows,'timings':samples,'scope':'Source-bound direct shaders and parity only; no live/hard24FPS acceptance.'}
(OUT/'owned_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
