"""Qualify native stroked-path filling before considering geometry reuse."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PySide6.QtGui import QPainterPathStroker
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from spacr.qt.widgets import ambient

app = QApplication([])
rows=[]
source=Path(ambient.__file__).read_bytes()
for size in [2.5, 1.0]:
    engines=[ambient.make_engine('data_art_fungal_growth','spacr','#101418',seed=42,resolution=2,density=3,size=size,blur=0) for _ in range(2)]
    for e in engines:
        e.set_max_pixels(3840*2160)
        e.set_time(95)
        e._fungal_raster_failed=True
    def fill_strokes(painter, paths):
        colors=engines[1].paint_colors
        for (hue, stroke, alpha), path in paths.items():
            stroker=QPainterPathStroker()
            stroker.setWidth(stroke)
            stroker.setCapStyle(Qt.RoundCap)
            stroker.setJoinStyle(Qt.RoundJoin)
            painter.fillPath(stroker.createStroke(path), ambient._with_alpha(colors[hue],alpha))
    engines[1]._paint_fungal_paths=fill_strokes
    frames=[e.shade(3840,2160) for e in engines]
    planes=[np.frombuffer(frame.constBits(),dtype=np.uint32) for frame in frames]
    differing=np.flatnonzero(planes[0]!=planes[1])
    row={'size':size,'clock':95,'dimensions':[[f.width(),f.height()] for f in frames],'different_pixels':int(len(differing)),'first_offsets':differing[:10].tolist(),'hashes':[hashlib.sha256(f.constBits()).hexdigest() for f in frames]}
    rows.append(row);print(json.dumps(row),flush=True)
    if len(differing):break
receipt={'ambient_sha256':hashlib.sha256(source).hexdigest(),'rows':rows,'accepted':all(r['different_pixels']==0 for r in rows),'scope':'Scratch method override only; no application/source/cache/budget edits. Native full-raster qualification only, not FPS acceptance.'}
Path(__file__).with_name('receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
