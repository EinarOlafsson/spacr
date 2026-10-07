import hashlib
import json
import types
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import QApplication
from spacr.qt.widgets import ambient

prototype=Path('/mnt/wd4tb/scratch/aurora-native-perf-20261007/prototype_clip.py').read_text().split('QApplication.instance() or QApplication([])')[0]
namespace={}
exec(prototype,namespace)
replacement=namespace['replacement']
QApplication.instance() or QApplication([])
rows=[]
for width,height,density,size,resolution in ((1280,720,1,1,1),(3840,2160,1,1,1),(3840,2160,3,3,2)):
    for palette,background in (('spacr','#101418'),('borealis','#101418'),('mono','#f3f2ee')):
        original=ambient.make_engine('aurora',palette,background,seed=7,density=density,size=size,resolution=resolution)
        candidate=ambient.make_engine('aurora',palette,background,seed=7,density=density,size=size,resolution=resolution)
        original.set_max_pixels(width*height)
        candidate.set_max_pixels(width*height)
        candidate._paint_field=types.MethodType(replacement,candidate)
        for clock in (0.0,3.33,9.5,60.2):
            original.set_time(clock)
            candidate.set_time(clock)
            a=original.shade(width,height)
            b=candidate.shade(width,height)
            assert a.width()==b.width()==width and a.height()==b.height()==height
            left=np.frombuffer(a.bits(),dtype=np.uint8)
            right=np.frombuffer(b.bits(),dtype=np.uint8)
            changed=np.any(left.reshape(-1,4)!=right.reshape(-1,4),axis=1)
            rows.append({'size':[width,height],'density':density,'element_size':size,'resolution':resolution,'palette':palette,'background':background,'clock':clock,'changed_pixels':int(changed.sum()),'max_byte_diff':int(np.max(np.abs(left.astype(np.int16)-right.astype(np.int16))))})
print(json.dumps({'source':ambient.__file__,'source_sha256':hashlib.sha256(Path(ambient.__file__).read_bytes()).hexdigest(),'rows':rows},indent=2))
