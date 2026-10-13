import ast
import hashlib
import json
import statistics
import subprocess
import textwrap
import time
import types
from pathlib import Path
from PySide6.QtWidgets import QApplication
from spacr.qt.widgets import ambient

old_source=subprocess.check_output(['git','show','ae7269a08c136ac35d488d9f657f82cbfd33c49a:spacr/qt/widgets/ambient.py'],text=True)
module=ast.parse(old_source)
cls=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='AuroraEngine')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_paint_field')
lines=old_source.splitlines(keepends=True)
namespace=dict(ambient.__dict__)
exec(textwrap.dedent(''.join(lines[method.lineno-1:method.end_lineno])),namespace)
old_method=namespace['_paint_field']
QApplication.instance() or QApplication([])
rows=[]
for width,height,density,size,resolution in ((3840,2160,1,1,1),(3840,2160,3,3,2)):
    for mode in ('old','new','new','old'):
        engine=ambient.make_engine('aurora','spacr','#101418',seed=7,density=density,size=size,resolution=resolution)
        engine.set_max_pixels(width*height)
        if mode=='old': engine._paint_field=types.MethodType(old_method,engine)
        durations=[]
        for index in range(10):
            engine.set_time(9+index/12)
            started=time.perf_counter()
            frame=engine.shade(width,height)
            if index>=3: durations.append((time.perf_counter()-started)*1000)
            assert frame.width()==width and frame.height()==height
        rows.append({'size':[width,height],'density':density,'size_control':size,'resolution':resolution,'mode':mode,'median_ms':round(statistics.median(durations),2),'max_ms':round(max(durations),2),'durations_ms':[round(x,2) for x in durations]})
print(json.dumps({'source':ambient.__file__,'current_source_sha256':hashlib.sha256(Path(ambient.__file__).read_bytes()).hexdigest(),'old_method_git_sha':'ae7269a08c136ac35d488d9f657f82cbfd33c49a','rows':rows},indent=2))
