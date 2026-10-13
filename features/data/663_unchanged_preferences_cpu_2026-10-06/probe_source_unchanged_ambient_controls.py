import faulthandler
import ast
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time

faulthandler.enable(all_threads=True)
from PySide6.QtCore import QCoreApplication, QEvent, QSettings
from PySide6.QtWidgets import QApplication
from spacr.qt import preferences as prefs
from spacr.qt.widgets import ambient

source = Path(ambient.__file__)
assert source.parent.parent.parent.parent == Path(os.environ['SPACR_SOURCE_ROOT'])
source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
app = QApplication([])
store = QSettings(os.environ['PROBE_STORE'], QSettings.IniFormat)
prefs._settings = lambda: store
prefs.set_ambient_animation('data_art_fungal_growth')
prefs.set_ambient_enabled(True)
prefs.set_ambient_palette('spacr')
prefs.set_ambient_resolution(2.0)
prefs.set_ambient_density(3.0)
prefs.set_ambient_size(2.5)
widget = ambient.AmbientWidget(theme='data_art_fungal_growth', palette='spacr',
                              resolution=2.0, density=3.0, size=2.5, seed=17)
widget.resize(3840, 2160)
widget.show()
app.processEvents()
widget.set_time(95.0)
gui_thread = threading.get_ident()
calls = []
original_shade = widget.engine.shade

def shade(*args):
    if threading.get_ident() == gui_thread:
        calls.append(time.perf_counter())
    return original_shade(*args)

widget.engine.shade = shade
names = ('set_blur','set_resolution','set_density','set_size_scale',
         'set_direction','set_speed')
current = {name:getattr(ambient.AmbientWidget,name) for name in names}
old_source = subprocess.check_output(
    ['git','show','711999b77d:spacr/qt/widgets/ambient.py'], text=True)
tree = ast.parse(old_source)
old_class = next(x for x in tree.body if isinstance(x,ast.ClassDef)
                 and x.name == 'AmbientWidget')
methods = [x for x in old_class.body if isinstance(x,ast.FunctionDef)
           and x.name in names]
assert len(methods) == 6
originals = {}
exec(compile(ast.Module(body=methods,type_ignores=[]),
             'frozen711-original-setters','exec'),ambient.__dict__,originals)

results = []
for mode in ('before','candidate','candidate','before'):
    for name in names:
        setattr(ambient.AmbientWidget,name,
                current[name] if mode=='candidate' else originals[name])
    rows=[]
    for _ in range(6):
        before=len(calls)
        started=time.perf_counter()
        prefs.apply_ambient_preferences(app)
        rows.append({'ms':1000*(time.perf_counter()-started),'gui_shades':len(calls)-before})
        app.processEvents()
    results.append({'mode':mode,'median_ms':statistics.median(x['ms'] for x in rows),'rows':rows})
for name,original in current.items():
    setattr(ambient.AmbientWidget,name,original)
widget.close()
assert not widget.shading_thread_alive()
widget.deleteLater()
QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
assert not app.allWidgets()
assert source_sha == hashlib.sha256(source.read_bytes()).hexdigest()
print(json.dumps({'source':str(source),'source_sha256':source_sha,
                  'before_source_sha256':hashlib.sha256(old_source.encode()).hexdigest(),
                  'comparison':'Exact six frozen711 setter methods versus current staged six methods; other code is current in both arms.',
                  'results':results,'closed':True},indent=2),flush=True)
