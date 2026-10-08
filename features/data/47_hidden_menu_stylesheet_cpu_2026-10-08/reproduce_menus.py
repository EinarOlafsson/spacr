import hashlib
import json
import time
from pathlib import Path
from PySide6.QtWidgets import QApplication,QMenu,QLabel
from spacr.qt import theme
app=QApplication([])
first=theme.stylesheet('dark')
second=theme.stylesheet('light')
roots=[]
for i in range(150):
    menu=QMenu()
    for j in range(12): menu.addAction(f'Action {j}')
    roots.append(menu)
app.processEvents()
rows=[]
import inspect
source=inspect.getsource(theme.apply_stylesheet_per_window)
guard='        if (_sheets_itself_before_it_shows(window)\n                and not window.isVisible()):\n            continue\n'
if guard in source:
    assert source.count(guard)==1
    source=source.replace(guard,'')
ns=dict(vars(theme))
exec(compile(source,'frozen-before-helper','exec'),ns)
original=ns['apply_stylesheet_per_window']
for variant in ('before','after','after','before'):
    theme._forget_window_stylesheets(app)
    if variant=='after':
        import inspect
        source=inspect.getsource(original)
        source=source.replace('for window in list(app.topLevelWidgets()):','for window in list(app.topLevelWidgets()):\n        if not window.isVisible() and _sheets_itself_before_it_shows(window):\n            continue')
        ns=dict(vars(theme))
        exec(compile(source,'scratch-menu-candidate','exec'),ns)
        fn=ns['apply_stylesheet_per_window']
    else: fn=original
    elapsed=[]
    for index in range(3):
        started=time.perf_counter()
        count=fn(app,first if index%2==0 else second)
        elapsed.append({'seconds':time.perf_counter()-started,'count':count})
    rows.append({'variant':variant,'timings':elapsed,'sheeted_hidden':sum(bool(m.styleSheet()) for m in roots)})
    root=roots[0]
    root.aboutToShow.emit()
    rows[-1]['latest_on_open']=root.styleSheet()==first
    root.hide()
for root in roots: root.deleteLater()
from PySide6.QtCore import QEvent
QApplication.sendPostedEvents(None,QEvent.DeferredDelete)
Path('/mnt/wd4tb/scratch/serial-field-fade-20261008/menus.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
