import gc
import json
import os
import sys
from PySide6 import __version__
from PySide6.QtCore import QObject
from PySide6.QtGui import QUndoCommand, QUndoStack

print(json.dumps({'python':sys.version,'qt_binding':__version__,'case':sys.argv[1]}),flush=True)
gc.disable()
parent = QObject() if sys.argv[1] == 'parented' else None
for iteration in range(30):
    action = QUndoCommand('ownership probe')
    history = QUndoStack(parent)
    history.self_reference = history
    history.push(action)
    if sys.argv[1] == 'cleared':
        history.clear()
    del action, history
    gc.collect()
print('completed 30 collections',flush=True)
