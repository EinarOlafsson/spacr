import gc
import json
import sys
from PySide6 import __version__
from PySide6.QtCore import QObject
from PySide6.QtGui import QUndoCommand, QUndoStack

case = sys.argv[1]
print(json.dumps({'python':sys.version,'qt_binding':__version__,'case':case}),flush=True)
gc.disable()
for iteration in range(30):
    owner = QObject() if 'parent' in case else None
    if 'command_first' in case:
        entry = QUndoCommand('temporary value')
        ledger = QUndoStack(owner)
    else:
        ledger = QUndoStack(owner)
        entry = QUndoCommand('temporary value')
    ledger.self_reference = ledger
    if owner is not None:
        owner.self_reference = owner
    ledger.push(entry)
    del entry, ledger, owner
    gc.collect()
print('completed 30 collections',flush=True)
