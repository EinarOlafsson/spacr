"""Run PYSIDE-3452's weakref lookup sequence using the installed QObject."""
import gc
import json
import weakref

from PySide6.QtCore import QObject
from shiboken6 import Shiboken

obj = QObject()
pointer = Shiboken.getCppPointer(obj)[0]
original = id(obj)
seen = {}


def callback(_ref):
    wrapped = Shiboken.wrapInstance(pointer, QObject)
    seen["id"] = id(wrapped)
    print(json.dumps({"event": "callback", "same_wrapper": seen["id"] == original,
                      "valid": Shiboken.isValid(wrapped)}), flush=True)


ref = weakref.ref(obj, callback)
print(json.dumps({"event": "before", "pointer": hex(pointer), "id": original}), flush=True)
del obj
gc.collect()
print(json.dumps({"event": "after", "weakref_dead": ref() is None,
                  "same_wrapper": seen.get("id") == original}), flush=True)
