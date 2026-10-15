import json, sys
import PySide6
from PySide6.QtCore import QObject, Signal, Slot, QCoreApplication, Qt, QTimer, qVersion
try:
 import numpy as np
 nested = np.arange(4).reshape(2, 2)
 nested_kind = "numpy.ndarray"
except ImportError:
 nested = object()
 nested_kind = "object"
class Sender(QObject):
 ready = Signal(dict)
class Receiver(QObject):
 @Slot(dict)
 def sync(self, value):
  collect("QObject.AutoConnection", value)
 @Slot(dict)
 def queued(self, value):
  collect("QObject.QueuedConnection", value)
app=QCoreApplication([])
sender=Sender(); receiver=Receiver(); rows=[]
original={"path":"/scratch/probe.tif","labels":nested,"count":1}
def collect(route,value):
 rows.append({"route":route,"outer_dict_same":value is original,"nested_value_same":value.get("labels") is nested,"keys":sorted(value),"value_type":type(value).__name__})
def plain(value):collect("plain_callable.AutoConnection",value)
sender.ready.connect(plain)
sender.ready.connect(receiver.sync)
sender.ready.connect(receiver.queued,Qt.QueuedConnection)
sender.ready.emit(original)
QTimer.singleShot(100,app.quit)
app.exec()
assert len(rows)==3
print(json.dumps({"python":sys.version,"pyside":PySide6.__version__,"qt":qVersion(),"pyside_module":PySide6.__file__,"signal_signature":sender.metaObject().method(sender.metaObject().indexOfSignal("ready(QVariantMap)")).methodSignature().data().decode() if sender.metaObject().indexOfSignal("ready(QVariantMap)")>=0 else str(sender.ready),"nested_kind":nested_kind,"rows":rows},indent=2))
