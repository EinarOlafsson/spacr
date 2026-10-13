import json
import sys
from PySide6.QtCore import QEvent, QObject, Signal
from PySide6.QtWidgets import QApplication, QWidget
from shiboken6 import getCppPointer, wrapInstance

app = QApplication.instance() or QApplication([])
keep = []

class RetainsArrivingChild(QObject):
    def __init__(self):
        super().__init__()
        self.held = []

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.ChildAdded:
            self.held.append(event.child())
        return False


def init(self, parent=None):
    QWidget.__init__(self, parent)
    self.delivered = []
    self.poked.connect(self.delivered.append)

rows = []
for name, poison in [('Child', False), ('QualifiedChild', False), ('PoisonedChild', True)]:
    Child = type(name, (QWidget,), {'poked': Signal(int), '__init__': init, '__module__': '__main__'})
    host = QWidget()
    keep.append(host)
    if poison:
        watcher = RetainsArrivingChild()
        keep.append(watcher)
        host.installEventFilter(watcher)
    child = Child(host)
    keep.append(child)
    meta = child.metaObject()
    declared = type(child).staticMetaObject
    inherits = meta.inherits(declared)
    wrapper_identity = wrapInstance(getCppPointer(child)[0], type(child)) is child
    signal = meta.indexOfSignal('poked(int)')
    found = len(host.findChildren(type(child)))
    child.poked.emit(7)
    row = {'constructed_class': name, 'poisoned': poison, 'instance_meta_name': meta.className(), 'class_static_meta_name': declared.className(), 'old_bare_name_guard': meta.className() == 'Child', 'metaobject_inheritance_guard': inherits, 'binding_manager_wrapper_identity': wrapper_identity, 'signal_index': signal, 'found_children': found, 'delivered': child.delivered}
    rows.append(row)
    print(json.dumps(row), flush=True)
    if poison:
        assert not wrapper_identity and not inherits and signal == -1 and found == 0 and child.delivered == []
    else:
        assert wrapper_identity and inherits and signal != -1 and found == 1 and child.delivered == [7]
    if name == 'QualifiedChild':
        assert meta.className() == name and not row['old_bare_name_guard']
print('Healthy renamed child identity accepted; original retained-child wrapper poisoning rejected.', flush=True)
