from PySide6.QtCore import QEvent, QObject, QTimer, Signal, qInstallMessageHandler
from PySide6.QtWidgets import QApplication, QWidget
import PySide6

messages = []
def handler(_mode, _context, message):
    if 'addMetaMethod' in (message or ''):
        messages.append(message)
qInstallMessageHandler(handler)
app = QApplication([])

class TouchesTheChild(QObject):
    def __init__(self):
        super().__init__()
        self.held = []
    def eventFilter(self, watched, event):
        if event.type() == QEvent.ChildAdded:
            self.held.append(event.child())
        return False

class IgnoresTheChild(QObject):
    def eventFilter(self, watched, event):
        return False

class Ticker(QWidget):
    poked = Signal(int)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.delivered = []
        self.shown = 0
        self.poked.connect(self.delivered.append)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)
    def showEvent(self, event):
        self.shown += 1
        super().showEvent(event)
    def _on_tick(self):
        pass

keep = []
for label, watcher in [('plain', None), ('retains_child', TouchesTheChild()), ('ignores_child', IgnoresTheChild())]:
    host = QWidget()
    keep.extend([host, watcher])
    if watcher is not None:
        host.installEventFilter(watcher)
    before = len(messages)
    child = Ticker(host)
    keep.append(child)
    child.poked.emit(7)
    host.show()
    child.show()
    app.processEvents()
    print(label, 'meta=', child.metaObject().className(), 'static=', Ticker.staticMetaObject.className(), 'signal=', child.metaObject().indexOfSignal('poked(int)'), 'found=', len(host.findChildren(Ticker)), 'delivered=', child.delivered, 'shown=', child.shown, 'warnings=', len(messages)-before, 'held=', len(watcher.held) if isinstance(watcher, TouchesTheChild) else 0)
print('version=', PySide6.__version__)
