"""Read-only native popup flags, active modality and parent teardown checks."""
import hashlib
import json
from pathlib import Path
import sys

from PySide6.QtCore import QCoreApplication, QEvent, QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QWidget
import shiboken6

repo = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repo))
from spacr.qt import dialogs
from spacr.qt.widgets import glass
from spacr.qt.widgets.availability_panel import AvailabilityPanel

paths = ['spacr/qt/dialogs.py', 'spacr/qt/widgets/glass.py',
         'spacr/qt/widgets/availability_panel.py']
hashes = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in paths}
app = QApplication([])
records = []
for treatment in ('detach', 'glass'):
    parent = QWidget()
    parent.show()
    popup = QDialog(parent)
    if treatment == 'detach':
        dialogs.detach_from_window_manager(popup)
    else:
        assert glass.make_frameless(popup)
    seen = []

    def accept():
        seen.append(app.activeModalWidget() is popup)
        assert popup.parentWidget() is parent
        assert popup.windowFlags() & Qt.WindowStaysOnTopHint
        popup.accept()

    QTimer.singleShot(20, accept)
    assert popup.exec() == QDialog.Accepted
    assert seen == [True]
    popup.setWindowModality(Qt.NonModal)
    popup.show()
    app.processEvents()
    assert app.activeModalWidget() is None
    assert popup.windowHandle().transientParent() is parent.windowHandle()
    assert not parent.windowFlags() & Qt.WindowStaysOnTopHint
    parent.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert not shiboken6.isValid(parent)
    assert not shiboken6.isValid(popup)
    records.append({'treatment': treatment, 'exec_accept': True,
                    'active_modal_during_exec': True, 'modeless_after': True,
                    'parent_destruction_owns_popup': True})
parent = QWidget()
parent.show()
panel = AvailabilityPanel()
for _ in range(2):
    panel.open_for(parent, [{'title': 'CPU diagnostic', 'reason': 'not installed',
                            'url': '', 'offer': None}])
    app.processEvents()
    assert panel.windowFlags() & Qt.WindowStaysOnTopHint
    assert panel.windowHandle().flags() & Qt.WindowStaysOnTopHint
    assert panel.parentWidget() is None
    assert app.activeModalWidget() is None
    assert panel.isVisible() and panel._filtering
    QTest.keyClick(panel, Qt.Key_Escape)
    assert not panel.isVisible() and not panel._filtering
panel.deleteLater()
parent.deleteLater()
QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
assert not shiboken6.isValid(panel)
assert not shiboken6.isValid(parent)
assert hashes == {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in paths}
receipt = {'source_sha256': hashes, 'records': records,
           'availability_reopens': 2, 'availability_escape_unwatches_app': True,
           'native_desktop_stacking_acceptance': False,
           'scope': 'Actual offscreen native flags/exec/ownership; no desktop WM z-order proof'}
(Path(__file__).parent / 'popup_contracts.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
