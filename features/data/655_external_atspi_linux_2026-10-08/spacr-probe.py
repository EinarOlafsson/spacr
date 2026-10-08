import os
import sys
from pathlib import Path

root = Path(os.environ['SPACR_N655_OUT'])
root.mkdir(parents=True, exist_ok=True)
os.environ['SPACR_HOME'] = str(root / 'private-home')
os.environ['SPACR_LOG_DIR'] = str(root / 'private-logs')
os.environ['SPACR_NETWORK_CONFIG'] = str(root / 'private-network.json')
from PySide6.QtCore import QCoreApplication, QSettings, QTimer
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QAccessible
from spacr.qt.app import MainWindow
from spacr.qt.preferences import PreferencesDialog

QCoreApplication.setOrganizationName('spacr-n655-audit')
QCoreApplication.setApplicationName('spacr-n655-native-audit')
QSettings.setDefaultFormat(QSettings.IniFormat)
QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(root))
QSettings.setPath(QSettings.NativeFormat, QSettings.UserScope, str(root))
app = QApplication.instance() or QApplication([])
app.setApplicationName('spacr-n655-native-audit')
QAccessible.setActive(True)
window = MainWindow(initial_app='__home__')
window.show()
dialog = PreferencesDialog(window)
dialog.show()
QAccessible.setRootObject(app)
print('INTERNAL', QAccessible.isActive(), QAccessible.queryAccessibleInterface(window).childCount(), QAccessible.queryAccessibleInterface(dialog).childCount(), flush=True)
print('READY', os.getpid(), window.windowTitle(), dialog.windowTitle(), flush=True)
QTimer.singleShot(1000, lambda: print('INTERNAL_EVENT', QAccessible.queryAccessibleInterface(app).childCount(), flush=True))
QTimer.singleShot(18000, app.quit)
status = app.exec()
dialog.close()
window.close()
print('EXIT', status, flush=True)
sys.exit(status)
