"""Expose the real figure heading on a private native X11 display."""
import ctypes
import ctypes.util
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(os.environ['N655_SOURCE_ROOT'])
OUT = Path(os.environ['N655_SCRATCH'])
sys.path.insert(0, str(ROOT))
os.environ['SPACR_HOME'] = str(OUT / 'spacr-home')
os.environ['SPACR_LOG_DIR'] = str(OUT / 'spacr-logs')
os.environ['SPACR_NETWORK_CONFIG'] = str(OUT / 'spacr-network.json')
from PySide6.QtCore import QSettings, QTimer, Qt
from PySide6.QtGui import QAccessible, QColor, QPixmap
from PySide6.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget
QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(OUT / 'settings'))
QSettings.setPath(QSettings.NativeFormat, QSettings.UserScope, str(OUT / 'settings'))
from spacr.qt import i18n, preferences, theme
from spacr.qt.widgets.figure_grid_view import FigureGridView

app = QApplication([])
app.setApplicationName('spaCR N655 figure heading native acceptance')
preferences.set_ambient_animation('none')
preferences.set_preload_policy('on_demand')
preferences.set_theme_choice('dark')
preferences.set_strong_focus(False)
theme.apply_qpalette(app)
app.setStyleSheet(theme.stylesheet())
host = QWidget()
host.setWindowTitle('Isolated spaCR figure heading')
layout = QVBoxLayout(host)
before = QPushButton('Before heading')
grid = FigureGridView()
after = QPushButton('After figures')
for widget in (before, grid, after):
    layout.addWidget(widget)
pixmap = QPixmap(120, 90)
pixmap.fill(QColor('#465767'))
grid.set_figures([pixmap], ['Plate A'], [('Run 2026-10-15 native acceptance', 0, 1)])
i18n.retranslate_widget_tree(host, 'en')
host.resize(560, 420)
host.show()
QAccessible.setActive(True)
events = []

def on_focus(old, new):
    events.append({'object_name': new.objectName() if new else None,
                   'class': type(new).__name__ if new else None,
                   'accessible_name': new.accessibleName() if new else None})
    (OUT / 'app-focus-events.json').write_text(json.dumps(events, indent=2) + '\n')

app.focusChanged.connect(on_focus)

def ready():
    x11 = ctypes.CDLL(ctypes.util.find_library('X11'))
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XSetInputFocus.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    x11.XFlush.argtypes = [ctypes.c_void_p]
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    display = x11.XOpenDisplay(None)
    assert display
    x11.XSetInputFocus(display, int(host.winId()), 2, 0)
    x11.XFlush(display)
    x11.XCloseDisplay(display)
    before.setFocus(Qt.TabFocusReason)
    header = grid._headers[0]
    assert header.isVisible()
    assert header.accessibleName() == header.text() == 'Run 2026-10-15 native acceptance'
    hashes = {}
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if name.startswith('spacr') and path:
            path = Path(path)
            if path.is_relative_to(ROOT) and path.is_file():
                hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / 'process-source.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print('READY ' + json.dumps({'pid': os.getpid(), 'heading': header.text(),
                                 'platform': app.platformName(), 'source_files': len(hashes)}), flush=True)

QTimer.singleShot(900, ready)
QTimer.singleShot(25000, app.quit)
app.exec()
host.close()
