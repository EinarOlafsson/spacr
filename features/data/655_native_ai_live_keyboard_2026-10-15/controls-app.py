"""Observe real AI/Live controls without directly activating or focusing them."""
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
from PySide6.QtCore import QEvent, QObject, QSettings, QTimer, Qt
from PySide6.QtGui import QAccessible
from PySide6.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget
QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(OUT / 'settings'))
QSettings.setPath(QSettings.NativeFormat, QSettings.UserScope, str(OUT / 'settings'))
from spacr.qt import i18n, preferences, theme
from spacr.qt.widgets.ai_toggle_label import AiToggleLabel

app = QApplication([])
app.setApplicationName('spaCR N655 native AI Live acceptance')
preferences.set_ambient_animation('none')
preferences.set_preload_policy('on_demand')
preferences.set_theme_choice('dark')
preferences.set_strong_focus(False)
theme.apply_qpalette(app)
app.setStyleSheet(theme.stylesheet())
host = QWidget()
host.setWindowTitle('Isolated spaCR AI and Live controls')
layout = QVBoxLayout(host)
before = QPushButton('Before controls')
ai = AiToggleLabel(text='AI')
live = AiToggleLabel(text='Live', tooltip='Preview updates for this isolated control.')
after = QPushButton('After controls')
controls = {'AI': ai, 'Live': live}
for widget in (before, ai, live, after):
    layout.addWidget(widget)
observations = {'signals': [], 'key_events': [], 'focus_events': []}

def dump():
    observations['checked'] = {name: control.isChecked() for name, control in controls.items()}
    target = OUT / 'app-observations.json'
    temporary = OUT / 'app-observations.tmp'
    temporary.write_text(json.dumps(observations, indent=2) + '\n')
    temporary.replace(target)

def changed(name, value):
    observations['signals'].append({'control': name, 'checked': value})
    dump()

class KeyObserver(QObject):
    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            name = next(name for name, control in controls.items() if control is obj)
            observations['key_events'].append({'control': name, 'key': event.key(),
                'modifiers': event.modifiers().value, 'auto_repeat': event.isAutoRepeat()})
            dump()
        return False

observer = KeyObserver(host)
for name, control in controls.items():
    control.installEventFilter(observer)
    control.toggled.connect(lambda checked, name=name: changed(name, checked))

def focused(old, new):
    name = next((name for name, control in controls.items() if control is new), None)
    observations['focus_events'].append({'control': name, 'class': type(new).__name__ if new else None})
    dump()

app.focusChanged.connect(focused)
i18n.retranslate_widget_tree(host, 'en')
host.resize(460, 280)
host.show()
QAccessible.setActive(True)

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
    assert all(control.isVisible() for control in controls.values())
    assert not any(control.isChecked() for control in controls.values())
    dump()
    hashes = {}
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if name.startswith('spacr') and path:
            path = Path(path)
            if path.is_relative_to(ROOT) and path.is_file():
                hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / 'process-source.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print('READY ' + json.dumps({'pid': os.getpid(), 'platform': app.platformName(),
                                'source_files': len(hashes), 'initial_states': observations['checked']}), flush=True)

QTimer.singleShot(900, ready)
QTimer.singleShot(25000, app.quit)
app.exec()
host.close()
