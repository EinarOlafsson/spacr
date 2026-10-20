import atexit
import json
import os
from pathlib import Path
import socket
import sys

folder = Path(__file__).parent
events = []
original_connect = socket.socket.connect
original_connect_ex = socket.socket.connect_ex
original_sendto = socket.socket.sendto
original_getaddrinfo = socket.getaddrinfo
def refuse(kind, value):
    events.append({'kind': kind, 'address': repr(value)})
    raise RuntimeError('Network prohibited in zoom-only CI baseline')
def connect(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        refuse('connect', address)
    return original_connect(self, address)
def connect_ex(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        refuse('connect_ex', address)
    return original_connect_ex(self, address)
def sendto(self, data, *args):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        refuse('sendto', args)
    return original_sendto(self, data, *args)
def getaddrinfo(host, *args, **kwargs):
    refuse('getaddrinfo', host)
socket.socket.connect = connect
socket.socket.connect_ex = connect_ex
socket.socket.sendto = sendto
socket.getaddrinfo = getaddrinfo
atexit.register(lambda: (folder / 'diagnostic-network.json').write_text(json.dumps(events, indent=2) + '\n'))
(folder / 'diagnostic-startup.json').write_text(json.dumps({'pid': os.getpid(), 'affinity': sorted(os.sched_getaffinity(0)), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'qt_platform': os.environ.get('QT_QPA_PLATFORM')}, indent=2) + '\n')

import importlib.util
import tempfile
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication
from spacr.qt import preferences as prefs
from spacr.qt import theme
from spacr.qt.widgets.ai_toggle_label import AiToggleLabel

spec = importlib.util.spec_from_file_location('zoom_original_helpers', Path.cwd() / 'tests/qt/test_zoom_reaches_text.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
results = {}
with tempfile.TemporaryDirectory(prefix='zoom-diagnostic-', dir=folder) as temporary:
    os.environ['XDG_CONFIG_HOME'] = temporary
    store = QSettings(str(Path(temporary) / 'prefs.ini'), QSettings.IniFormat)
    previous_settings = prefs._settings
    prefs._settings = lambda: store
    try:
        app = QApplication.instance() or QApplication([])
        theme.load_widget_qss_registrars()
        theme.apply_qpalette(app)
        app.setStyleSheet(theme.stylesheet())
        for text in ['Live', 'AI']:
            m._at(m.SMALL)
            toggle = AiToggleLabel(text=text)
            toggle.show()
            QApplication.processEvents()
            rows = []
            for scale in [m.SMALL, m.LARGE]:
                if scale != m.SMALL:
                    m._at(scale)
                    QApplication.processEvents()
                for phase in ['normal', 'cleared_focus']:
                    if phase == 'cleared_focus':
                        toggle.clearFocus()
                        QApplication.processEvents()
                    image = toggle.grab().toImage()
                    image.save(str(folder / (text + '-' + str(scale) + '-' + phase + '.png')))
                    rows.append({'scale': scale, 'phase': phase, 'hasFocus': toggle.hasFocus(),
                                 'font_metrics_height': m._text_height(toggle), 'ink_rows': m._ink_rows(toggle),
                                 'widget_size': [toggle.width(), toggle.height()],
                                 'size_hint': [toggle.sizeHint().width(), toggle.sizeHint().height()],
                                 'font_pixel_size': toggle.font().pixelSize(),
                                 'displayed_text': toggle.displayed_text()})
                toggle.setFocus(Qt.OtherFocusReason)
                QApplication.processEvents()
            results[text] = rows
            toggle.close()
            toggle.deleteLater()
            QApplication.processEvents()
    finally:
        prefs._settings = previous_settings
(folder / 'diagnostic.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results, indent=2), flush=True)
