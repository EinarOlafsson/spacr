"""Observe a real help grip; all resizing comes from the external AT-SPI client."""
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
from PySide6.QtCore import QSettings, QTimer
from PySide6.QtGui import QAccessible
from PySide6.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget
QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(OUT / 'settings'))
QSettings.setPath(QSettings.NativeFormat, QSettings.UserScope, str(OUT / 'settings'))
from spacr.qt import i18n, preferences, theme
from spacr.qt.widgets.hint_bar import HintBar
app = QApplication([])
app.setApplicationName('spaCR N655 native help grip value acceptance')
preferences.set_ambient_animation('none')
preferences.set_preload_policy('on_demand')
preferences.set_theme_choice('dark')
theme.apply_qpalette(app)
app.setStyleSheet(theme.stylesheet())
host = QWidget()
host.setWindowTitle('Isolated spaCR help height control')
layout = QVBoxLayout(host)
page = QWidget()
page_layout = QVBoxLayout(page)
page_layout.addWidget(QPushButton('Before help'))
page_layout.addStretch()
bar = HintBar('Help text for the isolated resize control.')
for widget in (page, bar, QPushButton('After help')):
    layout.addWidget(widget)
observations = {'commits': []}
def dump():
    observations.update(height=bar.height(), minimum=bar._minimum_help_height(),
        maximum=bar._maximum_help_height(), manual_height=bar._manual_height,
        keyboard_step=bar.fontMetrics().lineSpacing())
    path = OUT / 'app-observations.tmp'
    path.write_text(json.dumps(observations, indent=2) + '\n')
    path.replace(OUT / 'app-observations.json')
def committed(height):
    observations['commits'].append(height)
    dump()
bar.helpHeightCommitted.connect(committed)
i18n.retranslate_widget_tree(host, 'en')
host.resize(460, 700)
host.show()
QAccessible.setActive(True)
def ready():
    dump()
    hashes = {}
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if name.startswith('spacr') and path:
            path = Path(path)
            if path.is_relative_to(ROOT) and path.is_file():
                hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / 'process-source.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print('READY ' + json.dumps({'pid':os.getpid(),'platform':app.platformName(),
        'source_files':len(hashes),'initial':observations}), flush=True)
QTimer.singleShot(900, ready)
QTimer.singleShot(25000, app.quit)
app.exec()
host.close()
