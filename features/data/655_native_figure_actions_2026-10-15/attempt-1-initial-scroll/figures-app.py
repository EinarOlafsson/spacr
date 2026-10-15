"""Observe isolated real clear/fold controls driven only by external AT-SPI."""
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
os.environ['SPACR_NO_NETWORK'] = '1'
from PySide6.QtCore import QSettings, QTimer
from PySide6.QtGui import QAccessible, QColor, QImage, QPixmap
from PySide6.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget
QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(OUT / 'settings'))
QSettings.setPath(QSettings.NativeFormat, QSettings.UserScope, str(OUT / 'settings'))
from spacr.qt import i18n, preferences, theme
from spacr.qt.widgets.figure_queue import FigureQueue
from spacr.qt.widgets.figure_grid_view import FigureGridView
from matplotlib.figure import Figure
app = QApplication([])
app.setApplicationName('spaCR N655 isolated figure accessibility')
preferences.set_ambient_animation('none')
preferences.set_preload_policy('on_demand')
preferences.set_theme_choice('dark')
theme.apply_qpalette(app)
app.setStyleSheet(theme.stylesheet())
observations = {'clear_clicks': 0, 'fold_calls': 0}
class ObservedGrid(FigureGridView):
    def toggle_section(self, header):
        observations['fold_calls'] += 1
        return super().toggle_section(header)
host = QWidget()
host.setWindowTitle('Isolated Clear figures and section actions')
layout = QVBoxLayout(host)
queue = FigureQueue()
queue.setMinimumHeight(240)
image = QImage(80, 60, QImage.Format_RGB32)
image.fill(QColor('#465767'))
fixture = OUT / 'inert-figure.png'
assert image.save(str(fixture))
queue.add_figure(Figure(figsize=(1, 1)), str(fixture))
grid = ObservedGrid()
grid.setMinimumHeight(180)
pixmap = QPixmap.fromImage(image)
grid.set_figures([pixmap], ['Inert panel'], [('Native acceptance run', 0, 1)])
disable = QPushButton('Disable isolated controls')
finish = QPushButton('Finish isolated probe')
for widget in (queue, grid, disable, finish):
    layout.addWidget(widget)
def dump():
    header = grid._headers[0]
    observations.update(queue_count=queue.count(), list_count=queue._list.count(),
        collapsed=grid.is_section_collapsed('Native acceptance run', 0),
        expanded=header.is_expanded(), clear_enabled=queue._clear_label.isEnabled(),
        header_enabled=header.isEnabled(), visible_cells=sum(cell.isVisible() for cell in grid._cells),
        workspace=grid.workspace_state())
    temp = OUT / 'app-observations.tmp'
    temp.write_text(json.dumps(observations, indent=2) + '\n')
    temp.replace(OUT / 'app-observations.json')
def cleared():
    observations['clear_clicks'] += 1
    dump()
queue._clear_label.clicked.connect(cleared)
def disabled():
    queue._clear_label.setEnabled(False)
    grid._headers[0].setEnabled(False)
    dump()
disable.clicked.connect(disabled)
finish.clicked.connect(app.quit)
i18n.retranslate_widget_tree(host, 'en')
host.resize(700, 650)
host.show()
QAccessible.setActive(True)
timer = QTimer(host)
timer.timeout.connect(dump)
timer.start(100)
def hashes():
    result = {}
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if name.startswith('spacr') and path:
            path = Path(path)
            if path.is_relative_to(ROOT) and path.is_file():
                result[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result
def ready():
    dump()
    (OUT / 'process-source.json').write_text(json.dumps(hashes(), indent=2) + '\n')
    assert queue.count() == 1 and queue._list.count() == 1
    assert queue._clear_label.isVisible() and grid._headers[0].isVisible()
    assert grid._is_raised(grid._headers[0])
    print('READY ' + json.dumps({'pid': os.getpid(), 'platform': app.platformName(),
                                'initial': observations}), flush=True)
QTimer.singleShot(900, ready)
QTimer.singleShot(65000, app.quit)
app.exec()
timer.stop()
dump()
(OUT / 'process-source-final.json').write_text(json.dumps(hashes(), indent=2) + '\n')
host.close()
print('CLEAN_EXIT', flush=True)
