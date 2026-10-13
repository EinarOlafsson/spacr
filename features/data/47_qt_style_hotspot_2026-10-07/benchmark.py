import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QLineEdit, QWidget

from spacr.qt import preferences
from spacr.qt import theme

parser = argparse.ArgumentParser()
parser.add_argument('--widgets', type=int, default=0)
parser.add_argument('--python-objects', type=int, default=0)
parser.add_argument('--unchanged', action='store_true')
args = parser.parse_args()

app = QApplication.instance() or QApplication([])
with tempfile.TemporaryDirectory(prefix='spacr-field-fade-scan-') as temp:
    settings = QSettings(str(Path(temp) / 'prefs.ini'), QSettings.IniFormat)
    preferences._settings = lambda: settings
    unrelated = [{'index': index, 'pair': (index, index + 1)}
                 for index in range(args.python_objects)]
    root = QWidget()
    fields = [QLineEdit(root) for _ in range(args.widgets)]
    root.show()
    QApplication.processEvents()
    root.hide()
    QApplication.processEvents()
    timings = []
    teardown_timings = []
    preferences.set_theme('light')
    for index in range(5):
        if not args.unchanged:
            preferences.set_theme('dark' if index % 2 else 'light')
        start = time.perf_counter()
        preferences.apply_preferences_to_app(app)
        timings.append(time.perf_counter() - start)
        start = time.perf_counter()
        if not args.unchanged:
            theme._forget_window_stylesheets(app)
        teardown_timings.append(time.perf_counter() - start)
    print(json.dumps({'widgets': args.widgets, 'python_objects': len(unrelated),
                      'all_widgets': len(app.allWidgets()), 'unchanged': args.unchanged,
                      'seconds': timings, 'warm_median_s': statistics.median(timings[1:]),
                      'teardown_seconds': teardown_timings,
                      'teardown_warm_median_s': statistics.median(teardown_timings[1:])}))
