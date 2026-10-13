import hashlib
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory

root = Path.cwd()
out = Path(os.environ['SPACR_PROBE_OUT'])
out.mkdir(parents=True, exist_ok=True)
os.environ['SPACR_HOME'] = str(out / 'private-home')
os.environ['SPACR_LOG_DIR'] = str(out / 'private-logs')
os.environ['SPACR_NETWORK_CONFIG'] = str(out / 'private-network.json')

import numpy
import PySide6
from PySide6.QtCore import QEvent, QSettings, qInstallMessageHandler
from PySide6.QtWidgets import QApplication, QComboBox, QDialog, QDialogButtonBox, QLabel, QSlider, QVBoxLayout
from shiboken6 import isValid
from spacr.qt import preferences as prefs, theme
from spacr.qt.app import MainWindow
from spacr.qt.widgets import ambient, glass

records, messages, exceptions, producers = [], [], [], []


def handler(kind, context, message):
    if len(messages) < 256:
        row = {'type': str(kind), 'message': message, 'thread': threading.current_thread().name}
        if any(mark in message.lower() for mark in ('slot', 'wrapper', 'paint device', 'active painter', 'painter not active')):
            row['python_stack'] = traceback.format_stack(limit=15)
        messages.append(row)


def caught(kind, value, trace):
    exceptions.append({'type': kind.__name__, 'message': str(value), 'traceback': ''.join(traceback.format_exception(kind, value, trace))})
    print(json.dumps(exceptions[-1]), file=sys.stderr, flush=True)


sys.excepthook = caught
threading.excepthook = lambda args: caught(args.exc_type, args.exc_value, args.exc_traceback)
app = QApplication.instance() or QApplication([])
qInstallMessageHandler(handler)


def turn(seconds=.12):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.002)


def note(stage):
    widgets = [w for w in app.allWidgets() if isinstance(w, ambient.AmbientWidget)]
    snapshots = []
    for widget in widgets:
        producer = widget._producer_box[0]
        if producer is not None and producer not in producers:
            producers.append(producer)
        meta = widget.metaObject()
        snapshots.append({'theme': widget.theme(), 'visible': widget.isVisible(), 'popup': bool(widget.property('spacrPopupBackdrop')), 'parent_class': type(widget.parentWidget()).__name__ if widget.parentWidget() is not None else None, 'window_title': widget.window().windowTitle(), 'timer_active': widget._timer.isActive(),
                          'frames_painted': widget.frames_painted, 'frames_shaded': widget.frames_shaded(),
                          'time': widget.engine.time, 'wrapper_class': type(widget).__name__, 'meta_class': meta.className(),
                          'tick_index': meta.indexOfSlot('_on_tick()')})
    record = {'stage': stage, 'seconds': time.monotonic() - started, 'ambient': snapshots}
    records.append(record)
    print(json.dumps(record), flush=True)


with TemporaryDirectory(dir=out) as folder:
    ini = Path(folder) / 'preferences.ini'
    prefs._settings = lambda: QSettings(str(ini), QSettings.IniFormat)
    prefs._SAFE_MODE = False
    theme.enable_spaceout()
    prefs.set_language('en')
    prefs.set_theme_choice('glass')
    prefs.set_ambient_animation('data_art_spaceout_field')
    prefs.set_ambient_palette('spacr')
    prefs.set_ambient_density(1)
    prefs.set_ambient_resolution(1)
    prefs._set_ambient_gravity_radius(.4)
    prefs.set_popup_backdrop('drift')
    started = time.monotonic()
    window = MainWindow(initial_app='__home__')
    window.resize(1100, 760)
    window.show()
    turn(.4)
    note('actual_home_field')
    for cycle in range(3):
        dialog = prefs.PreferencesDialog(window)
        dialog.show()
        turn(.12)
        choice = dialog.findChild(QComboBox, 'AmbientTheme')
        assert choice is not None
        chosen = 'data_art_tissue_facets'
        assert choice.findData(chosen) >= 0
        choice.setCurrentIndex(choice.findData(chosen))
        dialog.findChild(QSlider, 'AmbientDensity').setValue((100, 200, 300)[cycle])
        dialog.findChild(QSlider, 'AmbientResolution').setValue((100, 200, 100)[cycle])
        dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Apply).click()
        turn(.12)
        question = dialog._apply_confirmation
        assert question is not None and question.isVisible()
        next(button for button in question.buttons() if button.text() == 'Keep').click()
        turn(.12)
        assert prefs.get_ambient_animation() == chosen
        assert prefs.get_popup_backdrop() == 'drift'
        note('preferences_apply_keep_' + str(cycle))
        dialog.reject()
        dialog.deleteLater()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        turn(.05)
        assert not isValid(dialog)
        popup = QDialog(window)
        popup.resize(500, 350)
        layout = QVBoxLayout(popup)
        layout.addWidget(QLabel('Independent popup backdrop', popup))
        popup.show()
        first = glass._install_the_backdrop(popup)
        assert isinstance(first, ambient.AmbientWidget)
        for palette in ('random', 'spacr'):
            first.set_palette(palette)
            turn(.08)
        note('independent_popup_drift_' + str(cycle))
        popup.close()
        popup.deleteLater()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        turn(.05)
        assert not isValid(popup) and not isValid(first)
        for key in ('data_art_spaceout_field', 'data_art_tissue_facets', 'blobs', 'data_art_spaceout_field'):
            prefs.set_ambient_animation(key)
            prefs.apply_ambient_preferences(app)
            turn(.08)
        window.hide()
        turn(.05)
        note('hidden_' + str(cycle))
        window.show()
        turn(.12)
        note('restarted_' + str(cycle))
    for key in ('measure', 'make_masks', '__home__'):
        window._on_nav_selected(key)
        turn(.12)
        note('navigate_' + key)
    window.close()
    window.deleteLater()
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    turn(.2)
    note('shutdown')
    report = {'scope': 'Bounded real Home/module and Preferences Apply switching, independent popup retirement and shutdown; negative reproduction is not a missing-slot fix or native crash acceptance.',
              'checkpoint': '8f245a658c9791debe9fe9848c5bf2c691bfb77e', 'source': str(Path(ambient.__file__).resolve()),
              'ambient_sha256': hashlib.sha256(Path(ambient.__file__).read_bytes()).hexdigest(),
              'imported_sources': {name: {'path': str(Path(sys.modules[name].__file__).resolve()), 'sha256': hashlib.sha256(Path(sys.modules[name].__file__).read_bytes()).hexdigest()} for name in ('spacr', 'spacr.qt.app', 'spacr.qt.preferences', 'spacr.qt.screens.app_screen', 'spacr.qt.widgets.ambient', 'spacr.qt.widgets.glass')},
              'pyside_module': str(Path(PySide6.__file__).resolve()),
              'python': sys.version, 'executable': sys.executable, 'Qt': PySide6.__version__, 'numpy': numpy.__version__,
              'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'), 'QT_QPA_PLATFORM': os.environ.get('QT_QPA_PLATFORM'),
              'records': records, 'messages': messages, 'exceptions': exceptions,
              'retired_producers': len(producers), 'alive_producers_after': sum(p.is_alive() for p in producers),
              'remaining_ambient_widgets': len([w for w in app.allWidgets() if isinstance(w, ambient.AmbientWidget)]),
              'window_native_destroyed': not isValid(window)}
    (out / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('Qt', 'ambient_sha256', 'exceptions', 'retired_producers', 'alive_producers_after', 'remaining_ambient_widgets', 'window_native_destroyed')}), flush=True)
    assert not exceptions
    assert not any('slot' in row['message'].lower() or 'wrapper' in row['message'].lower() for row in messages)
    assert report['alive_producers_after'] == 0 and report['remaining_ambient_widgets'] == 0
