import faulthandler
import gc
import threading
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import subprocess

faulthandler.enable(all_threads=True)
from PySide6.QtCore import QCoreApplication, QEvent, QSettings, QTimer
from PySide6.QtWidgets import QApplication, QComboBox, QDialogButtonBox, QSlider
from PySide6.QtGui import QCursor
from spacr.qt import preferences as prefs
from spacr.qt.app import MainWindow
from spacr.qt.widgets.ambient import AmbientWidget
from spacr.qt import gc_policy
import spacr.qt.widgets.ambient as ambient_module


def main():
    base = Path('/mnt/wd4tb/scratch/ambient-crash-20261006/random-palette-probe')
    base.mkdir(exist_ok=True)
    source_paths = {
        name: Path(__import__(module, fromlist=['__file__']).__file__)
        for name, module in (
            ('ambient', 'spacr.qt.widgets.ambient'),
            ('preferences', 'spacr.qt.preferences'),
            ('app', 'spacr.qt.app'))}
    source_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in source_paths.items()}
    source_commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=os.environ['SPACR_SOURCE_ROOT'],
        text=True).strip()

    def memory():
        return {key: int(value.split()[0]) for row in Path('/proc/self/status').read_text().splitlines()
                if ':' in row for key, value in [row.split(':', 1)]
                if key in ('VmRSS', 'VmHWM', 'RssAnon', 'RssFile')}

    print('source hashes', json.dumps(source_hashes, sort_keys=True), flush=True)
    print('source paths', json.dumps({key: str(path) for key, path in source_paths.items()}, sort_keys=True), flush=True)
    print('source commit', source_commit, flush=True)
    store = QSettings(str(base / ('app-prefs-' + sys.argv[1] + '.ini')), QSettings.IniFormat)
    prefs._settings = lambda: store
    try:
        from spacr.qt.first_run import mark_tour_seen
        mark_tour_seen()
    except Exception:
        pass
    app = QApplication.instance() or QApplication([])
    assert gc_policy.install(app)
    gui_thread = threading.get_ident()
    collections = []
    def record_collection(phase, info):
        if phase == 'stop':
            collections.append((threading.get_ident(), info['generation']))
    gc.callbacks.append(record_collection)
    print('source', ambient_module.__file__,
          hashlib.sha256(Path(ambient_module.__file__).read_bytes()).hexdigest(),
          flush=True)
    prefs.set_ambient_animation('data_art_genetic_advection')
    prefs.set_ambient_palette('random')
    prefs.set_ambient_resolution(1.0)
    prefs.set_ambient_enabled(True)
    prefs.set_refresh_news(False)
    window = MainWindow()
    window.resize(3840, 2160)
    window.show()
    QCursor.setPos(window.mapToGlobal(window.rect().center()))

    def pump(count=8):
        for _ in range(count):
            app.processEvents()
            time.sleep(0.02)

    pump()
    print('home', len(app.allWidgets()), json.dumps(memory()), flush=True)
    for key in ('mask', 'measure', 'annotate'):
        window._on_nav_selected(key)
        pump()
        ambient_widgets = [w for w in app.allWidgets() if isinstance(w, AmbientWidget)]
        print('screen', key, len(app.allWidgets()), len(ambient_widgets),
              [(w.width(), w.height(), w.shading_thread_alive()) for w in ambient_widgets], flush=True)
    for index in range(3):
        wanted_palette = ('random', 'spacr', 'random')[index]
        def save_dialog():
            dialog = app.activeModalWidget()
            assert dialog is not None, type(dialog)
            slider = dialog.findChild(QSlider, 'AmbientResolution')
            assert slider is not None
            slider.setValue(200 if index % 2 == 0 else 100)
            gravity = dialog.findChild(QSlider, 'AmbientGravityRadius')
            assert gravity is not None
            gravity.setValue((0, 65, 0)[index])
            palette = dialog.findChild(QComboBox, 'AmbientPalette')
            assert palette is not None
            palette_index = palette.findData(wanted_palette)
            assert palette_index >= 0, wanted_palette
            palette.setCurrentIndex(palette_index)
            print('save start', index, slider.value(), gravity.value(),
                  palette.currentData(), flush=True)
            dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
        QTimer.singleShot(200, save_dialog)
        window.show_preferences_on()
        pump()
        assert prefs._ambient_gravity_radius() == (0.0, 0.65, 0.0)[index]
        assert prefs.get_ambient_palette() == wanted_palette
        candidates = [w for w in app.allWidgets() if isinstance(w, AmbientWidget)
                      and w.isVisible() and w.window() is window]
        assert candidates
        visible = max(candidates, key=lambda w: w.width() * w.height())
        window.activateWindow()
        QCursor.setPos(visible.mapToGlobal(visible.rect().center()))
        pump(12)
        pointer = visible._data_art_pointer_for_tick()
        if index == 1:
            for _ in range(15):
                if visible._engine.pointer is not None:
                    break
                pump(1)
            assert visible._engine.pointer is not None
        hovered = app.widgetAt(QCursor.pos())
        print('save done', index, json.dumps({
            'widgets': len(app.allWidgets()),
            'ambient': len([w for w in app.allWidgets() if isinstance(w, AmbientWidget)]),
            'workers': len([w for w in app.allWidgets() if isinstance(w, AmbientWidget) and w.shading_thread_alive()]),
            'gravity': prefs._ambient_gravity_radius(),
            'palette': prefs.get_ambient_palette(),
            'cursor_poll': pointer,
            'engine_pointer': visible._engine.pointer,
            'engine_radius': visible._engine.gravity_radius,
            'cursor_preconditions': {
                'animating': visible._animating,
                'visible': visible.isVisible(),
                'interactive': visible._engine.interactive,
                'active_main': app.activeWindow() is window,
                'hovered_main': hovered is not None and hovered.window() is window,
                'inside': visible.rect().contains(visible.mapFromGlobal(QCursor.pos())),
            },
            'cursor_global': [QCursor.pos().x(), QCursor.pos().y()],
            'memory_kib': memory(),
        }), flush=True)
        window._on_nav_selected(('mask', 'measure', 'annotate')[index])
        window.resize(3200 if index % 2 else 3840, 1800 if index % 2 else 2160)
        pump()
    window.close()
    window.deleteLater()
    pump(20)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    pump(2)
    remaining = [w for w in app.allWidgets() if isinstance(w, AmbientWidget)]
    live = [w for w in remaining if w.shading_thread_alive()]
    print('after close', json.dumps({'widgets': len(app.allWidgets()), 'ambient': len(remaining), 'workers': len(live), 'memory_kib': memory()}), flush=True)
    assert not live
    assert not remaining
    assert not app.allWidgets()
    print('gui gc', json.dumps({'collections': len(collections), 'all_gui': all(tid == gui_thread for tid, _ in collections), 'generations': [generation for _, generation in collections]}), flush=True)
    gc.callbacks.remove(record_collection)
    print('done', flush=True)
    gc_policy.uninstall()
    assert source_hashes == {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in source_paths.items()}
    print('source hashes after', json.dumps(source_hashes, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
