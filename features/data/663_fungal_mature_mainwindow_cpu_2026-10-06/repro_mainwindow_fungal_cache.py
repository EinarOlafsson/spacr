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
import numpy as np

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
    base = Path(os.environ['SPACR_PROOF_DIR'])
    base.mkdir(parents=True, exist_ok=True)
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
    assert source_hashes['ambient'] == 'b2a191470947a2f832e7fe135835f05527e0a9e6fa1e7f41408108a4c412d855'
    retained_frames = []
    save_records = []
    hits = {}
    original_reuse = ambient_module._FungalGrowthEngine._reuse_fungal_raster

    def count_reuse(engine, *arguments):
        result = original_reuse(engine, *arguments)
        hits[id(engine)] = hits.get(id(engine), 0) + bool(result)
        return result

    ambient_module._FungalGrowthEngine._reuse_fungal_raster = count_reuse
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
    prefs.set_ambient_animation('data_art_fungal_growth')
    prefs.set_ambient_palette('random')
    prefs.set_ambient_resolution(2.0)
    prefs.set_ambient_density(3.0)
    prefs.set_ambient_size(2.5)
    prefs.set_ambient_blur(0)
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
            density = dialog.findChild(QSlider, 'AmbientDensity')
            size = dialog.findChild(QSlider, 'AmbientSize')
            assert density is not None and size is not None
            density.setValue(300)
            size.setValue(250)
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
        assert isinstance(visible._engine, ambient_module._FungalGrowthEngine)
        visible.set_time(95)
        for _ in range(90):
            pump(1)
            if index % 2 or hits.get(id(visible._engine), 0) >= 8:
                break
        assert visible._engine.size == 2.5 and visible._engine.density == 3.0
        if index % 2 == 0:
            assert hits.get(id(visible._engine), 0) >= 8
            assert visible._engine._fungal_rasters
            assert visible._engine.buffer_size(visible.width(), visible.height()) == (visible.width(), visible.height())
        pointer = None
        producer = visible._producer_box[0]
        assert producer is not None
        frame = producer.latest()
        assert frame is not None and not frame.isNull()
        words = np.frombuffer(
            frame.bits(), dtype=np.uint32,
            count=frame.bytesPerLine() // 4 * frame.height())
        non_opaque = int(np.count_nonzero((words >> 24) != 255))
        assert non_opaque == 0
        digest = hashlib.sha256(frame.constBits()).hexdigest()
        for old_frame, old_digest in retained_frames:
            assert hashlib.sha256(old_frame.constBits()).hexdigest() == old_digest
            assert np.all(np.frombuffer(old_frame.constBits(), dtype=np.uint32) >> 24 == 255)
        retained_frames.append((frame, digest))
        hovered = app.widgetAt(QCursor.pos())
        print('save done', index, json.dumps({
            'widgets': len(app.allWidgets()),
            'ambient': len([w for w in app.allWidgets() if isinstance(w, AmbientWidget)]),
            'workers': len([w for w in app.allWidgets() if isinstance(w, AmbientWidget) and w.shading_thread_alive()]),
            'gravity': prefs._ambient_gravity_radius(),
            'palette': prefs.get_ambient_palette(),
            'cursor_poll': pointer,
            'engine_pointer': getattr(visible._engine, 'pointer', None),
            'engine_radius': getattr(visible._engine, 'gravity_radius', None),
            'frame_size': [frame.width(), frame.height()],
            'frame_format': int(frame.format().value),
            'non_opaque_frame_words': non_opaque,
            'cursor_preconditions': {
                'animating': visible._animating,
                'visible': visible.isVisible(),
                'interactive': getattr(visible._engine, 'interactive', False),
                'active_main': app.activeWindow() is window,
                'hovered_main': hovered is not None and hovered.window() is window,
                'inside': visible.rect().contains(visible.mapFromGlobal(QCursor.pos())),
            },
            'cursor_global': [QCursor.pos().x(), QCursor.pos().y()],
            'memory_kib': memory(),
        }), flush=True)
        save_records.append({'index': index, 'palette': wanted_palette, 'resolution': visible._engine.resolution,
                             'density': visible._engine.density, 'size': visible._engine.size,
                             'native_viewport': [visible.width(), visible.height()],
                             'frame_size': [frame.width(), frame.height()],
                             'max_pixels': visible._engine.max_pixels,
                             'cache_hits': hits.get(id(visible._engine), 0),
                             'cache_entries': len(visible._engine._fungal_rasters),
                             'cache_array_bytes': sum(e[1].nbytes + e[2].nbytes for e in visible._engine._fungal_rasters.values()),
                             'alpha255': non_opaque == 0, 'frame_sha256': digest, 'prior_frames_unchanged': True,
                             'memory_kib': memory()})
        window._on_nav_selected(('mask', 'measure', 'annotate')[index])
        window.resize(3200 if index % 2 else 3840, 1800 if index % 2 else 2160)
        pump()
    for old_frame, old_digest in retained_frames:
        assert hashlib.sha256(old_frame.constBits()).hexdigest() == old_digest
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
    assert collections and all(tid == gui_thread for tid, _ in collections)
    ambient_threads = [t.name for t in threading.enumerate() if t.name == 'spacr-ambient-shade']
    assert not ambient_threads
    receipt = {'source_commit': source_commit, 'source_sha256': source_hashes, 'source_paths': {k: str(v) for k,v in source_paths.items()},
               'qt_platform': os.environ['QT_QPA_PLATFORM'], 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
               'saves': save_records, 'gui_gc_count': len(collections), 'all_gc_on_gui_thread': True,
               'after_close': {'widgets': len(app.allWidgets()), 'ambient': len(remaining), 'live_workers': len(live),
                               'ambient_threads': ambient_threads, 'memory_kib': memory()},
               'scope': 'actual fresh native4K softwareXvfb MainWindow Home/Mask/Measure/Annotate; three real modal Preferences Saves; maxdensity3/size2.5, detail200/100/200, Random/default/Random; no GPU/inference/manualgc/hardFPS/crashfix claim'}
    (base/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
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
