"""Check real native widget disposal and temporary packed-array ownership."""
import gc
import hashlib
import importlib.util
import json
import os
import resource
import sys
import weakref
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, os.environ['SPACR_PROOF_REPO'])
from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication, QWidget

importlib.import_module('spacr.qt.widgets')
spec = importlib.util.spec_from_file_location('spacr.qt.widgets._gc_packed', ROOT / 'after.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
app = QApplication([])
app.setQuitOnLastWindowClosed(False)
module._warm_packed_scatter()
kernel = module._COLORED_SCATTER
assert kernel is not None


def wait(milliseconds):
    loop = QEventLoop()
    timer = QTimer()
    timer.setTimerType(Qt.PreciseTimer)
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    timer.start(milliseconds)
    loop.exec()


records = []
for mode in ['fallback', 'compiled', 'compiled', 'fallback']:
    temporaries, empty_rank_sizes = [], []

    def capture(flat, ranks, *arguments):
        temporaries.extend([weakref.ref(flat), weakref.ref(ranks)])
        empty_rank_sizes.append(ranks.size)
        kernel(flat, ranks, *arguments)

    module._COLORED_SCATTER = capture if mode == 'compiled' else None
    host = QWidget()
    host.resize(3840, 2160)
    widget = module.AmbientWidget(host, theme='data_art_genetic_advection', palette='random',
                                  seed=42, resolution=2, density=1, fps=24, blur=0)
    widget.follow_parent()
    host.show()
    app.processEvents()
    wait(350)
    engine, producer = widget.engine, widget._producer_box[0]
    assert engine.buffer_size(3840, 2160) == (3840, 2160)
    current = producer.latest()
    assert current is not None and producer.frames_shaded > 0
    assert np.all(np.frombuffer(current.constBits(), np.uint32) >> 24 == 255)
    engine_ref, widget_ref, image_ref = weakref.ref(engine), weakref.ref(widget), weakref.ref(current)
    tables = engine._material_cache[('random_grain_palette', engine.dark)]
    table_refs = [weakref.ref(table) for table in tables]
    table_bytes = sum(table.nbytes for table in tables)
    frames = producer.frames_shaded
    widget.hide()
    wait(200)
    assert not producer.is_alive()
    widget.deleteLater()
    host.deleteLater()
    wait(10)
    del widget, host, engine, producer, current, tables
    gc.collect()
    assert engine_ref() is None and widget_ref() is None and image_ref() is None
    assert all(reference() is None for reference in table_refs + temporaries)
    assert all(size == 0 for size in empty_rank_sizes)
    records.append({'mode': mode, 'native_size': [3840, 2160], 'shaded_frames': frames,
                    'palette_cache_bytes': table_bytes, 'temporary_array_views': len(temporaries),
                    'empty_rank_arguments': empty_rank_sizes,
                    'worker_stopped': True, 'widget_engine_frame_tables_released': True,
                    'temporary_views_released': True,
                    'idle_rss_kib': int(Path('/proc/self/status').read_text().split('VmRSS:')[1].split()[0])})
module._COLORED_SCATTER = kernel
receipt = {'source_sha256': hashlib.sha256((ROOT / 'after.py').read_bytes()).hexdigest(),
           'scope': 'actual native widget/producer show, hide, delete and GC; compiler warmed',
           'records': records, 'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
(ROOT / 'gc_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
