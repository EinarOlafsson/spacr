"""Fresh-process real shader startup import boundary and exact kernel parity."""
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
importlib.import_module('spacr.qt.widgets')
path = ROOT / 'after.py'
spec = importlib.util.spec_from_file_location('spacr.qt.widgets._frozen_palette_startup', path)
ambient = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ambient
spec.loader.exec_module(ambient)

app = QApplication([])
assert 'numba' not in sys.modules and 'scipy' not in sys.modules
ambient._begin_ambient_startup()
engines = []
frames = []
for theme in ['data_art_impulse_lens', 'data_art_genetic_advection', 'data_art_chromatin_ribbon']:
    engine = ambient.make_engine(theme, 'random', '#101418', seed=42, resolution=2)
    engine.set_max_pixels(1920 * 1080)
    engine.set_time(17)
    image = engine.shade(1920, 1080)
    frames.append(image.bits().tobytes())
    engines.append(engine)
assert 'numba' not in sys.modules and 'scipy' not in sys.modules
assert not ambient._PACKED_SCATTER_STARTED and not ambient._SATIN_COMPILER.started
ambient._complete_ambient_startup()
started = time.perf_counter()
while ambient._ready_colored_scatter() is None or ambient._SATIN_COMPILER.ready() is None:
    assert time.perf_counter() - started < 30
    app.processEvents()
    time.sleep(.01)
assert 'numba' in sys.modules and 'scipy' in sys.modules
for engine, reference in zip(engines, frames):
    image = engine.shade(1920, 1080)
    assert image.bits().tobytes() == reference
receipt = {'source_sha256': hashlib.sha256(Path(ambient.__file__).read_bytes()).hexdigest(),
           'source_path': ambient.__file__, 'blocked_first_frames': len(frames),
           'numba_and_scipy_absent_until_complete': True,
           'compiled_full_frame_parity': True, 'one_grain_worker_for_both_kernels': ambient._PACKED_SCATTER_STARTED and ambient._COLORED_SCATTER is not None,
           'compile_seconds_after_complete': time.perf_counter() - started,
           'pixel_sha256': [hashlib.sha256(frame).hexdigest() for frame in frames]}
(ROOT / 'startup_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
