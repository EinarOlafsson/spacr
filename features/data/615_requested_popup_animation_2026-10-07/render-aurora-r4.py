from pathlib import Path
import hashlib
import json
import time
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage, QPainter, QColor
from spacr.qt.widgets.ambient import make_engine

app = QApplication.instance() or QApplication([])
root = Path.cwd()
out = Path('/media/carruthers/mnt3/codex/scratch/magnifier-acceptance-20261007/aurora-preview-r4')
out.mkdir(exist_ok=False)
receipt = {'source_sha256': hashlib.sha256((root / 'spacr/qt/widgets/ambient.py').read_bytes()).hexdigest(), 'frames': [], 'timings': []}
for density in (0.1, 1.0):
    engine = make_engine('aurora', 'spacr', '#10161c', seed=17, density=density)
    for second in (0, 4.5, 12):
        engine.set_time(second)
        image = QImage(1280, 720, QImage.Format_RGB32)
        image.fill(QColor('#10161c'))
        painter = QPainter(image)
        engine.paint(painter, image.width(), image.height())
        painter.end()
        path = out / f'aurora-density-{density}-t-{second}.png'
        assert image.save(str(path))
        receipt['frames'].append({'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    for size in ((1280, 720), (3840, 2160)):
        engine.set_max_pixels(size[0] * size[1])
        engine.shade(*size)
        costs = []
        for tick in range(12):
            engine.set_time(tick / 24)
            start = time.perf_counter()
            engine.shade(*size)
            costs.append((time.perf_counter() - start) * 1000)
        receipt['timings'].append({'density': density, 'size': size, 'shade_ms': costs, 'mean_ms': sum(costs) / len(costs), 'scope': 'CPU engine shading only, not native GUI FPS acceptance'})
(out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt['timings'], indent=2), flush=True)
