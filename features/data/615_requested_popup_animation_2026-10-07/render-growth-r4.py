from pathlib import Path
import json
import hashlib
import math
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage, QPainter, QColor
from spacr.qt.widgets.ambient import make_engine

app = QApplication.instance() or QApplication([])
out = Path('/media/carruthers/mnt3/codex/scratch/magnifier-acceptance-20261007/growth-preview-r4')
out.mkdir(exist_ok=False)
receipt = {'source_sha256': hashlib.sha256(Path('spacr/qt/widgets/ambient.py').read_bytes()).hexdigest(), 'frames': []}
for density in (0.1, 1.0):
    engine = make_engine('data_art_fungal_growth', 'spacr', '#10161c', seed=29, density=density)
    for second in (1, 7, 15, 21, 30, 45):
        engine.set_time(second)
        image = QImage(1280, 720, QImage.Format_RGB32)
        image.fill(QColor('#10161c'))
        painter = QPainter(image)
        engine.paint(painter, 1280, 720)
        painter.end()
        path = out / f'growth-density-{density}-t-{second}.png'
        assert image.save(str(path))
        geometry = engine.geometry(1280, 720)
        receipt['frames'].append({'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'edges': len(geometry), 'max_segment_pixels': max(math.hypot(edge[4] - edge[0], edge[5] - edge[1]) for edge in geometry)})
(out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(receipt, flush=True)
