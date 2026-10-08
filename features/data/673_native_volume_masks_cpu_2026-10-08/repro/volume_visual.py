import hashlib
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PySide6 import __version__ as qt_version
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication
from shiboken6 import isValid

from spacr.qt import mask_engine as engine
from spacr.qt.screens.make_masks import _VolumeMaskEditor
from spacr.qt.widgets.gate_spec import BoxGate
from spacr.tiff_io import write_tiff

out = Path('/mnt/wd4tb/scratch/gate-anchors-20261008')
app = QApplication.instance() or QApplication([])
records = []
with TemporaryDirectory(dir=out) as folder:
    image = np.arange(16 ** 3, dtype=np.uint16).reshape(16, 16, 16)
    source = Path(folder) / 'volume.tif'
    write_tiff(source, image, metadata={'axes': 'ZYX', 'voxel_spacing': [1., .5, .25], 'unit': 'um'})
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    session = engine.load_volume_image_and_mask(folder, source.name)
    editor = _VolumeMaskEditor(session)
    editor.show()
    for _ in range(4):
        app.processEvents()
    editor._panel.canvas.gate_drawn.emit(BoxGate.from_limits('(unnamed)', ('X', 'Y', 'Z'), [(0, 2), (0, 4), (0, 8)]))
    deadline = time.monotonic() + 5
    while editor._busy and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.001)
    assert not editor._busy and np.count_nonzero(session.mask) == 8 ** 3
    canvas = editor._panel.canvas
    canvas._view_angles = (31., -47., 0.)
    editor._density.setValue(4)
    canvas.render_now()
    for _ in range(4):
        app.processEvents()
    path = out / 'volume-native-anchors.png'
    assert editor.grab().save(str(path))
    records.append({'path': path.name, 'width': editor.width(), 'height': editor.height(),
                    'label': 1, 'voxels': int(np.count_nonzero(session.mask)), 'anchor_density': 4,
                    'axes': session.axes, 'spacing': session.spacing})
    canvas._selected_surfaces = {1, 5}
    canvas._surface_gate = '1'
    canvas._edit_axis = 'z'
    canvas.render_now()
    for _ in range(4):
        app.processEvents()
    path = out / 'volume-selected-surfaces.png'
    assert editor.grab().save(str(path))
    records.append({'path': path.name, 'selected_surface_groups': [1, 5], 'voxels_unchanged': int(np.count_nonzero(session.mask))})
    assert editor._save_mask()
    restored = engine.load_volume_image_and_mask(folder, source.name)
    assert np.array_equal(restored.mask, session.mask)
    assert restored.geometries == session.geometries
    assert original == hashlib.sha256(source.read_bytes()).hexdigest()
    editor.close()
    editor.deleteLater()
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    assert not isValid(editor) and editor._worker.idle()
    receipt = {'source_checkpoint': '2720b462e8a', 'Qt': qt_version,
               'scope': 'Actual offscreen native Qt dialog captures; not native-display, 4K animation, GPU, FPS or aesthetics acceptance.',
               'images': records, 'source_bytes_unchanged': True, 'exact_mask_geometry_roundtrip': True,
               'dialog_native_destroyed': True, 'worker_idle': True,
               'top_level_widgets_after': len(app.topLevelWidgets())}
    (out / 'volume-visual-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))
