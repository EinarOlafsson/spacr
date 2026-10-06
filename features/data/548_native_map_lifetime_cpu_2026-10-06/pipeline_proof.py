"""Compare actual native preprocessing outputs with frozen pre-fix IO callables."""
import ast
import hashlib
import json
import resource
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile
from spacr import convert, io
from tests.test_native_tzyx_batch_f548 import _settings

root = Path('/mnt/wd4tb/scratch/f548-io-map-lifetime-20261006')
mode = sys.argv[1]
current_source = Path(io.__file__).read_bytes()
frozen_source = (root / 'before_io.py').read_bytes()
if mode == 'before':
    names = {'_normalize_img_channels', '_close_private_memmap',
             '_native_nonzero_vector', '_preprocess_mapped_volume_series'}
    parsed = ast.parse(frozen_source)
    definitions = [node for node in parsed.body if isinstance(node, ast.FunctionDef)
                   and node.name in names]
    assert len(definitions) == len(names)
    exec(compile(ast.Module(body=definitions, type_ignores=[]),
                 '<exact-frozen-native-IO-callables>', 'exec'), io.__dict__)
with tempfile.TemporaryDirectory(prefix=f'pipeline-{mode}-', dir=root) as folder:
    folder = Path(folder)
    raw = folder / 'raw' / 'A01'
    raw.mkdir(parents=True)
    y, x = np.indices((1280, 1280), dtype=np.uint16)
    for channel in range(4):
        stack = np.stack([np.stack([
            ((x + 2*y + 31*t + 53*z + 71*channel) % 3000).astype(np.uint16)
            for z in range(2)]) for t in range(2)])
        tifffile.imwrite(raw / f'field01_C{channel + 1}.tif', stack,
                        metadata={'axes': 'TZYX'}, photometric='minisblack')
    converted = folder / 'converted'
    assert convert.convert_folder(dict(src=str(folder / 'raw'), dst=str(converted),
                                       z_handling='keep', preview_rows=0)).is_complete
    settings, returned = io.preprocess_img_data(_settings(
        converted, nucleus_channel=3, cell_channel=1, pathogen_channel=None,
        organelle_channel=None, lower_percentile=17))
    outputs = {}
    for path in sorted(converted.rglob('*')):
        if path.is_file() and path.suffix in ('.npy', '.npz'):
            outputs[str(path.relative_to(converted))] = {
                'size': path.stat().st_size,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    assert not list(converted.glob('.spacr-volume-series-*'))
    receipt = {'mode': mode, 'active_io_sha256': hashlib.sha256(
        frozen_source if mode == 'before' else current_source).hexdigest(),
        'frozen_callables': sorted(names) if mode == 'before' else None,
        'outputs': outputs, 'channels': settings['channels'],
        'cellpose_nucleus_channel': settings['cellpose_nucleus_channel'],
        'cellpose_cell_channel': settings['cellpose_cell_channel'],
        'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (root / f'pipeline-{mode}.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)
