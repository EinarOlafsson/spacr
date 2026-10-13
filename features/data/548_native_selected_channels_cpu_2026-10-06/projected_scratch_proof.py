import contextlib
import hashlib
import inspect
import io as sysio
import tempfile
from pathlib import Path

import numpy as np

from spacr import io
from tests.test_cov_io_normalize import _base_settings

source = inspect.getsource(io._normalize_img_channels)
assert 'normalized_stack[array_index, ..., channel] = arr_2d_normalized' in source
source = source.replace('def _normalize_img_channels(', 'def _projected_normalize(', 1)
source = source.replace('normalized_stack[array_index, ..., channel] = arr_2d_normalized',
                        'normalized_stack[array_index, ..., col_by_source[channel]] = arr_2d_normalized')
namespace = dict(io.__dict__)
exec(compile(source, '<projected prototype>', 'exec'), namespace)
projected = namespace['_projected_normalize']

for source_dtype in (np.uint16, np.float32):
 for save_dtype in (np.float16, np.float32):
  for lower in (0, 17, 63.5):
   rng = np.random.default_rng(548)
   raw = rng.integers(0, 1001, size=(2, 3, 17, 19, 5)).astype(source_dtype)
   raw[..., 4] = 0
   settings = _base_settings(
       lower_percentile=lower, pathogen_channel=2, organelle_channel=3,
       nucleus_background=1, nucleus_signal_to_noise=1,
       cell_background=1, cell_signal_to_noise=992,
       pathogen_background=1, pathogen_signal_to_noise=1000000,
       remove_background_pathogen=False,
       organelle_background=400, organelle_signal_to_noise=2.4725,
       remove_background_organelle=True)
   for selected in ([3, 0], [4, 2, 1], [3, 0, 4, 2, 1]):
    namespace['col_by_source'] = {channel: pos for pos, channel in enumerate(selected)}
    with contextlib.redirect_stdout(sysio.StringIO()):
     old = io._normalize_img_batch(raw.copy(), selected, save_dtype, settings)[..., selected]
     result = projected(np.zeros((*raw.shape[:-1], len(selected)), dtype=np.float32),
                        selected, save_dtype, settings,
                        lambda channel: raw[..., channel].copy())
    assert result.tobytes() == old.tobytes(), (source_dtype, save_dtype, lower, selected)
    with tempfile.TemporaryDirectory() as folder:
     p = Path(folder)
     io._save_npz_atomic(p / 'old.npz', data=old, filenames=['one.npy', 'two.npy'])
     io._save_npz_atomic(p / 'new.npz', data=result, filenames=['one.npy', 'two.npy'])
     assert hashlib.sha256((p / 'old.npz').read_bytes()).digest() == hashlib.sha256((p / 'new.npz').read_bytes()).digest()
print('36 projected normalization cases: exact arrays and compressed NPZ bytes')
