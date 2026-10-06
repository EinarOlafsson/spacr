import shutil
from pathlib import Path

import numpy as np
import tifffile

from tests.test_watch_timelapse_series_f548 import _converted_series

base = Path('/mnt/wd4tb/scratch/f548-native-memory-20261006')
source, rows = _converted_series(base / 'fixture')
side = 2048
ramp = ((np.arange(side, dtype=np.uint32)[:, None] * 3
         + np.arange(side, dtype=np.uint32)[None, :] * 5) % 2000).astype(np.uint16)
for row in rows:
    value = ramp + np.uint16(10 * int(row['channel']) + 7 * int(row['z'])
                             + 13 * int(row['t']))
    tifffile.imwrite(source / row['target'], value)
for name in ('baseline', 'optimized'):
    target = base / name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
print(source, len(rows), side, flush=True)
