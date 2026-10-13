import argparse
import shutil
from pathlib import Path

import numpy as np
import tifffile

from tests.test_watch_timelapse_series_f548 import _converted_series

parser = argparse.ArgumentParser(description='Create a fresh native memory fixture.')
parser.add_argument('--artifact-root', type=Path, required=True)
base = parser.parse_args().artifact_root.resolve()
if base.exists() and any(base.iterdir()):
    parser.error('artifact root must be empty; existing evidence is never removed')
base.mkdir(parents=True, exist_ok=True)
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
    shutil.copytree(source, target)
print(source, len(rows), side, flush=True)
