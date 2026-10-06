from pathlib import Path
import shutil
import sys

import numpy as np
import tifffile

from spacr import convert

root = Path(sys.argv[1])
raw = root / 'raw' / 'A01'
raw.mkdir(parents=True, exist_ok=True)
y, x = np.indices((1536, 1536), dtype=np.uint16)
for channel in range(4):
    stack = np.stack([
        np.stack([((x + 2 * y + 31 * time + 53 * z + 71 * channel) % 3000).astype(np.uint16)
                  for z in range(2)])
        for time in range(2)])
    tifffile.imwrite(raw / f'field01_C{channel + 1}.tif', stack,
                     metadata={'axes': 'TZYX'}, photometric='minisblack')
converted = root / 'converted'
assert convert.convert_folder(dict(src=str(root / 'raw'), dst=str(converted),
                                   z_handling='keep', preview_rows=0)).is_complete
for name in ('baseline', 'candidate'):
    shutil.copytree(converted, root / name)
shutil.rmtree(root / 'raw')
shutil.rmtree(converted)
