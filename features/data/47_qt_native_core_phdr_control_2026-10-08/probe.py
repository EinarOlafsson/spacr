"""Create a representative imported Qt/scientific process for ELF inspection."""

import json
import os
import signal
import sys

import imageio
import numpy
import pandas
import scipy
import skimage
import tifffile
from PySide6.QtWidgets import QApplication

from spacr.qt.screens import make_masks


app = QApplication([])
with open('/proc/self/maps', encoding='utf-8') as mappings:
    mapped_regions = sum(1 for _ in mappings)
print(json.dumps({
    'pid': os.getpid(),
    'executable': os.path.realpath(sys.executable),
    'maps': mapped_regions,
    'imports': [module.__name__ for module in
                (imageio, numpy, pandas, scipy, skimage, tifffile, make_masks)],
}), flush=True)
os.kill(os.getpid(), signal.SIGUSR1)
