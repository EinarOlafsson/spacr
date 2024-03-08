from spacr.version import version, version_str
import logging

from . import core
from . import io
from . import utils
from . import plot
from . import measure
from . import sim
from . import timelapse
from . import train
from . import mask_app
from . import annotate_app
from . import gui_mask
from . import gui_measure
from . import gui_utils
from . import logger

__all__ = [
    "core",
    "io",
    "utils",
    "plot",
    "measure",
    "sim",
    "timelapse",
    "train",
    "mask_app",
    "annotate_app",
    "gui_utils",
    "gui_mask",
    "gui_measure",
    "logger"
]

logging.basicConfig(filename='spacr.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')
