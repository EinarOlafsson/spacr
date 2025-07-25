import logging, os, builtins

def silent_print(*args, **kwargs):
    if "Welcome to CellposeSAM" in str(args[0]):
        return  # skip the banner
    return _original_print(*args, **kwargs)

_original_print = builtins.print
builtins.print = silent_print


from . import core
from . import io
from . import utils
from . import settings
from . import plot
from . import measure
from . import sequencing
from . import timelapse
from . import deep_spacr
from . import gui_utils
from . import gui_elements
from . import gui_core
from . import gui
from . import app_mask
from . import app_measure
from . import app_classify
from . import app_sequencing
from . import submodules
from . import ml
from . import toxo
from . import spacr_cellpose
from . import sp_stats
from . import logger

__all__ = [
    "core",
    "io",
    "utils",
    "settings",
    "plot",
    "measure",
    "sequencing",
    "timelapse",
    "deep_spacr",
    "gui_utils",
    "gui_elements",
    "gui_core",
    "gui",
    "app_mask",
    "app_measure",
    "app_classify",
    "app_sequencing",
    "submodules",
    "ml",
    "toxo",
    "spacr_cellpose",
    "sp_stats",
    "logger"
]

logging.basicConfig(filename='spacr.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

from .utils import download_models

download_models()