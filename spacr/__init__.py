from spacr.version import version, version_str
import logging
import torch

from . import core
from . import io
from . import utils
from . import plot
from . import measure
from . import sim
from . import sequencing
from . import timelapse
from . import deep_spacr
from . import annotate_app
from . import gui_utils
from . import mask_app
from . import gui_mask_app
from . import gui_measure_app
from . import gui_classify_app
from . import logger


__all__ = [
    "core",
    "io",
    "utils",
    "plot",
    "measure",
    "sim",
    "sequencing"
    "timelapse",
    "deep_spacr",
    "annotate_app",
    "gui_utils",
    "mask_app",
    "gui_mask_app",
    "gui_measure_app",
    "gui_classify_app",
    "logger"
]

# Check for CUDA GPU availability
if torch.cuda.is_available():
    from . import graph_learning
    __all__.append("graph_learning")
    logging.info("CUDA GPU detected. Graph learning module loaded.")
else:
    logging.info("No CUDA GPU detected. Graph learning module not loaded.")

logging.basicConfig(filename='spacr.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')
