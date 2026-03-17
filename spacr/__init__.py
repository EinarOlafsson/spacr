import logging, os, builtins, sys, io

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress cellpose banner (goes to print)
_original_print = builtins.print

def silent_print(*args, **kwargs):
    if args and any(s in str(args[0]) for s in ("Welcome to CellposeSAM", "cellpose version", "platform:", "python version:", "torch version:", "CPSAM is much larger", "We encourage users")):
        return
    return _original_print(*args, **kwargs)

builtins.print = silent_print

# Persistent stderr filter for Python-level noise
class _StderrFilter(io.TextIOBase):
    def __init__(self, real):
        self._real = real
    def write(self, s):
        if any(k in s for k in ("oneDNN", "TF_ENABLE_ONEDNN", "cpu_feature_guard", "TensorFlow binary is optimized", "CellposeSAM", "CPSAM is much larger", "We encourage users")):
            return len(s)
        return self._real.write(s)
    def flush(self):
        self._real.flush()
    def fileno(self):
        return self._real.fileno()
    def isatty(self):
        return self._real.isatty()

sys.stderr = _StderrFilter(sys.stderr)

# Suppress C-level stderr (TF writes directly to fd 2, bypassing Python)
def _suppress_fd_stderr(func):
    """Call func() with OS-level stderr silenced."""
    stderr_fd = 2
    saved_fd = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fd)
    os.close(devnull)
    try:
        return func()
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)

def _do_noisy_imports():
    try:
        import tensorflow
    except ImportError:
        pass
    try:
        import cellpose
    except ImportError:
        pass

_suppress_fd_stderr(_do_noisy_imports)


from . import core
from . import io
from . import utils
from . import settings
from . import plot
from . import measure
from . import sim
from . import sequencing
from . import timelapse
from . import deep_spacr
from . import app_annotate
from . import gui_utils
from . import gui_elements
from . import gui_core
from . import gui
from . import app_make_masks
from . import app_mask
from . import app_measure
from . import app_classify
from . import app_sequencing
from . import app_umap
from . import submodules
from . import ml
from . import toxo
from . import spacr_cellpose
from . import sp_stats
from . import spacrops
from . import object
from . import logger

__all__ = [
    "core",
    "io",
    "utils",
    "settings",
    "plot",
    "measure",
    "sim",
    "sequencing",
    "timelapse",
    "deep_spacr",
    "app_annotate",
    "gui_utils",
    "gui_elements",
    "gui_core",
    "gui",
    "app_make_masks",
    "app_mask",
    "app_measure",
    "app_classify",
    "app_sequencing",
    "app_umap",
    "submodules",
    "openai",
    "ml",
    "toxo",
    "spacr_cellpose",
    "sp_stats",
    "spacrops",
    "object",
    "logger"
]

logging.basicConfig(filename='spacr.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

from .utils import download_models

download_models()