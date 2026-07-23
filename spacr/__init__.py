from __future__ import annotations

import warnings as _warnings
from importlib import import_module
from typing import Final

from .version import __version__

# Third-party FutureWarnings that fire at import — noise the user
# can't act on. Silenced before the modules that trigger them import.
# (Users can re-enable with `warnings.filterwarnings("default")` in
# their own code.)
_warnings.filterwarnings(
    "ignore",
    message=r"The logit link alias is deprecated.*",
    category=FutureWarning,
    module=r"statsmodels\..*",
)
_warnings.filterwarnings(
    "ignore",
    message=r"You are using a Python version.*google\.api_core.*",
    category=FutureWarning,
)
_warnings.filterwarnings(
    "ignore",
    message=r"You are using a Python version.*",
    category=FutureWarning,
    module=r"google\..*",
)

_SUBMODULES: Final[tuple[str, ...]] = (
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
    "app_annotate",
    "app_make_masks",
    "app_mask",
    "app_measure",
    "app_classify",
    "app_sequencing",
    "app_umap",
    "submodules",
    "ml",
    "toxo",
    "spacr_cellpose",
    "spacrops",
    "sp_stats",
    "sim",
    "object",
    "cli_repro",
    "logger",
    "logging_util",
    "mask_io",
    "run_journal",
    "version",
)

__all__ = ["__version__", "download_models", *_SUBMODULES]


def __getattr__(name: str):
    """Lazily import declared submodules and the ``download_models`` helper on first access.

    :param name: Attribute name requested on the ``spacr`` package.
    :returns: Imported submodule or the ``download_models`` callable.
    :raises AttributeError: If ``name`` is neither a known submodule nor ``download_models``.
    """
    if name == "download_models":
        from .utils import download_models
        return download_models

    if name in _SUBMODULES:
        return import_module(f".{name}", __name__)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazy submodule names in ``dir(spacr)`` for tab-completion."""
    return sorted(set(globals()) | {"download_models"} | set(_SUBMODULES))