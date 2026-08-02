"""spaCR public package and version metadata."""

from __future__ import annotations

import warnings as _warnings
from importlib import import_module
from typing import Final

from .version import __version__

# Third-party FutureWarnings that fire at import — noise the user
# can't act on from inside spaCR. Silenced before the modules that trigger
# them import. The Statsmodels warning formerly listed here was fixed at its
# source by switching from the deprecated ``logit`` alias to ``Logit``.
# (Users can re-enable with `warnings.filterwarnings("default")` in
# their own code.)
_warnings.filterwarnings(
    "ignore",
    message=r"The pynvml package is deprecated\..*",
    category=FutureWarning,
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
    "schema",
    "database_schema",
    "database_concurrency",
    "io",
    "utils",
    "errors",
    "settings",
    "setting_animations",
    "plot",
    "measure",
    "measurement_schema",
    "sequencing",
    "timelapse",
    "tiff_io",
    "deep_spacr",
    "diameter",
    "feature_dict",
    "image_colors",
    "crops",
    "align",
    "convert",
    "foreign",
    "external_masks",
    "resume",
    "checkpoint",
    "cancellation",
    "zstack",
    "report",
    "train_compare",
    "hyperparam",
    "attribution",
    "agreement",
    "active_learning",
    "plate_qc",
    "seg_qc",
    "model_compare",
    "model_zoo",
    "batch",
    "batch_correction",
    "classifier_evaluation",
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
    "predictions",
    "toxo",
    "spacr_cellpose",
    "spacrops",
    "sp_stats",
    "sim",
    "object",
    "cli",
    "cli_database",
    "cli_leakage",
    "cli_plugins",
    "cli_remote",
    "cli_repro",
    "_v1_v2_bridge",
    "logger",
    "logging_util",
    "mask_io",
    # The shared filter/selection model the linked views are built on. Pure
    # pandas, no Qt, so it is usable headless and from a notebook too.
    "selection",
    # Diagnostic figures for a fitted regression.
    "regression_qc",
    # The spaCRPower port: `power_simulate` generates a synthetic pooled
    # screen, `power_model` fits the horseshoe-Poisson hit model to it. They
    # are separate modules because the simulator is cheap and dependency-free
    # while the model pulls in torch, and a parameter sweep re-runs the first
    # far more often than the second.
    "power_simulate",
    "power_model",
    "pipeline_v2",
    "plugins",
    "remote_execution",
    "run_journal",
    "notebook_export",
    "custom_features",
    "umap_annotations",
    "row_exclusions",
    "torch_artifacts",
    "validate",
    "updater",
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
