"""spaCR public package and version metadata."""

from __future__ import annotations

import os as _os
import warnings as _warnings
from importlib import import_module

_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from ._version import __version__

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

_warnings.filterwarnings(
    "ignore",
    message=r".*[Ss]parse invariant checks are implicitly disabled",
    category=UserWarning,
    module=r"cellpose(\.|$)",
)

_DOCUMENTED_SUBMODULES: tuple[str, ...] = (
    "api",
    "core",
    "schema",
    "database_schema",
    "database_concurrency",
    "io",
    "tabular",
    "utils",
    "errors",
    "settings",
    "setting_animations",
    "settings_spec",
    "settings_advisor",
    "plot",
    "measure",
    "measure_hooks",
    "roi",
    "illumination",
    "measurement_schema",
    "sequencing",
    "sequencing_qc",
    "read_background",
    "lineage",
    "timelapse",
    "tiff_io",
    "deep_spacr",
    "diameter",
    "feature_dict",
    "image_colors",
    "crops",
    "png_list",
    "regex_infer",
    "import_plan",
    "portable_paths",
    "picture_settings",
    "well_spec",
    "align",
    "convert",
    "foreign",
    "external_masks",
    "resume",
    "restart_state",
    "checkpoint",
    "normalization",
    "intensity_rescale",
    "install_profile",
    "umap_search",
    "cancellation",
    "zstack",
    "report",
    "train_compare",
    "hyperparam",
    "attribution",
    "attribution_columns",
    "agreement",
    "annotation_power",
    "annotation_umap_qc",
    "annotation_validation",
    "active_learning",
    "curation",
    "sudoku",
    "plate_qc",
    "seg_qc",
    "model_compare",
    "image_import",
    "image_stitch",
    "model_zoo",
    "batch",
    "batch_correction",
    "classifier_evaluation",
    "classifier_quality",
    "confusion",
    "submodules",
    "ml",
    "predictions",
    "toxo",
    "spacr_cellpose",
    "spacrops",
    "sp_stats",
    "sim",
    "object",
    "object_roles",
    "object_settings_table",
    "organelle_types",
    "ome_zarr",
    "omero",
    "cli",
    "cli_database",
    "cli_workspace",
    "doctor",
    "crashreport",
    "cli_leakage",
    "cli_download",
    "example_archives",
    "cli_plugins",
    "cli_remote",
    "cli_repro",
    "_v1_v2_bridge",
    "logger",
    "logging_util",
    "mask_io",
    "layers",
    "counting",
    "napari_bridge",
    "selection",
    "regression_qc",
    "regression_failure",
    "regression_summary",
    "localisation",
    "figure_sink",
    "control_names",
    "example_data_manifest",
    "example_data",
    "columns",
    "annotation",
    "regression_backends",
    "mixed_gpu",
    "rra",
    "group_lasso",
    "baseline",
    "cell_montage",
    "plate_measurements",
    "thresholds",
    "regression_spec",
    "refit",
    "power_simulate",
    "power_model",
    "hits",
    "profiler",
    "pipeline_v2",
    "plugins",
    "remote_execution",
    "runctx",
    "run_journal",
    "run_compare",
    "macro",
    "notebook_export",
    "methods_export",
    "custom_features",
    "umap_annotations",
    "row_exclusions",
    "torch_artifacts",
    "ports",
    "artifacts",
    "pipeline_graph",
    "chaining",
    "data_manager",
    "projects",
    "validate",
    "updater",
    "version",
    "classify",
    "classify_classes",
    "crop_source",
    "benchmark",
    "column_groups",
    "filters",
    "gate_library",
    "gpu_reduce",
    "merge_tables",
    "model_check",
    "openmp_guard",
    "surrogate",
    "guide_permutation",
    "hit_attribution",
    "hit_investigation",
    "training_basis",
    "multiple_testing",
    "volcano_style",
    "guide_concordance",
    "regression_diagnostics",
    "regression_search",
    "metadata_resolution",
    "multi_database",
    "measurement_scan",
    "gene_facts",
    "gene_tile",
    "gene_measurement_compare",
    "gene_measurement_sweep",
    "guide_attribution",
    "fit_resources",
    "parameter_sweep",
    "sweep_child",
    "trial_metrics",
    "workspace",
    "figure_style",
    "style_base",
    "dependent_join",
    "graph_types",
    "outlier_filter",
    "permutation_qc",
    "response_distribution",
    "run_recommendations",
    "stream_dataset",
    "well_scope",
)


def _submodules_on_disk() -> frozenset[str]:
    """Every ``spacr/*.py`` sitting beside this file, by module name.

    Returns nothing when the sources are not on a readable filesystem -- a
    PyInstaller bundle keeps the modules inside its archive, where there is
    no directory to scan -- which is why this widens the documented tuple
    rather than replacing it.
    """
    try:
        entries = _os.listdir(_os.path.dirname(_os.path.abspath(__file__)))
    except OSError:
        return frozenset()
    return frozenset(
        name[:-3] for name in entries
        if name.endswith(".py") and name not in ("__init__.py", "__main__.py")
    )


#: What ``getattr(spacr, name)`` will import. The documented tuple is the
#: floor; the directory is the authority. A hand-kept inventory of the files
#: in its own directory had drifted four separate times, each landing a
#: module that existed but could not be reached through the package, so the
#: names are taken from the directory whenever there is one to read.
_SUBMODULES: tuple[str, ...] = tuple(sorted(
    set(_DOCUMENTED_SUBMODULES) | _submodules_on_disk()
))

__all__ = [
    "__version__",
    "download_models",
    "MaskConfig",
    "MeasureConfig",
    "run_mask",
    "run_measure",
]

_FACADE_NAMES: frozenset[str] = frozenset({
    "MaskConfig", "MeasureConfig", "run_mask", "run_measure",
})


def download_models(repo_id="einarolafsson/models", retries=5, delay=5):
    """Download spaCR's optional model files on first use.

    The implementation is imported only when called, keeping ``import spacr``
    and wildcard imports lightweight.
    """
    from .utils import download_models as _download_models
    return _download_models(repo_id=repo_id, retries=retries, delay=delay)


def __getattr__(name: str):
    """Lazily import declared submodules and the ``download_models`` helper on first access.

    :param name: Attribute name requested on the ``spacr`` package.
    :returns: Imported submodule or the ``download_models`` callable.
    :raises AttributeError: If ``name`` is neither a known submodule nor ``download_models``.
    """
    if name in _FACADE_NAMES:
        return getattr(import_module(".api", __name__), name)

    if name in _SUBMODULES:
        return import_module(f".{name}", __name__)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazy submodule names in ``dir(spacr)`` for tab-completion."""
    return sorted(set(globals()) | _FACADE_NAMES | set(_SUBMODULES))


def _silence_glyph_logging() -> None:
    """Pin fontTools at WARNING as soon as spaCR is imported.

    ``fontTools.subset`` emits about forty INFO lines for every figure saved
    -- each glyph name and glyph ID, twice, for MATH then GSUB then glyf, then
    one line per font table. A regression run saves a dozen figures, so
    thousands of lines of glyph inventory bury the run's own output, and the
    line the user is actually looking for scrolls past unread.

    ``logging_util.QUIET_LOGGERS`` lists it too, but that only applies when
    ``setup_logging()`` runs, it short-circuits on ``_INITIALISED`` if
    something configured logging first, and a script or notebook that never
    calls it gets no protection at all. Doing it at import means importing
    spaCR is sufficient, whatever the startup order.

    This sets a floor, not a lock: anyone who genuinely wants glyph traces can
    lower the level again after importing.
    """
    import logging

    for name in ("fontTools", "fontTools.subset", "fontTools.ttLib"):
        logging.getLogger(name).setLevel(logging.WARNING)


_silence_glyph_logging()
